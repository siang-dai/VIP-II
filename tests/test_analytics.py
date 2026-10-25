import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

import vip2.analytics as analytics
from vip2.analytics import (asset_metrics, correlations, ewma_volatility, fourier_moment,
                            fourier_weights, historical_volatility, paired_prices, validate_prices)


def sample(n=650, seed=4):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=pd.bdate_range("2020-01-01", periods=n))


def test_hv_uses_22_returns_and_matches_direct_sample_std():
    p = sample()
    hv = historical_volatility(p)
    assert hv.first_valid_index() == p.index[22]
    expected = np.std(np.diff(np.log(p.iloc[-23:])), ddof=1) * np.sqrt(252) * 100
    assert hv.iloc[-1] == pytest.approx(expected)


def test_ewma_initialization_has_consistent_units_and_uses_latest_return():
    p = sample()
    r = np.diff(np.log(p))
    out = ewma_volatility(p)
    v = np.mean(r[:22] ** 2)
    assert out.iloc[22] == pytest.approx(np.sqrt(252 * v) * 100)
    v = .94 * v + .06 * r[22] ** 2
    assert out.iloc[23] == pytest.approx(np.sqrt(252 * v) * 100)


def test_fourier_weights_equal_independent_fft_reconstruction():
    rng = np.random.default_rng(2)
    q = rng.random(252)
    reflected = np.r_[q, q[::-1]]
    k = np.abs(np.fft.fftfreq(504) * 504)
    smoothed = np.fft.ifft(np.fft.fft(reflected) * np.maximum(1 - k / 9, 0)).real
    assert q @ fourier_weights() == pytest.approx(smoothed[251])
    assert fourier_weights().min() >= 0
    assert fourier_weights().sum() == pytest.approx(1)
    assert fourier_moment(pd.Series(np.ones(300))).iloc[-1] == pytest.approx(1)


def test_all_four_volatility_series_are_causal_and_scale_invariant():
    p = sample()
    full = asset_metrics(p)
    prefix = asset_metrics(p.iloc[:600])
    np.testing.assert_allclose(full.loc[prefix.index].iloc[:, 1:], prefix.iloc[:, 1:], rtol=1e-7, equal_nan=True)
    scaled = asset_metrics(p * 10)
    np.testing.assert_allclose(full[["hv", "ewma", "instantaneous"]], scaled[["hv", "ewma", "instantaneous"]], rtol=1e-9, equal_nan=True)
    assert (full.iloc[505:, 1:] >= 0).all().all()


def test_garch_restarts_failed_optimizer_and_accepts_only_converged_fit(monkeypatch):
    training = pd.Series(np.random.default_rng(7).normal(size=504))
    real_model = analytics.arch_model(training, mean="Zero", vol="GARCH", p=1, q=1,
                                      dist="normal", rescale=False)
    invalid = SimpleNamespace(convergence_flag=8, loglikelihood=-100.,
                              params=pd.Series([.05, .1, .85], index=["omega", "alpha[1]", "beta[1]"]),
                              conditional_volatility=np.ones(504),
                              optimization_result=SimpleNamespace(message="Positive directional derivative"))
    starts = []

    def fit(**kwargs):
        starts.append(kwargs["starting_values"])
        return invalid if len(starts) == 1 else real_model.fit(**kwargs)

    monkeypatch.setattr(analytics, "arch_model", lambda *a, **kw: SimpleNamespace(fit=fit))
    result = analytics.fit_garch(training)
    assert len(starts) == 2
    assert starts[0] is None
    assert np.all(starts[1] >= 0) and starts[1][1:].sum() < 1
    assert result.convergence_flag == 0 and np.isfinite(result.loglikelihood)


@pytest.mark.parametrize("status,params", [(8, [.05, .1, .85]), (0, [.05, .3, .9])])
def test_garch_exhausted_or_invalid_fits_abort(monkeypatch, status, params):
    calls = []
    invalid = SimpleNamespace(convergence_flag=status, loglikelihood=-100.,
                              params=pd.Series(params, index=["omega", "alpha[1]", "beta[1]"]),
                              conditional_volatility=np.ones(504),
                              optimization_result=SimpleNamespace(message="test"))

    def fit(**kwargs):
        calls.append(kwargs)
        return invalid

    monkeypatch.setattr(analytics, "arch_model", lambda *a, **kw: SimpleNamespace(fit=fit))
    with pytest.raises(RuntimeError, match="failed validation after 4 starts"):
        analytics.fit_garch(pd.Series(np.random.default_rng(7).normal(size=504)))
    assert len(calls) == 4


def test_correlations_identity_inverse_symmetry_and_bounds():
    a, b = sample(), sample(seed=8)
    same = correlations(a, a).dropna()
    np.testing.assert_allclose(same[["historical", "instantaneous"]], 1, atol=1e-10)
    inverse = correlations(a, 10000 / a).dropna()
    np.testing.assert_allclose(inverse[["historical", "instantaneous"]], -1, atol=1e-10)
    ab, ba = correlations(a, b), correlations(b, a)
    np.testing.assert_allclose(ab[["historical", "instantaneous"]], ba[["historical", "instantaneous"]], equal_nan=True)
    assert (ab[["historical", "instantaneous"]].dropna().abs() <= 1).all().all()
    short = correlations(a.iloc[:600], b.iloc[:600])
    np.testing.assert_allclose(ab.loc[short.index], short, equal_nan=True)


def test_pair_prices_align_before_return_calculation_and_do_not_fill():
    a, b = sample(), sample(seed=8).drop(sample().index[30])
    paired = paired_prices(a, b)
    assert len(paired) == len(b)
    expected = np.log(paired).diff().iloc[10:32].corr().iloc[0, 1]
    assert correlations(a, b).historical.iloc[31] == pytest.approx(expected)


def test_constant_series_correlation_is_undefined_not_zero():
    a = sample()
    out = correlations(a, pd.Series(100., index=a.index))
    assert out.historical.isna().all()
    assert out.instantaneous.isna().all()


@pytest.mark.parametrize("values", [[1, 0, 2], [1, -1, 2], [1, np.nan, 2], [1, np.inf, 2]])
def test_invalid_prices_are_rejected(values):
    with pytest.raises(ValueError):
        validate_prices(pd.Series(values))
