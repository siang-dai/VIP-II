"""Six estimators. Inputs are positive daily prices, never filled across holidays.

Fourier estimates use full-band covariance coefficients and Fejer reconstruction
on a *reflected* trailing window. Reflection avoids wrapping the oldest return
onto the endpoint. The resulting common nonnegative weights preserve PSD.
This is a documented daily-data adaptation, not a numerical copy of VIP I.
All values at t use data at or before t, including the GARCH refit schedule.
"""
import logging
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from arch import arch_model
from numpy.lib.stride_tricks import sliding_window_view

from .config import (ANNUALIZATION, EWMA_DECAY, FOURIER_MODES, FOURIER_WINDOW,
                     GARCH_REFIT, GARCH_WINDOW, HV_WINDOW)


def validate_prices(prices: pd.Series) -> pd.Series:
    p = prices.astype(float).sort_index()
    if p.index.has_duplicates or p.empty:
        raise ValueError("Prices must have unique dates and at least one observation")
    if not np.isfinite(p.to_numpy()).all() or (p <= 0).any():
        raise ValueError("Prices must be finite and strictly positive")
    return p


def returns(prices: pd.Series) -> pd.Series:
    return np.log(validate_prices(prices)).diff()


def historical_volatility(prices: pd.Series, window: int = HV_WINDOW) -> pd.Series:
    return returns(prices).rolling(window).std(ddof=1) * np.sqrt(ANNUALIZATION) * 100


def ewma_volatility(prices: pd.Series) -> pd.Series:
    r = returns(prices)
    out = pd.Series(np.nan, index=r.index)
    if len(r) <= HV_WINDOW:
        return out
    variance = float(np.mean(np.square(r.iloc[1:HV_WINDOW + 1])))
    out.iloc[HV_WINDOW] = np.sqrt(ANNUALIZATION * variance) * 100
    for i in range(HV_WINDOW + 1, len(r)):
        variance = EWMA_DECAY * variance + (1 - EWMA_DECAY) * r.iloc[i] ** 2
        out.iloc[i] = np.sqrt(ANNUALIZATION * variance) * 100
    return out


def fit_garch(training: pd.Series):
    """Fit unit-scaled returns, retrying SLSQP from valid deterministic starts.

    A different starting point can resolve a boundary-related optimizer failure.
    Every accepted result must still converge and satisfy the GARCH constraints;
    retries never replace a failed fit with fixed parameters or another model.
    """
    model = arch_model(training, mean="Zero", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    second_moment = float(np.mean(training ** 2))
    starts = [None] + [np.array([second_moment * (1 - a - b), a, b])
                       for a, b in ((.10, .85), (.10, .70), (.25, .50))]
    failure = "invalid fit"
    for attempt, start in enumerate(starts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = model.fit(starting_values=start, disp="off", show_warning=False,
                               options={"maxiter": 3000})
        params = fitted.params[["omega", "alpha[1]", "beta[1]"]].to_numpy()
        conditional = np.asarray(fitted.conditional_volatility)
        valid = (fitted.convergence_flag == 0 and np.isfinite(fitted.loglikelihood)
                 and np.isfinite(params).all() and (params >= 0).all()
                 and params[1:].sum() <= 1 + 1e-8
                 and np.isfinite(conditional).all() and (conditional > 0).all())
        if valid:
            if attempt:
                logging.getLogger(__name__).warning("GARCH converged after %d restart(s), window ending %s",
                                                    attempt, training.index[-1])
            return fitted
        failure = f"status {fitted.convergence_flag}: {fitted.optimization_result.message}"
    raise RuntimeError(f"GARCH fit failed validation after {len(starts)} starts ({failure})")


def garch_volatility(prices: pd.Series) -> pd.Series:
    """Next-session forecast after each close; trailing 504d fit every 21 sessions.

    Fit through t-1, update variance with t's return, and label forecast at its
    origin t. No future observation is used to estimate historical parameters.
    Failed optimization is an error, not a silently substituted estimate.
    """
    r = returns(prices).iloc[1:] * 100
    out = pd.Series(np.nan, index=prices.index)
    omega = alpha = beta = variance = np.nan
    for i in range(GARCH_WINDOW, len(r)):
        if (i - GARCH_WINDOW) % GARCH_REFIT == 0:
            training = r.iloc[i - GARCH_WINDOW:i]
            if np.std(training) < 1e-9:
                omega, alpha, beta, variance = 0., 0., 0., 0.
            else:
                # Unit-variance training improves optimizer conditioning for both
                # low-volatility bonds and high-volatility VIX. Convert back to
                # percent-return units before applying the daily recursion.
                scale = float(training.std(ddof=0))
                try:
                    fitted = fit_garch(training / scale)
                except RuntimeError as exc:
                    raise RuntimeError(f"GARCH did not converge at {r.index[i]}: {exc}") from exc
                omega, alpha, beta = (float(fitted.params[k]) for k in
                                      ("omega", "alpha[1]", "beta[1]"))
                omega *= scale ** 2
                variance = omega + alpha * training.iloc[-1] ** 2 + beta * (fitted.conditional_volatility.iloc[-1] * scale) ** 2
        variance = omega + alpha * r.iloc[i] ** 2 + beta * variance
        if not np.isfinite(variance) or variance < 0:
            raise RuntimeError("Invalid GARCH variance")
        out.loc[r.index[i]] = np.sqrt(variance * ANNUALIZATION)
    return out


@lru_cache(maxsize=8)
def fourier_weights(window: int = FOURIER_WINDOW, modes: int = FOURIER_MODES) -> np.ndarray:
    """Endpoint weights from a Fejer Fourier multiplier on reflected observations.

    For q = [r0²,...,r(n-1)²,r(n-1)²,...,r0²], smooth with frequency multiplier
    (1-|k|/(M+1)) for |k|<=M and evaluate at n-1. The FFT convolution theorem
    gives identical weights for r², s² and r*s, hence a PSD covariance matrix.
    """
    if window < 2 or not 0 < modes < window:
        raise ValueError("Require 0 < modes < window")
    size = 2 * window
    frequency = np.abs(np.fft.fftfreq(size) * size)
    multiplier = np.maximum(1 - frequency / (modes + 1), 0)
    kernel = np.fft.ifft(multiplier).real
    j = np.arange(window)
    weights = kernel[(window - 1 - j) % size] + kernel[(window - 1 - (size - 1 - j)) % size]
    if weights.min() < -1e-12:
        raise ValueError("Fejer kernel must be nonnegative")
    weights = np.maximum(weights, 0)
    return weights / weights.sum()


def fourier_moment(products: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=products.index)
    if len(products) >= FOURIER_WINDOW:
        windows = sliding_window_view(products.to_numpy(), FOURIER_WINDOW)
        out.iloc[FOURIER_WINDOW - 1:] = windows @ fourier_weights()
    return out


def instantaneous_volatility(prices: pd.Series) -> pd.Series:
    variance = fourier_moment(returns(prices) ** 2) * ANNUALIZATION
    return np.sqrt(variance) * 100


def paired_prices(first: pd.Series, second: pd.Series) -> pd.DataFrame:
    # Intersect prices BEFORE differencing: both returns then span identical dates.
    return pd.concat([validate_prices(first).rename("a"), validate_prices(second).rename("b")], axis=1).dropna()


def correlations(first: pd.Series, second: pd.Series) -> pd.DataFrame:
    prices = paired_prices(first, second)
    r = np.log(prices).diff()
    historical = r.a.rolling(HV_WINDOW).corr(r.b)
    aa, bb, ab = (fourier_moment(v) for v in (r.a ** 2, r.b ** 2, r.a * r.b))
    denominator = np.sqrt(aa * bb)
    spot = ab / denominator.where(denominator > 1e-20)
    # Only round-off is bounded; substantive violations fail validation.
    if (spot.dropna().abs() > 1 + 1e-8).any():
        raise RuntimeError("Fourier covariance lost positive semidefiniteness")
    return pd.DataFrame({"price_a": prices.a, "price_b": prices.b,
                         "historical": historical.clip(-1, 1), "instantaneous": spot.clip(-1, 1)})


def asset_metrics(prices: pd.Series) -> pd.DataFrame:
    p = validate_prices(prices)
    return pd.DataFrame({"price": p, "hv": historical_volatility(p),
                         "ewma": ewma_volatility(p), "garch": garch_volatility(p),
                         "instantaneous": instantaneous_volatility(p)})
