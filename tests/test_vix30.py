import numpy as np
import pandas as pd
import pytest

from vip2.vix30 import interpolate_vix30


def curve(days, values, date="2026-10-22"):
    return pd.DataFrame({"Date": date, "Days_to_Exp": days, "VIX": values})


def test_uses_nearest_brackets_and_total_variance_not_volatility_average():
    frame = curve([90, 40, 5, 20], [80, 30, 60, 20])
    result = interpolate_vix30(frame, pd.Timestamp("2026-10-22"))
    # Equal time weights, but first interpolate accumulated variance in years.
    expected = 100 * np.sqrt((0.5 * (20 / 365) * .2**2
                              + 0.5 * (40 / 365) * .3**2) / (30 / 365))
    assert result.iloc[0] == pytest.approx(expected)
    assert result.iloc[0] != pytest.approx(25)


def test_fractional_maturities_and_flat_curve():
    days = [27.760417, 28.03125, 31.03125, 35.03125]
    vols = [19.5593, 19.6289, 19.2165, 19.2593]
    result = interpolate_vix30(curve(days, vols), pd.Timestamp("2026-10-22"))
    accumulated = np.array(days) / 365 * (np.array(vols) / 100)**2
    expected = 100 * np.sqrt(np.interp(30, days, accumulated) / (30 / 365))
    assert result.iloc[0] == pytest.approx(expected)
    flat = interpolate_vix30(curve([27, 55], [31, 31]), pd.Timestamp("2026-10-22"))
    assert flat.iloc[0] == pytest.approx(31)


def test_exact_thirty_days_missing_brackets_and_future_dates():
    frame = pd.concat([
        curve([30], [22], "2026-10-20"),
        curve([1, 20], [40, 25], "2026-10-21"),
        curve([35, 55], [21, 23], "2026-10-22"),
        curve([30], [999], "2026-10-23"),
    ])
    result = interpolate_vix30(frame, pd.Timestamp("2026-10-22"))
    assert result.index.tolist() == list(pd.date_range("2026-10-20", periods=3))
    assert result.iloc[0] == 22
    assert result.iloc[1:].isna().all()  # No extrapolation or forward filling.


def test_missing_quote_does_not_hide_valid_brackets():
    frame = curve([0, 20, 29, 40], [100, 20, np.nan, 30])
    result = interpolate_vix30(frame, pd.Timestamp("2026-10-22"))
    assert result.iloc[0] == pytest.approx(np.sqrt(22000 / 30))


@pytest.mark.parametrize("frame, message", [
    (pd.DataFrame({"Date": []}), "required"),
    (curve([20, 40], [20, np.inf]), "Nonfinite"),
    (curve([20, 40], [-20, 30]), "Negative"),
    (curve([20, 20, 40], [20, 21, 30]), "Conflicting"),
    (curve([1, 20], [20, 30]), "No valid"),
])
def test_rejects_invalid_sources(frame, message):
    with pytest.raises(ValueError, match=message):
        interpolate_vix30(frame, pd.Timestamp("2026-10-22"))
