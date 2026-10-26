"""Constant 30-calendar-day VIX from the two Observatory term structures."""
import io
import time

import numpy as np
import pandas as pd
import requests

SOURCES = {
    "TAIEX": {
        "source": "TXO Observatory",
        "url": "https://raw.githubusercontent.com/siang-dai/txo-vix-iv/main/data/vix_term_structure_history.csv",
    },
    "SPX": {
        "source": "SPX Observatory",
        "url": "https://raw.githubusercontent.com/siang-dai/spx-vix-term-structure/main/data/vix_term_structure_history.csv",
    },
}


def interpolate_vix30(frame: pd.DataFrame, expected: pd.Timestamp) -> pd.Series:
    """Interpolate total variance; keep unbracketed dates missing, never extrapolate.

    Inputs are annualized VIX percentages and calendar Days_to_Exp (including
    fractional days for SPX). With d1 <= 30 <= d2 and w=(d2-30)/(d2-d1):
    VIX30 = sqrt((w*d1*VIX1**2 + (1-w)*d2*VIX2**2) / 30).
    The shared 365-day annualization and percentage conversion cancel here.
    This uses the Observatories' available maturities, not an official index feed.
    """
    required = ["Date", "Days_to_Exp", "VIX"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Missing required term-structure fields")
    data = frame[required].copy()
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d", errors="raise")
    if data["Date"].isna().any():
        raise ValueError("Missing term-structure dates")
    data = data.loc[data["Date"] <= expected]
    for column in ["Days_to_Exp", "VIX"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column]).any():
            raise ValueError(f"Nonfinite term-structure {column}")
    if (data["VIX"] < 0).any():
        raise ValueError("Negative term-structure VIX")
    dates = pd.DatetimeIndex(data["Date"].unique()).sort_values()
    result = pd.Series(np.nan, index=dates, name="vix30", dtype=float)
    data = data.dropna().loc[lambda x: x["Days_to_Exp"] > 0].drop_duplicates()
    if data.duplicated(["Date", "Days_to_Exp"]).any():
        raise ValueError("Conflicting quotes at the same maturity")
    for date, group in data.groupby("Date"):
        curve = group.set_index("Days_to_Exp")["VIX"].sort_index()
        if 30.0 in curve.index:
            result.loc[date] = curve.loc[30.0]
            continue
        near, far = curve.loc[curve.index < 30], curve.loc[curve.index > 30]
        if near.empty or far.empty:
            continue
        d1, d2 = near.index[-1], far.index[0]
        weight = (d2 - 30) / (d2 - d1)
        result.loc[date] = np.sqrt((weight * d1 * near.iloc[-1] ** 2
                                   + (1 - weight) * d2 * far.iloc[0] ** 2) / 30)
    if result.dropna().empty or not np.isfinite(result.dropna()).all():
        raise ValueError("No valid 30-day VIX observations")
    return result


def download_vix30(asset_id: str, expected: pd.Timestamp) -> pd.Series:
    error = None
    for attempt in range(3):
        try:
            response = requests.get(SOURCES[asset_id]["url"], timeout=45)
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            return interpolate_vix30(frame, expected)
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{asset_id} 30-day VIX: {error}") from error
