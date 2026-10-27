"""Official daily 30-calendar-day volatility indices, keyed by local date."""
import io
import json
import time

import numpy as np
import pandas as pd
import requests

from .sources import CBOE_URL

SOURCES = {
    "TAIEX": {
        "source": "TAIFEX",
        "index_name": "TAIWAN VIX",
        "url": "https://www.taifex.com.tw/indes/index.aspx/GetStockDayPrices",
        "reference_url": "https://www.taifex.com.tw/indes/index.aspx",
    },
    "SPX": {
        "source": "Cboe",
        "index_name": "Cboe VIX",
        "url": CBOE_URL,
        "reference_url": "https://www.cboe.com/tradable-products/vix/vix-historical-data",
    },
}


def clean_official_vix(dates, values, date_format: str, expected: pd.Timestamp) -> pd.Series:
    """Keep source trading dates as dates; never shift, fill or interpolate them."""
    index = pd.DatetimeIndex(pd.to_datetime(dates, format=date_format, errors="raise"))
    series = pd.Series(pd.to_numeric(values, errors="raise"), index=index, name="vix30", dtype=float)
    if index.isna().any() or index.has_duplicates:
        raise ValueError("Invalid or duplicate official VIX dates")
    series = series.loc[(index >= expected - pd.DateOffset(years=5)) & (index <= expected)].sort_index()
    if series.empty or not np.isfinite(series).all() or (series <= 0).any():
        raise ValueError("Missing or invalid official VIX observations")
    return series


def parse_taiwan_vix(payload: dict, expected: pd.Timestamp) -> pd.Series:
    data = payload.get("d")
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict) or not isinstance(data.get("TrendData"), list):
        raise ValueError("Missing official TAIWAN VIX daily history")
    rows = data["TrendData"]
    try:
        return clean_official_vix([row["Time"] for row in rows], [row["Price"] for row in rows],
                                  "%Y%m%d", expected)
    except KeyError as exc:
        raise ValueError("Missing official TAIWAN VIX date/price fields") from exc


def parse_cboe_vix(text: str, expected: pd.Timestamp) -> pd.Series:
    frame = pd.read_csv(io.StringIO(text))
    if not {"DATE", "CLOSE"}.issubset(frame.columns):
        raise ValueError("Missing official Cboe VIX date/close fields")
    return clean_official_vix(frame["DATE"].tolist(), frame["CLOSE"].tolist(), "%m/%d/%Y", expected)


def download_vix30(asset_id: str, expected: pd.Timestamp) -> pd.Series:
    error = None
    for attempt in range(3):
        try:
            url = SOURCES[asset_id]["url"]
            if asset_id == "TAIEX":
                response = requests.post(url, json={
                    "syid": "TAIWANVIX", "flag": "MS",
                    "startDate": (expected - pd.DateOffset(years=5)).strftime("%Y/%m/%d"),
                    "endDate": expected.strftime("%Y/%m/%d"),
                }, timeout=45)
                response.raise_for_status()
                return parse_taiwan_vix(response.json(), expected)
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            return parse_cboe_vix(response.text, expected)
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{asset_id} official 30-day VIX: {error}") from error
