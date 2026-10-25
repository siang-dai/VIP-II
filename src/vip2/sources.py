"""Daily source adapters; fail closed on truncated or stale responses."""
import io
import time
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .analytics import validate_prices
from .config import Asset, HISTORY_START

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
# Known closures not yet present in the bundled exchange calendar.
# 2026-10-23: typhoon closure, https://www.taifex.com.tw/cht/11/announcement
# 2026-10-26: observed Retrocession Day, https://wwwc.twse.com.tw/zh/trading/holiday.html
EXTRA_CLOSURES = {"XTAI": {"2026-10-23", "2026-10-26"}}


def completed_session(calendar: str, now: pd.Timestamp) -> pd.Timestamp:
    now = pd.Timestamp(now).tz_convert("UTC")
    cal = xcals.get_calendar(calendar, start=(now - pd.Timedelta(days=30)).date(), end=(now + pd.Timedelta(days=10)).date())
    closes = cal.schedule["close"]
    # Allow two hours for EOD sources to settle, including VIX's later close.
    eligible = closes[closes + pd.Timedelta(hours=2) <= now]
    eligible = eligible[~eligible.index.strftime("%Y-%m-%d").isin(EXTRA_CLOSURES.get(calendar, set()))]
    if eligible.empty:
        raise RuntimeError("No completed session in calendar")
    return pd.Timestamp(eligible.index[-1]).tz_localize(None).normalize()


def clean_prices(frame: pd.DataFrame, column: str, expected: pd.Timestamp) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Missing required field {column}")
    p = frame[column].copy()
    p.index = pd.DatetimeIndex(p.index).tz_localize(None).normalize()
    p = p.loc[(p.index >= pd.Timestamp(HISTORY_START)) & (p.index <= expected)].sort_index()
    # Reject missing prices, duplicate dates and invalid data; never forward-fill.
    p = validate_prices(p)
    if len(p) < 1800 or p.index.min() > expected - pd.DateOffset(years=7):
        raise ValueError("Source history is too short for 5Y display plus model warm-up")
    if p.index[-1] != expected:
        raise ValueError(f"Source is stale: {p.index[-1].date()}, expected {expected.date()}")
    if np.log(p).diff().abs().max() > 1.2:
        raise ValueError("Extreme price discontinuity; inspect source adjustments")
    return p


def download_asset(asset: Asset, expected: pd.Timestamp, cache: Path) -> tuple[pd.Series, str]:
    cache.mkdir(exist_ok=True, parents=True)
    yf.set_tz_cache_location(str(cache / "yfinance"))
    error = None
    for attempt in range(3):
        try:
            if asset.id == "VIX":
                response = requests.get(CBOE_URL, timeout=45)
                response.raise_for_status()
                frame = pd.read_csv(io.StringIO(response.text))
                frame.index = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
                return clean_prices(frame, "CLOSE", expected), "Cboe daily VIX history"
            ticker = yf.Ticker(asset.symbol)
            frame = ticker.history(start=HISTORY_START, end=(expected + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                   interval="1d", auto_adjust=False, actions=False, raise_errors=True,
                                   repair=False, timeout=45)
            # Some providers repeat the previous close on an exceptional closure.
            frame = frame.loc[~frame.index.strftime("%Y-%m-%d").isin(EXTRA_CLOSURES.get(asset.calendar, set()))]
            return clean_prices(frame, "Adj Close" if asset.adjusted else "Close", expected), "Yahoo Finance via yfinance"
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{asset.id}: {error}") from error
