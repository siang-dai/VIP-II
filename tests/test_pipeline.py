import json

import numpy as np
import pandas as pd
import pytest

from vip2 import update
from vip2.sources import clean_prices, completed_session


def test_market_close_delay_and_weekends():
    # Monday: Taiwan has closed, New York has not opened yet.
    now = pd.Timestamp("2025-06-09T08:00:00Z")
    assert completed_session("XTAI", now) == pd.Timestamp("2025-06-09")
    assert completed_session("XNYS", now) == pd.Timestamp("2025-06-06")
    assert completed_session("XNYS", pd.Timestamp("2025-06-09T21:00:00Z")) == pd.Timestamp("2025-06-06")
    assert completed_session("XNYS", pd.Timestamp("2025-06-09T23:00:00Z")) == pd.Timestamp("2025-06-09")


def test_taiwan_exceptional_closure_and_observed_holiday():
    assert completed_session("XTAI", pd.Timestamp("2026-10-25T12:00:00Z")) == pd.Timestamp("2026-10-22")
    assert completed_session("XTAI", pd.Timestamp("2026-10-26T12:00:00Z")) == pd.Timestamp("2026-10-22")


def test_source_rejects_stale_and_unadjusted_missing_field():
    dates = pd.bdate_range("2016-01-01", "2026-01-01")
    frame = pd.DataFrame({"Close": np.arange(len(dates)) + 100}, index=dates)
    assert clean_prices(frame, "Close", dates[-1]).index[-1] == dates[-1]
    with pytest.raises(ValueError, match="stale"):
        clean_prices(frame, "Close", dates[-1] + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="required field"):
        clean_prices(frame, "Adj Close", dates[-1])


def test_failed_download_preserves_last_good_release(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    manifest = tmp_path / "output/manifest.json"
    manifest.write_text('{"assets": [], "release": "last-good"}')
    monkeypatch.setattr(update, "ROOT", tmp_path)
    def fail(*args):
        raise RuntimeError("network unavailable")
    monkeypatch.setattr(update, "download_asset", fail)
    with pytest.raises(RuntimeError, match="network unavailable"):
        update.run(force=True)
    assert json.loads(manifest.read_text())["release"] == "last-good"
    assert list((tmp_path / "output").iterdir()) == [manifest]


def test_json_is_compact_strict_and_preserves_missing_values():
    df = pd.DataFrame({"x": [np.nan, .12345678]}, index=pd.date_range("2024-01-01", periods=2))
    value = json.loads(update.encode(update.table(df)))
    assert value["x"] == [None, .12346]
