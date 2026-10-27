import json
import hashlib

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


@pytest.fixture
def current_price_release(tmp_path, monkeypatch):
    output = tmp_path / "output"
    (output / "assets").mkdir(parents=True)
    dates = ["2026-10-20", "2026-10-21", "2026-10-22"]
    assets, payloads = [], []
    for asset in update.ASSETS:
        data = {"id": asset.id, "dates": dates,
                **{column: [1, 2, 3] for column in ["price", "hv", "ewma", "garch", "instantaneous"]}}
        meta = {"id": asset.id, "as_of": dates[-1], "file": f"assets/{asset.id}.json"}
        if asset.id in update.VIX30_SOURCES:
            data["vix30"] = [20, 21, None]
            meta["vix30"] = {"as_of": dates[1], "history_start": dates[0]}
        payload = update.encode(data)
        (output / meta["file"]).write_text(payload)
        assets.append(meta)
        payloads.append(payload)
    digest = hashlib.sha256("".join(payloads).encode()).hexdigest()[:16]
    (output / "manifest.json").write_text(update.encode({"assets": assets, "pairs": [], "release": digest}))
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update, "completed_session", lambda *args: pd.Timestamp(dates[-1]))
    def unexpected(*args):
        pytest.fail("Current price estimators must not be recalculated")
    monkeypatch.setattr(update, "calculate_price_metrics", unexpected)
    monkeypatch.setattr(update, "download_vix30", lambda *args: pd.Series(
        [20., 21., 22.], index=pd.to_datetime(dates), name="vix30"))
    return output


def test_late_vix_update_without_recalculating_prices(current_price_release):
    output = current_price_release
    before = {p.name: json.loads(p.read_text()) for p in (output / "assets").iterdir()}
    assert update.run()
    manifest = json.loads((output / "manifest.json").read_text())
    for meta in manifest["assets"]:
        data = json.loads((output / meta["file"]).read_text())
        old = before[f'{meta["id"]}.json']
        if meta["id"] in update.VIX30_SOURCES:
            assert data.pop("vix30") == [20, 21, 22]
            old.pop("vix30")
            assert meta["vix30"]["as_of"] == "2026-10-22"
        assert data == old
    snapshot = {p: p.read_bytes() for p in output.rglob("*.json")}
    assert not update.run()  # Identical source results do not create hourly commits.
    assert {p: p.read_bytes() for p in snapshot} == snapshot


def test_vix_failure_preserves_entire_release(current_price_release, monkeypatch):
    output = current_price_release
    snapshot = {p: p.read_bytes() for p in output.rglob("*.json")}
    download = update.download_vix30
    def fail_second(asset_id, expected):
        if asset_id == "SPX":
            raise RuntimeError("network unavailable")
        return download(asset_id, expected)
    monkeypatch.setattr(update, "download_vix30", fail_second)
    with pytest.raises(RuntimeError, match="network unavailable"):
        update.run()
    assert {p: p.read_bytes() for p in snapshot} == snapshot


def test_truncated_vix_history_is_not_published(current_price_release, monkeypatch):
    output = current_price_release
    snapshot = {p: p.read_bytes() for p in output.rglob("*.json")}
    monkeypatch.setattr(update, "download_vix30", lambda *args: pd.Series(
        [22.], index=pd.to_datetime(["2026-10-22"]), name="vix30"))
    with pytest.raises(ValueError, match="cover every published price date"):
        update.run()
    assert {p: p.read_bytes() for p in snapshot} == snapshot


def test_official_vix_gap_is_not_filled(current_price_release, monkeypatch):
    output = current_price_release
    snapshot = {p: p.read_bytes() for p in output.rglob("*.json")}
    monkeypatch.setattr(update, "download_vix30", lambda *args: pd.Series(
        [20., 22.], index=pd.to_datetime(["2026-10-20", "2026-10-22"]), name="vix30"))
    with pytest.raises(ValueError, match="cover every published price date"):
        update.run()
    assert {p: p.read_bytes() for p in snapshot} == snapshot


def test_vix_only_keeps_existing_price_dates(current_price_release, monkeypatch):
    # Newer market sessions need not be available yet in every price source.
    monkeypatch.setattr(update, "completed_session", lambda *args: pd.Timestamp("2026-10-27"))
    download = update.download_vix30
    requests = []
    def record(asset_id, expected):
        requests.append((asset_id, expected))
        return download(asset_id, expected)
    monkeypatch.setattr(update, "download_vix30", record)
    assert update.run(vix_only=True)
    assert requests == [("TAIEX", pd.Timestamp("2026-10-22")), ("SPX", pd.Timestamp("2026-10-22"))]
    manifest = json.loads((current_price_release / "manifest.json").read_text())
    assert all(a["as_of"] == "2026-10-22" for a in manifest["assets"])
    assert all(a["vix30"]["official"] for a in manifest["assets"] if a["id"] in update.VIX30_SOURCES)
