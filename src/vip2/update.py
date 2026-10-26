"""Publish small per-asset / per-pair JSON files, preserving the last good release."""
import argparse
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import asset_metrics, correlations
from .config import ASSETS
from .sources import completed_session, download_asset
from .vix30 import SOURCES as VIX30_SOURCES, download_vix30

ROOT = Path(__file__).resolve().parents[2]


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def table(frame: pd.DataFrame) -> dict:
    return {"dates": frame.index.strftime("%Y-%m-%d").tolist(), **{
        key: [round(float(x), 5) if np.isfinite(x) else None for x in frame[key]] for key in frame}}


def validate_metrics(frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Invalid metric dates")
    for col in columns:
        if not np.isfinite(frame[col].to_numpy()).all():
            raise ValueError(f"Missing/nonfinite values in published {col}")


def calculate_price_metrics(expected: dict) -> tuple[list, list, dict]:
    prices, assets, staged = {}, [], {}
    for asset in ASSETS:
        print(f"Fetching {asset.id}, completed session {expected[asset.id].date()}", flush=True)
        p, source = download_asset(asset, expected[asset.id], ROOT / ".cache")
        prices[asset.id] = p
        calculated = asset_metrics(p)
        cutoff = p.index[-1] - pd.DateOffset(years=5)
        shown = calculated.loc[calculated.index >= cutoff]
        validate_metrics(shown, ["price", "hv", "ewma", "garch", "instantaneous"])
        file = f"assets/{asset.id}.json"
        staged[file] = encode({"id": asset.id, **table(shown)})
        assets.append({**asdict(asset), "source": source, "as_of": p.index[-1].strftime("%Y-%m-%d"),
                       "history_start": p.index[0].strftime("%Y-%m-%d"), "file": file})
        print(f"Calculated {asset.id}: {len(shown)} published days", flush=True)

    pairs = []
    for a, b in itertools.combinations(ASSETS, 2):
        frame = correlations(prices[a.id], prices[b.id])
        shown = frame.loc[frame.index >= frame.index[-1] - pd.DateOffset(years=5)]
        validate_metrics(shown, ["price_a", "price_b", "historical", "instantaneous"])
        file = f"pairs/{a.id}--{b.id}.json"
        staged[file] = encode({"a": a.id, "b": b.id, **table(shown)})
        pairs.append({"a": a.id, "b": b.id, "file": file, "as_of": shown.index[-1].strftime("%Y-%m-%d"),
                      "cross_market": a.calendar != b.calendar})
    return assets, pairs, staged


def run(force: bool = False) -> bool:
    now = pd.Timestamp.now(tz="UTC")
    output = ROOT / "output"
    manifest_path = output / "manifest.json"
    expected = {a.id: completed_session(a.calendar, now) for a in ASSETS}
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous = {a["id"]: a for a in old.get("assets", [])}
    prices_current = all(previous.get(a.id, {}).get("as_of") == expected[a.id].strftime("%Y-%m-%d") for a in ASSETS)
    if not force and prices_current:
        assets, pairs = old["assets"], old["pairs"]
        staged = {item["file"]: (output / item["file"]).read_text(encoding="utf-8") for item in assets + pairs}
    else:
        assets, pairs, staged = calculate_price_metrics(expected)

    # Observatory updates can arrive after prices: check them even when the
    # price dates are unchanged, without refitting any of the existing models.
    for asset in assets:
        asset_id = asset["id"]
        if asset_id not in VIX30_SOURCES:
            continue
        series = download_vix30(asset_id, expected[asset_id])
        data = json.loads(staged[asset["file"]])
        aligned = series.reindex(pd.to_datetime(data["dates"]))
        valid = aligned.dropna()
        if valid.empty:
            raise ValueError(f"{asset_id}: no 30-day VIX on published price dates")
        prior = previous.get(asset_id, {}).get("vix30", {})
        first, last = valid.index[0].strftime("%Y-%m-%d"), valid.index[-1].strftime("%Y-%m-%d")
        if prior and (last < prior["as_of"] or first > max(prior["history_start"], data["dates"][0])):
            raise ValueError(f"{asset_id}: truncated 30-day VIX history")
        data["vix30"] = table(aligned.to_frame())["vix30"]
        staged[asset["file"]] = encode(data)
        asset["vix30"] = {**VIX30_SOURCES[asset_id], "history_start": first, "as_of": last,
                          "interpolation": "linear total variance", "horizon_days": 30}
        print(f"{asset_id} 30-day VIX: {len(valid)} observations, latest {last}", flush=True)

    digest = hashlib.sha256("".join(staged.values()).encode()).hexdigest()[:16]
    if not force and digest == old.get("release"):
        print("All completed sessions and 30-day VIX observations are already published.")
        return False
    manifest = {"schema": 1, "release": digest, "generated_at": now.isoformat(timespec="seconds"),
                "frequency": "daily", "display_years": 5, "assets": assets, "pairs": pairs,
                "methods": {"hv_window": 22, "ewma_lambda": 0.94, "garch_window": 504,
                            "garch_refit_sessions": 21, "garch_output": "next-session forecast",
                            "fourier_window": 252, "fourier_modes": 8,
                            "fourier_boundary": "reflected, trailing-only", "annualization": 252},
                "attribution": "https://github.com/ErikApostol/volatility_index_platform"}
    # No output is touched until every download, model and pair has passed validation.
    for file, content in staged.items():
        path = output / file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    manifest_path.write_text(encode(manifest), encoding="utf-8")
    print(f"Published {len(assets)} assets and {len(pairs)} pairs; {sum(len(x.encode()) for x in staged.values()):,} bytes total.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recalculate even if session dates are unchanged")
    run(parser.parse_args().force)
