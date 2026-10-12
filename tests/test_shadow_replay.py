import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay", ROOT / "scripts" / "05_shadow" / "run_selected_factor_shadow.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
START = pd.Timestamp("2026-10-01T00:00:00Z")
EPOCH = pd.Timestamp("1970-01-01T00:00:00Z")


def _ms(ts):
    """Epoch milliseconds, independent of pandas' datetime resolution."""
    return (pd.DatetimeIndex(ts) - EPOCH) // pd.Timedelta(milliseconds=1)


def _write_data(root: Path, *, crash_minute: int | None = None, funding: float = 0.0):
    (root / "klines").mkdir(parents=True)
    (root / "funding").mkdir()
    minutes = pd.date_range(START, periods=6 * 60, freq="1min")
    for symbol in SYMS:
        price = pd.Series(100.0, index=minutes)
        if crash_minute is not None and symbol == "BTCUSDT":
            # BTC falls 20% inside the first holding bar, then recovers by
            # the 30m close, so only an intrabar mark can see it.
            price.iloc[crash_minute:crash_minute + 3] = 80.0
        pd.DataFrame({
            "open_time": _ms(minutes),
            "open": price.values, "high": price.values, "low": price.values,
            "close": price.values, "quote_volume": 1e9,
        }).to_parquet(root / "klines" / f"{symbol}.parquet")
        pd.DataFrame({
            "funding_time": list(_ms([pd.Timestamp("2026-10-01T02:00:00Z")])),
            "funding_rate": [funding if symbol == "BTCUSDT" else 0.0],
        }).to_parquet(root / "funding" / f"{symbol}.parquet")


def _write_predictions(path: Path):
    rows = []
    for k in range(10):
        ts = START + pd.Timedelta(minutes=30 * k)
        for i, symbol in enumerate(SYMS):
            # BTC always ranks top -> long BTC, short XRP (25% of 4 = 1 each side).
            rows.append({"timestamp": ts.isoformat(), "symbol": symbol, "prediction": [4, 3, 2, 1][i]})
    pd.DataFrame(rows).to_csv(path, index=False)


def _run(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(replay, "SYMBOLS", SYMS)
    preds = tmp_path / "preds"
    preds.mkdir()
    _write_predictions(preds / "f1.csv")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "replay", "--predictions-dir", str(preds), "--data-dir", str(tmp_path / "data"),
        "--out-dir", str(out), "--rebalance-hours", "0.5,1",
        "--long-pct", "25", "--short-pct", "25", *extra,
    ])
    replay.main()
    return out


def test_intrabar_stop_is_seen_by_1m_marks(tmp_path, monkeypatch):
    _write_data(tmp_path / "data", crash_minute=40)
    out = _run(tmp_path, monkeypatch)
    meta = json.loads((out / "0.5h" / "f1.metrics.json").read_text())
    assert meta["shadow_entries"] >= 1
    ts = pd.read_csv(out / "0.5h" / "f1.timeseries.csv")
    marks = ts[ts["kind"] == "mark"]
    # The crash is caught at the 00:41 1m mark, before the 01:00 bar close.
    # It is large enough to hit the account-level daily stop first.
    assert marks["risk_events"].fillna("").str.contains("DAILY_ACCOUNT_HALT").any()
    first = pd.Timestamp(marks["timestamp"].iloc[0])
    assert first.hour == 0 and 40 <= first.minute < 50


def test_without_marks_the_same_crash_is_invisible(tmp_path, monkeypatch):
    _write_data(tmp_path / "data", crash_minute=40)
    out = _run(tmp_path, monkeypatch, "--mark-minutes", "0")
    meta = json.loads((out / "0.5h" / "f1.metrics.json").read_text())
    assert meta["shadow_entries"] == 0


def test_funding_is_booked_and_metrics_are_reported(tmp_path, monkeypatch):
    _write_data(tmp_path / "data", funding=0.001)
    out = _run(tmp_path, monkeypatch)
    meta = json.loads((out / "1h" / "f1.metrics.json").read_text())
    # Only BTC (the long) has positive funding at 02:00, so the long pays.
    assert meta["total_funding"] < 0
    assert meta["total_cost"] > 0
    assert meta["max_drawdown_pct"] <= 0
    assert "sharpe_annualized" in meta
    summary = pd.read_csv(out / "1h" / "summary.csv")
    assert list(summary["factor_id"]) == ["f1"]


def test_fractional_bar_rebalance_is_rejected(tmp_path, monkeypatch):
    _write_data(tmp_path / "data")
    with pytest.raises(SystemExit):
        _run(tmp_path, monkeypatch, "--rebalance-hours", "0.75")
