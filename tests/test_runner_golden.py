"""The restructured runner must behave exactly like the Developer_Crypto runner.

Both runners are driven through the same fake exchange, fake clock and command
line; every file they write (audit log, error log, engine and runner state)
must be byte-identical.  Scenarios cover dry-run, live orders with a price
crash (stop-loss -> SHADOW -> flatten), an order lost to a disconnect
(rollback + backoff), the CSV source and the generic JSONL source.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

import efficient_cryptos.execution.binance_adapter as adapter_mod
import efficient_cryptos.execution.cli as cli_mod
import efficient_cryptos.execution.runner as runner_mod
from tests.fake_exchange import FakeClock, FakeExchange, StopRun
from tests.legacy import live_runner_d59ce9e as legacy_mod

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = [f"C{i}USDT" for i in range(10)]
START = int(pd.Timestamp("2026-01-05T00:00:07Z").value // 1_000_000)
HOUR = 3_600_000
FILES = ("audit.jsonl", "runner_errors.jsonl", "engine_state.json", "runner_state.json", "latest.json")


def _run(which: str, out: Path, monkeypatch, argv: list[str], *, hours: float, signed: bool,
         crash=None, fail_orders=frozenset(), exchange: FakeExchange | None = None) -> dict[str, str]:
    """Run one runner. Pass ``exchange`` to continue on an existing account."""
    if exchange is None:
        clock = FakeClock(START, START + int(hours * HOUR))
        exchange = FakeExchange(clock, SYMBOLS, crash=crash, fail_order_numbers=set(fail_orders))
    else:
        clock = exchange.clock
        clock.end_ms = clock.ms + int(hours * HOUR)
    exchange.proxies = {}
    _run.last_exchange = exchange
    monkeypatch.setattr(time, "time", clock.time)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    fake_dt = clock.datetime_class()
    monkeypatch.setattr(legacy_mod, "datetime", fake_dt)
    monkeypatch.setattr(runner_mod, "datetime", fake_dt)
    monkeypatch.setattr(adapter_mod.requests, "Session", lambda: exchange)
    for name in ("BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET"):
        if signed:
            monkeypatch.setenv(name, "test-" + name.lower())
        else:
            monkeypatch.delenv(name, raising=False)
    full = argv + ["--output", str(out), "--symbols", ",".join(SYMBOLS)]
    try:
        if which == "legacy":
            monkeypatch.setattr(sys, "argv", ["live_runner"] + full)
            legacy_mod.main()
        else:
            cli_mod.main(full)
    except StopRun:
        pass
    return {name: (out / name).read_text() if (out / name).exists() else "" for name in FILES}


# Monitoring-only fields the restructured runner adds to each payload. They are
# removed before the byte-for-byte comparison; everything else must match.
NEW_PAYLOAD_KEYS = ("engine_totals",)


def _without_new_keys(name: str, text: str) -> str:
    if name == "audit.jsonl":
        lines = []
        for line in text.splitlines():
            record = json.loads(line)
            for key in NEW_PAYLOAD_KEYS:
                record.pop(key, None)
            lines.append(json.dumps(record, ensure_ascii=False))
        return "".join(line + "\n" for line in lines)
    if name == "latest.json":
        record = json.loads(text)
        for key in NEW_PAYLOAD_KEYS:
            record.pop(key, None)
        return json.dumps(record, indent=2, ensure_ascii=False)
    return text


def _compare(tmp_path, monkeypatch, argv, **kwargs):
    old = _run("legacy", tmp_path / "legacy", monkeypatch, argv, **kwargs)
    new = _run("new", tmp_path / "new", monkeypatch, argv, **kwargs)
    for name in FILES:
        assert _without_new_keys(name, new[name]) == old[name], \
            f"{name} differs between legacy and restructured runner"
    if "audit.jsonl" in FILES:
        assert all("engine_totals" in json.loads(line) for line in new["audit.jsonl"].splitlines()
                   if '"engine"' in line)
    return new


LIQUIDITY = ["--realtime-factor-id", "liquidity_crypto_templates_10k_v1_liquidity_0041_5",
             "--long-percent", "20", "--short-percent", "20", "--rebalance-bars", "2"]


def test_dry_run_unsigned_matches(tmp_path, monkeypatch):
    files = _compare(tmp_path, monkeypatch, LIQUIDITY + ["--poll-seconds", "600"], hours=3, signed=False)
    assert files["audit.jsonl"].count("\n") >= 15


def test_live_orders_with_crash_and_disconnect_match(tmp_path, monkeypatch):
    crash = ("*", START + 2 * HOUR, START + 3 * HOUR, 0.80)   # whole market -20%
    files = _compare(
        tmp_path, monkeypatch,
        LIQUIDITY + ["--poll-seconds", "300", "--live-orders", "--confirm-testnet-orders",
                     "--symbol-stop", "0.05", "--shadow-min-periods", "1",
                     "--shadow-positive-periods", "1", "--shadow-recovery-pnl-pct", "0"],
        hours=8, signed=True, crash=crash, fail_orders={5},
    )
    audit = files["audit.jsonl"]
    assert '"submitted": [{' in audit                       # orders were sent
    assert "SHADOW" in audit                                 # the crash triggered the risk engine
    assert "network_error" in files["runner_errors.jsonl"]   # the injected disconnect was handled


def test_csv_source_matches(tmp_path, monkeypatch):
    csv = tmp_path / "pred.csv"
    rows = [{"timestamp": "2026-01-04T23:30:00Z", "symbol": s, "prediction": i * 0.1}
            for i, s in enumerate(SYMBOLS)]
    pd.DataFrame(rows).to_csv(csv, index=False)
    _compare(tmp_path, monkeypatch, ["--signals", str(csv), "--long-percent", "20",
                                     "--short-percent", "20", "--poll-seconds", "900",
                                     "--live-orders", "--confirm-testnet-orders"], hours=1, signed=True)


def test_jsonl_source_matches(tmp_path, monkeypatch):
    jsonl = ROOT / "examples" / "crypto_factors.example.jsonl"
    _compare(tmp_path, monkeypatch, ["--realtime-factor-jsonl", str(jsonl), "--long-percent", "20",
                                     "--short-percent", "20", "--realtime-history-bars", "60",
                                     "--poll-seconds", "1800"], hours=2, signed=False)
