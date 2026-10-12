"""Alpha101/191 and ML sources plug into the same runner without re-sizing."""
import json
import signal
from pathlib import Path

import pytest

import efficient_cryptos.execution.runner as runner_mod
import efficient_cryptos.execution.sources.alpha_scanner as alpha_mod
import efficient_cryptos.common.binance_client as client_mod
from efficient_cryptos.execution.sources.base import AlphaSource, Sizing
from tests.test_runner_golden import SYMBOLS, _run

ROOT = Path(__file__).resolve().parents[1]
SIZING = Sizing(gross_leverage=3, margin_buffer_pct=0.2, long_percent=10, short_percent=10, min_assets=2)


class _Client:
    def close(self):
        pass


@pytest.fixture
def fake_alpha(monkeypatch):
    calls = []

    def scan(client, store, cfg, *, equity, workers=8):
        calls.append(equity)
        return {SYMBOLS[0]: 0.4 * equity, SYMBOLS[1]: 0.2 * equity, SYMBOLS[9]: -0.6 * equity}
    monkeypatch.setattr(alpha_mod, "scan_alpha_once", scan)
    monkeypatch.setattr(client_mod, "make_futures_client", lambda **kw: _Client())
    return calls


def test_alpha_source_keeps_its_own_sizing_and_caches_per_bar(fake_alpha, monkeypatch):
    import time
    monkeypatch.setattr(time, "time", lambda: 1_767_571_207.0)
    source = AlphaSource(ROOT / "config" / "default.yaml")
    first = source.fetch(None, SYMBOLS, 10_000.0, SIZING)
    second = source.fetch(None, SYMBOLS, 10_000.0, SIZING)
    assert first is second and len(fake_alpha) == 1                    # one scan per bar
    assert fake_alpha[0] == pytest.approx(8_000.0)                      # equity * (1 - buffer)
    assert {r["symbol"]: r["target_notional"] for r in first.rows} == {
        SYMBOLS[0]: 3_200.0, SYMBOLS[1]: 1_600.0, SYMBOLS[9]: -4_800.0}
    assert source.factor_id == "alpha101_alpha020"


def test_runner_trades_alpha_targets(fake_alpha, tmp_path, monkeypatch):
    out = tmp_path / "run"
    _run("new", out, monkeypatch, ["--alpha-config", str(ROOT / "config" / "default.yaml"),
                                   "--poll-seconds", "600", "--live-orders", "--confirm-testnet-orders"],
         hours=1, signed=True)
    last = json.loads((out / "audit.jsonl").read_text().splitlines()[-1])
    held = {p["symbol"]: float(p["positionAmt"]) for p in last["exchange_positions"]}
    assert held[SYMBOLS[0]] > 0 and held[SYMBOLS[1]] > 0 and held[SYMBOLS[9]] < 0
    assert set(held) == {SYMBOLS[0], SYMBOLS[1], SYMBOLS[9]}


def test_ml_source_uses_rank_targets(monkeypatch):
    pytest.importorskip("torch")
    import efficient_cryptos.execution.sources.ml_model as ml
    from efficient_cryptos.execution.sources.base import MlSource
    source = object.__new__(MlSource)
    source.bundle = type("B", (), {"args": {"interval": "30m"}})()
    source.k_long, source.m_short, source.history_bars = 2, 1, 260
    source.factor_id, source.proxy, source._cache, source._cache_key = "ml:test", None, None, None
    monkeypatch.setattr(client_mod, "make_futures_client", lambda **kw: _Client())
    monkeypatch.setattr(ml, "scan_ml_once", lambda client, bundle, **kw: ml.SignalResult(
        signal_time_ms=1_767_570_000_000, scores={"A": 3.0, "B": 2.0, "C": 1.0},
        targets=ml.rank_targets({"A": 3.0, "B": 2.0, "C": 1.0}, equity=kw["equity"],
                                k_long=2, m_short=1, gross_leverage=kw["gross_leverage"])))
    batch = source.fetch(None, [], 10_000.0, SIZING)
    assert {r["symbol"]: r["target_notional"] for r in batch.rows} == {"A": 6_000.0, "B": 6_000.0, "C": -12_000.0}


def test_sigterm_finishes_cycle_and_closes_on_exit(tmp_path, monkeypatch):
    """A stop request ends the loop after the current cycle and flattens the book."""
    original = runner_mod.LiveRunner.run_cycle
    def run_cycle(self):
        payload = original(self)
        handler = signal.getsignal(signal.SIGTERM)
        if callable(handler) and payload["exchange_positions"]:
            handler(signal.SIGTERM, None)
        return payload
    monkeypatch.setattr(runner_mod.LiveRunner, "run_cycle", run_cycle)
    out = tmp_path / "run"
    _run("new", out, monkeypatch, ["--realtime-factor-id", "liquidity_crypto_templates_10k_v1_liquidity_0041_5",
                                   "--long-percent", "20", "--short-percent", "20", "--poll-seconds", "300",
                                   "--live-orders", "--confirm-testnet-orders", "--close-on-exit"],
         hours=4, signed=True)
    exchange = _run.last_exchange
    lines = (out / "audit.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["type"] == "close_all"
    assert not any(p["qty"] for p in exchange.positions.values())
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.default_int_handler)
