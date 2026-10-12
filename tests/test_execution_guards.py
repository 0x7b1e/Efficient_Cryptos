"""Exchange-side guards: verbatim Predict_Cryptos risk controls + runner wiring."""
import json

import pandas as pd
import pytest

import efficient_cryptos.execution.cli as cli_mod
from efficient_cryptos.execution.guards import (
    ManualChangeGuard, _apply_live_risk_controls, _map_targets_for_execution,
    _snapshot_changes, position_snapshot,
)
from tests.test_runner_golden import HOUR, LIQUIDITY, START, _run


# ---------------------------------------------------- ported unit tests ----
def _cfg(state_file, take_profit_pct=0.30, stop_loss_pct=0.05):
    return {"risk": {
        "enabled": True, "per_symbol_stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct,
        "take_profit_block_for_day": True, "daily_halt_enabled": True, "total_drawdown_pct": 1.0,
        "daily_reset_utc_offset_hours": 8, "state_file": str(state_file),
    }}


def _position(unrealized_profit):
    return {"symbol": "BTCUSDT", "positionAmt": "1", "positionSide": "BOTH", "entryPrice": "100",
            "markPrice": "106", "notional": "106", "unRealizedProfit": str(unrealized_profit)}


def test_take_profit_closes_and_blocks_for_day(tmp_path):
    state_file = tmp_path / "risk.json"
    adjusted, notes = _apply_live_risk_controls({"BTCUSDT": 100.0}, [_position(30.0)], 1000.0, _cfg(state_file))
    assert adjusted["BTCUSDT"] == 0.0
    assert any("take profit" in note for note in notes)
    saved = json.loads(state_file.read_text())
    assert saved["blocked_reasons"]["BTCUSDT"] == "take_profit"
    again, _ = _apply_live_risk_controls({"BTCUSDT": 100.0}, [], 1000.0, _cfg(state_file))
    assert again["BTCUSDT"] == 0.0


@pytest.mark.parametrize("pnl,closed", [(-5.0, True), (-4.99, False), (29.99, False)])
def test_symbol_stop_and_take_thresholds(tmp_path, pnl, closed):
    adjusted, _ = _apply_live_risk_controls({"BTCUSDT": 100.0}, [_position(pnl)], 1000.0,
                                            _cfg(tmp_path / "risk.json"))
    assert (adjusted["BTCUSDT"] == 0.0) is closed


@pytest.mark.parametrize("nav", [949.0, 1300.0])
def test_daily_nav_halts_use_daily_opening_node(tmp_path, nav):
    cfg = _cfg(tmp_path / "risk.json", take_profit_pct=1.0, stop_loss_pct=1.0)
    cfg["risk"].update(daily_stop_loss_pct=0.05, daily_take_profit_pct=0.30)
    _apply_live_risk_controls({}, [], 1000.0, cfg)
    adjusted, notes = _apply_live_risk_controls({"BTCUSDT": 100.0}, [_position(0.0)], nav, cfg)
    assert adjusted["BTCUSDT"] == 0.0
    assert any("Daily NAV return from opening node" in note for note in notes)


def test_symbol_map_for_execution():
    cfg = {"execution": {"symbol_map": {"BTCUSDT": "BTCUSDC", "ETHUSDT": "ETHUSDC"}}}
    assert _map_targets_for_execution({"BTCUSDT": 100.0, "ETHUSDT": -50.0, "SOLUSDT": 25.0, "BTCUSDC": 10.0}, cfg) \
        == {"BTCUSDC": 110.0, "ETHUSDC": -50.0, "SOLUSDT": 25.0}


def test_manual_position_change_detection():
    assert _snapshot_changes({"BTCUSDT": 1.0}, {}) == ["BTCUSDT: 1 -> 0"]
    assert _snapshot_changes({"BTCUSDT": 1.0}, {"BTCUSDT": 0.5}) == ["BTCUSDT: 1 -> 0.5"]
    assert _snapshot_changes({"BTCUSDT": 1.0}, {"BTCUSDT": 1.0}) == []


def test_manual_guard_state_machine(tmp_path):
    guard = ManualChangeGuard(tmp_path / "g.json")
    assert guard.check({"A": 1.0}) == []            # not armed yet
    guard.arm({"A": 1.0})
    assert guard.check({"A": 1.0}) == []
    assert guard.check({"A": 0.4}) == ["A: 1 -> 0.4"]
    assert ManualChangeGuard(tmp_path / "g.json").halted   # persisted
    guard.reset()
    assert not ManualChangeGuard(tmp_path / "g.json").halted


def test_position_snapshot_skips_flat_and_marks_hedge_sides():
    rows = [{"symbol": "A", "positionAmt": "0", "positionSide": "BOTH"},
            {"symbol": "B", "positionAmt": "2", "positionSide": "BOTH"},
            {"symbol": "C", "positionAmt": "-1", "positionSide": "SHORT"}]
    assert position_snapshot(rows) == {"B": 2.0, "C:SHORT": -1.0}


# --------------------------------------------------------- runner wiring ----
LIVE = LIQUIDITY + ["--poll-seconds", "300", "--live-orders", "--confirm-testnet-orders"]


def _audit(out):
    return [json.loads(line) for line in (out / "audit.jsonl").read_text().splitlines()]


def test_manual_change_flattens_and_halts(tmp_path, monkeypatch):
    import tests.fake_exchange as fx
    original = fx.FakeExchange.account
    calls = {"n": 0}

    def account(self):   # after a few cycles someone closes C0USDT by hand
        calls["n"] += 1
        if calls["n"] == 6 and self.positions:
            symbol = sorted(s for s, p in self.positions.items() if p["qty"])[0]
            self.positions[symbol] = {"qty": 0.0, "entry": 0.0}
        return original(self)
    monkeypatch.setattr(fx.FakeExchange, "account", account)
    out = tmp_path / "run"
    _run("new", out, monkeypatch, LIVE + ["--manual-guard"], hours=6, signed=True)
    records = _audit(out)
    assert "manual_halt" in records[-1]
    assert not records[-1]["exchange_positions"]                 # flattened
    assert json.loads((out / "manual_guard_state.json").read_text())["halted"] is True
    with pytest.raises(SystemExit, match="Manual-change guard is halted"):
        _run("new", out, monkeypatch, LIVE + ["--manual-guard"], hours=1, signed=True)


def test_exchange_risk_controls_override_targets(tmp_path, monkeypatch):
    risk = tmp_path / "risk.yaml"
    risk.write_text(
        "risk:\n  enabled: true\n  per_symbol_stop_loss_pct: 0.05\n  take_profit_pct: 0.30\n"
        "  daily_halt_enabled: false\n")
    out = tmp_path / "run"
    _run("new", out, monkeypatch,
         LIVE + ["--exchange-risk-config", str(risk), "--symbol-stop", "0.99", "--daily-stop", "0.99",
                 "--portfolio-drawdown", "0.99"],
         hours=5, signed=True, crash=("*", START + 2 * HOUR, START + 3 * HOUR, 0.80))
    records = _audit(out)
    notes = [n for r in records for n in r.get("exchange_risk_notes", [])]
    assert (out / "exchange_risk_state.json").exists()           # state lives with the run
    assert any("stop loss from entry" in n for n in notes)
    assert all(r["engine"]["state"] == "LIVE" for r in records)  # the engine itself never stopped


def test_close_all_flattens_account(tmp_path, monkeypatch):
    out = tmp_path / "run"
    _run("new", out, monkeypatch, LIVE, hours=1, signed=True)
    exchange = _run.last_exchange
    assert any(p["qty"] for p in exchange.positions.values())       # the run left positions open
    _run("new", out, monkeypatch, ["--close-all", "--live-orders", "--confirm-testnet-orders"],
         hours=1, signed=True, exchange=exchange)
    assert not any(p["qty"] for p in exchange.positions.values())
    assert json.loads((out / "audit.jsonl").read_text().splitlines()[-1])["type"] == "close_all"


def test_close_all_requires_confirmation():
    with pytest.raises(SystemExit, match="Refusing testnet orders"):
        cli_mod.main(["--close-all"])


def test_exactly_one_signal_source_required():
    with pytest.raises(SystemExit, match="exactly one signal source"):
        cli_mod.main([])
