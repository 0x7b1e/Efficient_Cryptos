import math
import random

from efficient_cryptos.engine.risk_engine import Engine, EngineConfig


def _no_risk(**overrides):
    values = dict(
        initial_equity=10_000.0, gross_leverage=3.0, rebalance_bars=1,
        fee_bps=0.0, slippage_bps=0.0,
        per_symbol_stop_loss_pct=9.0, per_symbol_take_profit_pct=9.0,
        daily_stop_loss_pct=9.0, daily_take_profit_pct=9.0, portfolio_drawdown_pct=9.0,
    )
    values.update(overrides)
    return EngineConfig(**values)


def _bar(symbol, close, **extra):
    return {"symbol": symbol, "close": close, "quote_volume": 1e15, **extra}


def test_shadow_book_keeps_marking_after_stop():
    cfg = EngineConfig(
        initial_equity=1000.0,
        gross_leverage=1.0,
        rebalance_bars=1,
        daily_stop_loss_pct=0.99,
        portfolio_drawdown_pct=0.10,
        shadow_min_periods=2,
        shadow_positive_periods=1,
        shadow_recovery_pnl_pct=0.01,
    )
    e = Engine(cfg)
    bar = {"symbol": "BTCUSDT", "close": 100.0, "quote_volume": 1_000_000}
    e.step("2026-10-01T00:00:00Z", [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 500}], [bar])
    # A negative move takes the live account into shadow.
    event = e.step("2026-10-01T00:30:00Z", [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 500}], [{**bar, "close": 70.0}])
    assert event["state"] == "SHADOW"
    assert event["shadow_equity"] != 1.0
    assert not e.positions
    assert e.shadow_positions


def test_shadow_recovery_happens_only_after_periods():
    cfg = EngineConfig(
        initial_equity=1000.0, gross_leverage=1.0, rebalance_bars=1,
        daily_stop_loss_pct=.99, portfolio_drawdown_pct=.10,
        shadow_min_periods=2, shadow_positive_periods=1,
        shadow_recovery_pnl_pct=.001,
    )
    e = Engine(cfg)
    b = {"symbol": "BTCUSDT", "close": 100.0, "quote_volume": 1_000_000}
    sig = [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 500}]
    e.step("2026-10-01T00:00:00Z", sig, [b])
    e.step("2026-10-01T00:30:00Z", sig, [{**b, "close": 70.0}])
    assert e.state == "SHADOW"
    # Same UTC day: the intraday drawdown halted the account for the day,
    # so the shadow book may not hand control back yet.
    e.step("2026-10-01T01:00:00Z", sig, [{**b, "close": 71.0}])
    assert e.state == "SHADOW"
    # Next UTC day the block is lifted and the recovery rules apply.
    e.step("2026-10-02T00:00:00Z", sig, [{**b, "close": 72.0}])
    assert e.state == "LIVE"


def test_recovered_account_stays_live_after_portfolio_drawdown():
    """Regression: the frozen high-water mark used to bounce LIVE->SHADOW forever."""
    e = Engine(_no_risk(
        gross_leverage=5.0, portfolio_drawdown_pct=0.10,
        shadow_min_periods=1, shadow_positive_periods=1, shadow_recovery_pnl_pct=0.0,
    ))
    sig = [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 40_000}]
    e.step("2026-01-01T00:00:00Z", sig, [_bar("BTCUSDT", 100.0)])
    r = e.mark("2026-01-01T00:10:00Z", [_bar("BTCUSDT", 97.0)])
    assert "STOP_TO_SHADOW" in r["risk_events"]
    e.step("2026-01-02T00:00:00Z", sig, [_bar("BTCUSDT", 97.0)])
    e.step("2026-01-02T00:30:00Z", sig, [_bar("BTCUSDT", 100.0)])
    assert e.state == "LIVE"
    for i in range(4):
        r = e.step(f"2026-01-03T0{i}:00:00Z", sig, [_bar("BTCUSDT", 100.0 + i)])
        assert r["state"] == "LIVE", r["risk_events"]
    assert e.positions


def test_pnl_matches_quantity_accounting_exactly():
    e = Engine(_no_risk(rebalance_bars=100))
    sig = [
        {"symbol": "L", "signal": 1, "target_notional": 1_000.0},
        {"symbol": "S", "signal": -1, "target_notional": -1_000.0},
    ]
    e.step("2026-01-01T00:00:00Z", sig, [_bar("L", 100.0), _bar("S", 100.0)])
    e.step("2026-01-01T00:30:00Z", sig, [_bar("L", 110.0), _bar("S", 90.0)])
    e.step("2026-01-01T01:00:00Z", sig, [_bar("L", 99.0), _bar("S", 99.0)])
    # 10 units long 100->99 = -10, 10 units short 100->99 = +10.
    assert math.isclose(e.equity, 10_000.0, abs_tol=1e-9)
    assert math.isclose(e.positions["L"].notional, 990.0)
    assert math.isclose(e.positions["S"].notional, -990.0)


def test_zero_drift_prices_have_no_systematic_short_bias():
    """The old log-return formula inflated a short high-vol book by ~13%/week."""
    rng = random.Random(7)
    vols = {"L1": 0.008, "L2": 0.008, "S1": 0.02, "S2": 0.02}
    side = {"L1": 1, "L2": 1, "S1": -1, "S2": -1}
    diffs = []
    for _ in range(60):
        e = Engine(_no_risk(rebalance_bars=24))
        px = {s: 100.0 for s in vols}
        exact, qty = 10_000.0, {}
        for b in range(336):
            if b:
                new = {s: px[s] * math.exp(rng.gauss(0, v) - v * v / 2) for s, v in vols.items()}
                exact += sum(q * (new[s] - px[s]) for s, q in qty.items())
                px = new
            sig = [{"symbol": s, "signal": 0, "target_notional": side[s] * e.equity * 0.6} for s in vols]
            e.step("2026-01-01T00:00:00Z", sig, [_bar(s, px[s]) for s in vols])
            qty = {s: p.notional / px[s] for s, p in e.positions.items()}
        diffs.append(e.equity - exact)
    assert max(abs(d) for d in diffs) < 1e-6


def test_stop_loss_liquidation_pays_exit_cost():
    e = Engine(EngineConfig(
        initial_equity=10_000.0, rebalance_bars=100, fee_bps=5.0, slippage_bps=0.0,
        per_symbol_stop_loss_pct=0.05, daily_stop_loss_pct=9.0, portfolio_drawdown_pct=9.0,
    ))
    sig = [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 10_000.0}]
    e.step("2026-01-01T00:00:00Z", sig, [_bar("BTCUSDT", 100.0)])
    after_entry = e.equity                      # 10_000 - 5 bps entry
    r = e.mark("2026-01-01T00:10:00Z", [_bar("BTCUSDT", 94.0)])
    assert "STOP_LOSS:BTCUSDT" in r["risk_events"]
    assert e.state == "SHADOW" and not e.positions
    expected = after_entry - 600.0 - 9_400.0 * 5 / 10_000
    assert math.isclose(e.equity, expected, rel_tol=1e-12)
    assert r["cost"] > 0


def test_funding_long_pays_short_receives():
    e = Engine(_no_risk(rebalance_bars=100))
    sig = [
        {"symbol": "L", "signal": 1, "target_notional": 1_000.0},
        {"symbol": "S", "signal": -1, "target_notional": -2_000.0},
    ]
    e.step("2026-01-01T00:00:00Z", sig, [_bar("L", 10.0), _bar("S", 10.0)])
    e.step("2026-01-01T00:30:00Z", sig, [
        _bar("L", 10.0, funding_rate=0.001), _bar("S", 10.0, funding_rate=0.001),
    ])
    # long pays 1.0, short receives 2.0
    assert math.isclose(e.equity, 10_001.0)
    assert math.isclose(e.total_funding, 1.0)


def test_same_side_rebalance_keeps_entry_price_for_stops():
    e = Engine(_no_risk(rebalance_bars=1, per_symbol_stop_loss_pct=0.05))
    sig = [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 1_000.0}]
    e.step("2026-01-01T00:00:00Z", sig, [_bar("BTCUSDT", 100.0)])
    # Reducing keeps the original entry, so the stop is still measured from 100.
    e.step("2026-01-01T00:30:00Z",
           [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 500.0}],
           [_bar("BTCUSDT", 97.0)])
    assert e.positions["BTCUSDT"].entry_price == 100.0
    r = e.mark("2026-01-01T00:40:00Z", [_bar("BTCUSDT", 94.9)])
    assert "STOP_LOSS:BTCUSDT" in r["risk_events"]


def test_adding_to_a_position_averages_the_entry_like_the_exchange():
    e = Engine(_no_risk(rebalance_bars=1))
    sig = [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 1_000.0}]
    e.step("2026-01-01T00:00:00Z", sig, [_bar("BTCUSDT", 100.0)])   # 10 units @100
    e.step("2026-01-01T00:30:00Z", sig, [_bar("BTCUSDT", 97.0)])    # +30/97 units @97
    assert math.isclose(e.positions["BTCUSDT"].entry_price, (10 * 100.0 + 30.0) / (10 + 30 / 97))


def test_missing_price_is_a_warning_not_a_risk_event():
    e = Engine(_no_risk(rebalance_bars=100))
    sig = [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 1_000.0}]
    e.step("2026-01-01T00:00:00Z", sig, [_bar("BTCUSDT", 100.0)])
    r = e.mark("2026-01-01T00:10:00Z", [])
    assert r["warnings"] == "MISSING_PRICE:BTCUSDT"
    assert r["risk_events"] == ""
    assert "BTCUSDT" in e.positions


def test_day_does_not_flap_back_and_clear_blocks():
    e = Engine(_no_risk())
    e.mark("2026-01-02T00:00:05Z", [])
    e.blocked_symbols["__ACCOUNT__"] = "daily_risk"
    # The runner steps with the previous bar's open time after midnight.
    e.step("2026-01-01T23:30:00Z", [], [])
    assert e.day == "2026-01-02"
    assert "__ACCOUNT__" in e.blocked_symbols


def test_load_dict_round_trip_is_independent_copy():
    e = Engine(_no_risk())
    e.step("2026-01-01T00:00:00Z",
           [{"symbol": "BTCUSDT", "signal": 1, "target_notional": 1_000.0}],
           [_bar("BTCUSDT", 100.0)])
    snapshot = e.to_dict()
    e.mark("2026-01-01T00:10:00Z", [_bar("BTCUSDT", 120.0)])
    assert e.equity != snapshot["equity"]
    e.load_dict(snapshot)
    assert e.to_dict() == snapshot
    assert e.positions["BTCUSDT"].mark_price == 100.0
