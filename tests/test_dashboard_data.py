import json
import math

import numpy as np
import pandas as pd
import pytest

from efficient_cryptos.dashboard import data as d
from tests.test_runner_golden import HOUR, LIQUIDITY, START, _run


# ------------------------------------------------------------ factor sweep
def _write_factor(root, rebalance, factor, sharpe, ret, n=50, seed=0):
    folder = root / rebalance
    folder.mkdir(parents=True, exist_ok=True)
    metrics = {"factor_id": factor, "sharpe": sharpe, "return_pct": ret, "max_drawdown_pct": -5.0,
               "avg_turnover": 0.3, "total_cost": 12.0, "final_equity": 10000 * (1 + ret / 100), "bars": n,
               "signed_q10_minus_q1": 1e-4}
    (folder / f"{factor}.metrics.json").write_text(json.dumps(metrics))
    ts = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
    eq = 10000 * np.cumprod(1 + np.random.default_rng(seed).normal(0, 1e-3, n))
    pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S+00:00"), "equity": eq,
                  "drawdown": eq / np.maximum.accumulate(eq) - 1, "net_return": 0.0}).to_csv(
        folder / f"{factor}.timeseries.csv", index=False)
    layers = {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S+00:00")}
    for q in range(1, 11):
        layers[f"signed_q{q}"] = np.full(n, (q - 5.5) * 1e-4)
        layers[f"abs_q{q}"] = np.zeros(n)
    pd.DataFrame(layers).to_csv(folder / f"{factor}.layers.csv", index=False)


@pytest.fixture()
def sweeps(tmp_path):
    train, test = tmp_path / "outputs" / "sweep_train", tmp_path / "outputs" / "sweep_test"
    for reb, s in (("1h", 1.0), ("4h", 2.5), ("24h", -0.5)):
        _write_factor(train, reb, "fa", s, s * 3)
        _write_factor(train, reb, "fb", s / 2, s)
        _write_factor(test, reb, "fa", s / 3, s)
    (train / "4h" / "plots").mkdir()
    (train / "4h" / "broken.metrics.json").write_text("{not json")
    return tmp_path / "outputs", train, test


def test_scan_and_join(sweeps):
    base, train, test = sweeps
    assert d.find_sweep_roots(base) == sorted([train, test])
    frame = d.scan_sweep(train)
    assert len(frame) == 6 and frame.iloc[0][["factor_id", "rebalance", "sharpe"]].tolist() == ["fa", "4h", 2.5]
    assert d.sort_rebalances(frame["rebalance"]) == ["1h", "4h", "24h"]
    joined = d.join_periods(frame, d.scan_sweep(test))
    row = joined[(joined.factor_id == "fa") & (joined.rebalance == "4h")].iloc[0]
    assert row["test_sharpe"] == pytest.approx(2.5 / 3)
    assert math.isnan(joined[joined.factor_id == "fb"]["test_sharpe"].iloc[0])
    best = d.best_per_factor(frame)
    assert best.set_index("factor_id").loc["fb", "rebalance"] == "4h"
    stab = d.stability_table(frame).set_index("factor_id")
    assert stab.loc["fa", "positive_periods"] == 2 and stab.loc["fa", "periods"] == 3


def test_timeseries_and_layers(sweeps):
    _, train, _ = sweeps
    ts = d.load_timeseries(train / "4h" / "fa.timeseries.csv", max_points=10)
    assert 10 <= len(ts) <= 12 and str(ts["timestamp"].dt.tz) == "UTC"
    bars, cum = d.layer_summary(train / "4h" / "fa.layers.csv")
    assert bars["decile"].tolist()[0] == "Q1" and bars["decile"].tolist()[-1] == "Q10"
    assert bars["mean_bps"].iloc[-1] == pytest.approx(4.5)
    assert cum["Q10"].iloc[-1] == pytest.approx(math.expm1(50 * 4.5e-4) * 100)
    assert d.load_timeseries(train / "missing.csv").empty


def test_factor_meta(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps({"factor_id": "fa", "block": "volume", "formula": "x", "hypothesis": "h"}) + "\n\nbad\n")
    meta = d.load_factor_meta(path)
    assert meta.set_index("factor_id").loc["fa", "block"] == "volume"
    assert d.load_factor_meta(None).empty


# ------------------------------------------------------------ live runner
def test_tail_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(5000)))
    tail = d.read_jsonl_tail(path, 3)
    assert [r["i"] for r in tail] == [4997, 4998, 4999]
    assert len(d.tail_lines(path, 10_000)) == 5000


@pytest.fixture()
def live_dir(tmp_path, monkeypatch):
    out = tmp_path / "outputs" / "live_run"
    crash = ("*", START + 2 * HOUR, START + 3 * HOUR, 0.80)
    _run("new", out, monkeypatch,
         LIQUIDITY + ["--poll-seconds", "300", "--live-orders", "--confirm-testnet-orders",
                      "--symbol-stop", "0.05", "--shadow-min-periods", "1",
                      "--shadow-positive-periods", "1", "--shadow-recovery-pnl-pct", "0"],
         hours=8, signed=True, crash=crash, fail_orders={5})
    return out


def test_live_loaders_on_real_runner_output(live_dir):
    assert d.find_live_dirs(live_dir.parent.parent) == [live_dir]
    records = d.read_jsonl_tail(live_dir / "audit.jsonl", 10_000)
    audit = d.audit_frame(records)
    assert len(audit) == len(records) and audit["time"].is_monotonic_increasing
    assert {"LIVE", "SHADOW"} <= set(audit["state"])
    assert (audit["submitted"] > 0).any()
    with_positions = audit[audit["positions"] > 0]
    assert (with_positions["long_notional"] > 0).all() and (with_positions["short_notional"] > 0).all()

    spans = d.state_spans(audit)
    assert spans[0][0] == audit["time"].iloc[0] and spans[-1][1] == audit["time"].iloc[-1]
    assert any(s[2] == "SHADOW" for s in spans)

    events = d.event_frame(audit, records)
    assert (events["kind"] == "risk").any() and events["detail"].str.contains("STOP_LOSS").any()

    errors = d.errors_frame(live_dir / "runner_errors.jsonl")
    assert (errors["type"] == "network_error").any()

    orders = d.orders_frame(records)
    assert {"symbol", "side", "status"} <= set(orders.columns) and (orders["status"] == "FILLED").any()

    latest = d.read_json(live_dir / "latest.json")
    state = d.read_json(live_dir / "engine_state.json")
    summary = d.live_summary(latest, state, (live_dir / "latest.json").stat().st_mtime)
    assert summary["mode"] == "TESTNET_LIVE" and summary["equity"] > 0
    assert summary["seconds_since_update"] >= 0


def test_positions_and_targets_from_payload():
    latest = {"targets": {"AUSDT": 100.0, "BUSDT": -100.0},
              "exchange_positions": [{"symbol": "AUSDT", "notional": "95", "positionAmt": "1", "entryPrice": "95",
                                      "markPrice": "95", "unRealizedProfit": "0", "leverage": "3"}]}
    tva = d.targets_vs_actual(latest).set_index("symbol")
    assert tva.loc["AUSDT", "diff"] == pytest.approx(-5) and tva.loc["BUSDT", "actual"] == 0
    pos = d.positions_frame(latest, None)
    assert pos.iloc[0]["side"] == "LONG" and pos.iloc[0]["source"] == "exchange"
    # dry run: no exchange rows, fall back to the engine book
    engine = {"positions": {"CUSDT": {"notional": -200.0, "entry_price": 10.0, "mark_price": 9.0, "entry_time": "x"}}}
    pos = d.positions_frame({}, engine)
    assert pos.iloc[0]["unrealized_pnl"] == pytest.approx(20.0)  # short gains as price falls and pos.iloc[0]["source"] == "engine"
    summary = d.live_summary(latest, {"peak_equity": 200, "day_open_equity": 100, "equity": 100}, None)
    assert summary["net_exposure_pct"] == pytest.approx(100.0)  # only the long leg is filled


def test_orders_matched_by_client_order_index():
    record = {"engine": {"timestamp": "2026-01-05T00:00:00+00:00"},
              "orders": [{"symbol": "AUSDT", "side": "SELL", "quantity": 1, "reduce_only": True},
                         {"symbol": "AUSDT", "side": "SELL", "quantity": 2},
                         {"symbol": "BUSDT", "side": "BUY", "quantity": 0, "rejected": True, "reason": "min_notional"}],
              "submitted": [{"symbol": "AUSDT", "clientOrderId": "sh_abc_0_x", "status": "FILLED", "executedQty": "1"},
                            {"symbol": "AUSDT", "client_order_id": "sh_abc_1_y", "error": "margin", "order_error": True}]}
    orders = d.orders_frame([record])
    assert orders["status"].tolist() == ["FILLED", "ERROR", "rejected"]
    assert orders.loc[1, "error"] == "margin" and orders.loc[0, "executed_qty"] == 1
