"""Offline paper replay: signals CSV + 30m bars CSV through the Engine."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from efficient_cryptos.engine.risk_engine import Engine, EngineConfig

_FLOAT_ARGS = [
    ("initial-equity", 10000.0), ("gross-leverage", 3.0), ("margin-buffer-pct", 0.20),
    ("per-symbol-stop-loss-pct", 0.05), ("per-symbol-take-profit-pct", 0.30),
    ("daily-stop-loss-pct", 0.05), ("daily-take-profit-pct", 0.30),
    ("portfolio-drawdown-pct", 0.10), ("shadow-recovery-pnl-pct", 0.01),
    ("fee-bps", 5.0), ("slippage-bps", 1.0),
]
_INT_ARGS = [("rebalance-bars", 24), ("shadow-min-periods", 3), ("shadow-positive-periods", 2)]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="Shadow Testnet paper risk engine")
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--bars", type=Path, required=True,
                    help="timestamp,symbol,close,quote_volume[,funding_rate]")
    ap.add_argument("--output", type=Path, required=True)
    for name, default in _FLOAT_ARGS:
        ap.add_argument("--" + name, dest=name.replace("-", "_"), type=float, default=default)
    for name, default in _INT_ARGS:
        ap.add_argument("--" + name, dest=name.replace("-", "_"), type=int, default=default)
    ap.add_argument("--no-funding", action="store_true")
    a = ap.parse_args()

    by_time: dict[str, list[dict]] = {}
    for b in read_csv(a.bars):
        by_time.setdefault(b["timestamp"], []).append(b)
    sig_by_time: dict[str, list[dict]] = {}
    for s in read_csv(a.signals):
        sig_by_time.setdefault(s["timestamp"], []).append(s)

    fields = EngineConfig.__dataclass_fields__
    cfg = EngineConfig(**{k: v for k, v in vars(a).items() if k in fields},
                       funding_enabled=not a.no_funding)
    e = Engine(cfg)
    rows, orders = [], []
    peak, max_dd = e.equity, 0.0
    # Step only bars that carry signals: an empty signal set on a rebalance bar
    # would mean "flat", and rebalance_bars counts signal bars. PnL between
    # steps is still exact because positions are marked from their last price.
    for ts in sorted(set(by_time) & set(sig_by_time)):
        before = {k: v.notional for k, v in e.positions.items()}
        row = e.step(ts, sig_by_time[ts], by_time[ts])
        after = {k: v.notional for k, v in e.positions.items()}
        row["positions_json"] = json.dumps(after, sort_keys=True)
        rows.append(row)
        peak = max(peak, e.equity)
        max_dd = min(max_dd, e.equity / peak - 1.0)
        for s in sorted(set(before) | set(after)):
            # Pure price drift changes notional without trading; only report
            # entries, exits, flips and resizes at rebalance/risk events.
            traded = (s not in before or s not in after
                      or before[s] * after[s] < 0
                      or (row["cost"] and abs(after[s] - before[s]) > 1e-9 * max(1.0, abs(before[s]))))
            if traded:
                orders.append({"timestamp": ts, "symbol": s, "old_notional": before.get(s, 0.0),
                               "target_notional": after.get(s, 0.0), "state": e.state,
                               "risk_events": row["risk_events"]})

    a.output.mkdir(parents=True, exist_ok=True)
    for name, data in [("timeseries.csv", rows), ("orders.csv", orders)]:
        with (a.output / name).open("w", newline="", encoding="utf-8") as f:
            if data:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    metrics = {
        "final_equity": e.equity, "return_pct": (e.equity / cfg.initial_equity - 1) * 100,
        "max_drawdown_pct": max_dd * 100, "total_cost": e.total_cost,
        "total_funding": e.total_funding, "final_state": e.state,
        "shadow_equity": e.shadow_equity, "shadow_entries": e.shadow_entries,
        "shadow_exits": e.shadow_exits, "bars": len(rows), "orders": len(orders),
    }
    (a.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
