#!/usr/bin/env python3
"""Replay prediction CSVs through the same Engine the live runner uses.

Timeline per 30m signal bar ``T`` (bar open time, like the live runner):

* ``Engine.step(T)`` runs at the bar close ``T+30m`` using the 30m close, after
  realising PnL and any funding settled inside the bar.
* Between two steps ``Engine.mark`` runs every ``--mark-minutes`` on the
  completed 1m OHLC average (the live runner's TWAP risk price), so stops,
  daily halts and drawdown fire intrabar as they would live.
* Incomplete 30m bars (fewer than 30 one-minute rows) are not tradable,
  matching the live rule that a forming bar never enters the signal.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from efficient_cryptos.engine.risk_engine import Engine, EngineConfig  # noqa: E402

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,ZECUSDT,MOVRUSDT,NEARUSDT,QNTUSDT,XRPUSDT,DOGEUSDT,SUIUSDT,UNIUSDT,WLDUSDT,1000PEPEUSDT,BNBUSDT,LINKUSDT,ADAUSDT".split(",")
BAR = pd.Timedelta(minutes=30)
STEPS_PER_YEAR = 48 * 365
FUNDING_RATE_COLUMNS = ("funding_rate", "fundingRate", "rate", "last_funding_rate")


def _key(ts: pd.Timestamp) -> str:
    return str(pd.Timestamp(ts).tz_convert("UTC"))


def _funding_by_bar(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    """Map each funding settlement to the 30m bar whose close covers it."""
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    column = next((c for c in FUNDING_RATE_COLUMNS if c in frame.columns), None)
    if column is None or "funding_time" not in frame.columns:
        return {}
    times = pd.to_datetime(pd.to_numeric(frame["funding_time"], errors="coerce"), unit="ms", utc=True)
    rates = pd.to_numeric(frame[column], errors="coerce")
    result: dict[str, float] = {}
    for when, rate in zip(times, rates):
        if pd.isna(when) or pd.isna(rate):
            continue
        # Settlement at exactly 08:00 is paid by positions held through the
        # bar [07:30, 08:00), whose open time is floor(08:00 - 1ms).
        bar = (when - pd.Timedelta(milliseconds=1)).floor("30min")
        if start <= bar <= end:
            result[_key(bar)] = result.get(_key(bar), 0.0) + float(rate)
    return result


def load_market(data_dir: Path, timestamps: set[str], *, mark_minutes: int,
                use_funding: bool):
    """Return (30m bars by key, sorted mark times, mark bars by time)."""
    wanted = pd.to_datetime(list(timestamps), utc=True)
    start, end = wanted.min(), wanted.max()
    bars: dict[str, list[dict]] = {t: [] for t in timestamps}
    marks: dict[pd.Timestamp, list[dict]] = {}
    for symbol in SYMBOLS:
        path = data_dir / "klines" / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["ts"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        # One extra bar after the last signal so the final holding can be marked.
        frame = frame[(frame.ts >= start) & (frame.ts < end + 2 * BAR)].copy()
        if frame.empty:
            continue
        frame["bar"] = frame.ts.dt.floor("30min")
        grouped = frame.groupby("bar", sort=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), quote_volume=("quote_volume", "sum"),
            minutes=("open", "size"),
        ).reset_index()
        funding = _funding_by_bar(data_dir / "funding" / f"{symbol}.parquet", start, end + BAR) if use_funding else {}
        for row in grouped.to_dict("records"):
            key = _key(row.pop("bar"))
            if row.pop("minutes") < 30:
                continue
            if key in funding:
                row["funding_rate"] = funding[key]
            bars.setdefault(key, []).append({"symbol": symbol, **row})
        if mark_minutes > 0:
            twap = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
            close_time = frame["ts"] + pd.Timedelta(minutes=1)
            minute_of_day = close_time.dt.hour * 60 + close_time.dt.minute
            keep = (minute_of_day % mark_minutes == 0) & (twap > 0)
            for when, price in zip(close_time[keep], twap[keep]):
                marks.setdefault(when, []).append({"symbol": symbol, "close": float(price)})
    mark_times = sorted(marks)
    return bars, mark_times, marks


def signals_for(frame: pd.DataFrame, equity: float, gross_leverage: float, margin_buffer: float,
                long_pct: float, short_pct: float):
    result = []
    for ts, group in frame.groupby("timestamp", sort=True):
        g = group.dropna(subset=["prediction"]).sort_values("prediction")
        if len(g) < 2:
            continue
        nl = max(1, min(len(g), math.ceil(len(g) * long_pct / 100))) if long_pct else 0
        ns = max(1, min(len(g), math.ceil(len(g) * short_pct / 100))) if short_pct else 0
        if nl + ns > len(g):
            continue
        each = equity * (1 - margin_buffer) * gross_leverage / 2
        for _, r in (g.tail(nl).iterrows() if nl else []):
            result.append({"timestamp": ts, "symbol": r.symbol, "signal": float(r.prediction), "target_notional": each / nl})
        for _, r in (g.head(ns).iterrows() if ns else []):
            result.append({"timestamp": ts, "symbol": r.symbol, "signal": float(r.prediction), "target_notional": -each / ns})
    return result


def _config(args) -> EngineConfig:
    return EngineConfig(
        initial_equity=args.initial_equity, gross_leverage=args.gross_leverage,
        margin_buffer_pct=args.margin_buffer_pct, rebalance_bars=args.rebalance_bars,
        per_symbol_stop_loss_pct=args.symbol_stop, per_symbol_take_profit_pct=args.symbol_take,
        daily_stop_loss_pct=args.daily_stop, daily_take_profit_pct=args.daily_take,
        portfolio_drawdown_pct=args.portfolio_drawdown,
        shadow_min_periods=args.shadow_min_periods,
        shadow_positive_periods=args.shadow_positive_periods,
        shadow_recovery_pnl_pct=args.shadow_recovery_pnl_pct,
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        funding_enabled=not args.no_funding,
    )


def run_one(pred_path, market, args):
    bars_by_ts, mark_times, marks = market
    frame = pd.read_csv(pred_path)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).map(_key)
    frame = frame.dropna(subset=["prediction"])
    e = Engine(_config(args))
    rows = []
    timestamps = sorted((t for t in set(frame.timestamp) if bars_by_ts.get(t)),
                        key=lambda t: pd.Timestamp(t))
    by_ts = {ts: part for ts, part in frame.groupby("timestamp")}
    for index, ts in enumerate(timestamps):
        signals = signals_for(by_ts[ts], e.equity, args.gross_leverage, args.margin_buffer_pct,
                              args.long_pct, args.short_pct)
        row = e.step(ts, signals, bars_by_ts[ts])
        row["kind"] = "step"
        rows.append(row)
        if not mark_times:
            continue
        # Intrabar risk marks until the next step's bar close.
        step_time = pd.Timestamp(ts) + BAR
        next_step = (pd.Timestamp(timestamps[index + 1]) + BAR) if index + 1 < len(timestamps) else step_time + BAR
        lo = bisect.bisect_right(mark_times, step_time)
        hi = bisect.bisect_left(mark_times, next_step)
        for when in mark_times[lo:hi]:
            if not (e.positions or e.shadow_positions):
                break
            row = e.mark(_key(when), marks[when])   # same timestamp format as step rows
            if row["risk_events"]:
                row["kind"] = "mark"
                rows.append(row)
    return e, rows


def metrics(e: Engine, rows: list[dict], args) -> dict:
    steps = [r for r in rows if r["kind"] == "step"]
    equity = [args.initial_equity] + [r["equity"] for r in steps]
    peak, max_dd = equity[0], 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1.0)
    returns = [b / a - 1.0 for a, b in zip(equity, equity[1:]) if a > 0]
    sharpe = float("nan")
    if len(returns) > 1:
        mean = sum(returns) / len(returns)
        std = math.sqrt(sum((x - mean) ** 2 for x in returns) / (len(returns) - 1))
        if std > 0:
            sharpe = mean / std * math.sqrt(STEPS_PER_YEAR)
    return {
        "bars": len(steps), "final_equity": e.equity,
        "return_pct": (e.equity / args.initial_equity - 1) * 100,
        "max_drawdown_pct": max_dd * 100, "sharpe_annualized": sharpe,
        "total_cost": e.total_cost, "total_funding": e.total_funding,
        "final_state": e.state, "shadow_entries": e.shadow_entries,
        "shadow_exits": e.shadow_exits, "shadow_equity": e.shadow_equity,
    }


def _rebalance_bars(hours: float) -> int:
    bars = hours * 2.0
    if not math.isfinite(bars) or bars <= 0 or not math.isclose(bars, round(bars), abs_tol=1e-9):
        raise SystemExit(f"rebalance interval {hours:g}h is not a multiple of the 30m signal bar")
    return int(round(bars))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions_selected_2026_10_01_2026_10_07"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/shadow_selected_engine_test"))
    ap.add_argument("--rebalance-hours", default="1,2,4,8,12,24",
                    help="Comma-separated; each must be a multiple of 0.5h")
    ap.add_argument("--initial-equity", type=float, default=10000)
    ap.add_argument("--gross-leverage", type=float, default=3)
    ap.add_argument("--margin-buffer-pct", type=float, default=.20)
    ap.add_argument("--long-pct", type=float, default=10)
    ap.add_argument("--short-pct", type=float, default=10)
    ap.add_argument("--symbol-stop", type=float, default=.05)
    ap.add_argument("--symbol-take", type=float, default=.30)
    ap.add_argument("--daily-stop", type=float, default=.05)
    ap.add_argument("--daily-take", type=float, default=.30)
    ap.add_argument("--portfolio-drawdown", type=float, default=.10)
    ap.add_argument("--shadow-min-periods", type=int, default=3)
    ap.add_argument("--shadow-positive-periods", type=int, default=2)
    ap.add_argument("--shadow-recovery-pnl-pct", type=float, default=.01)
    ap.add_argument("--fee-bps", type=float, default=5.0, help="Taker fee per side, bps")
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--no-funding", action="store_true", help="Ignore funding parquet files")
    ap.add_argument("--mark-minutes", type=int, default=1,
                    help="Intrabar risk-mark interval on 1m TWAP; 0 disables (optimistic)")
    ap.add_argument("--sort-by", default="sharpe_annualized",
                    choices=["sharpe_annualized", "return_pct", "max_drawdown_pct"])
    args = ap.parse_args()
    if args.mark_minutes < 0 or (args.mark_minutes and 60 % args.mark_minutes):
        raise SystemExit("--mark-minutes must be 0 or a divisor of 60")
    hours_list = [float(x) for x in args.rebalance_hours.split(",") if x.strip()]
    bars_list = [_rebalance_bars(h) for h in hours_list]
    paths = sorted(args.predictions_dir.glob("*.csv"))
    if not paths:
        raise SystemExit("no prediction CSVs")
    all_ts = set()
    for p in paths:
        all_ts.update(pd.to_datetime(pd.read_csv(p, usecols=["timestamp"]).timestamp, utc=True).map(_key))
    market = load_market(args.data_dir, all_ts, mark_minutes=args.mark_minutes,
                         use_funding=not args.no_funding)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for hours, rebalance_bars in zip(hours_list, bars_list):
        args.rebalance_bars = rebalance_bars
        root = args.out_dir / f"{hours:g}h"
        root.mkdir(exist_ok=True)
        summary = []
        for i, p in enumerate(paths, 1):
            e, rows = run_one(p, market, args)
            factor = p.stem
            if rows:
                fields = sorted({k for r in rows for k in r})
                with (root / f"{factor}.timeseries.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
            meta = {"factor_id": factor, "rebalance_hours": hours, **metrics(e, rows, args)}
            (root / f"{factor}.metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            summary.append(meta)
            if i % 25 == 0:
                print(f"{hours:g}h {i}/{len(paths)}")
        # Descending for all three: drawdown is negative, so closest to 0 first.
        pd.DataFrame(summary).sort_values(args.sort_by, ascending=False, na_position="last") \
            .to_csv(root / "summary.csv", index=False)
        print(f"completed {hours:g}h factors={len(summary)}")


if __name__ == "__main__":
    main()
