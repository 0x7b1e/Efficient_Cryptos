#!/usr/bin/env python3
"""Train/evaluate a 30m cross-sectional model and report factor deciles + equal-weight PnL."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from efficient_cryptos.common.config import load_config
from efficient_cryptos.features.frame import load_feature_frame
from efficient_cryptos.ml.sections import build_sections, select_features
from efficient_cryptos.ml.models import build_model
from efficient_cryptos.backtest.metrics import summarize_metrics
from efficient_cryptos.common.signal_policy import signal_is_due, signal_is_expired
from efficient_cryptos.common.costs import CostModel
from efficient_cryptos.backtest.portfolio import factor_summary, lifecycle_pnl_costed
from efficient_cryptos.backtest.report import write_factor_report

ASSETS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT")

def split_sections(sections, train_frac=.7, val_frac=.15):
    n = len(sections); a = max(1, int(n * train_frac)); b = min(n - 1, max(a + 1, int(n * (train_frac + val_frac))))
    return sections[:a], sections[a:b], sections[b:]

def fit_linear(train_sections):
    x = np.concatenate([s.x.reshape(len(s.symbols), -1) for s in train_sections])
    y = np.concatenate([s.y for s in train_sections])
    x1 = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(x1, y, rcond=None)[0]
    return coef

def predict_linear(coef, sections):
    rows = []
    for s in sections:
        x = np.column_stack([np.ones(len(s.symbols)), s.x.reshape(len(s.symbols), -1)])
        pred = x @ coef
        rows.extend({"timestamp": s.timestamp, "symbol": sym, "prediction": float(p), "target": float(y)}
                    for sym, p, y in zip(s.symbols, pred, s.y))
    return pd.DataFrame(rows)

def equal_weight_pnl(frame, initial_cash=10000.0):
    rows = []
    for ts, part in frame.groupby("timestamp", sort=True):
        r = part["target"].astype(float)
        rows.append({"timestamp": ts, "period_return": float(r.mean()), "n_assets": len(r)})
    out = pd.DataFrame(rows)
    out["equity"] = initial_cash * (1.0 + out["period_return"].fillna(0)).cumprod()
    return out

def factor_deciles(frame):
    rows = []
    for ts, part in frame.groupby("timestamp", sort=True):
        if len(part) < 4: continue
        q = pd.qcut(part["prediction"].rank(method="first"), 10, labels=False, duplicates="drop")
        part = part.assign(decile=q)
        for decile, grp in part.groupby("decile"):
            rows.append({"timestamp": ts, "decile": int(decile) + 1, "mean_return": float(grp.target.mean()), "n": len(grp)})
    out = pd.DataFrame(rows)
    if out.empty: return out
    summary = out.groupby("decile").agg(mean_return=("mean_return", "mean"), observations=("mean_return", "size")).reset_index()
    return summary

def _side_count(n_assets, percent, fixed_count):
    if percent is None:
        return max(0, min(int(fixed_count), n_assets))
    percent = float(percent)
    if not 0.0 <= percent <= 100.0:
        raise ValueError("long/short percentage must be between 0 and 100")
    if percent == 0:
        return 0
    return max(1, min(n_assets, int(np.ceil(n_assets * percent / 100.0))))


def lifecycle_pnl(frame, *, interval_bars=1, hold_bars=1, k_long=2, m_short=1,
                  long_percent=None, short_percent=None,
                  gross_leverage=1.0,
                  initial_cash=10000.0, cost_model=None, include_costs=False,
                  include_funding=False):
    """Evaluate the same scheduled signal/expiry rule used by live execution."""
    if float(gross_leverage) <= 0:
        raise ValueError("gross leverage must be positive")
    if include_costs or include_funding:
        return lifecycle_pnl_costed(
            frame, interval_bars=interval_bars, hold_bars=hold_bars,
            k_long=k_long, m_short=m_short, long_percent=long_percent,
            short_percent=short_percent, gross_leverage=gross_leverage,
            initial_cash=initial_cash, cost_model=cost_model or CostModel(),
            include_costs=include_costs, include_funding=include_funding,
        )
    active = {}
    signal_time = None
    rows = []
    for ts, part in frame.groupby("timestamp", sort=True):
        ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
        if signal_is_due(ts_ms, 30, interval_bars):
            ordered = part.sort_values("prediction", ascending=False)
            n_assets = len(ordered)
            n_long = _side_count(n_assets, long_percent, k_long)
            n_short = _side_count(n_assets, short_percent, m_short)
            if n_long + n_short > n_assets:
                raise ValueError(
                    f"long/short selections overlap: {n_long}+{n_short}>{n_assets}; "
                    "lower the percentages"
                )
            longs = list(ordered.symbol.head(n_long))
            shorts = list(ordered.symbol.tail(n_short))
            active = {s: 1.0 for s in longs}
            active.update({s: -1.0 for s in shorts if s not in active})
            signal_time = ts_ms
        elif signal_is_expired(ts_ms, signal_time, 30, hold_bars):
            active = {}
            signal_time = None
        returns = part.set_index("symbol")["target"]
        pnl = (sum(active.get(symbol, 0.0) * float(value)
                   for symbol, value in returns.items()) / max(1, len(active))
               * float(gross_leverage)) if active else 0.0
        rows.append({"timestamp": ts, "period_return": pnl, "active_symbols": len(active)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["equity"] = initial_cash * (1.0 + out["period_return"]).cumprod()
    return out


def plot_pnl(baseline, lifecycle, path):
    """Save comparable equity curves for the baseline and lifecycle strategy."""
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    if not baseline.empty:
        ax.plot(pd.to_datetime(baseline["timestamp"], utc=True), baseline["equity"],
                label="Equal-weight baseline", linewidth=1.5)
    if not lifecycle.empty:
        ax.plot(pd.to_datetime(lifecycle["timestamp"], utc=True), lifecycle["equity"],
                label="Signal lifecycle", linewidth=1.8)
    ax.set_title("Cross-sectional PnL / Equity")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Equity (USDT)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", default=",".join(ASSETS))
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--label", choices=("30m", "utc_close"), default="30m")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--max-features", type=int, default=64)
    ap.add_argument("--min-symbols", type=int, default=4)
    ap.add_argument("--signal-interval-bars", type=int, default=1)
    ap.add_argument("--signal-hold-bars", type=int, default=1)
    ap.add_argument("--k-long", type=int, default=2)
    ap.add_argument("--m-short", type=int, default=1)
    ap.add_argument("--gross-leverage", type=float, default=1.0,
                    help="Gross notional multiplier for lifecycle PnL")
    ap.add_argument("--include-costs", action="store_true",
                    help="Include taker fees and participation-based slippage")
    ap.add_argument("--exclude-funding", action="store_true",
                    help="Do not charge funding events when cost model is enabled")
    ap.add_argument("--taker-fee-bps", type=float, default=None)
    ap.add_argument("--slippage-base-bps", type=float, default=None)
    ap.add_argument("--slippage-impact-coef", type=float, default=None)
    ap.add_argument("--max-slippage-bps", type=float, default=None)
    ap.add_argument("--long-percent", type=float, default=None,
                    help="Top percentage of each cross-section to long (0-100); overrides --k-long")
    ap.add_argument("--short-percent", type=float, default=None,
                    help="Bottom percentage of each cross-section to short (0-100); overrides --m-short")
    ap.add_argument("--out", default="outputs/evaluation")
    ap.add_argument("--layered-cache", action="store_true",
                    help="Use reusable base-feature and label caches")
    args = ap.parse_args(argv)
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    label_mode = "horizon" if args.label == "30m" else "utc_close"
    frame, candidates = load_feature_frame(args.data_dir, symbols, interval="30m", horizon=1,
                                           days=args.days, require_positioning=False,
                                           cache=True, label_mode=label_mode,
                                           layered_cache=args.layered_cache)
    features = select_features(frame, candidates, max_features=args.max_features)
    frame[features] = frame[features].fillna(0.0)
    sections = build_sections(frame, features, args.seq_len, args.min_symbols)
    train, val, test = split_sections(sections)
    pred = predict_linear(fit_linear(train), test)
    baseline = equal_weight_pnl(pred)
    defaults = load_config().get("costs", {})
    cost_model = CostModel(
        taker_fee_bps=float(args.taker_fee_bps if args.taker_fee_bps is not None else defaults.get("taker_fee_bps", 4.0)),
        slippage_base_bps=float(args.slippage_base_bps if args.slippage_base_bps is not None else defaults.get("slippage_base_bps", 1.0)),
        slippage_impact_coef=float(args.slippage_impact_coef if args.slippage_impact_coef is not None else defaults.get("slippage_impact_coef", 60.0)),
        max_slippage_bps=float(args.max_slippage_bps if args.max_slippage_bps is not None else defaults.get("max_slippage_bps", 80.0)),
    )
    lifecycle = lifecycle_pnl(
        pred, interval_bars=args.signal_interval_bars,
        hold_bars=args.signal_hold_bars, k_long=args.k_long, m_short=args.m_short,
        long_percent=args.long_percent, short_percent=args.short_percent,
        gross_leverage=args.gross_leverage, cost_model=cost_model,
        include_costs=args.include_costs,
        include_funding=args.include_costs and not args.exclude_funding,
    )
    deciles = factor_deciles(pred)
    out = Path(args.out) / args.label; out.mkdir(parents=True, exist_ok=True)
    pred.attrs["factor_id"] = "linear_probe"
    report = write_factor_report(pred, lifecycle, out / "factors", factor_id="linear_probe")
    unified_metrics = factor_summary(
        pred, interval_bars=args.signal_interval_bars, hold_bars=args.signal_hold_bars,
        k_long=args.k_long, m_short=args.m_short, long_percent=args.long_percent,
        short_percent=args.short_percent, gross_leverage=args.gross_leverage,
        initial_cash=10000.0, cost_model=cost_model,
        include_costs=args.include_costs,
        include_funding=args.include_costs and not args.exclude_funding,
    )
    pred.to_parquet(out / "predictions.parquet", index=False)
    baseline.to_csv(out / "baseline_equal_weight_pnl.csv", index=False)
    lifecycle.to_csv(out / "signal_lifecycle_pnl.csv", index=False)
    plot_pnl(baseline, lifecycle, out / "pnl_curve.png")
    deciles.to_csv(out / "factor_deciles.csv", index=False)
    summary = {"label": args.label, "symbols": list(symbols), "initial_cash": 10000.0,
               "model": "linear_probe", "features": len(features),
               "test_metrics": summarize_metrics(pred),
               "baseline_final_equity": float(baseline.equity.iloc[-1]) if len(baseline) else None,
               "baseline_return_pct": float((baseline.equity.iloc[-1] / 10000 - 1) * 100) if len(baseline) else None,
               "signal_interval_bars": args.signal_interval_bars,
               "signal_hold_bars": args.signal_hold_bars,
               "long_percent": args.long_percent,
               "short_percent": args.short_percent,
               "k_long": args.k_long,
               "m_short": args.m_short,
               "gross_leverage": args.gross_leverage,
               "lifecycle_final_equity": float(lifecycle.equity.iloc[-1]) if len(lifecycle) else None,
               "lifecycle_metrics": unified_metrics,
               "factor_report": report}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__": main()
