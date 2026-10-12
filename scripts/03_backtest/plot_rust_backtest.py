#!/usr/bin/env python3
"""Plot one Rust factor backtest.

The Rust core writes aggregate metrics, per-bar portfolio timeseries, and
cross-sectional layer returns.  This script turns those three files into one
diagnostic PNG without changing the backtest calculations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _cumret(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return (1.0 + values).cumprod() - 1.0


def _cum_logret(series: pd.Series) -> pd.Series:
    """Convert cumulative log returns to simple cumulative returns."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return values.cumsum().clip(lower=-700).map(lambda value: math.expm1(float(value)))


def _thin(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Downsample only for rendering; all CSV calculations remain untouched."""
    if max_points <= 0 or len(frame) <= max_points:
        return frame
    step = max(1, math.ceil(len(frame) / max_points))
    result = frame.iloc[::step].copy()
    if result.index[-1] != frame.index[-1]:
        result = pd.concat([result, frame.iloc[[-1]]])
    return result


def _plot_equity(ts: pd.DataFrame, metrics: dict, out: Path, max_points: int) -> None:
    ts = _thin(ts, max_points)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ax_equity, ax_pnl = axes[0]
    ax_dd, ax_cost = axes[1]
    ax_equity.plot(ts["timestamp"], ts["equity"], color="#1464f4", linewidth=1.0)
    ax_equity.set_title("Portfolio equity")
    ax_equity.set_ylabel("equity")
    ax_pnl.plot(ts["timestamp"], _cumret(ts["net_return"]) * 100.0,
                color="#159957", linewidth=0.9, label="net cumulative")
    ax_pnl.plot(ts["timestamp"], _cumret(ts["gross_return"]) * 100.0,
                color="#888888", linewidth=0.8, label="gross cumulative")
    ax_pnl.set_title("Cumulative return")
    ax_pnl.set_ylabel("return (%)")
    ax_pnl.legend(loc="best", fontsize=8)
    ax_dd.fill_between(ts["timestamp"], ts["drawdown"] * 100.0, 0.0,
                       color="#d62728", alpha=0.35)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("drawdown (%)")
    ax_cost.plot(ts["timestamp"], ts["turnover"], label="turnover", color="#9467bd", linewidth=0.8)
    ax_cost.set_title("Turnover and trading drag")
    ax_cost.set_ylabel("turnover")
    ax_cost2 = ax_cost.twinx()
    ax_cost2.plot(ts["timestamp"], ts["cost"], label="cost", color="#d62728", linewidth=0.8)
    ax_cost2.plot(ts["timestamp"], ts["funding"], label="funding", color="#ff7f0e", linewidth=0.8)
    ax_cost2.set_ylabel("cash amount")
    handles, labels = ax_cost.get_legend_handles_labels()
    handles2, labels2 = ax_cost2.get_legend_handles_labels()
    ax_cost.legend(handles + handles2, labels + labels2, loc="best", fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    factor = metrics.get("factor_id", out.stem)
    fig.suptitle(
        f"{factor} | return={metrics.get('return_pct')}%  "
        f"Sharpe={metrics.get('sharpe')}  maxDD={metrics.get('max_drawdown_pct')}%",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_layers(layers: pd.DataFrame, metrics: dict, out: Path, max_points: int) -> None:
    layers = _thin(layers, max_points)
    colors = plt.cm.viridis([i / 9.0 for i in range(10)])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for i, color in enumerate(colors, start=1):
        axes[0].plot(layers["timestamp"], _cum_logret(layers[f"signed_q{i}"]) * 100.0,
                     label=f"Q{i}", linewidth=0.75, color=color)
        axes[1].plot(layers["timestamp"], _cum_logret(layers[f"abs_q{i}"]) * 100.0,
                     label=f"abs Q{i}", linewidth=0.75, color=color)
    axes[0].set_title("Signed 10-layer cumulative returns")
    axes[1].set_title("Absolute-magnitude 10-layer cumulative returns")
    for axis in axes:
        axis.set_ylabel("cumulative return (%)")
        axis.grid(alpha=0.22)
        axis.legend(ncol=2, fontsize=7)
    factor = metrics.get("factor_id", out.stem)
    fig.suptitle(f"{factor} | signed/absolute cross-sectional layers")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=Path)
    parser.add_argument("--timeseries", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--kind", choices=("all", "equity", "layers"), default="all",
                        help="Render only the portfolio chart, only the layer chart, or both")
    parser.add_argument("--max-points", type=int, default=5000,
                        help="Maximum points rendered per line; CSV data is not changed")
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    ts = None
    layers = None
    if args.kind in ("all", "equity"):
        ts = pd.read_csv(args.timeseries, parse_dates=["timestamp"])
        if ts.empty:
            raise SystemExit(f"empty timeseries: {args.timeseries}")
    if args.kind in ("all", "layers"):
        layers = pd.read_csv(args.layers, parse_dates=["timestamp"])
        if layers.empty:
            raise SystemExit(f"empty layers: {args.layers}")

    if args.kind == "equity":
        _plot_equity(ts, metrics, args.out, args.max_points)
        print(f"wrote {args.out}")
        return 0
    if args.kind == "layers":
        _plot_layers(layers, metrics, args.out, args.max_points)
        print(f"wrote {args.out}")
        return 0

    ts = _thin(ts, args.max_points)
    layers = _thin(layers, args.max_points)
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=False)
    ax_equity, ax_pnl = axes[0]
    ax_dd, ax_signed = axes[1]
    ax_abs, ax_cost = axes[2]

    ax_equity.plot(ts["timestamp"], ts["equity"], color="#1464f4", linewidth=1.2)
    ax_equity.set_title("Portfolio equity")
    ax_equity.set_ylabel("equity")
    ax_equity.grid(alpha=0.25)

    ax_pnl.plot(ts["timestamp"], _cumret(ts["net_return"]) * 100.0,
                color="#159957", linewidth=1.0, label="net cumulative")
    ax_pnl.plot(ts["timestamp"], _cumret(ts["gross_return"]) * 100.0,
                color="#888888", linewidth=0.8, alpha=0.8, label="gross cumulative")
    ax_pnl.set_title("Cumulative return")
    ax_pnl.set_ylabel("return (%)")
    ax_pnl.legend(loc="best")
    ax_pnl.grid(alpha=0.25)

    ax_dd.fill_between(ts["timestamp"], ts["drawdown"] * 100.0, 0.0,
                       color="#d62728", alpha=0.35)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("drawdown (%)")
    ax_dd.grid(alpha=0.25)

    colors = plt.cm.viridis([i / 9.0 for i in range(10)])
    for i, color in enumerate(colors, start=1):
        ax_signed.plot(layers["timestamp"], _cum_logret(layers[f"signed_q{i}"]) * 100.0,
                       label=f"Q{i}", linewidth=0.8, color=color)
    ax_signed.set_title("Signed 10-layer cumulative returns")
    ax_signed.set_ylabel("return (%)")
    ax_signed.legend(loc="best")
    ax_signed.grid(alpha=0.25)

    for i, color in enumerate(colors, start=1):
        ax_abs.plot(layers["timestamp"], _cum_logret(layers[f"abs_q{i}"]) * 100.0,
                    label=f"abs Q{i}", linewidth=0.8, color=color)
    ax_abs.set_title("Absolute-magnitude 10-layer returns")
    ax_abs.set_ylabel("return (%)")
    ax_abs.legend(loc="best")
    ax_abs.grid(alpha=0.25)

    ax_cost.plot(ts["timestamp"], ts["turnover"], label="turnover", color="#9467bd", linewidth=0.8)
    ax_cost.set_ylabel("turnover")
    ax_cost.set_title("Turnover and trading drag")
    ax_cost.grid(alpha=0.25)
    ax_cost2 = ax_cost.twinx()
    ax_cost2.plot(ts["timestamp"], ts["cost"], label="cost", color="#d62728", linewidth=0.8)
    ax_cost2.plot(ts["timestamp"], ts["funding"], label="funding", color="#ff7f0e", linewidth=0.8)
    ax_cost2.set_ylabel("cash amount")
    handles, labels = ax_cost.get_legend_handles_labels()
    handles2, labels2 = ax_cost2.get_legend_handles_labels()
    ax_cost.legend(handles + handles2, labels + labels2, loc="best")

    factor = metrics.get("factor_id", args.metrics.stem)
    fig.suptitle(
        f"{factor} | return={metrics.get('return_pct')}%  "
        f"Sharpe={metrics.get('sharpe')}  maxDD={metrics.get('max_drawdown_pct')}%",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
