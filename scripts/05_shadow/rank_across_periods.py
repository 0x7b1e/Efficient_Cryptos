#!/usr/bin/env python3
"""Rank factors across all rebalance periods of a shadow replay.

A factor that only works at one rebalance frequency is usually noise. This
reads every ``<period>/summary.csv`` written by run_selected_factor_shadow.py
and ranks factors by how consistently they hold up across periods.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="--out-dir of run_selected_factor_shadow.py")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out-csv", type=Path)
    args = ap.parse_args()
    frames = []
    for path in sorted(args.results_dir.glob("*/summary.csv")):
        frame = pd.read_csv(path)
        frame["period"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no */summary.csv under {args.results_dir}")
    df = pd.concat(frames, ignore_index=True)
    rank = df.groupby("factor_id").agg(
        periods=("period", "nunique"),
        positive_periods=("return_pct", lambda x: int((x > 0).sum())),
        live_periods=("final_state", lambda x: int((x == "LIVE").sum())),
        mean_return=("return_pct", "mean"),
        median_return=("return_pct", "median"),
        mean_sharpe=("sharpe_annualized", "mean"),
        median_sharpe=("sharpe_annualized", "median"),
        worst_drawdown=("max_drawdown_pct", "min"),
        mean_cost=("total_cost", "mean"),
    ).reset_index().sort_values(
        ["positive_periods", "live_periods", "median_sharpe", "mean_return"], ascending=False)
    pd.set_option("display.width", 200)
    print(rank.head(args.top).to_string(index=False))
    print()
    print("best single runs:")
    print(df.sort_values("sharpe_annualized", ascending=False)[[
        "factor_id", "period", "return_pct", "max_drawdown_pct", "sharpe_annualized",
        "total_cost", "final_state"]].head(args.top).to_string(index=False))
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        rank.to_csv(args.out_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
