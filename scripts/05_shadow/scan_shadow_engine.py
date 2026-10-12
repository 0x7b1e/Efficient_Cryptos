#!/usr/bin/env python3
"""Scan and plot shadow-engine test outputs.

The shadow engine writes a compact metrics JSON and a timeseries CSV for each
factor/rebalance window.  This scanner intentionally does not treat the
counterfactual ``shadow_equity`` as live account equity: both are reported so
that risk-triggered flat periods remain visible.
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


def _thin(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(frame) <= max_points:
        return frame
    step = max(1, math.ceil(len(frame) / max_points))
    out = frame.iloc[::step].copy()
    if out.index[-1] != frame.index[-1]:
        out = pd.concat([out, frame.iloc[[-1]]])
    return out


def _max_drawdown(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float((x / x.cummax() - 1.0).min() * 100.0)


def _read_one(metrics_path: Path) -> dict | None:
    try:
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ts_path = metrics_path.with_name(metrics_path.name.replace(".metrics.json", ".timeseries.csv"))
    if not ts_path.exists():
        return None
    try:
        ts = pd.read_csv(ts_path)
        ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True, format="ISO8601")
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if ts.empty or "equity" not in ts:
        return None
    equity = pd.to_numeric(ts["equity"], errors="coerce")
    shadow = pd.to_numeric(ts.get("shadow_equity", pd.Series(dtype=float)), errors="coerce")
    states = ts.get("state", pd.Series(dtype=str)).astype(str)
    risk = ts.get("risk_events", pd.Series(dtype=str)).fillna("").astype(str)
    initial = float(equity.dropna().iloc[0]) if not equity.dropna().empty else 1.0
    final = float(equity.dropna().iloc[-1]) if not equity.dropna().empty else float("nan")
    shadow_final = float(shadow.dropna().iloc[-1]) if not shadow.dropna().empty else float("nan")
    row = {
        "rebalance_hours": float(meta.get("rebalance_hours", float(metrics_path.parent.name.rstrip("h")))) ,
        "factor_id": str(meta.get("factor_id", metrics_path.stem.replace(".metrics", ""))),
        "return_pct": float(meta.get("return_pct", (final / initial - 1.0) * 100.0)),
        "final_equity": final,
        "max_drawdown_pct": _max_drawdown(equity),
        "shadow_final_nav_pct": (shadow_final - 1.0) * 100.0 if math.isfinite(shadow_final) else float("nan"),
        "final_state": str(meta.get("final_state", states.iloc[-1] if len(states) else "")),
        "shadow_entries": int(meta.get("shadow_entries", 0)),
        "shadow_exits": int(meta.get("shadow_exits", 0)),
        "bars": int(meta.get("bars", len(ts))),
        "live_bars": int((states == "LIVE").sum()),
        "shadow_bars": int((states == "SHADOW").sum()),
        "stop_events": int(risk.str.contains("STOP_LOSS", regex=False).sum()),
        "take_events": int(risk.str.contains("TAKE_PROFIT", regex=False).sum()),
        "daily_halts": int(risk.str.contains("DAILY", regex=False).sum()),
        "metrics_path": str(metrics_path),
        "timeseries_path": str(ts_path),
    }
    return row


def _plot_rankings(summary: pd.DataFrame, out: Path) -> None:
    freqs = sorted(summary.rebalance_hours.dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for freq in freqs:
        group = summary[summary.rebalance_hours == freq].sort_values("return_pct", ascending=False)
        axes[0].plot(range(1, len(group) + 1), group.return_pct.to_numpy(), marker=".", ms=2,
                     linewidth=0.8, label=f"{freq:g}h")
        axes[1].scatter(group.max_drawdown_pct, group.return_pct, s=12, alpha=.55, label=f"{freq:g}h")
    axes[0].set_title("Shadow-engine return ranking")
    axes[0].set_xlabel("factor rank within rebalance window")
    axes[0].set_ylabel("live account return (%)")
    axes[1].set_title("Return vs maximum drawdown")
    axes[1].set_xlabel("max drawdown (%)")
    axes[1].set_ylabel("live account return (%)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_frequency(summary: pd.DataFrame, freq: float, out_dir: Path, top_n: int, max_points: int) -> None:
    group = summary[summary.rebalance_hours == freq].sort_values("return_pct", ascending=False).head(top_n)
    if group.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    colors = plt.cm.viridis([i / max(1, len(group) - 1) for i in range(len(group))])
    for color, (_, row) in zip(colors, group.iterrows()):
        ts = pd.read_csv(row.timeseries_path)
        ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True, format="ISO8601")
        ts = _thin(ts, max_points)
        label = f"{row.factor_id} ({row.return_pct:.1f}%)"
        axes[0, 0].plot(ts.timestamp, ts.equity, color=color, lw=1.0, label=label)
        if "shadow_equity" in ts:
            axes[0, 1].plot(ts.timestamp, ts.shadow_equity * 10000.0, color=color, lw=1.0, label=label)
        axes[1, 0].plot(ts.timestamp, ts.equity / ts.equity.iloc[0] - 1.0, color=color, lw=1.0, label=label)
        states = ts.state.astype(str) if "state" in ts else pd.Series("LIVE", index=ts.index)
        axes[1, 1].plot(ts.timestamp, (states == "SHADOW").astype(int), color=color, lw=.8, label=label)
    axes[0, 0].set_title(f"{freq:g}h live equity (top {top_n})")
    axes[0, 1].set_title(f"{freq:g}h counterfactual shadow NAV (top {top_n})")
    axes[1, 0].set_title("Live cumulative return")
    axes[1, 1].set_title("SHADOW state (1=shadow)")
    axes[1, 1].set_ylim(-.05, 1.05)
    for ax in axes.flat:
        ax.grid(alpha=.22)
        ax.legend(fontsize=6, ncol=2)
    fig.savefig(out_dir / f"{freq:g}h_top{top_n}_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--max-points", type=int, default=5000)
    args = ap.parse_args()
    out = args.out_dir or args.results_dir / "scan"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for p in sorted(args.results_dir.rglob("*.metrics.json")) if (r := _read_one(p))]
    if not rows:
        raise SystemExit("no complete metrics/timeseries pairs found")
    summary = pd.DataFrame(rows).sort_values(["rebalance_hours", "return_pct"], ascending=[True, False])
    summary.to_csv(out / "shadow_engine_scan.csv", index=False)
    summary.groupby("rebalance_hours", as_index=False).head(args.top_n).to_csv(out / "shadow_engine_top.csv", index=False)
    lines = ["# Shadow engine scan", "", f"- results: `{args.results_dir}`", f"- complete runs: {len(summary)}", ""]
    lines += ["| rebalance | factor | return % | max DD % | shadow NAV % | state | entries | exits | live bars | shadow bars |", "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for _, r in summary.head(100).iterrows():
        lines.append(f"| {r.rebalance_hours:g}h | `{r.factor_id}` | {r.return_pct:.4f} | {r.max_drawdown_pct:.4f} | {r.shadow_final_nav_pct:.4f} | {r.final_state} | {r.shadow_entries} | {r.shadow_exits} | {r.live_bars} | {r.shadow_bars} |")
    (out / "shadow_engine_scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_rankings(summary, out / "ranking_return_drawdown.png")
    for freq in sorted(summary.rebalance_hours.unique()):
        _plot_frequency(summary, float(freq), out, args.top_n, args.max_points)
    print(f"complete_runs={len(summary)} out={out}")
    print(summary.groupby("rebalance_hours").head(3)[["rebalance_hours", "factor_id", "return_pct", "max_drawdown_pct", "shadow_final_nav_pct", "final_state"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
