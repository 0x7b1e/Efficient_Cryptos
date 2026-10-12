"""Per-factor research reports and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from efficient_cryptos.backtest.portfolio import decile_profile


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_factor_report(predictions: pd.DataFrame, lifecycle: pd.DataFrame,
                        output_dir: str | Path, *, factor_id: str,
                        metrics: dict | None = None) -> dict[str, str]:
    """Write CSVs and a compact per-factor PNG report.

    The main chart is a ten-decile profile with explicit Top, Bottom and
    Top-Bottom spread. Additional panels show cumulative decile equity, signal
    IC, lifecycle equity/drawdown, turnover and trading costs.
    """
    root = Path(output_dir) / str(factor_id)
    root.mkdir(parents=True, exist_ok=True)
    profile = decile_profile(predictions, 10)
    profile.to_csv(root / "decile_observations.csv", index=False)
    if profile.empty:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "Insufficient valid cross-sectional observations", ha="center", va="center")
        ax.set_axis_off()
        fig.suptitle(f"Factor report: {factor_id}")
        _savefig(fig, root / "report.png")
        return {"report": str(root / "report.png")}

    avg = profile.groupby("decile", as_index=False).agg(
        mean_return=("mean_return", "mean"), observations=("mean_return", "size"),
    )
    avg["cumulative_return"] = avg["mean_return"].add(1.0).cumprod().sub(1.0)
    top = profile[profile["decile"] == 10].set_index("timestamp")["mean_return"]
    bottom = profile[profile["decile"] == 1].set_index("timestamp")["mean_return"]
    spread = pd.concat([top.rename("top"), bottom.rename("bottom")], axis=1).dropna()
    spread["top_bottom"] = spread["top"] - spread["bottom"]
    spread.to_csv(root / "top_bottom_spread.csv")
    avg.to_csv(root / "decile_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    x = avg["decile"].to_numpy()
    colors = ["#b2182b" if d == 1 else "#2166ac" if d == 10 else "#999999" for d in x]
    axes[0, 0].bar(x, avg["mean_return"] * 10_000, color=colors)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("10-decile forward return")
    axes[0, 0].set_xlabel("Prediction decile (1=bottom, 10=top)")
    axes[0, 0].set_ylabel("Mean return (bps)")

    for decile, values in profile.groupby("decile"):
        curve = (1.0 + values.sort_values("timestamp")["mean_return"]).cumprod()
        axes[0, 1].plot(values.sort_values("timestamp")["timestamp"], curve,
                        label=f"D{int(decile)}", linewidth=1.0,
                        color="#2166ac" if decile == 10 else "#b2182b" if decile == 1 else None)
    axes[0, 1].set_title("Decile cumulative return")
    axes[0, 1].legend(ncol=2, fontsize=7)
    axes[0, 1].grid(alpha=0.25)

    axes[0, 2].plot(spread.index, spread["top"] * 10_000, label="Top/D10")
    axes[0, 2].plot(spread.index, spread["bottom"] * 10_000, label="Bottom/D1")
    axes[0, 2].plot(spread.index, spread["top_bottom"] * 10_000, label="Top-Bottom", linewidth=1.5)
    axes[0, 2].axhline(0, color="black", linewidth=0.8)
    axes[0, 2].set_title("Top / Bottom spread")
    axes[0, 2].set_ylabel("Return (bps)")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.25)

    if lifecycle is not None and not lifecycle.empty:
        ts = pd.to_datetime(lifecycle["timestamp"], utc=True)
        equity = lifecycle["equity"]
        drawdown = equity / equity.cummax() - 1.0
        axes[1, 0].plot(ts, equity, color="#2166ac")
        axes[1, 0].set_title("Lifecycle equity")
        axes[1, 1].fill_between(ts, drawdown * 100, 0, color="#b2182b", alpha=0.6)
        axes[1, 1].set_title("Drawdown")
        axes[1, 1].set_ylabel("Drawdown (%)")
        axes[1, 2].plot(ts, lifecycle.get("turnover", 0), label="Turnover")
        if "cost" in lifecycle:
            axes[1, 2].plot(ts, lifecycle["cost"] * 10_000, label="Cost (bps)")
        if "funding_cost" in lifecycle:
            axes[1, 2].plot(ts, lifecycle["funding_cost"] * 10_000, label="Funding (bps)")
        axes[1, 2].set_title("Turnover / costs")
        axes[1, 2].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(f"Factor report: {factor_id}")
    _savefig(fig, root / "report.png")
    return {
        "report": str(root / "report.png"),
        "decile_summary": str(root / "decile_summary.csv"),
        "top_bottom": str(root / "top_bottom_spread.csv"),
    }
