"""IC/RankIC diagnostics and TensorBoard-ready summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_section_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for timestamp, part in frame.groupby("timestamp", sort=True):
        valid = part[["prediction", "target"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 2 or valid.nunique().min() < 2:
            ic = rank_ic = np.nan
        else:
            ic = valid["prediction"].corr(valid["target"], method="pearson")
            rank_ic = valid["prediction"].corr(valid["target"], method="spearman")
        rows.append({"timestamp": timestamp, "ic": ic, "rank_ic": rank_ic, "n": len(valid)})
    return pd.DataFrame(rows)


def summarize_metrics(frame: pd.DataFrame, quantile: float = 0.2) -> dict[str, float]:
    sections = per_section_metrics(frame)
    spreads, accuracies = [], []
    for _, part in frame.groupby("timestamp", sort=True):
        valid = part[["prediction", "target"]].dropna().sort_values("prediction")
        if not valid.empty:
            accuracies.append(float(((valid.prediction > 0) == (valid.target > 0)).mean()))
        if len(valid) >= 5:
            k = max(1, int(len(valid) * quantile))
            spreads.append(float(valid.target.iloc[-k:].mean() - valid.target.iloc[:k].mean()))
    ic_mean, rank_mean = sections.ic.mean(), sections.rank_ic.mean()
    ic_std, rank_std = sections.ic.std(), sections.rank_ic.std()
    spread_mean = float(np.mean(spreads)) if spreads else np.nan
    spread_std = float(np.std(spreads)) if spreads else np.nan
    return {
        "ic": float(ic_mean),
        "rank_ic": float(rank_mean),
        "icir": float(ic_mean / ic_std) if pd.notna(ic_std) and ic_std > 0 else np.nan,
        "rank_icir": float(rank_mean / rank_std) if pd.notna(rank_std) and rank_std > 0 else np.nan,
        "long_short_spread": spread_mean,
        "long_short_ir": float(spread_mean / spread_std) if pd.notna(spread_std) and spread_std > 0 else np.nan,
        "direction_accuracy": float(np.mean(accuracies)) if accuracies else np.nan,
        "sections": float(len(sections)),
        "samples": float(len(frame)),
    }
