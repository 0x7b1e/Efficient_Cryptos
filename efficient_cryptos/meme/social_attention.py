"""Timestamp-safe social attention aggregation.

This module intentionally only consumes an exported, timestamped dataset.  It
does not bypass Binance access controls or scrape a website implicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = {"timestamp", "symbol"}


def build_attention_features(posts: pd.DataFrame, *, freq: str = "1min", baseline: str = "6h") -> pd.DataFrame:
    missing = REQUIRED - set(posts.columns)
    if missing:
        raise ValueError(f"missing social columns: {sorted(missing)}")
    frame = posts.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor(freq)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    for column in ("engagement", "sentiment"):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    grouped = frame.groupby(["timestamp", "symbol"], sort=True)
    out = grouped.agg(
        mention_count=("symbol", "size"),
        unique_authors=("author", "nunique") if "author" in frame else ("symbol", "size"),
        engagement=("engagement", "sum"),
        sentiment=("sentiment", "mean"),
    ).reset_index()
    out = out.sort_values(["symbol", "timestamp"])
    g = out.groupby("symbol", sort=False)
    baseline_minutes = max(1, int(pd.Timedelta(baseline).total_seconds() // 60))
    min_periods = min(30, baseline_minutes)
    for column in ("mention_count", "unique_authors", "engagement"):
        # The input is already bucketed to ``freq``.  Use an integer rolling
        # window here so each symbol can be transformed independently without
        # requiring a fragile time-based MultiIndex.
        base = g[column].transform(
            lambda s, w=baseline_minutes: s.rolling(w, min_periods=min_periods).mean()
        )
        out[f"{column}_shock"] = np.log(out[column] / base.replace(0, np.nan))
    out["attention_score"] = (
        out["mention_count_shock"].fillna(0)
        + out["unique_authors_shock"].fillna(0)
        + out["engagement_shock"].fillna(0)
        + out["sentiment"].fillna(0).clip(-1, 1)
    )
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
