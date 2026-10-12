"""Short-horizon event feature construction from 1-minute bars."""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = {"timestamp", "symbol", "open", "high", "low", "close", "quote_volume"}


def _zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return series * 0.0
    return (series - mean) / std


def _cs_zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame[columns] = frame.groupby("timestamp", sort=False)[columns].transform(_zscore)
    return frame


def build_event_features(bars: pd.DataFrame, *, add_cross_sectional_scores: bool = True) -> pd.DataFrame:
    """Build causal 1m/5m event features.

    Optional columns ``trades`` and ``taker_buy_quote`` improve event quality;
    when absent, the corresponding features remain NaN rather than being
    silently fabricated.
    """
    missing = REQUIRED - set(bars.columns)
    if missing:
        raise ValueError(f"missing required bar columns: {sorted(missing)}")
    out = bars.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])
    for column in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    g = out.groupby("symbol", sort=False)
    close = pd.to_numeric(out["close"], errors="coerce").clip(lower=1e-12)
    quote = pd.to_numeric(out["quote_volume"], errors="coerce").clip(lower=0)
    out["log_ret_1m"] = g["close"].transform(lambda s: np.log(s.clip(lower=1e-12)).diff())
    for window in (3, 5, 10):
        out[f"ret_{window}m"] = g["close"].transform(
            lambda s, w=window: np.log(s.clip(lower=1e-12)).diff(w)
        )
    out["momentum_accel_1m"] = g["log_ret_1m"].diff()
    out["realized_vol_10m"] = g["log_ret_1m"].transform(
        lambda s: s.rolling(10, min_periods=5).std()
    )
    out["quote_volume_1m"] = quote
    out["quote_volume_5m"] = g["quote_volume"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    out["quote_volume_10m"] = g["quote_volume"].transform(
        lambda s: s.rolling(10, min_periods=10).sum()
    )
    out["quote_volume_baseline_60m"] = g["quote_volume"].transform(
        lambda s: s.rolling(60, min_periods=30).median()
    )
    out["volume_shock_5m"] = np.log(
        out["quote_volume_5m"] / out["quote_volume_baseline_60m"].replace(0, np.nan)
    )
    out["volume_shock_10m"] = np.log(
        out["quote_volume_10m"] / (out["quote_volume_baseline_60m"] * 2).replace(0, np.nan)
    )

    if "trades" in out:
        out["trade_count_5m"] = g["trades"].transform(lambda s: s.rolling(5, min_periods=5).sum())
        out["trade_count_baseline_60m"] = g["trades"].transform(lambda s: s.rolling(60, min_periods=30).median())
        out["trade_intensity_shock"] = np.log1p(out["trade_count_5m"] / out["trade_count_baseline_60m"].replace(0, np.nan))
    else:
        out["trade_intensity_shock"] = np.nan

    if "taker_buy_quote" in out:
        buy = pd.to_numeric(out["taker_buy_quote"], errors="coerce").clip(lower=0)
        sell = (quote - buy).clip(lower=0)
        out["taker_buy_ratio"] = buy / quote.replace(0, np.nan)
        out["order_flow_imbalance"] = (buy - sell) / (buy + sell).replace(0, np.nan)
        out["ofi_5m"] = g["order_flow_imbalance"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    else:
        out["taker_buy_ratio"] = np.nan
        out["order_flow_imbalance"] = np.nan
        out["ofi_5m"] = np.nan

    score_columns = ["ret_3m", "ret_5m", "volume_shock_5m", "trade_intensity_shock", "ofi_5m"]
    if add_cross_sectional_scores:
        available = [c for c in score_columns if c in out]
        out = _cs_zscore(out, available)
        out["event_score"] = (
            0.25 * out["ret_3m"].fillna(0)
            + 0.25 * out["ret_5m"].fillna(0)
            + 0.25 * out["volume_shock_5m"].fillna(0)
            + 0.15 * out["trade_intensity_shock"].fillna(0)
            + 0.10 * out["ofi_5m"].fillna(0)
        )
        out["event_percentile"] = out.groupby("timestamp")["event_score"].rank(pct=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_forward_labels(features: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 30)) -> pd.DataFrame:
    """Add future log-return labels; use only after event features are frozen."""
    out = features.copy().sort_values(["symbol", "timestamp"])
    g = out.groupby("symbol", sort=False)
    for horizon in horizons:
        out[f"future_ret_{horizon}m"] = g["close"].transform(
            lambda s, h=horizon: np.log(s.shift(-h) / s)
        )
    return out
