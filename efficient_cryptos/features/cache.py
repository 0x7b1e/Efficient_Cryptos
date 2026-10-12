"""Layered feature/label caches for repeated model and factor experiments.

The legacy cache stores features and one target together.  This module adds a
stable base-feature cache plus independent label caches so changing a model,
loss, sequence length, or forecast horizon does not rebuild the expensive
resampling/indicator layer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from efficient_cryptos.features.frame import POSITION_FEATURES, load_feature_frame
from efficient_cryptos.common.storage import LocalStore

CACHE_SCHEMA_VERSION = 2


def _symbols_key(symbols: Iterable[str]) -> str:
    value = "_".join(sorted(str(s).upper() for s in symbols))
    return hashlib.sha1(value.encode()).hexdigest()[:12]


def _source_signature(store: LocalStore, symbols: Iterable[str]) -> dict[str, list[int]]:
    signature: dict[str, list[int]] = {}
    for symbol in symbols:
        for kind, path in (("k", store.kline_path(symbol)), ("f", store.funding_path(symbol))):
            if path.exists():
                stat = path.stat()
                signature[f"{kind}:{str(symbol).upper()}"] = [
                    int(stat.st_size), int(stat.st_mtime_ns)
                ]
    return signature


def _base_paths(data_dir: str | Path, symbols: Iterable[str], interval: str,
                days: int | None) -> tuple[Path, Path]:
    suffix = "all" if days is None else str(int(days))
    stem = f"base_{interval}_d{suffix}_{_symbols_key(symbols)}"
    root = Path(data_dir) / "features"
    return root / f"{stem}.parquet", root / f"{stem}.json"


def _label_paths(data_dir: str | Path, symbols: Iterable[str], interval: str,
                 days: int | None, horizon: int, label_mode: str) -> tuple[Path, Path]:
    suffix = "all" if days is None else str(int(days))
    label = f"h{int(horizon)}" if label_mode == "horizon" else str(label_mode)
    stem = f"label_{interval}_{label}_d{suffix}_{_symbols_key(symbols)}"
    root = Path(data_dir) / "features"
    return root / f"{stem}.parquet", root / f"{stem}.json"


def _read_valid_cache(path: Path, meta_path: Path, expected: dict) -> pd.DataFrame | None:
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta != expected:
            return None
        return pd.read_parquet(path)
    except Exception:
        return None


def load_base_features(data_dir: str | Path, symbols: Iterable[str], *, interval: str = "30m",
                       days: int | None = 180, cache: bool = True,
                       rebuild_cache: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Load or build the expensive resampled/derived feature layer.

    The first build delegates to the reference feature implementation with a
    temporary horizon.  The temporary target is discarded; all price,
    volume, funding, positioning and market-relative columns are retained.
    """
    symbols = tuple(str(s).upper() for s in symbols)
    store = LocalStore(data_dir)
    path, meta_path = _base_paths(data_dir, symbols, interval, days)
    source_signature = _source_signature(store, symbols)
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "kind": "base_features",
        "symbols": list(symbols),
        "interval": interval,
        "days": days,
        "source_signature": source_signature,
    }
    if cache and not rebuild_cache:
        cached = _read_valid_cache(path, meta_path, expected)
        if cached is not None:
            excluded = _excluded_columns()
            return cached, [c for c in cached.columns if c not in excluded
                            and pd.api.types.is_numeric_dtype(cached[c])]

    # The existing implementation already performs leakage-safe feature
    # construction and cross-sectional normalization.  Build it once, then
    # remove only the target columns from the persisted base layer.
    frame, features = load_feature_frame(
        data_dir, symbols, interval=interval, horizon=1, days=days,
        require_positioning=False, cache=False, label_mode="horizon",
    )
    base = frame.drop(columns=["raw_target", "target"], errors="ignore")
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        base.to_parquet(path, index=False)
        meta_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return base, features


def _excluded_columns() -> set[str]:
    return {
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        "funding_time", "next_funding_time", "open_time", "raw_target", "target", "market_return",
        "funding_rate_raw",
    }


def _make_label(base: pd.DataFrame, horizon: int, label_mode: str) -> pd.DataFrame:
    columns = ["timestamp", "symbol", "close"]
    if label_mode == "funding_event" and "next_funding_time" in base.columns:
        columns.append("next_funding_time")
    frame = base[columns].copy()
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", sort=False)
    log_close = np.log(frame["close"].clip(lower=1e-12))
    if label_mode == "horizon":
        future = grouped["close"].transform(
            lambda series: np.log(series.clip(lower=1e-12)).shift(-int(horizon))
        )
    elif label_mode == "utc_close":
        utc_day = frame["timestamp"].dt.floor("D")
        day_close = frame.groupby([frame["symbol"], utc_day])["close"].transform("last")
        future = np.log(day_close.clip(lower=1e-12))
    elif label_mode == "funding_event":
        from efficient_cryptos.features.frame import _future_close_at_events
        event_close = _future_close_at_events(frame)
        future = np.log(event_close.clip(lower=1e-12))
    else:
        raise ValueError("label_mode must be 'horizon', 'utc_close' or 'funding_event'")
    frame["raw_target"] = future - log_close
    frame["target"] = frame["raw_target"] - frame.groupby("timestamp")["raw_target"].transform("mean")
    return frame[["timestamp", "symbol", "raw_target", "target"]]


def load_layered_feature_frame(data_dir: str | Path, symbols: Iterable[str], *, interval: str = "30m",
                               horizon: int = 1, days: int | None = 180,
                               require_positioning: bool = True, cache: bool = True,
                               rebuild_cache: bool = False,
                               label_mode: str = "horizon") -> tuple[pd.DataFrame, list[str]]:
    """Join a reusable base feature cache with a label-specific cache."""
    symbols = tuple(str(s).upper() for s in symbols)
    base, features = load_base_features(
        data_dir, symbols, interval=interval, days=days,
        cache=cache, rebuild_cache=rebuild_cache,
    )
    store = LocalStore(data_dir)
    path, meta_path = _label_paths(data_dir, symbols, interval, days, horizon, label_mode)
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "kind": "labels",
        "symbols": list(symbols),
        "interval": interval,
        "days": days,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "source_signature": _source_signature(store, symbols),
    }
    labels = None if rebuild_cache else _read_valid_cache(path, meta_path, expected)
    if labels is None:
        labels = _make_label(base, horizon, label_mode)
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            labels.to_parquet(path, index=False)
            meta_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")

    frame = base.merge(labels, on=["timestamp", "symbol"], how="inner", validate="one_to_one")
    if require_positioning:
        missing = [c for c in POSITION_FEATURES if c not in frame.columns]
        if missing:
            raise ValueError("Positioning columns are missing: " + ", ".join(missing))
        frame = frame[frame[POSITION_FEATURES].notna().all(axis=1)].copy()
        if frame.empty:
            raise ValueError("No rows contain the required long-short positioning inputs")
    frame = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return frame, features
