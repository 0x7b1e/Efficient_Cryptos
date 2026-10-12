"""Leakage-safe crypto features and labels (bars, funding, positioning -> one frame).

The torch-based sequence samples built from this frame live in ``ml/sections.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from efficient_cryptos.common.storage import LocalStore

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

POSITION_FEATURES = [
    "top_account_long_short_ratio", "top_account_long_share", "top_account_short_share",
    "top_position_long_short_ratio", "top_position_long_share", "top_position_short_share",
    "global_account_long_short_ratio", "global_account_long_share", "global_account_short_share",
    "taker_buy_sell_ratio", "taker_buy_volume_5m", "taker_sell_volume_5m",
]

CORE_FEATURES = [
    "log_open_price", "log_high_price", "log_low_price", "log_close_price",
    "log_ret_1", "log_ret_3", "log_ret_12", "intrabar_return", "open_gap", "high_low_range",
    "close_vwap_gap", "log_volume", "log_quote_volume", "log_trades",
    "volume_change", "quote_volume_change", "taker_buy_ratio", "taker_imbalance",
    "taker_buy_sell_quote_ratio", "taker_net_quote", "avg_trade_quote",
    "realized_vol_12", "realized_vol_48", "funding_rate", "funding_change",
] + POSITION_FEATURES

MULTISCALE_WINDOWS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 72, 96)
SLOW_WINDOWS = (6, 12, 24, 48, 72, 96)
FEATURE_SCHEMA_VERSION = 4


def interval_to_rule(interval: str) -> str:
    value, unit = int(interval[:-1]), interval[-1].lower()
    if value <= 0 or unit not in {"m", "h", "d"}:
        raise ValueError(f"Unsupported interval: {interval}")
    return f"{value}{ {'m': 'min', 'h': 'h', 'd': 'D'}[unit] }"


def _resample_symbol(klines: pd.DataFrame, funding: pd.DataFrame,
                     interval: str, symbol: str) -> pd.DataFrame:
    if klines.empty:
        return pd.DataFrame()
    df = klines.copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    numeric = [c for c in POSITION_FEATURES if c in df.columns]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    aggregations = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum", "trades": "sum",
        "taker_buy_base": "sum", "taker_buy_quote": "sum",
        **{col: "last" for col in numeric},
    }
    out = df.resample(interval_to_rule(interval), label="left", closed="left").agg(aggregations)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if numeric:
        # Positioning is published at 5m or slower. Carry the last public value
        # forward for at most 3 hours, never across large collection gaps.
        out[numeric] = out[numeric].ffill(limit=max(1, int(180 / _interval_minutes(interval))))
    if not funding.empty:
        f = funding[["funding_time", "funding_rate"]].copy()
        f["timestamp"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True)
        f["funding_rate"] = pd.to_numeric(f["funding_rate"], errors="coerce")
        f = f.dropna(subset=["timestamp"]).sort_values("timestamp")
        out = pd.merge_asof(
            out.reset_index().sort_values("timestamp"),
            f[["timestamp", "funding_rate"]].sort_values("timestamp"),
            on="timestamp", direction="backward", tolerance=pd.Timedelta("12h"),
        )
        # Keep the next *actual* settlement timestamp for the event label. It
        # is metadata for label construction only and is removed from the
        # generated-factor input, so it cannot leak into a factor expression.
        future_events = f[["timestamp", "funding_time"]].rename(
            columns={"timestamp": "next_event_lookup"}
        )
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            future_events.sort_values("next_event_lookup"),
            left_on="timestamp", right_on="next_event_lookup",
            direction="forward", allow_exact_matches=False,
            tolerance=pd.Timedelta("12h"),
        ).drop(columns=["next_event_lookup"], errors="ignore")
        out["next_funding_time"] = pd.to_datetime(
            out["funding_time"], unit="ms", utc=True, errors="coerce"
        )
        out = out.drop(columns=["funding_time"], errors="ignore").set_index("timestamp")
    else:
        out["funding_rate"] = np.nan
        out["next_funding_time"] = pd.NaT
    out["symbol"] = symbol
    return out.reset_index()


def _interval_minutes(interval: str) -> int:
    value, unit = int(interval[:-1]), interval[-1].lower()
    return value * {"m": 1, "h": 60, "d": 1440}[unit]


def _future_close_at_events(frame: pd.DataFrame) -> pd.Series:
    """Return the first available bar close at each row's next funding event."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if "next_funding_time" not in frame.columns:
        return result
    for _, index in frame.groupby("symbol", sort=False).groups.items():
        part = frame.loc[index].sort_values("timestamp")
        timestamps = pd.to_datetime(part["timestamp"], utc=True).astype("int64").to_numpy()
        event_times = pd.to_datetime(part["next_funding_time"], utc=True, errors="coerce")
        event_ns = event_times.astype("int64").to_numpy()
        valid = event_times.notna().to_numpy(copy=True)
        positions = np.searchsorted(timestamps, event_ns, side="left")
        valid &= positions < len(part)
        values = np.full(len(part), np.nan, dtype=float)
        if valid.any():
            closes = pd.to_numeric(part["close"], errors="coerce").to_numpy(dtype=float)
            values[valid] = closes[positions[valid]]
        result.loc[part.index] = values
    return result


def _derive_features(df: pd.DataFrame, horizon: int, label_mode: str = "horizon") -> pd.DataFrame:
    df = df.sort_values(["symbol", "timestamp"]).copy()
    grouped = df.groupby("symbol", sort=False)
    log_close = np.log(df["close"].clip(lower=1e-12))
    df["log_open_price"] = np.log(df["open"].clip(lower=1e-12))
    df["log_high_price"] = np.log(df["high"].clip(lower=1e-12))
    df["log_low_price"] = np.log(df["low"].clip(lower=1e-12))
    df["log_close_price"] = log_close
    df["log_ret_1"] = grouped["close"].transform(lambda s: np.log(s.clip(lower=1e-12)).diff())
    df["log_ret_3"] = grouped["close"].transform(lambda s: np.log(s.clip(lower=1e-12)).diff(3))
    df["log_ret_12"] = grouped["close"].transform(lambda s: np.log(s.clip(lower=1e-12)).diff(12))
    df["intrabar_return"] = np.log(df["close"].clip(lower=1e-12) / df["open"].clip(lower=1e-12))
    previous_close = grouped["close"].shift(1)
    df["open_gap"] = np.log(df["open"].clip(lower=1e-12) / previous_close.clip(lower=1e-12))
    df["high_low_range"] = np.log(df["high"].clip(lower=1e-12) / df["low"].clip(lower=1e-12))
    vwap = df["quote_volume"] / df["volume"].replace(0, np.nan)
    df["close_vwap_gap"] = np.log(df["close"].clip(lower=1e-12) / vwap.clip(lower=1e-12))
    df["log_volume"] = np.log1p(df["volume"])
    df["log_quote_volume"] = np.log1p(df["quote_volume"])
    df["log_trades"] = np.log1p(df["trades"])
    df["volume_change"] = grouped["volume"].transform(lambda s: np.log1p(s).diff())
    df["quote_volume_change"] = grouped["quote_volume"].transform(lambda s: np.log1p(s).diff())
    df["taker_buy_ratio"] = df["taker_buy_quote"] / df["quote_volume"].replace(0, np.nan)
    df["taker_imbalance"] = 2.0 * df["taker_buy_ratio"] - 1.0
    taker_sell_quote = (df["quote_volume"] - df["taker_buy_quote"]).clip(lower=0)
    df["taker_buy_sell_quote_ratio"] = df["taker_buy_quote"] / taker_sell_quote.replace(0, np.nan)
    df["taker_net_quote"] = df["taker_buy_quote"] - taker_sell_quote
    df["realized_vol_12"] = grouped["log_ret_1"].transform(lambda s: s.rolling(12).std())
    df["realized_vol_48"] = grouped["log_ret_1"].transform(lambda s: s.rolling(48).std())
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["funding_change"] = grouped["funding_rate"].diff()
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    candle_top = df[["open", "close"]].max(axis=1)
    candle_bottom = df[["open", "close"]].min(axis=1)
    df["candle_body_ratio"] = (df["close"] - df["open"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - candle_top) / candle_range
    df["lower_wick_ratio"] = (candle_bottom - df["low"]) / candle_range
    df["close_location"] = (df["close"] - df["low"]) / candle_range
    df["avg_trade_quote"] = df["quote_volume"] / df["trades"].replace(0, np.nan)
    df["taker_avg_trade_quote"] = df["taker_buy_quote"] / df["trades"].replace(0, np.nan)

    for window in MULTISCALE_WINDOWS:
        min_periods = max(2, window // 2)
        df[f"log_ret_{window}"] = grouped["close"].transform(
            lambda s, w=window: np.log(s.clip(lower=1e-12)).diff(w)
        )
        df[f"price_z_{window}"] = grouped["close"].transform(
            lambda s, w=window, m=min_periods: (
                (np.log(s.clip(lower=1e-12)) - np.log(s.clip(lower=1e-12)).rolling(w, min_periods=m).mean())
                / np.log(s.clip(lower=1e-12)).rolling(w, min_periods=m).std().replace(0, np.nan)
            )
        )
        df[f"realized_vol_{window}"] = grouped["log_ret_1"].transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).std()
        )
        df[f"downside_vol_{window}"] = grouped["log_ret_1"].transform(
            lambda s, w=window, m=min_periods: s.clip(upper=0).pow(2).rolling(w, min_periods=m).mean().pow(0.5)
        )
        df[f"range_mean_{window}"] = grouped["high_low_range"].transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean()
        )
        df[f"volume_z_{window}"] = grouped["log_volume"].transform(
            lambda s, w=window, m=min_periods: (s - s.rolling(w, min_periods=m).mean())
            / s.rolling(w, min_periods=m).std().replace(0, np.nan)
        )
        df[f"quote_volume_z_{window}"] = grouped["log_quote_volume"].transform(
            lambda s, w=window, m=min_periods: (s - s.rolling(w, min_periods=m).mean())
            / s.rolling(w, min_periods=m).std().replace(0, np.nan)
        )
        df[f"trades_z_{window}"] = grouped["log_trades"].transform(
            lambda s, w=window, m=min_periods: (s - s.rolling(w, min_periods=m).mean())
            / s.rolling(w, min_periods=m).std().replace(0, np.nan)
        )
        df[f"taker_imbalance_mean_{window}"] = grouped["taker_imbalance"].transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean()
        )
        df[f"taker_imbalance_std_{window}"] = grouped["taker_imbalance"].transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).std()
        )

    for window in SLOW_WINDOWS:
        min_periods = max(3, window // 2)
        df[f"ema_gap_{window}"] = grouped["close"].transform(
            lambda s, w=window: s / s.ewm(span=w, adjust=False).mean() - 1.0
        )
        delta = grouped["close"].diff()
        avg_up = delta.clip(lower=0).groupby(df["symbol"]).transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean()
        )
        avg_down = (-delta.clip(upper=0)).groupby(df["symbol"]).transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean()
        )
        df[f"rsi_{window}"] = avg_up / (avg_up + avg_down).replace(0, np.nan)
        illiquidity = df["log_ret_1"].abs() / df["quote_volume"].replace(0, np.nan)
        df[f"amihud_{window}"] = illiquidity.groupby(df["symbol"]).transform(
            lambda s, w=window, m=min_periods: s.rolling(w, min_periods=m).mean()
        )
        corr_name = f"ret_volume_corr_{window}"
        df[corr_name] = np.nan
        for _, index in df.groupby("symbol", sort=False).groups.items():
            part = df.loc[index]
            df.loc[index, corr_name] = part["log_ret_1"].rolling(
                window, min_periods=min_periods
            ).corr(part["quote_volume_change"]).to_numpy()

    for lag in (1, 2, 4, 8, 16, 32):
        df[f"funding_delta_{lag}"] = grouped["funding_rate"].diff(lag)
    for window in (12, 48, 96):
        min_periods = max(3, window // 2)
        df[f"funding_z_{window}"] = grouped["funding_rate"].transform(
            lambda s, w=window, m=min_periods: (s - s.rolling(w, min_periods=m).mean())
            / s.rolling(w, min_periods=m).std().replace(0, np.nan)
        )

    for column in [c for c in POSITION_FEATURES if c in df.columns]:
        values = pd.to_numeric(df[column], errors="coerce")
        if "volume" in column:
            df[f"{column}_log"] = np.log1p(values.clip(lower=0))
            source = f"{column}_log"
        elif "ratio" in column:
            df[f"{column}_log"] = np.log(values.clip(lower=1e-12))
            source = f"{column}_log"
        else:
            source = column
        source_group = df.groupby("symbol", sort=False)[source]
        for lag in (1, 3, 12):
            df[f"{column}_delta_{lag}"] = source_group.diff(lag)
        for window in (12, 48):
            min_periods = max(3, window // 2)
            df[f"{column}_z_{window}"] = source_group.transform(
                lambda s, w=window, m=min_periods: (s - s.rolling(w, min_periods=m).mean())
                / s.rolling(w, min_periods=m).std().replace(0, np.nan)
            )

    # Relative-to-market dynamics: useful in a cross-sectional target even
    # though the common market return itself would normalize to zero.
    df["market_return"] = df.groupby("timestamp")["log_ret_1"].transform("mean")
    for window in (12, 48, 96):
        min_periods = max(4, window // 2)
        beta_name, corr_name = f"market_beta_{window}", f"market_corr_{window}"
        df[beta_name] = np.nan
        df[corr_name] = np.nan
        for _, index in df.groupby("symbol", sort=False).groups.items():
            part = df.loc[index]
            market_var = part["market_return"].rolling(window, min_periods=min_periods).var()
            df.loc[index, beta_name] = (
                part["log_ret_1"].rolling(window, min_periods=min_periods).cov(part["market_return"])
                / market_var.replace(0, np.nan)
            ).to_numpy()
            df.loc[index, corr_name] = part["log_ret_1"].rolling(
                window, min_periods=min_periods
            ).corr(part["market_return"]).to_numpy()
    if label_mode == "horizon":
        next_log_close = grouped["close"].transform(
            lambda s: np.log(s.clip(lower=1e-12)).shift(-horizon)
        )
    elif label_mode == "utc_close":
        # Label each 30m bar by the return from its close to the final close of
        # the same UTC day. The final bar has zero raw return by construction.
        utc_day = df["timestamp"].dt.floor("D")
        day_close = df.groupby([df["symbol"], utc_day])["close"].transform("last")
        next_log_close = np.log(day_close.clip(lower=1e-12))
    elif label_mode == "funding_event":
        event_close = _future_close_at_events(df)
        next_log_close = np.log(event_close.clip(lower=1e-12))
    else:
        raise ValueError("label_mode must be 'horizon', 'utc_close' or 'funding_event'")
    df["raw_target"] = next_log_close - log_close
    df["target"] = df["raw_target"] - df.groupby("timestamp")["raw_target"].transform("mean")
    return df


def _feature_cache_path(data_dir: str | Path, symbols: Iterable[str], interval: str,
                        horizon: int, days: int | None, label_mode: str = "horizon") -> Path:
    key = "_".join(sorted(str(s).upper() for s in symbols))
    import hashlib
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    suffix = "all" if days is None else str(int(days))
    label = f"h{horizon}" if label_mode == "horizon" else str(label_mode)
    return Path(data_dir) / "features" / f"{interval}_{label}_d{suffix}_{digest}.parquet"


def _source_signature(store: LocalStore, symbols: Iterable[str]) -> dict[str, list[int]]:
    """Cheap cache invalidation signature for source parquet files."""
    signature: dict[str, list[int]] = {}
    for symbol in symbols:
        for kind, path in (("k", store.kline_path(symbol)), ("f", store.funding_path(symbol))):
            if path.exists():
                stat = path.stat()
                signature[f"{kind}:{symbol}"] = [int(stat.st_size), int(stat.st_mtime_ns)]
    return signature


def load_feature_frame(data_dir: str | Path, symbols: Iterable[str], *, interval: str = "15m",
                       horizon: int = 1, days: int | None = 30,
                       require_positioning: bool = True, cache: bool = True,
                       rebuild_cache: bool = False,
                       label_mode: str = "horizon",
                       layered_cache: bool = False) -> tuple[pd.DataFrame, list[str]]:
    if layered_cache:
        # Import lazily to keep the legacy cache path dependency-free and to
        # avoid a module cycle: cache_layers uses this function for its first
        # base-feature build.
        from efficient_cryptos.features.cache import load_layered_feature_frame
        return load_layered_feature_frame(
            data_dir, symbols, interval=interval, horizon=horizon, days=days,
            require_positioning=require_positioning, cache=cache,
            rebuild_cache=rebuild_cache, label_mode=label_mode,
        )
    symbols = tuple(str(s).upper() for s in symbols)
    store = LocalStore(data_dir)
    source_signature = _source_signature(store, symbols)
    cache_path = _feature_cache_path(data_dir, symbols, interval, horizon, days, label_mode)
    metadata_path = cache_path.with_suffix(".json")
    if cache and not rebuild_cache and cache_path.exists() and metadata_path.exists():
        try:
            import json
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (int(meta.get("feature_schema_version", -1)) == FEATURE_SCHEMA_VERSION
                    and meta.get("source_signature") == source_signature):
                frame = pd.read_parquet(cache_path)
                return frame, [str(x) for x in meta["features"]]
        except Exception:
            pass
    frames = []
    for symbol in symbols:
        klines = store.load_klines(symbol)
        if klines.empty:
            continue
        if days is not None:
            cutoff = int(klines.open_time.max()) - days * 86_400_000
            klines = klines[klines.open_time >= cutoff]
        frames.append(_resample_symbol(klines, store.load_funding(symbol), interval, symbol))
    if not frames:
        raise ValueError("No local kline data found for the selected symbols")
    frame = _derive_features(pd.concat(frames, ignore_index=True), horizon, label_mode)
    # Preserve the exchange funding rate for cash accounting. The model-facing
    # ``funding_rate`` feature is cross-sectionally normalized below and must
    # never be reused as an actual payment rate by the backtester.
    frame["funding_rate_raw"] = frame["funding_rate"]
    present_positioning = [c for c in POSITION_FEATURES if c in frame.columns]
    missing_positioning = [c for c in POSITION_FEATURES if c not in frame.columns]
    if require_positioning and missing_positioning:
        raise ValueError(
            "Positioning columns are missing. Run scripts/01_data/download_data.py without "
            f"--skip-positioning. Missing: {', '.join(missing_positioning)}"
        )
    excluded = {
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        "funding_time", "next_funding_time", "open_time", "raw_target", "target", "market_return",
        "funding_rate_raw",
    }
    features = [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    if require_positioning:
        feature_valid = frame[present_positioning].notna().all(axis=1)
        frame = frame[feature_valid].copy()
        if frame.empty:
            raise ValueError(
                "No rows contain the required long-short positioning inputs. "
                "Download recent positioning data first."
            )
    # Recompute the market-neutral target on the actually tradable universe.
    frame["target"] = frame["raw_target"] - frame.groupby("timestamp")["raw_target"].transform("mean")
    # Cross-sectional normalization uses only information available at that bar.
    def zscore(series: pd.Series) -> pd.Series:
        # Match live inference exactly (population standard deviation).
        std = series.std(ddof=0)
        return (series - series.mean()) / std if pd.notna(std) and std > 1e-12 else series * 0.0
    frame[features] = frame.groupby("timestamp")[features].transform(zscore)
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)
    frame = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
        import json
        metadata_path.write_text(json.dumps({"features": features, "symbols": symbols,
                                             "interval": interval, "horizon": horizon,
                                             "label_mode": label_mode, "days": days,
                                             "feature_schema_version": FEATURE_SCHEMA_VERSION,
                                             "source_signature": source_signature},
                                            ensure_ascii=False, indent=2), encoding="utf-8")
    return frame, features


def add_alpha_features(frame: pd.DataFrame, families: Iterable[str], alpha_names: Iterable[str],
                       max_features: int = 24) -> tuple[pd.DataFrame, list[str]]:
    """Compute selected Alpha101/191 formulas in memory and merge them.

    This intentionally does not create an alpha cache under ``data`` so the
    data directory remains limited to klines and funding.
    """
    from efficient_cryptos.factors.alpha import alpha_class

    families = [x.strip().lower() for x in families if x.strip()]
    requested = [x.strip().lower() for x in alpha_names if x.strip()]
    if not families:
        return frame, []
    fields = {}
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        fields[column] = frame.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    fields["amount"] = fields["quote_volume"]
    fields["vwap"] = fields["amount"] / fields["volume"].replace(0, np.nan)
    benchmark_open = fields["open"].get("BTCUSDT", fields["open"].mean(axis=1))
    benchmark_close = fields["close"].get("BTCUSDT", fields["close"].mean(axis=1))
    fields["benchmark_open"] = pd.DataFrame(
        {symbol: benchmark_open for symbol in fields["close"].columns}, index=fields["close"].index
    )
    fields["benchmark_close"] = pd.DataFrame(
        {symbol: benchmark_close for symbol in fields["close"].columns}, index=fields["close"].index
    )
    popbo_input = pd.concat({key: fields[key] for key in (
        "open", "close", "high", "low", "volume", "amount", "vwap",
        "benchmark_open", "benchmark_close",
    )}, axis=1)
    added = []
    out = frame
    for family in families:
        cls = alpha_class(family)
        available = sorted(m for m in dir(cls) if m.startswith("alpha") and callable(getattr(cls, m)))
        names = available if not requested or requested == ["all"] else [x for x in requested if x in available]
        for name in names:
            if max_features > 0 and len(added) >= max_features:
                break
            feature_name = f"{family}_{name}"
            try:
                factor = getattr(cls(popbo_input), name)()
                factor.index = pd.to_datetime(factor.index, utc=True)
                try:
                    stacked = factor.stack(future_stack=True)
                except TypeError:
                    stacked = factor.stack(dropna=False)
                long = stacked.rename(feature_name).rename_axis(
                    index=["timestamp", "symbol"]
                ).reset_index()
                out = out.merge(long, on=["timestamp", "symbol"], how="left")
                added.append(feature_name)
            except Exception as exc:
                warnings.warn(f"Skipping {feature_name}: {type(exc).__name__}: {exc}")
        if max_features > 0 and len(added) >= max_features:
            break
    if added:
        def zscore(series: pd.Series) -> pd.Series:
            std = series.std(ddof=0)
            return (series - series.mean()) / std if pd.notna(std) and std > 1e-12 else series * 0.0
        out[added] = out.groupby("timestamp")[added].transform(zscore)
        out[added] = out[added].replace([np.inf, -np.inf], np.nan)
    return out, added
