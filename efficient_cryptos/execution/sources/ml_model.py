"""Live inference for a trained cross-sectional crypto sequence model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch

from efficient_cryptos.features.frame import POSITION_FEATURES, _derive_features, _resample_symbol
from efficient_cryptos.ml.models import build_model
from efficient_cryptos.common.storage import LocalStore
from efficient_cryptos.features.frame import load_feature_frame
from efficient_cryptos.common.signal_policy import signal_is_due


DEFAULT_MODEL_DIR = Path("outputs/crypto_t1/resnet_mse_ic_spread_30m_t1_seed42")
POSITION_ENDPOINTS = {
    "topLongShortAccountRatio": {
        "longShortRatio": "top_account_long_short_ratio",
        "longAccount": "top_account_long_share",
        "shortAccount": "top_account_short_share",
    },
    "topLongShortPositionRatio": {
        "longShortRatio": "top_position_long_short_ratio",
        "longAccount": "top_position_long_share",
        "shortAccount": "top_position_short_share",
        "longPosition": "top_position_long_share",
        "shortPosition": "top_position_short_share",
    },
    "globalLongShortAccountRatio": {
        "longShortRatio": "global_account_long_short_ratio",
        "longAccount": "global_account_long_share",
        "shortAccount": "global_account_short_share",
    },
    "takerlongshortRatio": {
        "buySellRatio": "taker_buy_sell_ratio",
        "buyVol": "taker_buy_volume_5m",
        "sellVol": "taker_sell_volume_5m",
    },
}


@dataclass(frozen=True)
class LiveModel:
    model: Any
    args: Dict[str, Any]
    features: tuple[str, ...]
    device: torch.device

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(
            symbol.strip().upper()
            for symbol in str(self.args["symbols"]).split(",")
            if symbol.strip()
        )


@dataclass(frozen=True)
class SignalResult:
    signal_time_ms: int
    scores: Dict[str, float]
    targets: Dict[str, float]


def choose_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(
            "use --device auto, cuda, mps, or cpu"
        )
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA GPU is available")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but Apple GPU support is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _checkpoint_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "__dict__"):
        return dict(vars(raw))
    raise TypeError("Checkpoint args must be a dict or argparse namespace")


def load_live_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    device: str = "auto",
) -> LiveModel:
    path = Path(model_dir).expanduser()
    if path.is_file():
        checkpoint_path = path
    else:
        checkpoint_path = path / "best_model.pt"
        if not checkpoint_path.exists() and (path / "model.txt").exists():
            checkpoint_path = path / "model.txt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    # Tree models do not use PyTorch devices; neural checkpoints require GPU.
    target_device = torch.device("cpu") if checkpoint_path.suffix == ".txt" else choose_device(device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=target_device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
    if checkpoint_path.name == "model.txt" or checkpoint_path.suffix == ".txt":
        import json
        meta = json.loads((checkpoint_path.parent / "features.json").read_text(encoding="utf-8"))
        args = _checkpoint_args(meta["args"])
        features = tuple(str(name) for name in meta["selected"])
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Loading an lgbm model requires lightgbm") from exc
        return LiveModel(model=lgb.Booster(model_file=str(checkpoint_path)), args=args,
                         features=features, device=torch.device("cpu"))
    args = _checkpoint_args(checkpoint["args"])
    features = tuple(str(name) for name in checkpoint["features"])
    model = build_model(
        args["model"],
        int(args["seq_len"]),
        len(features),
        int(args.get("hidden_dim", 128)),
        float(args.get("dropout", 0.1)),
        int(args.get("resnet_blocks", 3)),
        int(args.get("attention_layers", 2)),
        int(args.get("attention_heads", 8)),
        float(args.get("moe_tau", 1.0)),
        int(args.get("moe_expert_hidden", 8)),
        int(args.get("tlob_layers", 4)),
        int(args.get("tlob_heads", 1)),
        str(args.get("foundation_model_name", "")),
        not bool(args.get("unfreeze_foundation", False)),
    ).to(target_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return LiveModel(model=model, args=args, features=features, device=target_device)


def interval_minutes(interval: str) -> int:
    value, unit = int(interval[:-1]), interval[-1].lower()
    return value * {"m": 1, "h": 60, "d": 1440}[unit]


def latest_closed_bar_open_ms(now_ms: int | None, bar_minutes: int) -> int:
    current = int(now_ms if now_ms is not None else time.time() * 1000)
    bar_ms = int(bar_minutes) * 60_000
    return current // bar_ms * bar_ms - bar_ms


def is_scheduled_signal(signal_time_ms: int, bar_minutes: int, rebalance_bars: int) -> bool:
    return signal_is_due(signal_time_ms, bar_minutes, rebalance_bars)


def _funding_frame(rows: Iterable[Dict[str, Any]], symbol: str) -> pd.DataFrame:
    data = []
    for row in rows:
        funding_time = row.get("fundingTime", row.get("funding_time"))
        funding_rate = row.get("fundingRate", row.get("funding_rate"))
        if funding_time is None or funding_rate is None:
            continue
        data.append({
            "funding_time": int(funding_time),
            "symbol": symbol,
            "funding_rate": float(funding_rate),
        })
    return pd.DataFrame(data, columns=["funding_time", "symbol", "funding_rate"])


def _needs_positioning(features: Iterable[str]) -> bool:
    return any(base in feature for feature in features for base in POSITION_FEATURES)


def _paged_klines(client: Any, symbol: str, interval: str, start_ms: int,
                   end_ms: int, step_ms: int) -> list[list[Any]]:
    rows_all: list[list[Any]] = []
    cursor = int(start_ms)
    while cursor <= end_ms:
        rows = client.klines(symbol, interval, limit=1500, start_ms=cursor,
                             end_ms=end_ms, futures=True)
        if not rows:
            break
        rows_all.extend(rows)
        last = int(rows[-1][0])
        if last < cursor:
            break
        cursor = last + step_ms
        if len(rows) < 1500:
            break
    # Network retries or boundary semantics can return a duplicate row.
    by_time = {int(row[0]): row for row in rows_all}
    return [by_time[key] for key in sorted(by_time)]


def _fetch_positioning(client: Any, symbol: str, start_ms: int,
                       end_ms: int) -> pd.DataFrame:
    frames = []
    step_ms = 5 * 60_000
    for endpoint, mapping in POSITION_ENDPOINTS.items():
        rows_all = []
        cursor = start_ms
        while cursor <= end_ms:
            rows = client.futures_data_history(
                endpoint, symbol, period="5m", limit=500,
                start_ms=cursor, end_ms=end_ms,
            )
            if not rows:
                break
            rows_all.extend(rows)
            last = int(rows[-1]["timestamp"])
            if last < cursor:
                break
            cursor = last + step_ms
            if len(rows) < 500:
                break
        records = []
        for row in rows_all:
            item = {"open_time": int(row["timestamp"])}
            for source, destination in mapping.items():
                if source in row:
                    item[destination] = float(row[source])
            records.append(item)
        if records:
            frames.append(pd.DataFrame(records).drop_duplicates("open_time", keep="last"))
    if not frames:
        return pd.DataFrame(columns=["open_time"])
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="open_time", how="outer")
    return result


def fetch_feature_frame(
    client: Any,
    bundle: LiveModel,
    *,
    now_ms: int | None = None,
    history_bars: int = 260,
) -> tuple[pd.DataFrame, int]:
    interval = str(bundle.args["interval"])
    bar_minutes = interval_minutes(interval)
    bar_ms = bar_minutes * 60_000
    signal_time_ms = latest_closed_bar_open_ms(now_ms, bar_minutes)
    end_ms = signal_time_ms + bar_ms - 1
    history_bars = max(int(history_bars), int(bundle.args["seq_len"]) + 100)
    needs_positioning = _needs_positioning(bundle.features)
    frames = []
    for symbol in bundle.symbols:
        if needs_positioning:
            if bar_minutes < 5 or bar_minutes % 5:
                raise ValueError("positioning features require an interval divisible by 5m")
            source_interval, source_minutes = "5m", 5
            source_bars = history_bars * (bar_minutes // source_minutes)
            start_ms = signal_time_ms - max(0, source_bars - bar_minutes // source_minutes) * 60_000 * source_minutes
            rows = _paged_klines(
                client, symbol, source_interval, start_ms, end_ms,
                source_minutes * 60_000,
            )
        else:
            rows = client.klines(
                symbol, interval, limit=min(history_bars, 1500),
                end_ms=end_ms, futures=True,
            )
        klines = LocalStore.raw_klines_to_df(rows)
        latest_source_open = signal_time_ms + bar_ms - (5 * 60_000 if needs_positioning else bar_ms)
        klines = klines[klines["open_time"] <= latest_source_open]
        if needs_positioning:
            positioning = _fetch_positioning(
                client, symbol, int(klines["open_time"].min()), end_ms
            ) if not klines.empty else pd.DataFrame()
            if not positioning.empty:
                klines = klines.merge(positioning, on="open_time", how="left")
        if klines.empty or int(klines["open_time"].max()) != latest_source_open:
            raise RuntimeError(f"{symbol} has no closed {interval} candle at {signal_time_ms}")
        start_ms = int(klines["open_time"].min()) - 24 * 3_600_000
        funding_rows = client.funding_history(
            symbol, limit=1000, start_ms=start_ms, end_ms=end_ms
        )
        resampled = _resample_symbol(
            klines,
            _funding_frame(funding_rows, symbol),
            interval,
            symbol,
        )
        frames.append(resampled)
    raw = _derive_features(
        pd.concat(frames, ignore_index=True), int(bundle.args["horizon"])
    )
    missing = [feature for feature in bundle.features if feature not in raw.columns]
    if missing:
        raise ValueError(f"Live feature frame is missing: {', '.join(missing)}")

    def zscore(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        std = numeric.std(ddof=0)
        if numeric.notna().sum() <= 1 or not np.isfinite(std) or std <= 1e-12:
            return numeric * 0.0
        return (numeric - numeric.mean()) / std

    raw[list(bundle.features)] = raw.groupby("timestamp")[list(bundle.features)].transform(
        zscore
    )
    raw[list(bundle.features)] = raw[list(bundle.features)].replace(
        [np.inf, -np.inf], np.nan
    )
    return raw.sort_values(["timestamp", "symbol"]).reset_index(drop=True), signal_time_ms


def load_cached_live_features(data_dir: str | Path, bundle: LiveModel, *, days: int = 30,
                              rebuild_cache: bool = False) -> tuple[pd.DataFrame, int]:
    """Load the same feature schema used in training from the local Parquet cache."""
    frame, _ = load_feature_frame(data_dir, bundle.symbols,
                                  interval=str(bundle.args["interval"]),
                                  horizon=int(bundle.args["horizon"]), days=days,
                                  require_positioning=False, cache=True,
                                  rebuild_cache=rebuild_cache)
    # Offline caches can lag behind wall-clock time.  Use the latest complete
    # timestamp actually present in the cache instead of assuming that local
    # data reaches the current bar; otherwise inference fails with a misleading
    # "lacks a complete model window" error.
    if frame.empty or "timestamp" not in frame.columns:
        raise ValueError("Cached feature frame is empty or missing timestamp")
    latest_timestamp = pd.to_datetime(frame["timestamp"], utc=True).max()
    if pd.isna(latest_timestamp):
        raise ValueError("Cached feature frame has no valid timestamps")
    signal_time_ms = int(latest_timestamp.timestamp() * 1000)
    return frame, signal_time_ms


def latest_model_input(
    frame: pd.DataFrame,
    bundle: LiveModel,
    signal_time_ms: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    signal_time = pd.to_datetime(signal_time_ms, unit="ms", utc=True)
    seq_len = int(bundle.args["seq_len"])
    step = pd.Timedelta(minutes=interval_minutes(str(bundle.args["interval"])))
    windows = []
    symbols = []
    for symbol in bundle.symbols:
        part = frame[
            (frame["symbol"] == symbol) & (frame["timestamp"] <= signal_time)
        ].sort_values("timestamp").tail(seq_len)
        if len(part) != seq_len or part["timestamp"].iloc[-1] != signal_time:
            raise RuntimeError(f"{symbol} lacks a complete model window at {signal_time}")
        deltas = part["timestamp"].diff().dropna()
        if not bool((deltas == step).all()):
            raise RuntimeError(f"{symbol} model window contains missing {step} bars")
        values = part[list(bundle.features)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            bad = np.argwhere(~np.isfinite(values))[0]
            raise RuntimeError(
                f"{symbol} has invalid live feature {bundle.features[int(bad[1])]}"
            )
        windows.append(values)
        symbols.append(symbol)
    tensor = torch.from_numpy(np.stack(windows)).to(bundle.device)
    return tensor, tuple(symbols)


def rank_targets(
    scores: Dict[str, float],
    *,
    equity: float,
    k_long: int = 2,
    m_short: int = 1,
    gross_leverage: float = 1.0,
) -> Dict[str, float]:
    if equity <= 0 or gross_leverage <= 0:
        raise ValueError("equity and gross_leverage must be positive")
    if k_long < 0 or m_short < 0 or k_long + m_short <= 0:
        raise ValueError("k_long and m_short must be non-negative and not both zero")
    if k_long + m_short > len(scores):
        raise ValueError("Requested long/short count exceeds available model symbols")
    ordered = sorted(scores, key=scores.get, reverse=True)
    longs = ordered[:k_long]
    shorts = [symbol for symbol in reversed(ordered) if symbol not in longs][:m_short]
    side_gross = float(gross_leverage) / 2.0
    targets: Dict[str, float] = {}
    if longs:
        targets.update({symbol: equity * side_gross / len(longs) for symbol in longs})
    if shorts:
        targets.update({symbol: -equity * side_gross / len(shorts) for symbol in shorts})
    return targets


@torch.no_grad()
def scan_ml_once(
    client: Any,
    bundle: LiveModel,
    *,
    equity: float,
    k_long: int = 2,
    m_short: int = 1,
    gross_leverage: float = 1.0,
    history_bars: int = 260,
    now_ms: int | None = None,
) -> SignalResult:
    frame, signal_time_ms = fetch_feature_frame(
        client, bundle, now_ms=now_ms, history_bars=history_bars
    )
    tensor, symbols = latest_model_input(frame, bundle, signal_time_ms)
    if str(bundle.args.get("model", "")).lower() == "lgbm":
        predictions = bundle.model.predict(tensor.detach().cpu().numpy().reshape(len(symbols), -1))
    else:
        predictions = bundle.model(tensor).detach().cpu().numpy()
    scores = {symbol: float(score) for symbol, score in zip(symbols, predictions)}
    targets = rank_targets(
        scores,
        equity=equity,
        k_long=k_long,
        m_short=m_short,
        gross_leverage=gross_leverage,
    )
    return SignalResult(signal_time_ms=signal_time_ms, scores=scores, targets=targets)
