"""Signal sources for the live runner.

Every source answers one question per cycle: *what is the latest completed
signal and which target rows does it produce?*  It returns a
:class:`SignalBatch`; the runner then feeds the rows to the risk engine.

==================  =========================================================
``csv``             latest cross-section of a prediction/target CSV
``liquidity``       built-in ``liquidity_0041_5`` factor from completed 30m bars
``jsonl``           any generated factor record, same sandbox as the backtests
``alpha``           Alpha101/191 scan with its own selection and sizing
``ml``              trained sequence model with its own k-long / m-short sizing
==================  =========================================================

``csv``/``liquidity``/``jsonl`` size Top/Bottom percentiles with equal notionals.
``alpha``/``ml`` keep the selection and sizing rules they were developed with;
their notionals go to the engine unchanged (the engine still caps gross).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from efficient_cryptos.execution.sources.csv_signals import read_signal_csv
from efficient_cryptos.execution.sources.liquidity_factor import (
    APPROVED_FACTOR, compute_liquidity_0041_5, targets_from_predictions,
)


@dataclass(frozen=True)
class Sizing:
    """Portfolio sizing shared by all percentile-based sources."""
    gross_leverage: float
    margin_buffer_pct: float
    long_percent: float
    short_percent: float
    min_assets: int


@dataclass
class SignalBatch:
    timestamp: str                       # bar open time of the signal, ISO-8601 UTC
    rows: list[dict[str, Any]]           # target rows: symbol, signal, target_notional
    quote_volumes: dict[str, float] = field(default_factory=dict)
    symbols: list[str] | None = None     # replaces the runner's symbol list (CSV mode)


class SignalSource(Protocol):
    factor_id: str

    def fetch(self, adapter: Any, symbols: list[str], equity: float, sizing: Sizing) -> SignalBatch:
        ...


def _target_rows(timestamp: str, targets: dict[str, float]) -> list[dict[str, Any]]:
    return [{"timestamp": timestamp, "symbol": symbol, "signal": float(notional),
             "target_notional": float(notional)}
            for symbol, notional in sorted(targets.items()) if notional]


class CsvSource:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.factor_id = "csv"

    def fetch(self, adapter, symbols, equity, sizing):
        timestamp, rows = read_signal_csv(
            self.path, equity=equity, gross_leverage=sizing.gross_leverage,
            margin_buffer_pct=sizing.margin_buffer_pct, long_percent=sizing.long_percent,
            short_percent=sizing.short_percent, min_assets=sizing.min_assets,
        )
        return SignalBatch(timestamp, rows, {}, sorted({str(row["symbol"]).upper() for row in rows}))


class LiquiditySource:
    def __init__(self):
        self.factor_id = APPROVED_FACTOR

    def fetch(self, adapter, symbols, equity, sizing):
        snapshot = compute_liquidity_0041_5(adapter.realtime_factor_klines(symbols, interval="30m", limit=20))
        rows = targets_from_predictions(
            snapshot, equity=equity, gross_leverage=sizing.gross_leverage,
            margin_buffer_pct=sizing.margin_buffer_pct, long_percent=sizing.long_percent,
            short_percent=sizing.short_percent, min_assets=sizing.min_assets,
        )
        return SignalBatch(snapshot.timestamp, rows, snapshot.quote_volumes)


class JsonlSource:
    def __init__(self, path: Path, record_id: str | None, history_bars: int):
        self.path = Path(path)
        self.record_id = record_id
        self.history_bars = int(history_bars)
        self.factor_id = record_id or str(path)

    def fetch(self, adapter, symbols, equity, sizing):
        from efficient_cryptos.execution.sources.jsonl_factor import compute_generic_factor
        snapshot = compute_generic_factor(
            adapter.realtime_factor_klines(symbols, interval="30m", limit=self.history_bars),
            factor_jsonl=self.path, factor_id=self.record_id,
        )
        rows = targets_from_predictions(
            snapshot, equity=equity, gross_leverage=sizing.gross_leverage,
            margin_buffer_pct=sizing.margin_buffer_pct, long_percent=sizing.long_percent,
            short_percent=sizing.short_percent, min_assets=sizing.min_assets,
        )
        return SignalBatch(snapshot.timestamp, rows, snapshot.quote_volumes)


class AlphaSource:
    """Alpha101/191 live scan (``alpha_live`` section of the YAML config).

    The scan downloads 1m history for the whole universe, so it only runs once
    per completed bar; later polls in the same bar reuse the cached targets.
    """

    def __init__(self, config_path: Path, proxy: str | None = None):
        from efficient_cryptos.common.config import load_config
        from efficient_cryptos.execution.sources.alpha_scanner import _alpha_cfg
        self.cfg = load_config(config_path)
        acfg = _alpha_cfg(self.cfg)
        self.bar_min = int(acfg["bar_min"])
        self.factor_id = acfg["generated_factor_id"] or f"{acfg['family']}_{acfg['alpha']}"
        self.proxy = proxy
        self._cache: SignalBatch | None = None

    def fetch(self, adapter, symbols, equity, sizing):
        import time
        from efficient_cryptos.common.binance_client import make_futures_client
        from efficient_cryptos.common.storage import LocalStore
        from efficient_cryptos.execution.sources.alpha_scanner import _last_closed_minute, scan_alpha_once
        bar_open_ms = _last_closed_minute(int(time.time() * 1000), self.bar_min)
        timestamp = datetime.fromtimestamp(bar_open_ms / 1000, tz=timezone.utc).isoformat()
        if self._cache is not None and self._cache.timestamp == timestamp:
            return self._cache
        client = make_futures_client(testnet=False, proxy=self.proxy)
        try:
            targets = scan_alpha_once(client, LocalStore(self.cfg["general"]["data_dir"]), self.cfg,
                                      equity=equity * (1.0 - sizing.margin_buffer_pct))
        finally:
            client.close()
        if not targets:
            raise RuntimeError("Alpha scan returned no targets; keeping the previous signal")
        self._cache = SignalBatch(timestamp, _target_rows(timestamp, targets))
        return self._cache


class MlSource:
    """Trained sequence model (``scripts/07_ml/train_crypto_t1.py`` output)."""

    def __init__(self, model_dir: Path, *, device: str = "auto", k_long: int = 2,
                 m_short: int = 1, history_bars: int = 260, proxy: str | None = None):
        from efficient_cryptos.execution.sources.ml_model import load_live_model
        self.bundle = load_live_model(model_dir, device=device)
        self.k_long, self.m_short, self.history_bars = int(k_long), int(m_short), int(history_bars)
        self.factor_id = f"ml:{Path(model_dir).name}"
        self.proxy = proxy
        self._cache: SignalBatch | None = None
        self._cache_key: str | None = None

    def fetch(self, adapter, symbols, equity, sizing):
        import time
        from efficient_cryptos.common.binance_client import make_futures_client
        from efficient_cryptos.execution.sources.ml_model import (
            interval_minutes, latest_closed_bar_open_ms, scan_ml_once,
        )
        bar_minutes = interval_minutes(str(self.bundle.args["interval"]))
        bar_open_ms = latest_closed_bar_open_ms(int(time.time() * 1000), bar_minutes)
        cache_key = str(bar_open_ms)
        if self._cache is not None and self._cache_key == cache_key:
            return self._cache
        client = make_futures_client(testnet=False, proxy=self.proxy)
        try:
            result = scan_ml_once(
                client, self.bundle, equity=equity * (1.0 - sizing.margin_buffer_pct),
                k_long=self.k_long, m_short=self.m_short,
                gross_leverage=sizing.gross_leverage, history_bars=self.history_bars,
            )
        finally:
            client.close()
        signal_ts = datetime.fromtimestamp(result.signal_time_ms / 1000, tz=timezone.utc).isoformat()
        self._cache = SignalBatch(signal_ts, _target_rows(signal_ts, result.targets))
        self._cache_key = cache_key
        return self._cache
