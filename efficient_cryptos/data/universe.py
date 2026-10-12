"""Liquidity-based symbol selection for Binance data downloads."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List


def liquidity_universe(
    ticker_24hr: List[dict],
    *,
    quote_asset: str,
    top_n: int,
    min_quote_volume_usdt: float,
    exclude: List[str],
) -> List[str]:
    """Select the most liquid quote-asset pairs from Binance 24h tickers."""
    excluded = {symbol.upper() for symbol in exclude}
    rows = []
    for ticker in ticker_24hr:
        symbol = str(ticker.get("symbol", "")).upper()
        if not symbol.endswith(quote_asset) or symbol in excluded:
            continue
        quote_volume = float(ticker.get("quoteVolume", 0.0) or 0.0)
        if quote_volume >= min_quote_volume_usdt:
            rows.append((symbol, quote_volume))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in rows[:top_n]]


def live_liquidity_universe(
    ticker_24hr: List[dict],
    *,
    quote_asset: str,
    top_n: int,
    min_quote_volume_usdt: float,
    exclude: List[str],
) -> List[str]:
    """Backward-compatible name used by the restored live scanner."""
    return liquidity_universe(
        ticker_24hr,
        quote_asset=quote_asset,
        top_n=top_n,
        min_quote_volume_usdt=min_quote_volume_usdt,
        exclude=exclude,
    )


def universe_snapshot_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "universe" / "top40.json"


def load_universe_snapshot(data_dir: str | Path) -> List[str]:
    path = universe_snapshot_path(data_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [str(s).upper() for s in payload.get("symbols", [])]
    except (OSError, ValueError, TypeError):
        return []


def save_universe_snapshot(data_dir: str | Path, symbols: Iterable[str], *, quote_asset: str,
                           top_n: int, ticker_time_ms: int | None = None) -> Path:
    path = universe_snapshot_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": [str(s).upper() for s in symbols],
        "quote_asset": quote_asset,
        "top_n": int(top_n),
        "selected_at_ms": int(ticker_time_ms or time.time() * 1000),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
