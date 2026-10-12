"""Dynamic meme-event universe helpers.

The universe is deliberately based on tradable symbols and current liquidity,
not on a fixed historical list.  This reduces survivorship bias and allows new
listed meme contracts to enter the monitor without changing source code.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import Iterable, Mapping


DEFAULT_MAJOR = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "TRXUSDT", "TONUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "POLUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT",
    "UNIUSDT", "AAVEUSDT", "SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
}
DEFAULT_STABLE = {"USDCUSDT", "BUSDUSDT", "FDUSDUSDT", "TUSDUSDT", "USDEUSDT"}
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


@dataclass(frozen=True)
class MemeUniverseConfig:
    top_n: int = 200
    min_quote_volume_24h: float = 1_000_000.0
    min_price: float = 0.0
    exclude_major: bool = True
    min_listing_days: int = 7


def is_candidate_symbol(symbol: str, *, exclude: Iterable[str] = ()) -> bool:
    symbol = str(symbol).upper().strip()
    if not re.fullmatch(r"[A-Z0-9]+USDT", symbol):
        return False
    if symbol in DEFAULT_STABLE or symbol in set(map(str.upper, exclude)):
        return False
    if symbol.endswith(LEVERAGED_SUFFIXES):
        return False
    return True


def build_universe(tickers: Iterable[Mapping], config: MemeUniverseConfig | None = None,
                   *, exclude: Iterable[str] = ()) -> list[dict]:
    """Return ranked, JSON-serialisable candidates from Binance 24h tickers.

    ``tickers`` may contain Binance REST fields (symbol, quoteVolume, lastPrice,
    openTime).  Major coins are excluded by default, but the exact exclusions
    are recorded in the resulting snapshot for reproducibility.
    """
    config = config or MemeUniverseConfig()
    excluded = set(map(str.upper, exclude))
    if config.exclude_major:
        excluded |= DEFAULT_MAJOR
    rows = []
    for ticker in tickers:
        symbol = str(ticker.get("symbol", "")).upper()
        if not is_candidate_symbol(symbol, exclude=excluded):
            continue
        try:
            quote_volume = float(ticker.get("quoteVolume", ticker.get("quote_volume", 0.0)))
            last_price = float(ticker.get("lastPrice", ticker.get("last_price", 0.0)))
        except (TypeError, ValueError):
            continue
        if quote_volume < config.min_quote_volume_24h or last_price < config.min_price:
            continue
        rows.append({
            "symbol": symbol,
            "quote_volume_24h": quote_volume,
            "last_price": last_price,
            "open_time": ticker.get("openTime", ticker.get("open_time")),
            "source": "binance_futures_24h",
        })
    rows.sort(key=lambda row: row["quote_volume_24h"], reverse=True)
    return rows[: max(1, int(config.top_n))]


def save_snapshot(rows: list[dict], path: str | Path, *, config: MemeUniverseConfig | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "universe_type": "dynamic_non_major_usdt_futures",
        "config": asdict(config or MemeUniverseConfig()),
        "symbols": [row["symbol"] for row in rows],
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_snapshot(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x).upper() for x in payload]
    return [str(x).upper() for x in payload.get("symbols", [])]
