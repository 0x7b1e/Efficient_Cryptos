"""Local storage for candles and funding rates keyed by symbol and timestamp.

Layout (parquet for fast backtests):
  data/klines/<SYMBOL>.parquet     One 1m kline per row, keyed by open_time
  data/funding/<SYMBOL>.parquet    One funding event per row, keyed by funding_time
Upserts deduplicate on `open_time`, so each minute can be stored once per symbol.
Binance positioning series are kept as extra columns in the kline parquet, so
``data`` still contains only the ``klines`` and ``funding`` data families.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# Standard kline columns aligned with Binance `/klines`
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"]


class LocalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "klines").mkdir(parents=True, exist_ok=True)
        (self.root / "funding").mkdir(parents=True, exist_ok=True)

    # ---------------- Klines ----------------
    def kline_path(self, symbol: str) -> Path:
        return self.root / "klines" / f"{symbol.upper()}.parquet"

    @staticmethod
    def raw_klines_to_df(rows: Sequence[Sequence]) -> pd.DataFrame:
        """Convert raw Binance klines into a normalized DataFrame."""
        df = pd.DataFrame(rows, columns=list(range(12))).iloc[:, :11]
        df.columns = KLINE_COLS
        df["open_time"] = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        df["trades"] = df["trades"].astype("int64")
        for c in NUMERIC_COLS:
            df[c] = df[c].astype("float64")
        return df

    def upsert_klines(self, symbol: str, df: pd.DataFrame) -> int:
        """Merge new klines into the local file, deduplicating by open_time."""
        path = self.kline_path(symbol)
        if path.exists():
            old = pd.read_parquet(path)
            # New OHLCV rows do not contain positioning columns.  Preserve any
            # existing enrichment values when a candle is refreshed.
            merged = old.set_index("open_time").combine_first(df.set_index("open_time"))
            fresh = df.set_index("open_time")
            merged.update(fresh)
            df = merged.reset_index()
        df = (df.drop_duplicates("open_time", keep="last")
                .sort_values("open_time")
                .reset_index(drop=True))
        df.to_parquet(path, index=False)
        return len(df)

    def upsert_kline_features(self, symbol: str, df: pd.DataFrame) -> int:
        """Merge sparse timestamped features into a symbol's kline parquet.

        ``df`` must contain ``open_time``.  Values from the new frame overwrite
        the same feature at matching timestamps, while OHLCV and other existing
        feature columns are retained.
        """
        if df.empty:
            return len(self.load_klines(symbol))
        if "open_time" not in df.columns:
            raise ValueError("kline feature frame must contain open_time")
        path = self.kline_path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"Download klines before features: {path}")
        old = pd.read_parquet(path).set_index("open_time")
        new = df.drop_duplicates("open_time", keep="last").set_index("open_time")
        for col in new.columns:
            if col not in old.columns:
                old[col] = float("nan")
            old.loc[new.index.intersection(old.index), col] = new.loc[
                new.index.intersection(old.index), col
            ].to_numpy()
        out = old.reset_index().sort_values("open_time").reset_index(drop=True)
        out.to_parquet(path, index=False)
        return len(out)

    def load_klines(self, symbol: str, start_ms: Optional[int] = None,
                    end_ms: Optional[int] = None) -> pd.DataFrame:
        path = self.kline_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=KLINE_COLS)
        df = pd.read_parquet(path)
        if start_ms is not None:
            df = df[df["open_time"] >= start_ms]
        if end_ms is not None:
            df = df[df["open_time"] <= end_ms]
        return df.sort_values("open_time").reset_index(drop=True)

    def list_symbols(self) -> List[str]:
        return sorted(p.stem.upper() for p in (self.root / "klines").glob("*.parquet"))

    # ---------------- Funding ----------------
    def funding_path(self, symbol: str) -> Path:
        return self.root / "funding" / f"{symbol.upper()}.parquet"

    def upsert_funding(self, symbol: str, df: pd.DataFrame) -> int:
        path = self.funding_path(symbol)
        if df.empty:
            return 0
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = (df.drop_duplicates("funding_time", keep="last")
                .sort_values("funding_time").reset_index(drop=True))
        df.to_parquet(path, index=False)
        return len(df)

    def load_funding(self, symbol: str) -> pd.DataFrame:
        path = self.funding_path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=["funding_time", "symbol", "funding_rate"])
        return pd.read_parquet(path)

    def load_futures_exchange_info(self) -> Dict[str, Any]:
        """Read a legacy metadata cache without creating a metadata directory."""
        path = self.root / "metadata" / "futures_exchange_info.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
