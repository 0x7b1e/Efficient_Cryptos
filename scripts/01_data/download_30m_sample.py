#!/usr/bin/env python3
"""Download compact 30m OHLCV data for a named cross-sectional universe."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from efficient_cryptos.common.binance_client import make_futures_client
from efficient_cryptos.common.storage import LocalStore

BAR_MS = 30 * 60_000

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,HYPEUSDT")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--proxy", default=None)
    args = ap.parse_args()
    store = LocalStore(args.data_dir)
    client = make_futures_client(testnet=False, proxy=args.proxy, timeout=20, retries=5)
    end_ms = (int(time.time() * 1000) // BAR_MS) * BAR_MS - BAR_MS
    start_ms = end_ms - int(args.days) * 86_400_000
    try:
        for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
            cursor = start_ms
            total = 0
            while cursor <= end_ms:
                rows = client.klines(symbol, "30m", limit=1000, start_ms=cursor, end_ms=end_ms)
                if not rows:
                    break
                frame = store.raw_klines_to_df(rows)
                total = store.upsert_klines(symbol, frame)
                last = int(rows[-1][0])
                if last < cursor:
                    break
                cursor = last + BAR_MS
                if len(rows) < 1000:
                    break
            print(f"{symbol}: {total} rows")
    finally:
        client.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
