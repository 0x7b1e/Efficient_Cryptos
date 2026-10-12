#!/usr/bin/env python3
"""Download Binance USD-M Futures 1m klines for a meme universe.

This is intentionally separate from the Top40 downloader.  It writes one
parquet per symbol under the requested output directory and can resume from
the local last candle.  The public API is used; no order/account endpoints are
called.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from efficient_cryptos.common.binance_client import make_futures_client  # noqa: E402
from efficient_cryptos.common.storage import LocalStore  # noqa: E402
from efficient_cryptos.meme.universe import (  # noqa: E402
    MemeUniverseConfig,
    build_universe,
    load_snapshot,
    save_snapshot,
)

MINUTE_MS = 60_000


def date_ms(value: str, *, end: bool = False) -> int:
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59)
    return int(dt.timestamp() * 1000)


def download_one(symbol: str, out_dir: Path, start_ms: int, end_ms: int,
                 proxy: str | None, attempts: int, backoff: float,
                 skip_funding: bool) -> tuple[str, int, int]:
    store = LocalStore(out_dir)
    path = store.kline_path(symbol)
    if path.exists():
        old = store.load_klines(symbol)
        if not old.empty:
            start_ms = max(start_ms, int(old["open_time"].max()) + MINUTE_MS)
    client = make_futures_client(proxy=proxy, retries=attempts, backoff=backoff)
    try:
        rows = []
        cursor = start_ms
        while cursor <= end_ms:
            last_exc = None
            for attempt in range(attempts + 1):
                try:
                    batch = client.klines(symbol, "1m", limit=1500, start_ms=cursor, end_ms=end_ms)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < attempts:
                        time.sleep(backoff * (attempt + 1))
            if last_exc is not None:
                raise last_exc
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            if last_open < cursor:
                break
            cursor = last_open + MINUTE_MS
            if len(batch) < 1500:
                break
        candles = len(store.raw_klines_to_df(rows)) if rows else 0
        if rows:
            store.upsert_klines(symbol, store.raw_klines_to_df(rows))
        funding_count = 0
        if not skip_funding:
            cursor = start_ms
            funding_rows = []
            while cursor <= end_ms:
                batch = client.funding_history(symbol, limit=1000, start_ms=cursor, end_ms=end_ms)
                if not batch:
                    break
                funding_rows.extend(batch)
                last = int(batch[-1]["fundingTime"])
                if last < cursor:
                    break
                cursor = last + 1
                if len(batch) < 1000:
                    break
            if funding_rows:
                import pandas as pd
                funding = pd.DataFrame({
                    "funding_time": [int(x["fundingTime"]) for x in funding_rows],
                    "symbol": symbol,
                    "funding_rate": [float(x["fundingRate"]) for x in funding_rows],
                })
                funding_count = store.upsert_funding(symbol, funding)
        return symbol, candles, funding_count
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", help="Universe snapshot JSON")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--out-dir", default="data_meme")
    ap.add_argument("--proxy")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--skip-funding", action="store_true")
    ap.add_argument("--refresh-universe", action="store_true",
                    help="If no --symbols/--universe, fetch Binance 24h tickers and build a snapshot")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--min-quote-volume", type=float, default=1_000_000.0)
    args = ap.parse_args()
    if not args.symbols and not args.universe and not args.refresh_universe:
        ap.error("one of --symbols, --universe, or --refresh-universe is required")
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    if args.universe:
        symbols = load_snapshot(args.universe)
    if args.refresh_universe and not symbols:
        client = make_futures_client(proxy=args.proxy, retries=args.attempts, backoff=args.backoff)
        try:
            rows = build_universe(client.ticker_24hr(), MemeUniverseConfig(
                top_n=args.top_n, min_quote_volume_24h=args.min_quote_volume,
            ))
        finally:
            client.close()
        symbols = [row["symbol"] for row in rows]
        snapshot = Path(args.out_dir) / "universe_snapshot.json"
        save_snapshot(rows, snapshot)
        print(f"refreshed_universe={len(symbols)} snapshot={snapshot}", flush=True)
    symbols = list(dict.fromkeys(symbols))
    start_ms, end_ms = date_ms(args.start_date), date_ms(args.end_date, end=True)
    out_dir = Path(args.out_dir)
    print(f"Downloading {len(symbols)} Futures symbols, 1m, {args.start_date} -> {args.end_date}", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_one, symbol, out_dir, start_ms, end_ms, args.proxy,
                        max(0, args.attempts), args.backoff, args.skip_funding): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, candles, funding = future.result()
                print(f"DONE {sym} candles={candles} funding={funding}", flush=True)
            except Exception as exc:
                failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {symbol}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if failures:
        failure_path = out_dir / "meme_download_failures.json"
        import json
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"failed={len(failures)} report={failure_path}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
