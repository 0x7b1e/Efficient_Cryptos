#!/usr/bin/env python3
"""Download Binance candles, funding and positioning inputs into local storage.

The script first ranks liquid symbols from the 24h ticker, then paginates each
symbol's 1m candles and funding history. Everything is upserted into local
parquet files so the command can be re-run incrementally.

Usage:
  python scripts/download_data.py --days 180
  python scripts/download_data.py --incremental
  python scripts/download_data.py --symbols BTCUSDT,ETHUSDT --days 60
"""
from __future__ import annotations

import argparse
import datetime as dt
import socket
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from efficient_cryptos.common.binance_client import make_futures_client  # noqa: E402
from efficient_cryptos.common.config import load_config  # noqa: E402
from efficient_cryptos.common.storage import LocalStore  # noqa: E402
from efficient_cryptos.data.universe import (  # noqa: E402
    liquidity_universe, load_universe_snapshot, save_universe_snapshot,
)

INTERVAL_MS = 60_000
POSITION_PERIOD = "5m"
POSITION_INTERVAL_MS = 5 * INTERVAL_MS
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
POSITION_ANCHOR_COLUMNS = [
    "top_account_long_short_ratio",
    "top_position_long_short_ratio",
    "global_account_long_short_ratio",
    "taker_buy_sell_ratio",
]


def _check_socks5_proxy(proxy: Optional[str], timeout: float) -> None:
    """Quickly verify that the local SOCKS5 proxy responds to a handshake."""
    if not proxy:
        return
    parsed = urlparse(proxy)
    if parsed.scheme not in ("socks5", "socks5h"):
        return
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        raise RuntimeError(f"Invalid proxy address: {proxy}")

    try:
        with socket.create_connection((host, port), timeout=min(timeout, 5.0)) as sock:
            sock.settimeout(min(timeout, 5.0))
            sock.sendall(b"\x05\x01\x00")
            data = sock.recv(2)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to connect to SOCKS5 proxy {host}:{port}: {exc}\n"
            "Make sure the proxy client is running and the port/protocol are correct."
        ) from exc

    if data != b"\x05\x00":
        raise RuntimeError(
            f"SOCKS5 proxy {host}:{port} did not complete a no-auth handshake, returned={data!r}.\n"
            "The port may not be SOCKS5, or the proxy may require auth / be stuck."
        )


def _last_closed_minute_ms() -> int:
    """Only download closed 1m candles so the current incomplete candle is not saved."""
    now_ms = int(time.time() * 1000)
    return (now_ms // INTERVAL_MS) * INTERVAL_MS - INTERVAL_MS


def _date_range_ms(start_date: str, end_date: str) -> tuple[int, int]:
    """Parse inclusive UTC calendar dates into Binance's half-open millisecond range."""
    try:
        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date)
    except ValueError as exc:
        raise ValueError("dates must use YYYY-MM-DD format") from exc
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")
    start_dt = dt.datetime.combine(start, dt.time.min, tzinfo=dt.timezone.utc)
    # Include the complete end date, up to the last closed minute.
    end_exclusive = dt.datetime.combine(end + dt.timedelta(days=1), dt.time.min,
                                        tzinfo=dt.timezone.utc)
    end_ms = int(end_exclusive.timestamp() * 1000) - INTERVAL_MS
    return int(start_dt.timestamp() * 1000), end_ms


def _last_local_ms(store: LocalStore, symbol: str, kind: str) -> Optional[int]:
    """Read the last local timestamp. kind must be 'klines' or 'funding'."""
    if kind == "klines":
        df = store.load_klines(symbol)
        col = "open_time"
    elif kind == "funding":
        df = store.load_funding(symbol)
        col = "funding_time"
    else:
        raise ValueError(f"Unknown kind: {kind}")
    if df.empty or col not in df.columns:
        return None
    return int(df[col].max())


def _global_latest_ms(store: LocalStore, kind: str) -> Optional[int]:
    """Read the latest timestamp across the entire local dataset."""
    vals = []
    for sym in store.list_symbols():
        ts = _last_local_ms(store, sym, kind)
        if ts is not None:
            vals.append(ts)
    return max(vals) if vals else None


def _last_complete_positioning_ms(store: LocalStore, symbol: str) -> Optional[int]:
    """Last timestamp through which every positioning family is present."""
    df = store.load_klines(symbol)
    if df.empty or any(col not in df.columns for col in POSITION_ANCHOR_COLUMNS):
        return None
    latest = []
    for col in POSITION_ANCHOR_COLUMNS:
        valid = df.loc[pd.to_numeric(df[col], errors="coerce").notna(), "open_time"]
        if valid.empty:
            return None
        latest.append(int(valid.max()))
    return min(latest)


def download_klines(client, store: LocalStore, symbol: str, start_ms: int, end_ms: int) -> int:
    """Fetch 1m candles for [start, end] in pages."""
    total = 0
    cursor = start_ms
    while cursor < end_ms:
        rows = client.klines(symbol, "1m", limit=1000, start_ms=cursor, end_ms=end_ms)
        if not rows:
            break
        df = store.raw_klines_to_df(rows)
        total = store.upsert_klines(symbol, df)
        last_open = int(rows[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + INTERVAL_MS
        time.sleep(0.05)  # Light throttling to avoid rate limits
    return total


def download_funding(client, store: LocalStore, symbol: str, start_ms: int, end_ms: int) -> int:
    """Fetch all funding events in [start_ms, end_ms] in 1000-row pages."""
    cursor = start_ms
    total = 0
    while cursor <= end_ms:
        rows = client.funding_history(symbol, limit=1000, start_ms=cursor, end_ms=end_ms)
        if not rows:
            break
        df = pd.DataFrame([
            {"funding_time": int(r["fundingTime"]), "symbol": symbol,
             "funding_rate": float(r["fundingRate"])}
            for r in rows
        ])
        total = store.upsert_funding(symbol, df)
        last_time = max(int(r["fundingTime"]) for r in rows)
        if last_time < cursor or last_time >= end_ms or len(rows) < 1000:
            break
        cursor = last_time + 1
        time.sleep(0.05)
    return total


def download_positioning(client, store: LocalStore, symbol: str,
                         start_ms: int, end_ms: int) -> int:
    """Download all supported 5m positioning series and merge into klines."""
    frames = []
    for endpoint, mapping in POSITION_ENDPOINTS.items():
        cursor = start_ms
        rows_all = []
        recent_only = False
        while cursor <= end_ms:
            try:
                rows = client.futures_data_history(
                    endpoint, symbol, period=POSITION_PERIOD, limit=500,
                    start_ms=None if recent_only else cursor, end_ms=end_ms,
                )
            except RuntimeError as exc:
                # Binance positioning endpoints expose only a rolling history
                # and reject older startTime values with -1130. Retry once
                # without startTime so the currently available recent rows are
                # still merged into the local candles.
                if '"code":-1130' not in str(exc) or recent_only:
                    raise
                recent_only = True
                rows = client.futures_data_history(
                    endpoint, symbol, period=POSITION_PERIOD, limit=500,
                    start_ms=None, end_ms=end_ms,
                )
            if not rows:
                break
            rows_all.extend(rows)
            if recent_only:
                break
            last = int(rows[-1]["timestamp"])
            if last < cursor:
                break
            cursor = last + POSITION_INTERVAL_MS
            time.sleep(0.05)
        if not rows_all:
            continue
        records = []
        for row in rows_all:
            rec = {"open_time": int(row["timestamp"])}
            for source, dest in mapping.items():
                if source in row:
                    rec[dest] = float(row[source])
            records.append(rec)
        frames.append(pd.DataFrame(records).drop_duplicates("open_time", keep="last"))
    if not frames:
        return 0
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="open_time", how="outer")
    store.upsert_kline_features(symbol, merged)
    return len(merged)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Binance historical data locally")
    ap.add_argument("--config", default=None)
    ap.add_argument("--days", type=int, default=30,
                    help="Lookback days for first download or non-incremental mode; also used as backfill when no local data exists")
    ap.add_argument("--start-date", default=None,
                    help="UTC start date, inclusive (YYYY-MM-DD); use with --end-date")
    ap.add_argument("--end-date", default=None,
                    help="UTC end date, inclusive (YYYY-MM-DD); use with --start-date")
    ap.add_argument("--incremental", action="store_true",
                    help="Continue from the dataset-wide latest timestamp for the current liquidity top_n; good for daily updates")
    ap.add_argument("--per-symbol-incremental", action="store_true",
                    help="Backfill from each symbol's own last timestamp; useful for fixing single-symbol gaps")
    ap.add_argument("--include-existing", action="store_true",
                    help="Deprecated compatibility flag; top40 remains the managed universe")
    ap.add_argument("--refresh-universe", action="store_true",
                    help="Refresh the liquidity top40 snapshot; otherwise daily updates reuse it")
    ap.add_argument("--symbols", default="", help="Comma-separated; leave empty to auto-select by liquidity")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Take the top N symbols by quote volume; default comes from config universe.top_n (40)")
    ap.add_argument("--min-volume", type=float, default=None,
                    help="24h quote-volume floor in USDT; default comes from config min_quote_volume_usdt")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--timeout", type=float, default=10.0, help="Per-request HTTP timeout in seconds")
    ap.add_argument("--retries", type=int, default=3, help="Retry count for network failures")
    ap.add_argument("--skip-positioning", action="store_true",
                    help="Do not download large-trader/global long-short ratios and taker ratio")
    args = ap.parse_args()

    cfg = load_config(args.config)
    store = LocalStore(cfg["general"]["data_dir"])
    try:
        _check_socks5_proxy(args.proxy, args.timeout)
    except RuntimeError as exc:
        print(f"Network precheck failed: {exc}", file=sys.stderr)
        return 2
    client = make_futures_client(
        proxy=args.proxy,
        timeout=args.timeout,
        retries=args.retries,
    )
    try:
        end_ms = _last_closed_minute_ms()
        if bool(args.start_date) != bool(args.end_date):
            raise ValueError("--start-date and --end-date must be provided together")
        if args.incremental and (args.start_date or args.end_date):
            raise ValueError("date ranges cannot be combined with --incremental")
        if args.start_date and args.end_date:
            fallback_start_ms, requested_end_ms = _date_range_ms(args.start_date, args.end_date)
            if requested_end_ms > end_ms:
                raise ValueError("requested end date includes future or incomplete candles")
            end_ms = requested_end_ms
            range_desc = f"UTC {args.start_date} through {args.end_date}"
        else:
            fallback_start_ms = end_ms - args.days * 24 * 60 * INTERVAL_MS
            range_desc = f"lookback {args.days} days"
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            u = cfg["universe"]
            # CLI takes precedence; fall back to the config file
            top_n = args.top_n if args.top_n is not None else u["top_n"]
            min_vol = args.min_volume if args.min_volume is not None else u["min_quote_volume_usdt"]
            snapshot = load_universe_snapshot(cfg["general"]["data_dir"])
            if args.incremental and snapshot and not args.refresh_universe:
                symbols = snapshot[:top_n]
                print(f"Using fixed liquidity universe snapshot: {len(symbols)} symbols")
            else:
                try:
                    tickers = client.ticker_24hr()
                except Exception as exc:
                    print(
                        "Failed to fetch Binance 24h tickers.\n"
                        f"base_url={client.base_url} proxy={args.proxy or 'none'} "
                        f"timeout={args.timeout}s retries={args.retries}\n"
                        f"Error: {type(exc).__name__}: {exc}", file=sys.stderr,
                    )
                    return 2
                symbols = liquidity_universe(
                    tickers, quote_asset=cfg["general"]["quote_asset"], top_n=top_n,
                    min_quote_volume_usdt=min_vol, exclude=u.get("exclude", []))
                save_universe_snapshot(cfg["general"]["data_dir"], symbols,
                                       quote_asset=cfg["general"]["quote_asset"], top_n=top_n)
                print(f"Refreshed liquidity universe: top_n={top_n}, 24h floor={min_vol:,.0f} USDT")
        if args.incremental and args.per_symbol_incremental:
            raise ValueError("--incremental and --per-symbol-incremental are mutually exclusive")

        global_kline_start = None
        global_funding_start = None
        if args.incremental:
            latest_kline = _global_latest_ms(store, "klines")
            latest_funding = _global_latest_ms(store, "funding")
            global_kline_start = (latest_kline + INTERVAL_MS) if latest_kline is not None else fallback_start_ms
            global_funding_start = (latest_funding + 1) if latest_funding is not None else fallback_start_ms

        mode = (
            "dataset-wide incremental update"
            if args.incremental else
            "per-symbol incremental backfill"
            if args.per_symbol_incremental else
            range_desc
        )
        print(f"Preparing to download {len(symbols)} symbols, mode={mode}")
        if args.incremental:
            print(
                "Local latest timestamps: "
                f"candles={pd.to_datetime(global_kline_start - INTERVAL_MS, unit='ms', utc=True) if global_kline_start else 'none'} "
                f"funding={pd.to_datetime(global_funding_start - 1, unit='ms', utc=True) if global_funding_start else 'none'}"
            )
        for i, sym in enumerate(symbols, 1):
            if args.incremental:
                kline_start = global_kline_start
                funding_start = global_funding_start
            elif args.per_symbol_incremental:
                last_kline = _last_local_ms(store, sym, "klines")
                kline_start = (last_kline + INTERVAL_MS) if last_kline is not None else fallback_start_ms
                last_funding = _last_local_ms(store, sym, "funding")
                funding_start = (last_funding + 1) if last_funding is not None else fallback_start_ms
            else:
                kline_start = fallback_start_ms
                funding_start = fallback_start_ms

            if kline_start <= end_ms:
                before_k = len(store.load_klines(sym))
                n = download_klines(client, store, sym, kline_start, end_ms)
                after_k = len(store.load_klines(sym))
                new_k = max(0, after_k - before_k)
            else:
                n = len(store.load_klines(sym))
                new_k = 0

            if funding_start <= end_ms:
                before_f = len(store.load_funding(sym))
                f = download_funding(client, store, sym, funding_start, end_ms)
                after_f = len(store.load_funding(sym))
                new_f = max(0, after_f - before_f)
            else:
                f = len(store.load_funding(sym))
                new_f = 0

            positioning_n = 0
            if not args.skip_positioning:
                # Binance's positioning endpoints expose only a rolling recent
                # history. Asking for the requested start is safe; unavailable
                # older rows simply return empty and are never fabricated.
                try:
                    last_positioning = _last_complete_positioning_ms(store, sym)
                    positioning_start = (
                        last_positioning + POSITION_INTERVAL_MS
                        if last_positioning is not None and
                        (args.incremental or args.per_symbol_incremental)
                        else fallback_start_ms
                    )
                    positioning_n = download_positioning(
                        client, store, sym, positioning_start, end_ms
                    )
                except Exception as exc:
                    print(f"Warning: {sym} positioning download failed: {exc}", file=sys.stderr)

            print(
                f"  [{i}/{len(symbols)}] {sym}: candles={n} new={new_k}, "
                f"funding={f} new={new_f}, positioning_rows={positioning_n}"
            )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
