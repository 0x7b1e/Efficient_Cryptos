#!/usr/bin/env python3
"""Build meme-event features from local 1m parquet kline files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from efficient_cryptos.meme.features import add_forward_labels, build_event_features  # noqa: E402
from efficient_cryptos.meme.universe import load_snapshot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline-dir", default="data/klines")
    ap.add_argument("--universe", required=True)
    ap.add_argument("--start-date")
    ap.add_argument("--end-date")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-labels", action="store_true")
    args = ap.parse_args()
    symbols = load_snapshot(args.universe)
    parts = []
    for symbol in symbols:
        path = Path(args.kline_dir) / f"{symbol}.parquet"
        if not path.exists():
            print(f"SKIP {symbol}: missing {path}")
            continue
        df = pd.read_parquet(path)
        if "open_time" in df:
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        elif "timestamp" not in df:
            raise ValueError(f"{path} has neither open_time nor timestamp")
        keep = [c for c in ("timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote") if c in df]
        df = df[keep].copy()
        df["symbol"] = symbol
        if args.start_date:
            df = df[df.timestamp >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            df = df[df.timestamp < pd.Timestamp(args.end_date, tz="UTC") + pd.Timedelta(days=1)]
        parts.append(df)
    if not parts:
        raise SystemExit("no kline files found")
    frame = build_event_features(pd.concat(parts, ignore_index=True))
    if args.with_labels:
        frame = add_forward_labels(frame)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    print(f"rows={len(frame)} symbols={frame.symbol.nunique()} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
