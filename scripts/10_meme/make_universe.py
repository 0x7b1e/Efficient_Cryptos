#!/usr/bin/env python3
"""Create a reproducible meme-event universe from a saved 24h ticker JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from efficient_cryptos.meme.universe import MemeUniverseConfig, build_universe, save_snapshot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--min-quote-volume", type=float, default=1_000_000.0)
    ap.add_argument("--include-major", action="store_true")
    args = ap.parse_args()
    payload = json.loads(Path(args.tickers_json).read_text(encoding="utf-8"))
    tickers = payload.get("data", payload) if isinstance(payload, dict) else payload
    rows = build_universe(tickers, MemeUniverseConfig(
        top_n=args.top_n,
        min_quote_volume_24h=args.min_quote_volume,
        exclude_major=not args.include_major,
    ))
    save_snapshot(rows, args.out)
    print(f"symbols={len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
