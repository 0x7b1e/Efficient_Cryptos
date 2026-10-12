#!/usr/bin/env python3
"""Build reusable base-feature and label caches without training a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.features.cache import load_layered_feature_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", required=True,
                    help="Comma-separated symbols")
    ap.add_argument("--interval", default="30m")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--label-mode", choices=("horizon", "utc_close"), default="horizon")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        ap.error("--symbols must contain at least one symbol")
    frame, features = load_layered_feature_frame(
        args.data_dir, symbols, interval=args.interval, horizon=args.horizon,
        days=args.days, require_positioning=False, cache=True,
        rebuild_cache=args.rebuild_cache, label_mode=args.label_mode,
    )
    print(json.dumps({
        "rows": len(frame),
        "symbols": len(symbols),
        "features": len(features),
        "interval": args.interval,
        "horizon": args.horizon,
        "days": args.days,
        "label_mode": args.label_mode,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
