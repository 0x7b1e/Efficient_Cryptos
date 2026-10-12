#!/usr/bin/env python3
"""Incrementally collect whitelisted KOL tweets into the local store."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.kol.collect_once import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
