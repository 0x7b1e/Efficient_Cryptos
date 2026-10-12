#!/usr/bin/env python3
"""Paper replay: a signals CSV and a bars CSV through the risk engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.engine.paper_replay import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
