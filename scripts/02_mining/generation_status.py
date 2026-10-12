#!/usr/bin/env python3
"""Report completed and missing factor-generation batches."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.mining.status import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
