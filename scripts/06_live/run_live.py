#!/usr/bin/env python3
"""Risk-managed live runner (dry-run by default). See docs/06_live_trading.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.execution.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
