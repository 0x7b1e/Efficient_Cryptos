#!/usr/bin/env python3
"""Generate factor records with DeepSeek / OpenAI / Grok (see docs/02_factor_mining.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.mining.generator import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
