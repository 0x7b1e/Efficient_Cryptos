#!/usr/bin/env python3
"""Run the shared DPSK generator with the DEX prompt contract by default."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "efficient_cryptos.mining.generator"]
    args = sys.argv[1:]
    if "--prompt-root" not in args:
        command += ["--prompt-root", str(root / "efficient_cryptos/mining/prompts/dex")]
    if "--blocks" not in args:
        command += ["--blocks", "flow", "liquidity", "microstructure", "volume", "momentum"]
    command += args
    return subprocess.run(command, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
