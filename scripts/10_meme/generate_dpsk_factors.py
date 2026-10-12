#!/usr/bin/env python3
"""Launch the existing restricted DPSK generator for meme-event blocks."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", required=True)
    ap.add_argument("--total-candidates", type=int, default=100)
    ap.add_argument("--output-jsonl", default="outputs/meme_event_factors.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # Use the dedicated meme-event prompt contract and restrict the blocks to
    # fields produced by efficient_cryptos/meme/features.py.
    prompt_root = ROOT / "efficient_cryptos" / "mining" / "prompts" / "meme"
    command = [sys.executable, "-m", "efficient_cryptos.mining.generator",
               "--provider", args.provider, "--model", args.model,
               "--prompt-root", str(prompt_root),
               "--blocks", "momentum", "volume", "liquidity", "microstructure",
               "--fold-id", "meme_event_1m", "--train-start", "20240101", "--train-end", "20251231",
               "--validation-start", "20260101", "--validation-end", "20260331",
               "--total-candidates", str(args.total_candidates),
               "--output-jsonl", args.output_jsonl,
               "--rejected-jsonl", str(Path(args.output_jsonl).with_suffix(".rejected.jsonl")),
               "--registry-md", str(Path(args.output_jsonl).with_suffix(".md")),
               "--result-root", "outputs/meme_event_factor_results"]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
