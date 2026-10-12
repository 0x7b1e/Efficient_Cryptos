"""Report completed and missing factor-generation batches."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from efficient_cryptos.mining.generator import BLOCKS
from efficient_cryptos.mining.registry import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--fold-id", default="fold_1")
    parser.add_argument("--blocks", nargs="+", default=BLOCKS)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--total-candidates", type=int, default=2000)
    args = parser.parse_args()

    if args.blocks == ["all"]:
        args.blocks = BLOCKS.copy()
    slots = len(args.blocks) * args.batches
    base, remainder = divmod(args.total_candidates, slots)
    records = read_jsonl(args.jsonl) if args.jsonl.exists() else []
    counts = Counter(str(record.get("batch_id", "")) for record in records)

    missing: list[tuple[str, int, int]] = []
    slot = 0
    for block in args.blocks:
        for batch in range(1, args.batches + 1):
            expected = base + (1 if slot < remainder else 0)
            batch_id = f"{args.fold_id}_{block}_{batch:04d}"
            saved = counts[batch_id]
            if saved < expected:
                missing.append((batch_id, saved, expected - saved))
            slot += 1

    print(f"saved_records={len(records)}")
    print(f"target_candidates={args.total_candidates}")
    print(f"remaining_candidates={sum(item[2] for item in missing)}")
    print(f"missing_batches={len(missing)}")
    for batch_id, saved, needed in missing:
        print(f"{batch_id}: saved={saved} needed={needed}")


if __name__ == "__main__":
    main()
