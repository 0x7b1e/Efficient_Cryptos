#!/usr/bin/env python3
"""Extract unique selected factor records from a larger factor JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def selected_ids(roots: list[Path]) -> set[str]:
    result: set[str] = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("*.metrics.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                factor_id = str(item.get("factor_id") or path.name.removesuffix(".metrics.json"))
                if factor_id:
                    result.add(factor_id)
            except (OSError, ValueError):
                pass
        for path in root.rglob("*.csv"):
            if not (path.name.startswith("selected") or path.name == "factor_metrics.csv"):
                continue
            try:
                frame = pd.read_csv(path, usecols=["factor_id"])
                result.update(str(value) for value in frame.factor_id.dropna() if str(value))
            except (OSError, ValueError, pd.errors.ParserError):
                pass
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected-dir", action="append", type=Path, required=True)
    ap.add_argument("--factor-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    args = ap.parse_args()
    wanted = selected_ids(args.selected_dir)
    if not wanted:
        raise SystemExit("no factor IDs found in selected directories")
    found: dict[str, dict] = {}
    with args.factor_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            factor_id = str(record.get("factor_id", ""))
            if factor_id in wanted and factor_id not in found:
                found[factor_id] = record
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(
            f"{len(missing)} selected factor IDs are missing from {args.factor_jsonl}: "
            + ", ".join(missing[:10])
        )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for factor_id in sorted(found):
            handle.write(json.dumps(found[factor_id], ensure_ascii=False) + "\n")
    print(f"selected_unique={len(wanted)} written={len(found)} out={args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
