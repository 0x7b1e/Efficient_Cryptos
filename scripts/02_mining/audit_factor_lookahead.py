#!/usr/bin/env python3
"""Audit a factor JSONL for look-ahead and for code whose results changed.

For every record this reports:

* ``leak``     -- ``validate_factor_code`` now rejects it (full-history
                  aggregation per symbol/day, full-column statistics, reversed
                  slicing, backward fill, negative shift, ...).  Any backtest of
                  such a factor is contaminated with future data.
* ``changed``  -- the code passes, but uses a helper/alias whose behaviour was
                  fixed, so its predictions differ from earlier exports:
                  ``share`` (was divided by whole-history volume),
                  ``rank()``/``zscore()`` (were full-sample, now per timestamp),
                  ``ddb_rolling_*`` (mixed different coins, now per symbol).
* ``warning``  -- ``df[col].rolling/shift/diff/pct_change`` without grouping by
                  symbol. Rows are sorted by [timestamp, symbol], so these mix
                  different coins. Not look-ahead, but almost certainly a bug.
* ``ok``       -- none of the above.

Usage:
    python scripts/audit_factor_lookahead.py \\
        --factor-jsonl outputs/crypto_factor_templates_10000.jsonl \\
        --out-csv outputs/factor_lookahead_audit.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.factors.sandbox import validate_factor_code  # noqa: E402

CHANGED_HELPERS = {"rank", "zscore", *(f"ddb_rolling_{m}" for m in ("mean", "sum", "std", "min", "max"))}
TIME_SERIES_METHODS = {"rolling", "shift", "diff", "pct_change", "ewm", "expanding"}


def _changed_and_warnings(code: str) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    warnings: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return changed, warnings
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CHANGED_HELPERS:
            changed.append(f"{node.func.id}()")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "df":
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if key == "share":
                changed.append("df['share']")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in TIME_SERIES_METHODS:
            receiver = node.func.value
            # df["x"].rolling(...) directly on a column, with no groupby in between.
            if isinstance(receiver, ast.Subscript) and isinstance(receiver.value, ast.Name) \
                    and receiver.value.id == "df":
                warnings.append(f"ungrouped df[...].{node.func.attr}()")
    return sorted(set(changed)), sorted(set(warnings))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--factor-jsonl", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()
    rows = []
    with args.factor_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            code = str(record.get("code", ""))
            reason = validate_factor_code(code)
            changed, warnings = _changed_and_warnings(code)
            status = "leak" if reason else "changed" if changed else "warning" if warnings else "ok"
            rows.append({
                "factor_id": record.get("factor_id", ""), "block": record.get("block", ""),
                "status": status, "leak_reason": reason or "",
                "changed": ";".join(changed), "warnings": ";".join(warnings),
            })
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["factor_id"])
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["status"] for row in rows)
    print(json.dumps({"factors": len(rows), **{k: counts.get(k, 0) for k in ("leak", "changed", "warning", "ok")},
                      "out_csv": str(args.out_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
