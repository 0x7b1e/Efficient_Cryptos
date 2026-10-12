#!/usr/bin/env python3
"""Build a wide ML feature cache from selected factor prediction CSVs.

Selection directories contain Rust metrics/plots.  This script extracts their
unique factor IDs, finds the matching Python-exported prediction CSVs, and
stores one causal feature column per factor keyed by timestamp and symbol.
Rebalance variants of the same factor are deliberately deduplicated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:160] or "factor"


def selected_ids(directories: Iterable[Path]) -> list[str]:
    result: set[str] = set()
    for root in directories:
        if not root.exists():
            raise FileNotFoundError(f"selected directory not found: {root}")
        for path in root.rglob("*.metrics.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                factor_id = str(payload.get("factor_id") or path.name.removesuffix(".metrics.json"))
                if factor_id:
                    result.add(factor_id)
            except (OSError, ValueError):
                continue
        for name in ("selected_factors.csv", "selected.csv", "factor_metrics.csv"):
            for path in root.rglob(name):
                try:
                    frame = pd.read_csv(path, usecols=["factor_id"])
                    result.update(str(x) for x in frame.factor_id.dropna() if str(x))
                except (OSError, ValueError, pd.errors.ParserError):
                    continue
    if not result:
        raise ValueError("no selected factor IDs found in the supplied directories")
    return sorted(result)


def prediction_index(directories: Iterable[Path], wanted: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {factor_id: [] for factor_id in wanted}
    safe_to_id = {safe_name(factor_id): factor_id for factor_id in wanted}
    for root in directories:
        if not root.exists():
            raise FileNotFoundError(f"prediction directory not found: {root}")
        for path in root.rglob("*.csv"):
            factor_id = safe_to_id.get(path.stem)
            if factor_id is None:
                continue
            result[factor_id].append(path)
    return result


def transform_prediction(frame: pd.DataFrame, method: str) -> pd.Series:
    values = pd.to_numeric(frame["prediction"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    grouped = values.groupby(frame["timestamp"], sort=False)
    if method == "raw":
        return values
    if method == "cs_rank":
        return grouped.rank(pct=True) - 0.5
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (values - mean) / std


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected-dir", action="append", type=Path, required=True,
                    help="selected Rust result directory; repeat for each group")
    ap.add_argument("--prediction-dir", action="append", type=Path, required=True,
                    help="factor prediction CSV directory; train and test dirs may both be supplied")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--transform", choices=["cs_zscore", "cs_rank", "raw"], default="cs_zscore")
    ap.add_argument("--allow-missing", action="store_true",
                    help="write the available subset instead of failing when selected predictions are absent")
    args = ap.parse_args()

    factors = selected_ids(args.selected_dir)
    locations = prediction_index(args.prediction_dir, set(factors))
    missing = [factor_id for factor_id in factors if not locations[factor_id]]
    if missing and not args.allow_missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"missing prediction CSV for {len(missing)} factors: {preview}")

    columns: list[pd.DataFrame] = []
    used: list[str] = []
    for number, factor_id in enumerate(factors, 1):
        paths = locations[factor_id]
        if not paths:
            print(f"missing={number}/{len(factors)} factor={factor_id}", flush=True)
            continue
        pieces = []
        for path in paths:
            try:
                part = pd.read_csv(path, usecols=["timestamp", "symbol", "prediction"])
                pieces.append(part)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                print(f"skip unreadable prediction={path}: {exc}", flush=True)
        if not pieces:
            continue
        long = pd.concat(pieces, ignore_index=True)
        long["timestamp"] = pd.to_datetime(long["timestamp"], utc=True, errors="coerce")
        long["symbol"] = long["symbol"].astype(str).str.upper()
        long = long.dropna(subset=["timestamp", "symbol"]).drop_duplicates(
            ["timestamp", "symbol"], keep="last"
        )
        column = f"factor__{safe_name(factor_id)}"
        long[column] = transform_prediction(long, args.transform).astype("float32")
        columns.append(long[["timestamp", "symbol", column]].set_index(["timestamp", "symbol"]))
        used.append(factor_id)
        print(f"loaded={number}/{len(factors)} factor={factor_id} rows={len(long)}", flush=True)

    if not columns:
        raise SystemExit("no factor predictions were loaded")
    wide = pd.concat(columns, axis=1, join="outer").sort_index().reset_index()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(args.out, index=False)
    metadata = {
        "schema_version": 1,
        "kind": "selected_factor_features",
        "transform": args.transform,
        "selected_dirs": [str(path) for path in args.selected_dir],
        "prediction_dirs": [str(path) for path in args.prediction_dir],
        "selected_factor_count": len(factors),
        "feature_factor_count": len(used),
        "missing_factor_ids": missing,
        "factor_ids": used,
        "rows": len(wide),
        "start": str(wide.timestamp.min()),
        "end": str(wide.timestamp.max()),
    }
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"features={len(used)} rows={len(wide)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
