#!/usr/bin/env python3
"""Report exact time coverage of Binance kline and funding Parquet files."""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import pandas as pd


def to_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)
    values = pd.to_numeric(series, errors="coerce")
    # Binance timestamps are normally milliseconds. Handle seconds defensively.
    unit = "s" if values.dropna().abs().max() < 10_000_000_000 else "ms"
    return pd.to_datetime(values, unit=unit, utc=True, errors="coerce")


def inspect_file(path: str, column: str, expected_minutes: int | None = None) -> dict:
    try:
        frame = pd.read_parquet(path, columns=[column])
    except Exception as exc:
        print(f"ERROR\t{path}\t{type(exc).__name__}: {exc}")
        return {"status": "error", "path": path, "error": f"{type(exc).__name__}: {exc}"}
    if frame.empty or column not in frame:
        print(f"EMPTY\t{path}")
        return {"status": "empty", "path": path}

    ts = to_utc(frame[column]).dropna().sort_values().drop_duplicates()
    if ts.empty:
        print(f"EMPTY\t{path}")
        return {"status": "empty", "path": path}

    gaps = pd.Series(ts).diff().dropna()
    gap_count = 0
    largest_gap = pd.Timedelta(0)
    if not gaps.empty:
        largest_gap = gaps.max()
        if expected_minutes is not None:
            gap_count = int((gaps > pd.Timedelta(minutes=expected_minutes * 1.5)).sum())

    symbol = Path(path).stem
    kind = "KLINE" if expected_minutes == 1 else "FUNDING"
    record = {
        "status": "ok",
        "kind": kind,
        "symbol": symbol,
        "rows": len(ts),
        "first_utc": ts.iloc[0].isoformat(),
        "last_utc": ts.iloc[-1].isoformat(),
        "file_mtime_utc": pd.Timestamp.fromtimestamp(os.path.getmtime(path), tz="UTC").isoformat(),
        "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
        "gaps_gt_expected": gap_count,
        "largest_step": str(largest_gap),
    }
    print(
        f"{kind}\t{symbol}\trows={len(ts)}\t"
        f"first={ts.iloc[0].isoformat()}\tlast={ts.iloc[-1].isoformat()}\t"
        f"file_mtime={pd.Timestamp.fromtimestamp(os.path.getmtime(path), tz='UTC').isoformat()}\t"
        f"size_mb={os.path.getsize(path)/1024/1024:.2f}\t"
        f"gaps_gt_expected={gap_count}\tlargest_step={largest_gap}"
    )
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="data",
        help="Root containing klines/ and funding/",
    )
    ap.add_argument("--symbol", default="", help="Optional symbol, e.g. BTCUSDT")
    ap.add_argument("--output-csv", default="", help="Also write the report to this CSV path")
    args = ap.parse_args()

    root = Path(args.data_root)
    symbol = args.symbol.upper().strip()
    patterns = [
        (root / "klines" / (f"{symbol}.parquet" if symbol else "*.parquet"), "open_time", 1),
        (root / "funding" / (f"{symbol}.parquet" if symbol else "*.parquet"), "funding_time", 480),
    ]
    found = 0
    records = []
    print("kind\tsymbol\trows\tfirst\tlast\tfile_mtime\tsize_mb\tgaps_gt_expected\tlargest_step")
    for pattern, column, minutes in patterns:
        files = sorted(glob.glob(str(pattern)))
        for path in files:
            found += 1
            records.append(inspect_file(path, column, minutes))
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["status", "kind", "symbol", "rows", "first_utc", "last_utc",
                  "file_mtime_utc", "size_mb", "gaps_gt_expected", "largest_step",
                  "path", "error"]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(f"CSV report: {output}")
    if not found:
        print(f"No Parquet files found under {root}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
