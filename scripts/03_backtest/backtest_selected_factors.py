#!/usr/bin/env python3
"""Backtest selected factor/rebalance pairs with Rust and render both charts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FREQUENCY_BARS = {"1h": 2, "2h": 4, "4h": 8, "8h": 16, "12h": 24, "24h": 48}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--plot-max-points", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rust-bin", type=Path,
                        default=Path("rust_backtest/target/release/crypto-factor-backtest"))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    with args.selection_csv.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    tasks = []
    seen = set()
    missing = []
    for record in records:
        factor = str(record.get("factor_id", "")).strip()
        frequency = str(record.get("rebalance", "")).strip()
        key = (factor, frequency)
        if not factor or frequency not in FREQUENCY_BARS or key in seen:
            continue
        seen.add(key)
        prediction = args.predictions_dir / f"{factor}.csv"
        if not prediction.is_file():
            missing.append(str(prediction))
            continue
        tasks.append((factor, frequency, FREQUENCY_BARS[frequency], prediction))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_manifest.json").write_text(json.dumps({
        "selection_csv": str(args.selection_csv),
        "predictions_dir": str(args.predictions_dir),
        "selected_pairs": len(seen),
        "runnable_pairs": len(tasks),
        "missing_predictions": missing,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing:
        print(f"missing_predictions={len(missing)}", flush=True)
        for path in missing:
            print(f"MISSING {path}", flush=True)

    root = Path.cwd()
    plot_script = root / "scripts/03_backtest/plot_rust_backtest.py"

    def run(task: tuple[str, str, int, Path]) -> str:
        factor, frequency, bars, prediction = task
        result_dir = args.output_dir / frequency
        plot_dir = result_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        metrics = result_dir / f"{factor}.metrics.json"
        layers = result_dir / f"{factor}.layers.csv"
        timeseries = result_dir / f"{factor}.timeseries.csv"
        expected = (metrics, layers, timeseries)
        if args.force or not all(path.is_file() and path.stat().st_size for path in expected):
            subprocess.run([
                str(args.rust_bin), "--input", str(prediction),
                "--output", str(metrics), "--layers-output", str(layers),
                "--timeseries-output", str(timeseries),
                "--long-percent", "10", "--short-percent", "10",
                "--signal-interval-bars", str(bars), "--signal-hold-bars", str(bars),
                "--gross-leverage", "1", "--include-costs", "--include-funding",
            ], check=True, stdout=subprocess.DEVNULL)
        for kind in ("equity", "layers"):
            plot = plot_dir / f"{factor}.{kind}.png"
            if args.force or not plot.is_file() or not plot.stat().st_size:
                subprocess.run([
                    sys.executable, str(plot_script),
                    "--kind", kind, "--max-points", str(args.plot_max_points),
                    "--metrics", str(metrics), "--layers", str(layers),
                    "--timeseries", str(timeseries), "--out", str(plot),
                ], check=True, stdout=subprocess.DEVNULL)
        return f"{frequency} {factor}"

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                print(f"DONE {future.result()}", flush=True)
            except Exception as exc:
                factor, frequency, _, _ = futures[future]
                failures.append({"factor_id": factor, "rebalance": frequency, "error": str(exc)})
                print(f"FAILED {frequency} {factor}: {exc}", flush=True)
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"complete={len(tasks) - len(failures)} failed={len(failures)} missing={len(missing)}")
    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
