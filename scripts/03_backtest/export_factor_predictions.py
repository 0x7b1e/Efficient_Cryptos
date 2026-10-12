#!/usr/bin/env python3
"""Export generated-factor predictions for the standalone Rust evaluator.

The Rust core intentionally consumes a narrow, dependency-free CSV contract.
This bridge keeps execution of legacy Python factor code in Python while the
portfolio, layering, cost and metric stage can be moved to Rust.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.factors.sandbox import evaluate_factor_record, load_factor_records  # noqa: E402
from efficient_cryptos.backtest.funding import funding_event_rates  # noqa: E402
from efficient_cryptos.features.frame import _interval_minutes, load_feature_frame  # noqa: E402

# Rust contract. ``raw_target`` lets Rust book exact simple returns;
# ``funding_event_rate`` is the settlement paid by each row's holding window.
EXPORT_COLUMNS = [
    "factor_id", "timestamp", "symbol", "prediction", "target", "raw_target",
    "quote_volume", "funding_rate", "funding_event_rate",
]


# Linux workers use fork so this read-only frame is inherited copy-on-write
# instead of being serialized once per factor. Each worker still receives its
# own pandas temporaries while evaluating a factor.
_WORK_FRAME: pd.DataFrame | None = None
_WORK_INPUT: pd.DataFrame | None = None
_WORK_OUT_DIR: Path | None = None
_WORK_TOTAL = 0


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:160] or "factor"


def _slice(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if not start and not end:
        return frame
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= ts >= pd.Timestamp(start, tz="UTC")
    if end:
        mask &= ts < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return frame.loc[mask].copy()


def _export_one(task: tuple[int, dict]) -> dict:
    if _WORK_FRAME is None or _WORK_INPUT is None or _WORK_OUT_DIR is None:
        raise RuntimeError("export worker was not initialized")
    number, record = task
    factor_id = str(record.get("factor_id", f"factor_{number:05d}"))
    try:
        predictions = evaluate_factor_record(_WORK_FRAME, record, prepared_input=_WORK_INPUT)
        predictions.insert(0, "factor_id", factor_id)
        path = _WORK_OUT_DIR / f"{_safe_name(factor_id)}.csv"
        predictions[EXPORT_COLUMNS].to_csv(path, index=False)
        return {"number": number, "factor_id": factor_id, "path": str(path),
                "rows": len(predictions), "ok": True}
    except Exception as exc:
        return {"number": number, "factor_id": factor_id, "path": "", "rows": 0,
                "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factor-jsonl", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--interval", default="30m")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument(
        "--label-mode", choices=["horizon", "utc_close", "funding_event"],
        default="horizon",
        help="target label; funding_event means return to the next actual funding_time",
    )
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--start", type=int, default=0,
                    help="Zero-based starting factor index; e.g. --start 200 --limit 200 exports records 201-400")
    ap.add_argument("--out-dir", default="outputs/rust_factor_predictions")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Number of factor export worker processes; 2 is a safe starting point")
    ap.add_argument("--backend", choices=["process", "thread"], default="process",
                    help="thread shares the feature frame and uses much less RAM; process is faster only with spare memory")
    ap.add_argument("--resume", action="store_true",
                    help="skip factors whose non-empty CSV already exists in --out-dir")
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    if args.start < 0:
        ap.error("--start must be >= 0")

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        ap.error("--symbols must contain at least one symbol")
    print(f"loading features: {len(symbols)} symbols, interval={args.interval}, days={args.days} "
          f"(first run for a new symbol set rebuilds the cache under {args.data_dir}/features/, "
          "which can take several minutes)", flush=True)
    t0 = time.time()
    frame, _ = load_feature_frame(
        args.data_dir, symbols, interval=args.interval, horizon=args.horizon,
        days=args.days, require_positioning=False, cache=True,
        rebuild_cache=args.rebuild_cache, layered_cache=True,
        label_mode=args.label_mode,
    )
    print(f"features loaded in {time.time() - t0:.0f}s: {len(frame)} rows", flush=True)
    frame = _slice(frame, args.train_start, args.train_end)
    if frame.empty:
        print(f"ERROR: no rows between --train-start {args.train_start} and --train-end {args.train_end}; "
              "check --days and the data coverage", flush=True)
        return 1
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    print(f"window {ts.min()} -> {ts.max()}: {len(frame)} rows, {frame['symbol'].nunique()} symbols", flush=True)
    if args.label_mode != "horizon" or args.horizon != 1:
        # Layer/IC statistics are fine; only the Rust *portfolio* PnL is affected.
        print(f"WARNING: label {args.label_mode} (horizon={args.horizon}) spans several bars, but the Rust "
              "evaluator books one bar of PnL per row, so portfolio return/Sharpe overlap and are overstated. "
              "Use the layer statistics, or --label-mode horizon --horizon 1 for portfolio backtests.",
              flush=True)
    frame["funding_event_rate"] = funding_event_rates(
        frame, args.data_dir, bar_minutes=_interval_minutes(args.interval), symbols=symbols,
    )
    records = load_factor_records(args.factor_jsonl)
    records = records[args.start:]
    if args.limit is not None:
        records = records[:args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global _WORK_FRAME, _WORK_INPUT, _WORK_OUT_DIR, _WORK_TOTAL
    # Build compatibility aliases once. Previously every factor copied the
    # full 500k+ row frame in every worker, which could exhaust RAM and kill a
    # process pool halfway through a 1,000-factor export.
    from efficient_cryptos.factors.sandbox import _factor_input
    _WORK_FRAME, _WORK_INPUT, _WORK_OUT_DIR, _WORK_TOTAL = (
        frame, _factor_input(frame), out_dir, len(records)
    )
    all_records = records
    tasks = list(enumerate(records, 1))
    if args.resume:
        pending = []
        skipped = 0
        for number, record in tasks:
            factor_id = str(record.get("factor_id", f"factor_{number:05d}"))
            path = out_dir / f"{_safe_name(factor_id)}.csv"
            if path.exists() and path.stat().st_size > 0:
                skipped += 1
            else:
                pending.append((number, record))
        tasks = pending
        if skipped:
            print(f"resume_skipped={skipped}/{len(all_records)} existing factor CSVs", flush=True)
    manifest = []
    manifest_path = out_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(existing_manifest, list):
                manifest.extend(existing_manifest)
        except (OSError, ValueError):
            pass
    print(f"evaluating {len(tasks)} factors with jobs={args.jobs} backend={args.backend}", flush=True)
    failures = []
    if args.jobs == 1:
        results = (_export_one(task) for task in tasks)
        for result in results:
            if result["ok"]:
                manifest.append({k: result[k] for k in ("factor_id", "path", "rows")})
                print(f"exported={result['number']}/{len(all_records)} factor={result['factor_id']} rows={result['rows']}", flush=True)
            else:
                failures.append({k: result[k] for k in ("factor_id", "error")})
                print(f"failed={result['number']}/{len(all_records)} factor={result['factor_id']} error={result['error']}", flush=True)
    else:
        if args.backend == "thread":
            # Threads share the read-only feature/alias frames.  Pandas and
            # numpy release the GIL in the expensive rolling/groupby paths,
            # while avoiding one large temporary frame per child process.
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                results = pool.map(_export_one, tasks)
                for result in results:
                    if result["ok"]:
                        manifest.append({k: result[k] for k in ("factor_id", "path", "rows")})
                        print(f"exported={result['number']}/{len(all_records)} factor={result['factor_id']} rows={result['rows']}", flush=True)
                    else:
                        failures.append({k: result[k] for k in ("factor_id", "error")})
                        print(f"failed={result['number']}/{len(all_records)} factor={result['factor_id']} error={result['error']}", flush=True)
            results = None
        else:
            try:
                context = mp.get_context("fork")
            except ValueError as exc:
                raise SystemExit("--jobs > 1 requires a platform with fork multiprocessing") from exc
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
                results = pool.map(_export_one, tasks)
                for result in results:
                    if result["ok"]:
                        manifest.append({k: result[k] for k in ("factor_id", "path", "rows")})
                        print(f"exported={result['number']}/{len(all_records)} factor={result['factor_id']} rows={result['rows']}", flush=True)
                    else:
                        failures.append({k: result[k] for k in ("factor_id", "error")})
                        print(f"failed={result['number']}/{len(all_records)} factor={result['factor_id']} error={result['error']}", flush=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
