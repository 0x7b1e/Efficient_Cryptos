#!/usr/bin/env python3
"""Evaluate generated factor JSONL with the Python backtester (IC, deciles, lifecycle PnL).

This is the common factor backtest entry point.  It uses the same feature
loader, target definition, lifecycle, fees, slippage and funding model as the
built-in cross-sectional evaluator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.backtest.portfolio import factor_summary, lifecycle_pnl_costed  # noqa: E402
from efficient_cryptos.factors.sandbox import evaluate_factor_record, load_factor_records  # noqa: E402
from efficient_cryptos.backtest.report import write_factor_report  # noqa: E402
from efficient_cryptos.common.config import load_config  # noqa: E402
from efficient_cryptos.common.costs import CostModel  # noqa: E402
from efficient_cryptos.features.frame import load_feature_frame  # noqa: E402


def _date_slice(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if not start and not end:
        return frame
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= timestamp >= pd.Timestamp(start, tz="UTC")
    if end:
        mask &= timestamp < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return frame.loc[mask].copy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factor-jsonl", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", required=True,
                    help="Comma-separated Binance symbols used for the cross-section")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--interval", default="30m")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument(
        "--label-mode", choices=["horizon", "utc_close", "funding_event"],
        default="horizon",
        help="target label; funding_event means return to the next actual funding_time",
    )
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--validation-start")
    ap.add_argument("--validation-end")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--k-long", type=int, default=2)
    ap.add_argument("--m-short", type=int, default=1)
    ap.add_argument("--long-percent", type=float)
    ap.add_argument("--short-percent", type=float)
    ap.add_argument("--signal-interval-bars", type=int, default=1)
    ap.add_argument("--signal-hold-bars", type=int, default=1)
    ap.add_argument("--gross-leverage", type=float, default=1.0)
    ap.add_argument("--initial-cash", type=float, default=None)
    ap.add_argument("--include-costs", action="store_true")
    ap.add_argument("--exclude-funding", action="store_true")
    ap.add_argument("--taker-fee-bps", type=float)
    ap.add_argument("--slippage-base-bps", type=float)
    ap.add_argument("--slippage-impact-coef", type=float)
    ap.add_argument("--max-slippage-bps", type=float)
    ap.add_argument("--min-abs-ic", type=float, default=0.001)
    ap.add_argument("--min-abs-icir", type=float, default=0.25)
    ap.add_argument("--min-validation-abs-ic", type=float, default=0.0)
    ap.add_argument("--min-validation-abs-icir", type=float, default=0.0)
    ap.add_argument("--min-coverage", type=float, default=0.30)
    ap.add_argument("--corr-threshold", type=float, default=0.80)
    ap.add_argument("--corr-sample-size", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--keep-all", action="store_true",
                    help="Report all runnable factors without metric/correlation filtering")
    ap.add_argument("--save-predictions", action="store_true",
                    help="Persist one full prediction parquet per factor")
    ap.add_argument("--layered-cache", action="store_true",
                    help="Use reusable base-feature and label caches")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel factor workers; start with 2-4 to bound memory")
    ap.add_argument("--out", default="outputs/generated_factors")
    args = ap.parse_args(argv)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        ap.error("--symbols must contain at least one symbol")
    cfg = load_config()
    initial_cash = float(args.initial_cash or cfg["general"].get("initial_cash", 10000.0))
    costs = cfg.get("costs", {})
    model = CostModel(
        taker_fee_bps=float(args.taker_fee_bps if args.taker_fee_bps is not None else costs.get("taker_fee_bps", 4.0)),
        slippage_base_bps=float(args.slippage_base_bps if args.slippage_base_bps is not None else costs.get("slippage_base_bps", 1.0)),
        slippage_impact_coef=float(args.slippage_impact_coef if args.slippage_impact_coef is not None else costs.get("slippage_impact_coef", 60.0)),
        max_slippage_bps=float(args.max_slippage_bps if args.max_slippage_bps is not None else costs.get("max_slippage_bps", 80.0)),
    )
    frame, _ = load_feature_frame(
        args.data_dir, symbols, interval=args.interval, horizon=args.horizon,
        days=args.days, require_positioning=False, cache=True,
        layered_cache=args.layered_cache, label_mode=args.label_mode,
    )
    records = load_factor_records(args.factor_jsonl)
    if args.limit is not None:
        records = records[:args.limit]
    if args.jobs <= 0:
        ap.error("--jobs must be positive")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = []
    samples: dict[str, np.ndarray] = {}
    records_by_id: dict[str, dict] = {}
    def evaluate_one(record: dict) -> tuple[dict | None, dict | None, np.ndarray | None, dict | None]:
        factor_id = str(record.get("factor_id", "factor_unknown"))
        try:
            predictions = evaluate_factor_record(frame, record)
            predictions.attrs["factor_id"] = factor_id
            if args.save_predictions:
                predictions.to_parquet(out_dir / f"{factor_id}.predictions.parquet", index=False)
            ranked = predictions.groupby("timestamp")["prediction"].rank(pct=True)
            count = min(max(1, int(args.corr_sample_size)), len(ranked))
            indices = np.linspace(0, len(ranked) - 1, count).round().astype(int)
            sample = ranked.iloc[indices].to_numpy(dtype=np.float32)
            sample = np.nan_to_num(sample, nan=0.5, posinf=1.0, neginf=0.0)
            summary = {
                "factor_id": factor_id,
                "block": record.get("block"),
                "formula": record.get("formula"),
                "direction": record.get("direction"),
                "train": factor_summary(
                    predictions, interval_bars=args.signal_interval_bars,
                    hold_bars=args.signal_hold_bars, k_long=args.k_long,
                    m_short=args.m_short, long_percent=args.long_percent,
                    short_percent=args.short_percent, gross_leverage=args.gross_leverage,
                    initial_cash=initial_cash, cost_model=model,
                    include_costs=args.include_costs,
                    include_funding=args.include_costs and not args.exclude_funding,
                    start=args.train_start, end=args.train_end,
                ),
                "validation": factor_summary(
                    predictions, interval_bars=args.signal_interval_bars,
                    hold_bars=args.signal_hold_bars, k_long=args.k_long,
                    m_short=args.m_short, long_percent=args.long_percent,
                    short_percent=args.short_percent, gross_leverage=args.gross_leverage,
                    initial_cash=initial_cash, cost_model=model,
                    include_costs=args.include_costs,
                    include_funding=args.include_costs and not args.exclude_funding,
                    start=args.validation_start, end=args.validation_end,
                ),
            }
            report_frame = _date_slice(
                predictions,
                args.validation_start or args.train_start,
                args.validation_end or args.train_end,
            )
            lifecycle = lifecycle_pnl_costed(
                report_frame, interval_bars=args.signal_interval_bars,
                hold_bars=args.signal_hold_bars, k_long=args.k_long,
                m_short=args.m_short, long_percent=args.long_percent,
                short_percent=args.short_percent, gross_leverage=args.gross_leverage,
                initial_cash=initial_cash, cost_model=model,
                include_costs=args.include_costs,
                include_funding=args.include_costs and not args.exclude_funding,
            )
            report = write_factor_report(
                report_frame, lifecycle, out_dir / "factors", factor_id=factor_id,
                metrics=summary,
            )
            summary["report"] = report
            return summary, None, sample, dict(record)
        except Exception as exc:
            return None, {"factor_id": factor_id, "error": f"{type(exc).__name__}: {exc}"}, None, None

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(evaluate_one, record): (number, record)
                   for number, record in enumerate(records, start=1)}
        for future in as_completed(futures):
            number, record = futures[future]
            summary, failure, sample, original = future.result()
            if summary is not None:
                summaries.append(summary)
                samples[summary["factor_id"]] = sample
                records_by_id[summary["factor_id"]] = original
            if failure is not None:
                failures.append(failure)
            print(f"progress={number}/{len(records)} factor={record.get('factor_id')} "
                  f"status={'ok' if summary is not None else 'failed'} workers={args.jobs}", flush=True)
    summaries.sort(key=lambda item: item["factor_id"])
    def metric(item, split, key, default=np.nan):
        try:
            return float(item.get(split, {}).get(key, default))
        except (TypeError, ValueError):
            return float(default)

    if args.keep_all:
        candidates = list(summaries)
    else:
        candidates = [
            item for item in summaries
            if abs(metric(item, "train", "ic", 0.0)) >= args.min_abs_ic
            and abs(metric(item, "train", "icir", 0.0)) >= args.min_abs_icir
            and metric(item, "train", "coverage", 0.0) >= args.min_coverage
            and abs(metric(item, "validation", "ic", 0.0)) >= args.min_validation_abs_ic
            and abs(metric(item, "validation", "icir", 0.0)) >= args.min_validation_abs_icir
        ]
    candidates.sort(
        key=lambda item: abs(metric(item, "validation", "icir",
                                    metric(item, "train", "icir", 0.0))),
        reverse=True,
    )
    selected_summaries = []
    selected_samples: list[np.ndarray] = []
    for item in candidates:
        vector = samples[item["factor_id"]]
        if not args.keep_all and any(
            np.isfinite(corr := np.corrcoef(vector, prior)[0, 1])
            and abs(corr) > args.corr_threshold
            for prior in selected_samples
        ):
            continue
        selected_summaries.append(item)
        selected_samples.append(vector)
        if not args.keep_all and args.top_k > 0 and len(selected_summaries) >= args.top_k:
            break
    selected_ids = {item["factor_id"] for item in selected_summaries}
    selected_records = []
    for item in selected_summaries:
        record = records_by_id[item["factor_id"]]
        record["status"] = "predict_selected"
        record["predict_metrics"] = {"train": item["train"], "validation": item["validation"]}
        selected_records.append(record)
    with (out_dir / "selected.jsonl").open("w", encoding="utf-8") as handle:
        for record in selected_records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    for item in summaries:
        item["selection_status"] = (
            "selected" if item["factor_id"] in selected_ids else "rejected_by_metrics_or_correlation"
        )
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {
            "factor_id": item["factor_id"],
            "block": item.get("block"),
            "train_ic": item["train"].get("ic"),
            "train_icir": item["train"].get("icir"),
            "validation_ic": item["validation"].get("ic"),
            "validation_icir": item["validation"].get("icir"),
            "train_return_pct": item["train"].get("return_pct"),
            "validation_return_pct": item["validation"].get("return_pct"),
            "validation_final_equity": item["validation"].get("final_equity"),
            "validation_top_decile_return": item["validation"].get("top_decile_return"),
            "validation_bottom_decile_return": item["validation"].get("bottom_decile_return"),
            "validation_top_bottom_spread": item["validation"].get("top_bottom_spread"),
            "validation_top_bottom_ir": item["validation"].get("top_bottom_ir"),
            "validation_max_drawdown_pct": item["validation"].get("max_drawdown_pct"),
            "validation_sharpe": item["validation"].get("sharpe"),
            "validation_avg_turnover": item["validation"].get("avg_turnover"),
            "validation_total_cost": item["validation"].get("total_cost"),
            "report_png": item.get("report", {}).get("report") if isinstance(item.get("report"), dict) else "",
            "train_coverage": item["train"].get("coverage"),
            "validation_coverage": item["validation"].get("coverage"),
            "selection_status": item.get("selection_status"),
            "error": "",
        }
        for item in summaries
    ] + [
        {"factor_id": item["factor_id"], "error": item["error"]}
        for item in failures
    ]).to_csv(out_dir / "factor_metrics.csv", index=False)
    print(json.dumps({"evaluated": len(summaries), "selected": len(selected_records),
                      "failed": len(failures), "out": str(out_dir.resolve())},
                     ensure_ascii=False, indent=2))
    return 0 if summaries or not records else 1


if __name__ == "__main__":
    raise SystemExit(main())
