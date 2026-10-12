#!/usr/bin/env python3
"""Sweep Alpha101/191 formulas through the same Rust core as generated factors.

Python computes only Alpha predictions; Rust performs the portfolio lifecycle,
ten-layer output, log-return equity, leverage, fees, slippage, and funding.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.factors.alpha import alpha_class, alpha_names
from efficient_cryptos.backtest.funding import funding_event_rates
from efficient_cryptos.features.frame import _interval_minutes, load_feature_frame


def _popbo_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the MultiIndex panel expected by external popbo alpha classes."""
    fields: dict[str, pd.DataFrame] = {}
    for column in ("open", "close", "high", "low", "volume", "quote_volume"):
        fields[column] = frame.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    fields["amount"] = fields["quote_volume"]
    fields["vwap"] = fields["amount"] / fields["volume"].replace(0, np.nan)
    benchmark_open = fields["open"].get("BTCUSDT", fields["open"].mean(axis=1))
    benchmark_close = fields["close"].get("BTCUSDT", fields["close"].mean(axis=1))
    fields["benchmark_open"] = pd.DataFrame(
        {symbol: benchmark_open for symbol in fields["close"].columns},
        index=fields["close"].index,
    )
    fields["benchmark_close"] = pd.DataFrame(
        {symbol: benchmark_close for symbol in fields["close"].columns},
        index=fields["close"].index,
    )
    return pd.concat(
        {key: fields[key] for key in (
            "open", "close", "high", "low", "volume", "amount", "vwap",
            "benchmark_open", "benchmark_close",
        )},
        axis=1,
    )


def _factor_frame(frame: pd.DataFrame, panel: pd.DataFrame, family: str, name: str) -> pd.DataFrame:
    cls = alpha_class(family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        value = getattr(cls(panel), name)()
    if isinstance(value, pd.Series):
        value = value.to_frame()
    value.index = pd.to_datetime(value.index, utc=True)
    try:
        stacked = value.stack(future_stack=True)
    except TypeError:
        stacked = value.stack(dropna=False)
    factor = stacked.rename("prediction").rename_axis(index=["timestamp", "symbol"]).reset_index()
    result = frame.merge(factor, on=["timestamp", "symbol"], how="inner")
    result["prediction"] = pd.to_numeric(result["prediction"], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def _test_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    return (ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))


def _rust_layer_metrics(path: Path) -> dict[str, float]:
    """Derive cumulative layer values from the exact Rust layers CSV."""
    layers = pd.read_csv(path)
    result: dict[str, float] = {}
    for prefix in ("signed", "abs"):
        for q in (1, 10):
            column = f"{prefix}_q{q}"
            if column not in layers:
                continue
            values = pd.to_numeric(layers[column], errors="coerce").fillna(0.0)
            cumulative = float(np.expm1(values.sum()) * 100.0)
            result[f"{prefix}_q{q}_cumulative_pct"] = cumulative
    if "abs_q10_cumulative_pct" in result and "abs_q1_cumulative_pct" in result:
        result["abs_q10_minus_q1_cumulative_pct"] = (
            result["abs_q10_cumulative_pct"] - result["abs_q1_cumulative_pct"]
        )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--families", default="alpha101,alpha191")
    ap.add_argument("--alphas", default="all", help="Names or all")
    ap.add_argument("--interval", default="30m")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--days", type=int, default=1100)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--rebalance-hours", default="1,2,4,8,12,24")
    ap.add_argument("--long-percent", type=float, default=10.0)
    ap.add_argument("--short-percent", type=float, default=10.0)
    ap.add_argument("--gross-leverage", type=float, default=1.0)
    ap.add_argument("--min-symbols", type=int, default=8)
    ap.add_argument("--taker-fee-bps", type=float, default=None)
    ap.add_argument("--slippage-base-bps", type=float, default=None)
    ap.add_argument("--slippage-impact-coef", type=float, default=None)
    ap.add_argument("--max-slippage-bps", type=float, default=None)
    ap.add_argument("--rust-bin", default="rust_backtest/target/release/crypto-factor-backtest")
    ap.add_argument("--keep-artifacts", action="store_true",
                    help="keep per-alpha prediction CSVs alongside Rust outputs")
    ap.add_argument("--out-dir", default="outputs/alpha_sweep_test")
    ap.add_argument("--min-pnl-pct", type=float, default=0.0)
    ap.add_argument("--min-abs-q10-pct", type=float, default=150.0)
    args = ap.parse_args(argv)

    symbols = tuple(x.strip().upper() for x in args.symbols.split(",") if x.strip())
    families = tuple(x.strip().lower() for x in args.families.split(",") if x.strip())
    requested = {x.strip().lower() for x in args.alphas.split(",") if x.strip()}
    frame, _ = load_feature_frame(
        args.data_dir, symbols, interval=args.interval, horizon=args.horizon,
        days=args.days, require_positioning=False, cache=True, layered_cache=True,
    )
    keep = ["timestamp", "symbol", "target", "raw_target", "quote_volume"]
    if "funding_rate_raw" in frame:
        keep.append("funding_rate_raw")
    elif "funding_rate" in frame:
        keep.append("funding_rate")
    frame = frame[keep].copy()
    if "funding_rate" not in frame:
        frame["funding_rate"] = pd.to_numeric(frame.get("funding_rate_raw"), errors="coerce")
    elif "funding_rate_raw" in frame:
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate_raw"], errors="coerce")
    frame = frame[_test_mask(frame, args.test_start, args.test_end)].copy()
    frame["funding_event_rate"] = funding_event_rates(
        frame, args.data_dir, bar_minutes=_interval_minutes(args.interval),
    )
    panel_frame, _ = load_feature_frame(
        args.data_dir, symbols, interval=args.interval, horizon=args.horizon,
        days=args.days, require_positioning=False, cache=True, layered_cache=True,
    )
    panel = _popbo_panel(panel_frame[["timestamp", "symbol", "open", "close", "high", "low", "volume", "quote_volume"]])

    rust_bin = Path(args.rust_bin)
    if not rust_bin.exists():
        raise SystemExit(
            f"Rust binary not found: {rust_bin}. Build with: "
            "cargo build --release --manifest-path rust_backtest/Cargo.toml"
        )
    rebalances = [max(1, int(float(x) * 2)) for x in args.rebalance_hours.split(",") if x.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_root = out / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for family in families:
        names = alpha_names(family) if requested == {"all"} else sorted(requested & set(alpha_names(family)))
        for name in names:
            factor_id = f"{family}_{name}"
            try:
                predictions = _factor_frame(frame, panel, family, name)
                counts = predictions.dropna(subset=["prediction", "target"]).groupby("timestamp")["symbol"].nunique()
                valid_times = counts[counts >= args.min_symbols].index
                predictions = predictions[predictions["timestamp"].isin(valid_times)].copy()
                prediction_csv = artifact_root / f"{factor_id}.csv"
                predictions.insert(0, "factor_id", factor_id)
                predictions[["factor_id", "timestamp", "symbol", "prediction", "target",
                             "raw_target", "quote_volume", "funding_rate",
                             "funding_event_rate"]].to_csv(prediction_csv, index=False)
                for bars in rebalances:
                    hours = bars / 2.0
                    frequency = f"{hours:g}h"
                    result_dir = out / frequency
                    result_dir.mkdir(parents=True, exist_ok=True)
                    prefix = result_dir / factor_id
                    metrics_path = prefix.with_suffix(".metrics.json")
                    layers_path = prefix.with_suffix(".layers.csv")
                    timeseries_path = prefix.with_suffix(".timeseries.csv")
                    cmd = [
                        str(rust_bin), "--input", str(prediction_csv),
                        "--output", str(metrics_path),
                        "--layers-output", str(layers_path),
                        "--timeseries-output", str(timeseries_path),
                        "--long-percent", str(args.long_percent),
                        "--short-percent", str(args.short_percent),
                        "--signal-interval-bars", str(bars),
                        "--signal-hold-bars", str(bars),
                        "--gross-leverage", str(args.gross_leverage),
                        "--include-costs", "--include-funding",
                    ]
                    for flag, value in (
                        ("--taker-fee-bps", args.taker_fee_bps),
                        ("--slippage-base-bps", args.slippage_base_bps),
                        ("--slippage-impact-coef", args.slippage_impact_coef),
                        ("--max-slippage-bps", args.max_slippage_bps),
                    ):
                        if value is not None:
                            cmd.extend([flag, str(value)])
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    metrics.update(_rust_layer_metrics(layers_path))
                    metrics.update({"family": family, "alpha": name, "factor_id": factor_id,
                                    "rebalance_hours": hours, "backend": "rust"})
                    rows.append(metrics)
                if not args.keep_artifacts:
                    prediction_csv.unlink(missing_ok=True)
                print(f"DONE {factor_id} sections={len(valid_times)}", flush=True)
            except Exception as exc:  # keep one bad external formula from stopping the sweep
                failures.append({"factor_id": factor_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {factor_id}: {type(exc).__name__}: {exc}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(out / "alpha_sweep_test.csv", index=False)
    if result.empty:
        selected = result
    else:
        selected = result[
            (pd.to_numeric(result.get("return_pct"), errors="coerce") > args.min_pnl_pct)
            & (pd.to_numeric(result.get("abs_q10_cumulative_pct"), errors="coerce").abs() > args.min_abs_q10_pct)
        ].sort_values(["return_pct", "abs_q10_cumulative_pct"], ascending=False)
    selected.to_csv(out / "alpha_sweep_test_selected.csv", index=False)
    (out / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "families": families, "symbols": symbols, "test_start": args.test_start,
        "test_end": args.test_end, "rows": len(result), "selected": len(selected),
        "selection": {"pnl_positive": args.min_pnl_pct, "abs_q10_cumulative_pct": args.min_abs_q10_pct,
                       "backend": "rust"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"complete={len(result)} selected={len(selected)} failed={len(failures)} out={out}")
    return 1 if failures and result.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
