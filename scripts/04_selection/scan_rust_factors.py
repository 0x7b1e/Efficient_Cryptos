#!/usr/bin/env python3
"""Scan Rust backtest directories and rank factors with formula metadata.

The scanner reads every ``*.metrics.json`` below a directory.  If the matching
``*.layers.csv`` exists, it adds average Q1/Q10 layer returns, including the
absolute-value and direction-aligned layers.  It can therefore compare all
rebalance frequencies in one command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            factor_id = str(record.get("factor_id", ""))
            if factor_id:
                result[factor_id] = record
    return result


def _layer_means(path: Path) -> dict[str, float | None]:
    columns = [
        "abs_q1", "abs_q10", "abs_q1_directional", "abs_q10_directional",
    ]
    if not path.exists():
        return {column: None for column in columns}
    try:
        frame = pd.read_csv(path, usecols=columns)
    except (OSError, ValueError, pd.errors.ParserError):
        return {column: None for column in columns}
    return {
        column: _finite(pd.to_numeric(frame[column], errors="coerce").dropna().mean())
        for column in columns
    }


def _layer_cumulative_pct(path: Path, column: str) -> float | None:
    """Layers store log excess returns; convert the full curve to simple %."""
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=[column])
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    total = float(values.sum())
    return _finite(math.expm1(max(-700.0, min(700.0, total))) * 100.0)


def _row(metrics_path: Path, metadata: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    factor_id = str(metrics.get("factor_id") or metrics_path.name.removesuffix(".metrics.json"))
    layers_path = metrics_path.with_name(metrics_path.name.replace(".metrics.json", ".layers.csv"))
    layer_means = _layer_means(layers_path)
    relative = metrics_path.parent
    frequency = relative.name
    if frequency in {"outputs", "."}:
        frequency = "unknown"
    record = metadata.get(factor_id, {})
    row: dict[str, Any] = {
        "rebalance": frequency,
        "factor_id": factor_id,
        "metrics_path": str(metrics_path),
        "layers_path": str(layers_path) if layers_path.exists() else "",
        "return_pct": _finite(metrics.get("return_pct")),
        "final_equity": _finite(metrics.get("final_equity")),
        "sharpe": _finite(metrics.get("sharpe")),
        "max_drawdown_pct": _finite(metrics.get("max_drawdown_pct")),
        "avg_turnover": _finite(metrics.get("avg_turnover")),
        "total_cost": _finite(metrics.get("total_cost")),
        "signed_q10_minus_q1": _finite(metrics.get("signed_q10_minus_q1")),
        "abs_q10_minus_q1": _finite(metrics.get("abs_q10_minus_q1")),
        "abs_q10_directional_return": _finite(metrics.get("abs_q10_directional_return")),
        "abs_q1_mean": layer_means["abs_q1"],
        "abs_q10_mean": layer_means["abs_q10"],
        "abs_q1_directional_mean": layer_means["abs_q1_directional"],
        "abs_q10_directional_mean": layer_means["abs_q10_directional"],
        "abs_q1_cumulative_pct": _layer_cumulative_pct(layers_path, "abs_q1"),
        "abs_q10_cumulative_pct": _layer_cumulative_pct(layers_path, "abs_q10"),
        "signed_q1_cumulative_pct": _layer_cumulative_pct(layers_path, "signed_q1"),
        "signed_q10_cumulative_pct": _layer_cumulative_pct(layers_path, "signed_q10"),
        "name": record.get("name", ""),
        "block": record.get("block", ""),
        "hypothesis": record.get("hypothesis", ""),
        "direction": record.get("direction", ""),
        "formula": record.get("formula", ""),
        "normalized_formula": record.get("normalized_formula", ""),
        "required_columns": ",".join(map(str, record.get("required_columns", []) or [])),
        "lookback_bars": ",".join(map(str, record.get("lookback_bars", []) or [])),
    }
    return row


def _passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    def at_least(name: str, threshold: float | None) -> bool:
        value = _finite(row.get(name))
        return threshold is None or value is not None and value >= threshold

    def at_most(name: str, threshold: float | None) -> bool:
        value = _finite(row.get(name))
        return threshold is None or value is not None and value <= threshold

    return (
        (not args.pnl_positive or (_finite(row.get("return_pct")) is not None and row["return_pct"] > 0))
        and at_least("abs_q10_cumulative_pct", args.abs_q10_cum_pct_min)
        and at_least("abs_q10_mean", args.abs_q10_min)
        and at_least("abs_q10_directional_mean", args.abs_q10_directional_min)
        and at_most("abs_q1_mean", args.abs_q1_max)
        and at_most("abs_q1_directional_mean", args.abs_q1_directional_max)
        and at_least("return_pct", args.return_min)
        and at_least("sharpe", args.sharpe_min)
        and at_least("max_drawdown_pct", args.max_drawdown_pct_min)
    )


def _markdown(rows: list[dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Rust 因子筛选结果",
        "",
        f"- 扫描目录：`{args.results_dir}`",
        f"- 条件：`abs_q10_mean >= {args.abs_q10_min}`，`abs_q1_mean <= {args.abs_q1_max}`，"
        f"`abs_q10_cumulative_pct >= {args.abs_q10_cum_pct_min}`，"
        f"`PnL positive = {args.pnl_positive}`，`return_pct >= {args.return_min}`，"
        f"`Sharpe >= {args.sharpe_min}`",
        "",
        "| rebalance | factor | PnL+ | return% | Sharpe | maxDD% | abs Q10 cum% | abs Q1 | abs Q10 | abs Q10 directional | 公式 |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(name: str) -> str:
            value = row.get(name)
            return "" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
        formula = str(row.get("normalized_formula") or row.get("formula") or "").replace("|", "\\|")
        lines.append(
            f"| {row['rebalance']} | `{row['factor_id']}` | {'yes' if row.get('return_pct', 0) > 0 else 'no'} | {fmt('return_pct')} | "
            f"{fmt('sharpe')} | {fmt('max_drawdown_pct')} | {fmt('abs_q10_cumulative_pct')} | {fmt('abs_q1_mean')} | "
            f"{fmt('abs_q10_mean')} | {fmt('abs_q10_directional_mean')} | `{formula}` |"
        )
        lines.extend([
            "",
            f"### {row['factor_id']} ({row['rebalance']})",
            f"- 经济假设：{row.get('hypothesis') or '未提供'}",
            f"- 方向：{row.get('direction') or '未提供'}",
            f"- 公式：`{row.get('formula') or '未提供'}`",
            f"- 规范化公式：`{row.get('normalized_formula') or '未提供'}`",
            f"- 所需列：`{row.get('required_columns') or '未提供'}`；窗口：`{row.get('lookback_bars') or '未提供'}`",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--factor-jsonl", type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-markdown", type=Path)
    parser.add_argument("--abs-q10-min", type=float)
    parser.add_argument("--abs-q10-directional-min", type=float)
    parser.add_argument("--abs-q10-cum-pct-min", type=float,
                        help="Minimum final cumulative abs Q10 layer return in percent; log returns are converted with exp(sum)-1")
    parser.add_argument("--abs-q1-max", type=float)
    parser.add_argument("--abs-q1-directional-max", type=float)
    parser.add_argument("--return-min", type=float)
    parser.add_argument("--sharpe-min", type=float)
    parser.add_argument("--max-drawdown-pct-min", type=float,
                        help="Metrics max_drawdown_pct is negative; e.g. -30 allows drawdowns no worse than 30%%")
    parser.add_argument("--pnl-positive", action="store_true",
                        help="Require corrected portfolio return_pct > 0")
    parser.add_argument("--sort-by", default="abs_q10_directional_mean")
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    metadata = _metadata(args.factor_jsonl)
    rows = []
    for metrics_path in sorted(args.results_dir.rglob("*.metrics.json")):
        row = _row(metrics_path, metadata)
        if row is not None and _passes(row, args):
            rows.append(row)
    rows.sort(key=lambda row: (_finite(row.get(args.sort_by)) is None,
                               _finite(row.get(args.sort_by)) or 0.0),
              reverse=not args.ascending)
    if args.limit is not None:
        rows = rows[:args.limit]
    fields = [
        "rebalance", "factor_id", "return_pct", "final_equity", "sharpe",
        "max_drawdown_pct", "avg_turnover", "total_cost", "signed_q10_minus_q1",
        "abs_q10_minus_q1", "abs_q10_directional_return", "abs_q1_mean",
        "abs_q10_mean", "abs_q1_directional_mean", "abs_q10_directional_mean",
        "abs_q1_cumulative_pct", "abs_q10_cumulative_pct",
        "signed_q1_cumulative_pct", "signed_q10_cumulative_pct",
        "name", "block", "hypothesis", "direction", "formula", "normalized_formula",
        "required_columns", "lookback_bars", "metrics_path", "layers_path",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = args.out_markdown or args.out_csv.with_suffix(".md")
    _markdown(rows, markdown_path, args)
    print(f"matched={len(rows)} csv={args.out_csv} markdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
