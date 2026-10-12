#!/usr/bin/env python3
"""Measure redundancy among factor prediction CSVs in one directory.

Correlation is computed on aligned ``timestamp,symbol,prediction`` rows, not on
equity curves. A high-correlation connected component is treated as one factor
family and one representative is selected for downstream ML.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prediction_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.glob("*.csv")):
        try:
            cols = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if {"timestamp", "symbol", "prediction"} <= cols:
            files.append(path)
    return files


def load_series(path: Path, index: pd.MultiIndex) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp", "symbol", "prediction"])
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame.drop_duplicates(["timestamp", "symbol"]).set_index(["timestamp", "symbol"])
    return frame["prediction"].reindex(index).rename(path.stem)


def score_map(metrics_dir: Path | None, field: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not metrics_dir:
        return scores
    for path in metrics_dir.glob("*.metrics.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = float(payload.get(field, float("nan")))
            if np.isfinite(value):
                scores[path.name.removesuffix(".metrics.json")] = value
        except Exception:
            continue
    return scores


def clusters(corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    names = list(corr.columns)
    graph = {name: set() for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            value = corr.loc[left, right]
            if np.isfinite(value) and abs(float(value)) >= threshold:
                graph[left].add(right); graph[right].add(left)
    result = []
    unseen = set(names)
    while unseen:
        root = min(unseen); stack = [root]; component = []
        while stack:
            node = stack.pop()
            if node not in unseen:
                continue
            unseen.remove(node); component.append(node); stack.extend(graph[node] & unseen)
        result.append(sorted(component))
    return sorted(result, key=lambda group: (len(group) == 1, group[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=.90)
    ap.add_argument("--method", choices=("pearson", "spearman"), default="spearman")
    ap.add_argument("--max-rows", type=int, default=200_000,
                    help="deterministic common-row cap to avoid loading huge factor panels")
    ap.add_argument("--min-overlap", type=int, default=1000)
    ap.add_argument("--metrics-dir", help="optional Rust result directory for representative ranking")
    ap.add_argument("--score-field", default="abs_q10_cumulative_pct")
    args = ap.parse_args()
    if not 0 < args.threshold < 1:
        raise SystemExit("--threshold must be in (0,1)")
    files = prediction_files(Path(args.predictions_dir))
    if not files:
        raise SystemExit("no prediction CSVs with timestamp,symbol,prediction found")
    first = pd.read_csv(files[0], usecols=["timestamp", "symbol"])
    first = first.drop_duplicates(["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])
    if args.max_rows > 0 and len(first) > args.max_rows:
        step = int(np.ceil(len(first) / args.max_rows))
        first = first.iloc[::step].head(args.max_rows)
    index = pd.MultiIndex.from_frame(first)
    panel = pd.concat([load_series(path, index) for path in files], axis=1).dropna(axis=1, how="all")
    if panel.shape[1] < 2:
        raise SystemExit("at least two usable factor columns are required")
    corr = panel.corr(method=args.method, min_periods=args.min_overlap)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out / f"{args.method}.csv")
    pairs = []
    names = list(corr.columns)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            value = corr.loc[left, right]
            if np.isfinite(value) and abs(float(value)) >= args.threshold:
                overlap = int(panel[[left, right]].dropna().shape[0])
                pairs.append({"factor_a": left, "factor_b": right, "correlation": float(value), "abs_correlation": abs(float(value)), "overlap": overlap})
    pairs_frame = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False) if pairs else pd.DataFrame(columns=["factor_a", "factor_b", "correlation", "abs_correlation", "overlap"])
    pairs_frame.to_csv(out / "high_correlation_pairs.csv", index=False)
    scores = score_map(Path(args.metrics_dir) if args.metrics_dir else None, args.score_field)
    rows = []; selected = []
    for cluster_id, group in enumerate(clusters(corr, args.threshold), 1):
        representative = max(group, key=lambda name: (scores.get(name, float("-inf")), -len(name)))
        selected.append(representative)
        for factor in group:
            rows.append({"cluster_id": cluster_id, "factor_id": factor, "cluster_size": len(group),
                         "representative": representative, "selected": factor == representative,
                         "score": scores.get(factor, np.nan)})
    clusters_frame = pd.DataFrame(rows)
    clusters_frame.to_csv(out / "clusters.csv", index=False)
    (out / "selected_factors.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    summary = {"factors": int(panel.shape[1]), "rows": int(len(panel)), "method": args.method,
               "threshold": args.threshold, "clusters": len(selected), "redundant_factors": int(panel.shape[1] - len(selected)),
               "selected": selected}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

