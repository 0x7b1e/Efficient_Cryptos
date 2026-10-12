#!/usr/bin/env python3
"""Correlation of kline returns across symbols, to help choose a trading pool.

Example:
  python scripts/01_data/kline_correlation.py --data-dir data --interval 30m \
      --start 2026-01-01 --out-dir outputs/kline_corr

Outputs in --out-dir:
  corr_raw.csv / corr_resid.csv   correlation matrices, ordered by cluster
  heatmap_raw.png / heatmap_resid.png
  overlap_bars.csv                bars that each pair has in common
  high_corr_pairs.csv             pairs with |corr| >= --pair-threshold
  symbol_stats.csv                coverage, volatility, liquidity, avg/max correlation, beta to reference
  rolling_avg_corr.csv            each symbol's mean correlation to the others per --rolling-days window
  pool_decisions.csv              why each symbol was kept or dropped by the greedy selector
  suggested_pool.json             {"symbols": [...]} - same format as data/universe/top40.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from efficient_cryptos.data.correlation import (  # noqa: E402
    cluster_order, correlation, discover_kline_files, greedy_pool, high_corr_pairs, interval_minutes,
    load_panel, overlap_counts, plot_heatmap, residualize, rolling_average_corr, session_mask, symbol_stats,
)


def parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    path = Path(value)
    if path.suffix == ".json" and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload["symbols"] if isinstance(payload, dict) else payload)
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_hours(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    lo, hi = (int(x) for x in value.split("-"))
    if not (0 <= lo <= 23 and 1 <= hi <= 24):
        raise argparse.ArgumentTypeError("--utc-hours must look like 13-20")
    return lo, hi % 24 if hi == 24 else hi


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", action="append", default=None,
                   help="Directory containing klines/<SYMBOL>.parquet; repeat to mix sources (default: data)")
    p.add_argument("--symbols", default=None,
                   help="Comma list, or a universe JSON such as data/universe/top40.json (default: every file)")
    p.add_argument("--interval", default="30m", help="Bar size for returns, e.g. 5m, 30m, 1h, 4h, 1d")
    p.add_argument("--start", default=None, help="UTC start date/time, inclusive")
    p.add_argument("--end", default=None, help="UTC end date/time, exclusive")
    p.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--residual", choices=["market", "btc", "none"], default="market",
                   help="market: subtract cross-sectional mean (backtest target); btc: regress out --reference")
    p.add_argument("--reference", default="BTCUSDT", help="Reference symbol for beta and --residual btc")
    p.add_argument("--min-overlap", type=int, default=200, help="Minimum common bars for a pair's correlation")
    p.add_argument("--weekdays-only", action="store_true", help="Drop Saturday/Sunday bars (UTC)")
    p.add_argument("--utc-hours", type=parse_hours, default=None,
                   help="Keep bars whose open hour is in [start, end), e.g. 13-20 for US cash hours")
    p.add_argument("--rolling-days", type=float, default=30.0, help="Window for rolling average correlation; 0 disables")
    p.add_argument("--cluster-corr", type=float, default=0.7,
                   help="Symbols with average-linkage correlation above this share a cluster id")
    p.add_argument("--pair-threshold", type=float, default=0.8, help="List pairs with |corr| at or above this")
    p.add_argument("--select-on", choices=["raw", "resid"], default="resid",
                   help="Which matrix the greedy pool selector uses")
    p.add_argument("--max-corr", type=float, default=0.6,
                   help="Greedy selector drops a symbol whose |corr| to a kept symbol reaches this")
    p.add_argument("--rank-by", default="median_daily_quote_volume",
                   choices=["median_daily_quote_volume", "coverage", "avg_corr_raw", "avg_corr_resid", "ann_vol"],
                   help="Order in which the greedy selector considers symbols")
    p.add_argument("--min-coverage", type=float, default=0.8,
                   help="Selector drops symbols with fewer valid bars than this share of the window")
    p.add_argument("--min-daily-quote-volume", type=float, default=0.0,
                   help="Selector drops symbols below this median daily quote volume (USDT)")
    p.add_argument("--pool-size", type=int, default=0, help="Stop after this many symbols (0 = no limit)")
    p.add_argument("--must-include", default="", help="Comma list always kept in the pool, e.g. BTCUSDT,ETHUSDT")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out-dir", default="outputs/kline_corr")
    return p


def utc(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dirs = [Path(d) for d in (args.data_dir or ["data"])]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    minutes = interval_minutes(args.interval)

    files = discover_kline_files(data_dirs, parse_symbols(args.symbols))
    if len(files) < 2:
        print(f"need at least 2 kline files, found {len(files)} under {', '.join(map(str, data_dirs))}")
        return 1
    print(f"loading {len(files)} symbols at {args.interval} ...")
    panel = load_panel(files, minutes, utc(args.start), utc(args.end))
    returns = panel.returns
    mask = session_mask(returns.index, args.weekdays_only, args.utc_hours)
    returns = returns.loc[mask]
    panel.quote_volume = panel.quote_volume.loc[mask]
    returns = returns.loc[:, returns.notna().sum() >= 3]
    if returns.shape[1] < 2 or returns.empty:
        print("not enough overlapping data after filters")
        return 1

    reference = args.reference if args.reference in returns else None
    if args.residual == "btc" and reference is None:
        print(f"--residual btc needs --reference {args.reference} in the panel")
        return 1
    resid_returns = residualize(returns, args.residual, reference)

    overlap = overlap_counts(returns)
    raw = correlation(returns, args.method, args.min_overlap)
    resid = correlation(resid_returns, args.method, args.min_overlap)

    select_matrix = resid if args.select_on == "resid" else raw
    order, clusters = cluster_order(raw, args.cluster_corr)
    raw = raw.loc[order, order]
    resid_order = [s for s in order if s in resid.columns]
    resid = resid.loc[resid_order, resid_order]
    overlap = overlap.loc[order, order]

    rolling = None
    if args.rolling_days > 0:
        window_bars = max(10, int(round(args.rolling_days * 1440 / minutes)))
        rolling = rolling_average_corr(returns, window_bars, args.method, min(args.min_overlap, window_bars // 2))
        rolling.index.name = "window_start"
        rolling.to_csv(out / "rolling_avg_corr.csv")

    stats = symbol_stats(panel, raw, resid, returns, reference, rolling)
    stats["cluster"] = stats["symbol"].map(clusters)
    stats["order"] = stats["symbol"].map({s: i for i, s in enumerate(order)})
    stats = stats.sort_values("order").drop(columns="order")

    must = [s.strip() for s in args.must_include.split(",") if s.strip()]
    decisions = greedy_pool(stats, select_matrix, args.max_corr, rank_by=args.rank_by,
                            min_coverage=args.min_coverage, min_daily_quote_volume=args.min_daily_quote_volume,
                            pool_size=args.pool_size, must_include=must)
    pool = decisions.loc[decisions["decision"] == "selected", "symbol"].tolist()

    other_name = "resid" if args.residual != "none" else "raw"
    pairs = high_corr_pairs(raw, overlap, args.pair_threshold, resid, other_name)
    resid_pairs = high_corr_pairs(resid, overlap, args.pair_threshold, raw, "raw")

    raw.to_csv(out / "corr_raw.csv")
    resid.to_csv(out / "corr_resid.csv")
    overlap.to_csv(out / "overlap_bars.csv")
    pairs.to_csv(out / "high_corr_pairs.csv", index=False)
    resid_pairs.to_csv(out / "high_corr_pairs_resid.csv", index=False)
    stats.to_csv(out / "symbol_stats.csv", index=False)
    decisions.to_csv(out / "pool_decisions.csv", index=False)
    meta = {
        "symbols": pool,
        "source": "kline_correlation",
        "interval": args.interval,
        "start": str(returns.index.min()), "end": str(returns.index.max()),
        "method": args.method, "residual": args.residual, "select_on": args.select_on,
        "max_corr": args.max_corr, "min_coverage": args.min_coverage,
        "weekdays_only": args.weekdays_only, "utc_hours": args.utc_hours,
    }
    (out / "suggested_pool.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_plots:
        window = f"{returns.index.min():%Y-%m-%d} - {returns.index.max():%Y-%m-%d}"
        plot_heatmap(raw, out / "heatmap_raw.png", f"{args.method} corr of {args.interval} log returns, {window}")
        plot_heatmap(resid, out / "heatmap_resid.png",
                     f"{args.method} corr after removing {args.residual} ({args.interval}), {window}")

    pd.set_option("display.width", 200)
    print(f"window {returns.index.min()} -> {returns.index.max()}, {len(returns)} bars, {returns.shape[1]} symbols")
    off = raw.to_numpy(dtype=float)[~np.eye(len(raw), dtype=bool)]
    roff = resid.to_numpy(dtype=float)[~np.eye(len(resid), dtype=bool)]
    print(f"median pairwise corr: raw {np.nanmedian(off):.3f}, resid {np.nanmedian(roff):.3f}")
    show = ["symbol", "cluster", "coverage", "ann_vol", "median_daily_quote_volume", "avg_corr_raw",
            "avg_corr_resid", "max_corr_peer", "max_corr_raw"] + (["corr_to_ref", "beta_to_ref"] if reference else [])
    print(stats[[c for c in show if c in stats]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if not pairs.empty:
        print(f"\npairs with |raw corr| >= {args.pair_threshold}:")
        print(pairs.head(20).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not resid_pairs.empty:
        print(f"\npairs with |resid corr| >= {args.pair_threshold} (near-duplicate bets):")
        print(resid_pairs.head(20).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    dropped = decisions[decisions["decision"] != "selected"]
    print(f"\nsuggested pool ({len(pool)}, {args.select_on} |corr| < {args.max_corr}): {','.join(pool)}")
    if not dropped.empty:
        print(dropped.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
