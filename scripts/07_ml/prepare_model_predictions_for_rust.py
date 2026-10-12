#!/usr/bin/env python3
"""Add market-volume/funding columns to model predictions for Rust backtests.

Neural ``predict_frame`` artifacts intentionally contain only timestamp,
symbol, prediction and target.  Rust needs quote_volume for realistic impact
slippage; when it is absent it assumes zero volume and applies max slippage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from efficient_cryptos.backtest.funding import funding_event_rates  # noqa: E402
from efficient_cryptos.features.frame import _interval_minutes, load_feature_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--interval", default="30m")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--days", type=int, default=1100)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    predictions = pd.read_parquet(args.predictions)
    required = {"timestamp", "symbol", "prediction", "target"}
    missing = required - set(predictions.columns)
    if missing:
        raise SystemExit(f"prediction parquet missing columns: {sorted(missing)}")
    symbols = tuple(value.strip().upper() for value in args.symbols.split(",") if value.strip())
    market, _ = load_feature_frame(
        args.data_dir, symbols, interval=args.interval, horizon=args.horizon,
        days=args.days, require_positioning=False, cache=True,
        layered_cache=True,
    )
    # ``load_feature_frame`` cross-sectionally standardizes model features.
    # Rust needs the exchange's raw funding rate for cash-flow accounting;
    # using the standardized ``funding_rate`` can turn a few-bps rate into an
    # order-one number and explode the equity curve.
    funding_column = "funding_rate_raw" if "funding_rate_raw" in market.columns else "funding_rate"
    market = market[["timestamp", "symbol", "quote_volume", "raw_target", funding_column]].copy()
    market = market.rename(columns={funding_column: "funding_rate"})
    market["funding_event_rate"] = funding_event_rates(
        market, args.data_dir, bar_minutes=_interval_minutes(args.interval),
    )
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions["symbol"] = predictions["symbol"].astype(str).str.upper()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market["symbol"] = market["symbol"].astype(str).str.upper()
    for column in ("quote_volume", "funding_rate", "raw_target", "funding_event_rate"):
        if column in predictions:
            predictions = predictions.drop(columns=column)
    result = predictions.merge(market, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    result = result[["timestamp", "symbol", "prediction", "target", "raw_target",
                     "quote_volume", "funding_rate", "funding_event_rate"]]
    result["funding_event_rate"] = pd.to_numeric(result["funding_event_rate"], errors="coerce").fillna(0.0)
    result["quote_volume"] = pd.to_numeric(result["quote_volume"], errors="coerce").fillna(0.0)
    result["funding_rate"] = pd.to_numeric(result["funding_rate"], errors="coerce").fillna(0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    result.to_csv(args.out, index=False)
    zero_volume = int((result["quote_volume"] <= 0).sum())
    print(f"rows={len(result)} symbols={result.symbol.nunique()} zero_quote_volume={zero_volume} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
