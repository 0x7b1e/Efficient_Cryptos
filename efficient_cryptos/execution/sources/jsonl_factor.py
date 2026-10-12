"""Causal execution of a generated JSONL factor on live completed bars."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Same sandbox, validator and aliases as the research backtests.
from efficient_cryptos.factors.sandbox import evaluate_factor_record, load_factor_records
from efficient_cryptos.execution.sources.liquidity_factor import SignalSnapshot


def _record(path: Path, factor_id: str | None) -> dict[str, Any]:
    records = load_factor_records(path)
    if factor_id:
        matches = [r for r in records if str(r.get("factor_id")) == factor_id]
        if not matches:
            raise ValueError(f"factor_id not found in {path}: {factor_id}")
        return matches[0]
    if len(records) != 1:
        raise ValueError("--realtime-factor-record-id is required when JSONL contains multiple factors")
    return records[0]


def _frame(klines: dict[str, list[list[Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, values in klines.items():
        for row in values:
            if len(row) < 8:
                continue
            rows.append({
                "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "symbol": symbol.upper(), "open": float(row[1]),
                "high": float(row[2]), "low": float(row[3]),
                "close": float(row[4]), "volume": float(row[5]),
                "quote_volume": float(row[7]), "funding_rate": np.nan,
                "funding_rate_raw": np.nan, "target": np.nan,
                "raw_target": np.nan,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no completed bars returned for generic factor")
    frame = frame.sort_values(["symbol", "timestamp"]).drop_duplicates(
        ["symbol", "timestamp"], keep="last"
    ).reset_index(drop=True)
    grouped = frame.groupby("symbol", sort=False)
    frame["log_ret_1"] = grouped["close"].transform(lambda x: np.log(x / x.shift(1)))
    return frame


def compute_generic_factor(
    klines: dict[str, list[list[Any]]], *, factor_jsonl: Path,
    factor_id: str | None,
) -> SignalSnapshot:
    record = _record(factor_jsonl, factor_id)
    frame = _frame(klines)
    result = evaluate_factor_record(frame, record)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction"])
    if result.empty:
        raise ValueError(f"factor {record.get('factor_id')} produced no finite predictions")
    counts = result.groupby("timestamp")["symbol"].count()
    max_count = counts.max()
    # Prefer the newest bar when several timestamps have the same cross-section
    # coverage; never accidentally trade an older tie because of index order.
    latest = max(ts for ts, count in counts.items() if count == max_count)
    selected = result[result["timestamp"] == latest].copy()
    timestamp = pd.Timestamp(latest).tz_convert("UTC").isoformat()
    rows = [
        {"timestamp": timestamp, "symbol": str(r.symbol),
         "prediction": float(r.prediction), "signal": float(r.prediction)}
        for r in selected.itertuples()
    ]
    volumes = frame[frame["timestamp"] == latest].set_index("symbol")["quote_volume"]
    return SignalSnapshot(timestamp, rows, {s: float(volumes.get(s, 0.0)) for s in selected["symbol"]})
