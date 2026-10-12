"""Causal realtime implementation of the approved liquidity_0041_5 factor."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any


APPROVED_FACTOR = "liquidity_crypto_templates_10k_v1_liquidity_0041_5"


@dataclass(frozen=True)
class SignalSnapshot:
    timestamp: str
    rows: list[dict[str, Any]]
    quote_volumes: dict[str, float]


def _zscore(values: dict[str, float]) -> dict[str, float]:
    valid = {k: v for k, v in values.items() if math.isfinite(v)}
    if len(valid) < 2:
        return {}
    mean = sum(valid.values()) / len(valid)
    variance = sum((v - mean) ** 2 for v in valid.values()) / max(1, len(valid) - 1)
    std = math.sqrt(variance)
    if not math.isfinite(std) or std <= 0:
        return {}
    return {k: -(v - mean) / (std + 1e-9) for k, v in valid.items()}


def compute_liquidity_0041_5(klines: dict[str, list[list[Any]]]) -> SignalSnapshot:
    """Use only completed bars up through t to calculate the factor at t."""
    raw: dict[str, float] = {}
    quote_volumes: dict[str, float] = {}
    timestamps: dict[str, int] = {}
    for symbol, rows in klines.items():
        if len(rows) < 13:
            continue
        rows = sorted(rows, key=lambda row: int(row[0]))
        qv = [float(row[7]) for row in rows]
        current = sum(qv[-12:]) / 12.0
        previous = sum(qv[-13:-1]) / 12.0
        raw[symbol] = (current - previous) / (previous + 1.0)
        quote_volumes[symbol] = qv[-1]
        timestamps[symbol] = int(rows[-1][0])
    if not timestamps:
        raise ValueError("not enough completed 30m bars to calculate realtime factor")
    common = max(set(timestamps.values()), key=lambda ts: sum(v == ts for v in timestamps.values()))
    aligned = {s: raw[s] for s in raw if timestamps[s] == common}
    prediction = _zscore(aligned)
    timestamp = datetime.fromtimestamp(common / 1000, tz=timezone.utc).isoformat()
    rows = [{"timestamp": timestamp, "symbol": symbol, "prediction": value, "signal": value}
            for symbol, value in sorted(prediction.items())]
    return SignalSnapshot(timestamp, rows, {s: quote_volumes[s] for s in prediction})


def targets_from_predictions(snapshot: SignalSnapshot, *, equity: float,
                             gross_leverage: float, margin_buffer_pct: float,
                             long_percent: float, short_percent: float,
                             min_assets: int = 8) -> list[dict[str, Any]]:
    rows = sorted(snapshot.rows, key=lambda row: float(row["prediction"]))
    n = len(rows)
    if n < min_assets:
        raise ValueError(f"insufficient realtime cross-section: {n} < {min_assets}")
    nl = max(1, math.ceil(n * long_percent / 100.0)) if long_percent else 0
    ns = max(1, math.ceil(n * short_percent / 100.0)) if short_percent else 0
    if nl + ns > n:
        raise ValueError(f"long/short selections overlap: {nl}+{ns}>{n}")
    budget = equity * (1.0 - margin_buffer_pct) * gross_leverage / 2.0
    result: list[dict[str, Any]] = []
    for row in rows[-nl:] if nl else []:
        result.append({**row, "target_notional": budget / nl})
    for row in rows[:ns] if ns else []:
        result.append({**row, "target_notional": -budget / ns})
    return result
