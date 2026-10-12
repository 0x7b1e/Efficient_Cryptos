"""Read the latest cross-section from a prediction or target CSV."""
from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path


def read_signal_csv(path: Path, *, equity: float, gross_leverage: float,
                    margin_buffer_pct: float, long_percent: float,
                    short_percent: float, min_assets: int = 0) -> tuple[str, list[dict]]:
    """Return ``(latest_timestamp, target_rows)`` from a CSV.

    Two formats are accepted:

    * ``timestamp,symbol,signal,target_notional`` -- targets are used as given;
    * ``timestamp,symbol,prediction`` -- Top/Bottom ``long_percent`` /
      ``short_percent`` of the latest cross-section get equal notionals of
      ``equity * (1 - margin_buffer_pct) * gross_leverage / 2`` per side.
    """
    # ``utf-8-sig`` also accepts ordinary UTF-8 while removing a BOM that is
    # commonly added by spreadsheet applications (otherwise the first header
    # is silently parsed as ``﻿timestamp``).
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("signal CSV is empty; expected header timestamp,symbol,signal,target_notional")
    required = {"timestamp", "symbol"}
    columns = set(rows[0])
    missing = sorted(required - columns)
    if missing:
        actual = ",".join(str(column) for column in rows[0] if column is not None)
        raise ValueError(
            "signal CSV is missing required column(s): " + ", ".join(missing)
            + "; expected timestamp,symbol and either signal,target_notional or prediction"
            + (f"; found: {actual}" if actual else "")
        )
    rows = [row for row in rows if any(str(value or "").strip() for value in row.values())]
    if not rows:
        raise ValueError("signal CSV contains no data rows")

    def parse_ts(value: str) -> datetime:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)

    latest_dt = max(parse_ts(row["timestamp"]) for row in rows)
    latest = latest_dt.isoformat()
    latest_rows = [row for row in rows if parse_ts(row["timestamp"]) == latest_dt]
    if "target_notional" in columns:
        if "signal" not in columns:
            raise ValueError("target_notional input also requires signal")
        return latest, latest_rows
    if "prediction" not in columns:
        raise ValueError("signal CSV must contain either target_notional or prediction")
    latest_rows = [r for r in latest_rows if str(r.get("prediction", "")).strip() not in {"", "nan", "None"}]
    if len(latest_rows) < min_assets:
        raise ValueError(f"insufficient signal cross-section: {len(latest_rows)} < {min_assets}")
    latest_rows.sort(key=lambda r: float(r["prediction"]))
    n = len(latest_rows)
    nl = max(1, min(n, math.ceil(n * long_percent / 100))) if long_percent else 0
    ns = max(1, min(n, math.ceil(n * short_percent / 100))) if short_percent else 0
    if nl + ns > n:
        raise ValueError(f"long/short selections overlap: {nl}+{ns}>{n}")
    budget = equity * (1.0 - margin_buffer_pct) * gross_leverage / 2.0
    result = []
    for row in latest_rows[-nl:] if nl else []:
        item = dict(row); item["signal"] = item["prediction"]; item["target_notional"] = budget / nl; result.append(item)
    for row in latest_rows[:ns] if ns else []:
        item = dict(row); item["signal"] = item["prediction"]; item["target_notional"] = -budget / ns; result.append(item)
    return latest, result
