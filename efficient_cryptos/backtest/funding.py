"""Map real funding settlements onto the prediction rows that hold through them.

Row ``t`` (bar open time) of a prediction CSV is traded at the close of bar
``t`` and its ``target`` is the move from that close to the next close, so the
position chosen on row ``t`` is held during ``(t + bar, t + 2 * bar]`` (the Rust
evaluator books one bar of PnL per row, so it assumes ``horizon == 1``).
A settlement at time ``s`` is paid by the position held at ``s``.  We therefore
attach every settlement to the row whose holding window contains it.

This uses the actual ``funding_time`` of each event, so 1h/4h/8h settlement
schedules are all handled.  It is cash accounting only and is never exposed to
factor code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from efficient_cryptos.common.storage import LocalStore


def funding_event_rates(frame: pd.DataFrame, data_dir: str | Path, *,
                        bar_minutes: int = 30,
                        symbols: Iterable[str] | None = None) -> pd.Series:
    """Return the summed settlement rate paid by each row's holding window.

    The result is aligned to ``frame.index``; rows without a settlement are 0.
    """
    result = pd.Series(0.0, index=frame.index, dtype=float)
    if frame.empty:
        return result
    store = LocalStore(data_dir)
    bar_ns = int(bar_minutes) * 60 * 1_000_000_000
    hold_ns = bar_ns
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    row_ns = ((timestamps - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(1, "ns")).to_numpy(dtype=np.int64)
    wanted = {str(s).upper() for s in (symbols or frame["symbol"].unique())}
    for symbol, index in frame.groupby(frame["symbol"].astype(str).str.upper(), sort=False).groups.items():
        if symbol not in wanted:
            continue
        events = store.load_funding(symbol)
        if events.empty or "funding_time" not in events or "funding_rate" not in events:
            continue
        event_ms = pd.to_numeric(events["funding_time"], errors="coerce")
        rate = pd.to_numeric(events["funding_rate"], errors="coerce")
        ok = event_ms.notna() & rate.notna()
        event_ns = event_ms[ok].to_numpy(dtype=np.int64) * 1_000_000
        rate = rate[ok].to_numpy(dtype=float)
        # Holding window of row t is (t + bar, t + bar + hold]; the owning row
        # for settlement s is the bar open t = ceil((s - bar - hold) / bar) * bar.
        owner = -((-(event_ns - bar_ns - hold_ns)) // bar_ns) * bar_ns
        by_owner = pd.Series(rate).groupby(owner).sum()
        positions = frame.index.get_indexer(index)
        values = by_owner.reindex(row_ns[positions]).fillna(0.0).to_numpy()
        result.loc[index] = values
    return result
