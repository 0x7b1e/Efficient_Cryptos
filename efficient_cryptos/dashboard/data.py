"""Read-only loaders for the dashboard.

Everything here reads files that the pipeline already writes; nothing talks to
Binance and nothing writes. The Streamlit pages in ``app/`` are thin wrappers
around these functions, so they can be tested without a browser.

Factor backtests (``scripts/03_backtest/sweep_rust_rebalance.sh``)::

    <sweep_root>/<rebalance>/<factor>.metrics.json
    <sweep_root>/<rebalance>/<factor>.timeseries.csv   equity, drawdown, turnover per bar
    <sweep_root>/<rebalance>/<factor>.layers.csv       per-bar decile log excess returns

Live runner (``scripts/06_live/run_live.py --output DIR``)::

    DIR/latest.json            last cycle payload
    DIR/audit.jsonl            one payload per cycle (can grow large; read from the tail)
    DIR/engine_state.json      risk engine state
    DIR/runner_errors.jsonl    network / cycle errors with retry delays
    DIR/exchange_risk_state.json, DIR/manual_guard_state.json   optional guards
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "return_pct", "sharpe", "max_drawdown_pct", "avg_turnover", "total_cost", "final_equity", "bars",
    "signed_q10_minus_q1", "abs_q10_minus_q1", "abs_q10_directional_return",
]
META_COLUMNS = ["name", "block", "direction", "hypothesis", "formula", "normalized_formula"]


# ---------------------------------------------------------------- helpers
def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def rebalance_hours(label: str) -> float:
    """'30m' -> 0.5, '4h' -> 4, '1d' -> 24; unknown labels sort last."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([mhd])", str(label).strip().lower())
    if not match:
        return math.inf
    value, unit = float(match.group(1)), match.group(2)
    return value * {"m": 1 / 60, "h": 1.0, "d": 24.0}[unit]


def sort_rebalances(labels: Iterable[str]) -> list[str]:
    return sorted(set(labels), key=lambda x: (rebalance_hours(x), str(x)))


def dir_signature(root: str | Path, pattern: str = "*") -> tuple:
    """Cheap change detector used as a cache key: (count, newest mtime) of matching dirs/files."""
    root = Path(root)
    if not root.exists():
        return (0, 0)
    newest, count = root.stat().st_mtime_ns, 0
    for path in root.glob(pattern):
        try:
            newest = max(newest, path.stat().st_mtime_ns)
            count += 1
        except OSError:
            continue
    return (count, newest)


def find_sweep_roots(base: str | Path, max_depth: int = 3) -> list[Path]:
    """Directories below ``base`` that contain ``<rebalance>/*.metrics.json`` (or metrics directly)."""
    base = Path(base)
    if not base.exists():
        return []
    roots: set[Path] = set()
    base_depth = len(base.parts)
    for dirpath, dirnames, filenames in os.walk(base):
        depth = len(Path(dirpath).parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        if any(name.endswith(".metrics.json") for name in filenames):
            here = Path(dirpath)
            roots.add(here.parent if rebalance_hours(here.name) != math.inf else here)
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in {"plots", "logs"}]
    return sorted(roots)


# ---------------------------------------------------------- factor results
def scan_sweep(root: str | Path) -> pd.DataFrame:
    """One row per (factor, rebalance) with the Rust metrics and file paths."""
    root = Path(root)
    rows = []
    candidates = list(root.glob("*/*.metrics.json")) + list(root.glob("*.metrics.json"))
    for path in candidates:
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        factor_id = str(metrics.get("factor_id") or path.name.removesuffix(".metrics.json"))
        rebalance = path.parent.name if path.parent != root else "unknown"
        stem = path.name.removesuffix(".metrics.json")
        row = {"factor_id": factor_id, "rebalance": rebalance}
        for column in METRIC_COLUMNS:
            row[column] = _num(metrics.get(column))
        row["timeseries_path"] = str(path.with_name(f"{stem}.timeseries.csv"))
        row["layers_path"] = str(path.with_name(f"{stem}.layers.csv"))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["factor_id", "rebalance", *METRIC_COLUMNS, "timeseries_path", "layers_path"])
    if frame.empty:
        return frame
    frame["rebalance_hours"] = frame["rebalance"].map(rebalance_hours)
    return frame.sort_values(["sharpe", "factor_id"], ascending=[False, True], na_position="last").reset_index(drop=True)


def load_factor_meta(jsonl: str | Path | None) -> pd.DataFrame:
    """factor_id -> name/block/direction/hypothesis/formula from a factor JSONL."""
    columns = ["factor_id", *META_COLUMNS]
    if not jsonl or not Path(jsonl).is_file():
        return pd.DataFrame(columns=columns)
    rows = []
    with open(jsonl, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            factor_id = record.get("factor_id")
            if not factor_id:
                continue
            row = {"factor_id": str(factor_id)}
            for key in META_COLUMNS:
                value = record.get(key, "")
                row[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if not row["formula"] and isinstance(record.get("code"), str):
                row["formula"] = record["code"]
            rows.append(row)
    return pd.DataFrame(rows, columns=columns).drop_duplicates("factor_id", keep="last")


def join_periods(train: pd.DataFrame, test: pd.DataFrame | None) -> pd.DataFrame:
    """Add ``test_*`` metric columns, matched on (factor_id, rebalance)."""
    if test is None or test.empty:
        return train
    keep = ["factor_id", "rebalance", "return_pct", "sharpe", "max_drawdown_pct", "avg_turnover",
            "timeseries_path", "layers_path"]
    renamed = test[keep].rename(columns={c: f"test_{c}" for c in keep[2:]})
    out = train.merge(renamed, on=["factor_id", "rebalance"], how="left")
    return out


def best_per_factor(frame: pd.DataFrame, by: str = "sharpe") -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.sort_values(by, ascending=False, na_position="last")
    return ordered.drop_duplicates("factor_id", keep="first").reset_index(drop=True)


def stability_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per factor: how many rebalance periods are profitable, mean/min Sharpe, worst drawdown."""
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("factor_id")
    out = pd.DataFrame({
        "periods": grouped["rebalance"].count(),
        "positive_periods": grouped["return_pct"].apply(lambda s: int((s > 0).sum())),
        "mean_return_pct": grouped["return_pct"].mean(),
        "median_sharpe": grouped["sharpe"].median(),
        "min_sharpe": grouped["sharpe"].min(),
        "worst_drawdown_pct": grouped["max_drawdown_pct"].min(),
    })
    return out.sort_values(["positive_periods", "median_sharpe"], ascending=False).reset_index()


def load_timeseries(path: str | Path, max_points: int = 4000) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "timestamp" not in frame:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601", errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if len(frame) > max_points:
        # Keep the shape (and the worst drawdown point) while thinning for the browser.
        step = int(math.ceil(len(frame) / max_points))
        keep = set(range(0, len(frame), step)) | {len(frame) - 1}
        if "drawdown" in frame:
            keep.add(int(frame["drawdown"].to_numpy().argmin()))
        frame = frame.iloc[sorted(keep)]
    return frame.reset_index(drop=True)


def layer_summary(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decile mean per bar (bps) and cumulative simple return (%) of each signed decile.

    Layers store per-bar *log excess* returns, so the cumulative curve is
    ``exp(cumsum) - 1``. Returns (bars, cumulative) frames.
    """
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.read_csv(path)
    signed = [c for c in frame.columns if re.fullmatch(r"signed_q\d+", c)]
    signed.sort(key=lambda c: int(c.removeprefix("signed_q")))
    if not signed:
        return pd.DataFrame(), pd.DataFrame()
    values = frame[signed].apply(pd.to_numeric, errors="coerce")
    bars = pd.DataFrame({
        "decile": [c.removeprefix("signed_").upper() for c in signed],
        "mean_bps": (values.mean() * 1e4).to_numpy(),
        "hit_rate": (values > 0).mean().to_numpy(),
    })
    cumulative = np.expm1(values.fillna(0.0).cumsum().clip(-700, 700)) * 100.0
    cumulative.columns = [c.removeprefix("signed_").upper() for c in signed]
    cumulative.insert(0, "timestamp", pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601", errors="coerce"))
    return bars, cumulative


# ------------------------------------------------------------ live runner
def tail_lines(path: str | Path, max_lines: int, block: int = 1 << 16) -> list[str]:
    """Last ``max_lines`` lines of a file without reading all of it."""
    path = Path(path)
    if not path.is_file() or max_lines <= 0:
        return []
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        data = b""
        while end > 0 and data.count(b"\n") <= max_lines:
            start = max(0, end - block)
            handle.seek(start)
            data = handle.read(end - start) + data
            end = start
    lines = data.decode("utf-8", errors="replace").splitlines()
    if end > 0:
        lines = lines[1:]  # first line is probably partial
    return [line for line in lines[-max_lines:] if line.strip()]


def read_jsonl_tail(path: str | Path, max_lines: int) -> list[dict]:
    out = []
    for line in tail_lines(path, max_lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue  # a line being written right now
        if isinstance(value, dict):
            out.append(value)
    return out


def read_json(path: str | Path) -> dict | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def find_live_dirs(base: str | Path, max_depth: int = 3) -> list[Path]:
    base = Path(base)
    if not base.exists():
        return []
    found = []
    base_depth = len(base.parts)
    for dirpath, dirnames, filenames in os.walk(base):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []
        if "latest.json" in filenames or "audit.jsonl" in filenames:
            found.append(Path(dirpath))
            dirnames[:] = []
    return sorted(found)


def _side_notional(record: dict) -> tuple[float, float]:
    """(long, short) notional; exchange positions when signed, otherwise engine targets."""
    long_, short = 0.0, 0.0
    positions = record.get("exchange_positions")
    if positions:
        for row in positions:
            value = _num(row.get("notional"))
            if math.isnan(value):
                continue
            if value > 0:
                long_ += value
            else:
                short += -value
        return long_, short
    for value in (record.get("targets") or {}).values():
        value = _num(value)
        if value > 0:
            long_ += value
        elif value < 0:
            short += -value
    return long_, short


def audit_frame(records: list[dict]) -> pd.DataFrame:
    """One row per runner cycle."""
    rows = []
    for record in records:
        engine = record.get("engine")
        if not isinstance(engine, dict):
            continue
        long_, short = _side_notional(record)
        orders = record.get("orders") or []
        submitted = record.get("submitted") or []
        equity = _num(record.get("binance_equity"))
        if math.isnan(equity):
            equity = _num(engine.get("equity"))
        rows.append({
            "time": engine.get("timestamp") or record.get("timestamp"),
            "signal_timestamp": record.get("timestamp"),
            "signal_is_new": bool(record.get("signal_is_new")),
            "mode": record.get("mode", ""),
            "factor_id": record.get("factor_id", ""),
            "state": engine.get("state", ""),
            "equity": equity,
            "available": _num(record.get("binance_available_balance")),
            "engine_equity": _num(engine.get("equity")),
            "shadow_equity": _num(engine.get("shadow_equity")),
            "net_return": _num(engine.get("net_return")),
            "cost": _num(engine.get("cost")),
            "funding": _num(engine.get("funding")),
            "risk_events": engine.get("risk_events") or "",
            "warnings": engine.get("warnings") or "",
            "positions": int(_num(engine.get("positions")) if not math.isnan(_num(engine.get("positions"))) else 0),
            "long_notional": long_,
            "short_notional": short,
            "net_notional": long_ - short,
            "orders": len(orders),
            "rejected_orders": sum(1 for o in orders if o.get("rejected")),
            "submitted": len(submitted),
            "exchange_risk_notes": "; ".join(map(str, record.get("exchange_risk_notes") or [])),
            "manual_halt": json.dumps(record["manual_halt"], ensure_ascii=False) if record.get("manual_halt") else "",
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="ISO8601", errors="coerce")
    return frame.dropna(subset=["time"]).sort_values("time", kind="stable").reset_index(drop=True)


def event_frame(audit: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Risk events, warnings, guard notes, manual halts and close-all actions in time order."""
    rows = []
    if not audit.empty:
        for column, kind in (("risk_events", "risk"), ("warnings", "warning"),
                             ("exchange_risk_notes", "exchange_risk"), ("manual_halt", "manual_halt")):
            hits = audit[audit[column].astype(str) != ""]
            for _, row in hits.iterrows():
                rows.append({"time": row["time"], "kind": kind, "state": row["state"], "detail": row[column]})
        rejected = audit[audit["rejected_orders"] > 0]
        for _, row in rejected.iterrows():
            rows.append({"time": row["time"], "kind": "order_rejected", "state": row["state"],
                         "detail": f"{row['rejected_orders']} rejected of {row['orders']}"})
    for record in records:
        if record.get("type") == "close_all":
            rows.append({"time": pd.to_datetime(record.get("timestamp"), utc=True, errors="coerce"),
                         "kind": "close_all", "state": "", "detail": f"{len(record.get('orders') or [])} orders"})
    frame = pd.DataFrame(rows, columns=["time", "kind", "state", "detail"])
    if frame.empty:
        return frame
    return frame.sort_values("time", ascending=False, kind="stable").reset_index(drop=True)


def state_spans(audit: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Contiguous (start, end, state) spans for shading LIVE/SHADOW on charts."""
    if audit.empty:
        return []
    spans = []
    states = audit["state"].tolist()
    times = audit["time"].tolist()
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            end_time = times[i] if i < len(states) else times[-1]
            spans.append((times[start], end_time, states[start]))
            start = i
    return spans


def positions_frame(latest: dict | None, engine_state: dict | None) -> pd.DataFrame:
    """Current positions: exchange rows when signed, otherwise the engine's simulated book."""
    columns = ["symbol", "side", "notional", "quantity", "entry_price", "mark_price", "unrealized_pnl",
               "leverage", "source"]
    rows = []
    for row in (latest or {}).get("exchange_positions") or []:
        notional = _num(row.get("notional"))
        rows.append({
            "symbol": row.get("symbol"), "side": "LONG" if notional > 0 else "SHORT",
            "notional": notional, "quantity": _num(row.get("positionAmt")),
            "entry_price": _num(row.get("entryPrice")), "mark_price": _num(row.get("markPrice")),
            "unrealized_pnl": _num(row.get("unRealizedProfit")), "leverage": _num(row.get("leverage")),
            "source": "exchange",
        })
    if not rows and engine_state:
        for symbol, pos in (engine_state.get("positions") or {}).items():
            notional = _num(pos.get("notional"))
            entry, mark = _num(pos.get("entry_price")), _num(pos.get("mark_price"))
            pnl = notional * (mark / entry - 1.0) if entry and not math.isnan(entry) and not math.isnan(mark) else math.nan
            rows.append({
                "symbol": symbol, "side": "LONG" if notional > 0 else "SHORT", "notional": notional,
                "quantity": math.nan, "entry_price": entry, "mark_price": mark, "unrealized_pnl": pnl,
                "leverage": math.nan, "source": "engine",
            })
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("notional", ascending=False).reset_index(drop=True) if not frame.empty else frame


def targets_vs_actual(latest: dict | None) -> pd.DataFrame:
    latest = latest or {}
    targets = {str(k).upper(): _num(v) for k, v in (latest.get("targets") or {}).items()}
    actual: dict[str, float] = {}
    for row in latest.get("exchange_positions") or []:
        symbol = str(row.get("symbol", "")).upper()
        actual[symbol] = actual.get(symbol, 0.0) + _num(row.get("notional"))
    symbols = sorted(set(targets) | set(actual))
    frame = pd.DataFrame({
        "symbol": symbols,
        "target": [targets.get(s, 0.0) for s in symbols],
        "actual": [actual.get(s, 0.0) for s in symbols],
    })
    if frame.empty:
        return frame
    frame["diff"] = frame["actual"] - frame["target"]
    frame["diff_pct_of_target"] = np.where(frame["target"].abs() > 0,
                                           frame["diff"] / frame["target"].abs() * 100.0, np.nan)
    return frame


def orders_frame(records: list[dict], limit: int = 200) -> pd.DataFrame:
    """Flattened recent orders joined with their submission result."""
    rows = []
    for record in reversed(records):
        engine = record.get("engine") or {}
        # Results carry clientOrderId "sh_<cycle>_<plan index>_<digest>", which identifies
        # the plan row even when one symbol has two orders (hedge-mode reduce + open).
        by_index: dict[int, dict] = {}
        by_symbol: dict[str, dict] = {}
        for result in record.get("submitted") or []:
            if not isinstance(result, dict):
                continue
            client_id = str(result.get("clientOrderId") or result.get("client_order_id") or "")
            match = re.match(r"sh_[^_]+_(\d+)_", client_id)
            if match:
                by_index[int(match.group(1))] = result
            by_symbol.setdefault(str(result.get("symbol", "")).upper(), result)
        for index, order in enumerate(record.get("orders") or []):
            symbol = str(order.get("symbol", "")).upper()
            result = by_index.get(index) or ({} if by_index else by_symbol.get(symbol, {}))
            rows.append({
                "time": engine.get("timestamp") or record.get("timestamp"),
                "symbol": symbol, "side": order.get("side"), "quantity": _num(order.get("quantity")),
                "price": _num(order.get("price")), "target_notional": _num(order.get("target_notional")),
                "reduce_only": bool(order.get("reduce_only")), "rejected": bool(order.get("rejected")),
                "reason": order.get("reason") or "",
                "status": ("ERROR" if result.get("order_error") else result.get("status", ""))
                          if result else ("rejected" if order.get("rejected") else "not submitted"),
                "error": result.get("error", ""),
                "avg_price": _num(result.get("avgPrice")), "executed_qty": _num(result.get("executedQty")),
            })
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["time"] = pd.to_datetime(frame["time"], utc=True, format="ISO8601", errors="coerce")
    return frame


def errors_frame(path: str | Path, max_lines: int = 500) -> pd.DataFrame:
    records = read_jsonl_tail(path, max_lines)
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "type", "error", "retry_in_seconds"])
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), utc=True, format="ISO8601", errors="coerce")
    return frame.sort_values("timestamp", ascending=False).reset_index(drop=True)


def live_summary(latest: dict | None, engine_state: dict | None, latest_mtime: float | None,
                 now: pd.Timestamp | None = None) -> dict[str, Any]:
    latest, state = latest or {}, engine_state or {}
    engine = latest.get("engine") or {}
    equity = _num(latest.get("binance_equity"))
    if math.isnan(equity):
        equity = _num(engine.get("equity", state.get("equity")))
    peak = _num(state.get("peak_equity"))
    day_open = _num(state.get("day_open_equity"))
    now = now or pd.Timestamp.now(tz="UTC")
    age = (now - pd.Timestamp(latest_mtime, unit="s", tz="UTC")).total_seconds() if latest_mtime else math.nan
    long_, short = _side_notional(latest)
    gross = long_ + short
    return {
        "mode": latest.get("mode", ""),
        "factor_id": latest.get("factor_id", ""),
        "state": engine.get("state") or state.get("state", ""),
        "equity": equity,
        "available": _num(latest.get("binance_available_balance")),
        "day_pnl_pct": (equity / day_open - 1.0) * 100.0 if day_open and not math.isnan(day_open) else math.nan,
        # The engine's peak can lag the live NAV by one cycle; a drawdown is never positive.
        "drawdown_pct": min(0.0, (equity / peak - 1.0) * 100.0) if peak and not math.isnan(peak) else math.nan,
        "signal_timestamp": latest.get("timestamp", ""),
        "seconds_since_update": age,
        "long_notional": long_, "short_notional": short,
        "net_notional": long_ - short,
        "gross_leverage": gross / equity if equity and not math.isnan(equity) else math.nan,
        "net_exposure_pct": (long_ - short) / gross * 100.0 if gross else math.nan,
        "blocked_symbols": state.get("blocked_symbols") or {},
        "total_cost": _num(state.get("total_cost")),
        "total_funding": _num(state.get("total_funding")),
        "shadow_periods": state.get("shadow_periods"),
        "shadow_positive_periods": state.get("shadow_positive_periods"),
    }
