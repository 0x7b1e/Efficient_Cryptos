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


def metrics_kind(payload: dict) -> str:
    """'shadow' for engine replays (run_selected_factor_shadow.py), 'rust' for the Rust backtest."""
    return "shadow" if "final_state" in payload or "shadow_entries" in payload else "rust"


def find_sweep_roots(base: str | Path, max_depth: int = 3, kind: str | None = "rust") -> list[Path]:
    """Directories below ``base`` with ``<rebalance>/*.metrics.json`` (or metrics directly).

    ``kind`` keeps only Rust sweeps ("rust") or engine replays ("shadow"); both
    use the same layout, so one metrics file per directory is peeked at.
    """
    base = Path(base)
    if not base.exists():
        return []
    roots: set[Path] = set()
    base_depth = len(base.parts)
    for dirpath, dirnames, filenames in os.walk(base):
        depth = len(Path(dirpath).parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        sample = next((name for name in filenames if name.endswith(".metrics.json")), None)
        if sample:
            here = Path(dirpath)
            payload = read_json(here / sample) or {}
            if kind is None or metrics_kind(payload) == kind:
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
        if not isinstance(metrics, dict) or metrics_kind(metrics) != "rust":
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
    # Share of the gross (pre-cost) PnL eaten by fees + slippage. Funding is not in
    # metrics.json; the factor detail adds it from the timeseries.
    initial = frame["final_equity"] / (1.0 + frame["return_pct"] / 100.0)
    gross = frame["final_equity"] - initial + frame["total_cost"]
    frame["cost_pct_of_equity"] = frame["total_cost"] / initial * 100.0
    frame["cost_share_of_gross_pct"] = np.where(gross > 0, frame["total_cost"] / gross * 100.0, np.nan)
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


# ------------------------------------------------------------ costs
def _cost_summary(gross: float, cost: float, funding_paid: float, initial: float) -> dict[str, float]:
    net = gross - cost - funding_paid
    drag = cost + funding_paid
    return {
        "gross_pnl": gross, "cost": cost, "funding_paid": funding_paid, "net_pnl": net,
        "initial_equity": initial,
        "gross_return_pct": gross / initial * 100.0 if initial else math.nan,
        "cost_pct_of_equity": cost / initial * 100.0 if initial else math.nan,
        "funding_pct_of_equity": funding_paid / initial * 100.0 if initial else math.nan,
        "net_return_pct": net / initial * 100.0 if initial else math.nan,
        # Only meaningful when the strategy made money before costs.
        "cost_share_of_gross_pct": drag / gross * 100.0 if gross > 0 else math.nan,
    }


def rust_costs(ts: pd.DataFrame, bars_per_year: float = 48 * 365) -> tuple[pd.DataFrame, dict[str, float]]:
    """Cumulative gross PnL, fees+slippage, funding and net PnL from a Rust timeseries.

    Rust books ``net = gross - cost/equity - funding/equity`` per bar with
    ``cost`` and ``funding`` in dollars (funding > 0 means paid).
    """
    if ts.empty or "pnl" not in ts:
        return pd.DataFrame(), {}
    cost = pd.to_numeric(ts.get("cost", 0.0), errors="coerce").fillna(0.0)
    funding = pd.to_numeric(ts.get("funding", 0.0), errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(ts["pnl"], errors="coerce").fillna(0.0)
    gross = pnl + cost + funding
    curves = pd.DataFrame({
        "timestamp": ts["timestamp"],
        "gross_pnl": gross.cumsum(), "cost": cost.cumsum(), "funding_paid": funding.cumsum(),
        "net_pnl": pnl.cumsum(),
    })
    equity0 = float(pd.to_numeric(ts["equity"], errors="coerce").iloc[0]) - float(pnl.iloc[0])
    summary = _cost_summary(float(gross.sum()), float(cost.sum()), float(funding.sum()), equity0)
    if "turnover" in ts:
        summary["annual_turnover"] = float(pd.to_numeric(ts["turnover"], errors="coerce").mean() * bars_per_year)
    return curves, summary


def scan_shadow(root: str | Path) -> pd.DataFrame:
    """One row per (factor, rebalance) from run_selected_factor_shadow.py outputs."""
    root = Path(root)
    rows = []
    for path in list(root.glob("*/*.metrics.json")) + list(root.glob("*.metrics.json")):
        payload = read_json(path)
        if not payload or metrics_kind(payload) != "shadow":
            continue
        config = payload.get("config") or {}
        stem = path.name.removesuffix(".metrics.json")
        initial = _num(config.get("initial_equity"))
        if math.isnan(initial):
            ret = _num(payload.get("return_pct"))
            initial = _num(payload.get("final_equity")) / (1 + ret / 100) if not math.isnan(ret) else math.nan
        cost, funding_received = _num(payload.get("total_cost")), _num(payload.get("total_funding"))
        final = _num(payload.get("final_equity"))
        gross = final - initial + cost - (0.0 if math.isnan(funding_received) else funding_received)
        rows.append({
            "factor_id": str(payload.get("factor_id") or stem),
            "rebalance": path.parent.name if path.parent != root else "unknown",
            "return_pct": _num(payload.get("return_pct")),
            "sharpe": _num(payload.get("sharpe_annualized")),
            "max_drawdown_pct": _num(payload.get("max_drawdown_pct")),
            "final_state": payload.get("final_state", ""),
            "shadow_entries": _num(payload.get("shadow_entries")),
            "shadow_exits": _num(payload.get("shadow_exits")),
            "shadow_equity": _num(payload.get("shadow_equity")),
            "total_cost": cost,
            "funding_paid": -funding_received if not math.isnan(funding_received) else math.nan,
            "cost_pct_of_equity": cost / initial * 100 if initial else math.nan,
            "cost_share_of_gross_pct": (cost - (funding_received or 0.0)) / gross * 100 if gross > 0 else math.nan,
            "bars": _num(payload.get("bars")),
            "gross_leverage": _num(config.get("gross_leverage")),
            "timeseries_path": str(path.with_name(f"{stem}.timeseries.csv")),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["rebalance_hours"] = frame["rebalance"].map(rebalance_hours)
    return frame.sort_values(["sharpe", "factor_id"], ascending=[False, True], na_position="last").reset_index(drop=True)


def shadow_timeseries(path: str | Path, initial_equity: float, gross_leverage: float | None = None,
                      max_points: int = 6000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engine replay rows -> (per-step frame, risk-event frame).

    The per-step frame has live equity, shadow equity, state, cumulative live
    cost/funding (exact when the replay recorded ``total_cost``), and exposure
    / margin use when the replay recorded notionals.
    """
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "timestamp" not in frame:
        return pd.DataFrame(), pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601", errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    events = frame[frame.get("risk_events", pd.Series("", index=frame.index)).fillna("").astype(str) != ""]
    events = events[["timestamp", "state", "risk_events", "equity", "shadow_equity"]].copy() if not events.empty else pd.DataFrame()
    steps = frame[frame.get("kind", "step") == "step"].copy() if "kind" in frame else frame.copy()
    equity = pd.to_numeric(steps["equity"], errors="coerce")
    previous = equity.shift(1).fillna(initial_equity)
    if "total_cost" in steps:
        steps["cum_cost"] = pd.to_numeric(steps["total_cost"], errors="coerce")
        steps["cum_funding_paid"] = -pd.to_numeric(steps["total_funding"], errors="coerce")
        steps["cost_exact"] = True
    else:
        # Older replays: rebuild from per-row fractions on the live book.
        live = steps["state"] == "LIVE"
        steps["cum_cost"] = (pd.to_numeric(steps["cost"], errors="coerce").fillna(0) * previous).where(live, 0).cumsum()
        steps["cum_funding_paid"] = (-pd.to_numeric(steps.get("funding", 0.0), errors="coerce").fillna(0)).where(live, 0).cumsum()
        steps["cost_exact"] = False
    steps["net_pnl"] = equity - initial_equity
    steps["gross_pnl"] = steps["net_pnl"] + steps["cum_cost"] + steps["cum_funding_paid"]
    if "long_notional" in steps:
        gross = pd.to_numeric(steps["long_notional"], errors="coerce") + pd.to_numeric(steps["short_notional"], errors="coerce")
        steps["gross_notional"] = gross
        steps["net_notional"] = pd.to_numeric(steps["long_notional"], errors="coerce") - pd.to_numeric(steps["short_notional"], errors="coerce")
        steps["gross_leverage"] = gross / equity.where(equity > 0)
        if gross_leverage and gross_leverage > 0:
            # Binance initial margin is notional / leverage; share of equity locked as margin.
            steps["margin_used_pct"] = gross / gross_leverage / equity.where(equity > 0) * 100.0
    if len(steps) > max_points:
        step = int(math.ceil(len(steps) / max_points))
        keep = set(range(0, len(steps), step)) | {len(steps) - 1}
        keep |= set(np.flatnonzero(steps["state"].ne(steps["state"].shift()).to_numpy()))  # keep state switches
        steps = steps.iloc[sorted(keep)]
    return steps.reset_index(drop=True), events.reset_index(drop=True)


def frame_state_spans(frame: pd.DataFrame, time_col: str = "timestamp") -> list[tuple]:
    if frame.empty:
        return []
    return state_spans(frame.rename(columns={time_col: "time"}))


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
        totals = record.get("engine_totals") or {}
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
            # Exact engine running totals (runner writes them since the dashboard was
            # added); NaN for older audit lines, which fall back to an estimate below.
            "total_cost": _num(totals.get("total_cost")),
            "total_funding": _num(totals.get("total_funding")),
            "shadow_long_notional": _num(totals.get("shadow_long_notional")),
            "shadow_short_notional": _num(totals.get("shadow_short_notional")),
            # Estimate for old lines: cost is a fraction of the cycle's start equity.
            "cost_estimate_usd": _num(engine.get("cost")) * (equity if not math.isnan(equity) else _num(engine.get("equity")))
                                 if engine.get("state") == "LIVE" else 0.0,
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
    # audit.jsonl is written in cycle order. A new-signal cycle is stamped with the
    # signal bar's *open* time, which is earlier than the poll cycles just before
    # it, so sorting would shuffle cycles; keep file order and a monotone clock.
    frame = frame.dropna(subset=["time"]).reset_index(drop=True)
    frame["time"] = frame["time"].cummax()
    exact = frame["total_cost"].notna()
    frame["cost_exact"] = exact
    if exact.all():
        # Costs inside the loaded window, from the engine's own running totals.
        first = frame.iloc[0]
        frame["cum_cost_usd"] = frame["total_cost"] - first["total_cost"]
        frame["cum_funding_paid_usd"] = -(frame["total_funding"] - first["total_funding"])
    else:
        frame["cum_cost_usd"] = pd.to_numeric(frame["cost_estimate_usd"], errors="coerce").fillna(0.0).cumsum()
        frame["cum_funding_paid_usd"] = 0.0
    gross = frame["long_notional"] + frame["short_notional"]
    frame["gross_leverage"] = gross / frame["equity"].where(frame["equity"] > 0)
    # Binance's own view of locked margin: NAV minus what is still free.
    frame["margin_used"] = frame["equity"] - frame["available"]
    frame["margin_used_pct"] = frame["margin_used"] / frame["equity"].where(frame["equity"] > 0) * 100.0
    return frame


def risk_config(latest: dict | None) -> dict[str, Any]:
    """Risk settings the runner recorded in ``engine_totals.config`` (empty for old runs)."""
    return dict(((latest or {}).get("engine_totals") or {}).get("config") or {})


def recovery_progress(engine_state: dict | None, min_periods: int, positive_periods: int,
                      recovery_pnl_pct: float) -> dict[str, Any]:
    """How far a SHADOW book is from the three recovery conditions."""
    state = engine_state or {}
    shadow_equity = _num(state.get("shadow_equity"))
    periods = int(_num(state.get("shadow_periods")) or 0) if not math.isnan(_num(state.get("shadow_periods"))) else 0
    positive = int(_num(state.get("shadow_positive_periods")) or 0) if not math.isnan(_num(state.get("shadow_positive_periods"))) else 0
    pnl_pct = (shadow_equity - 1.0) * 100.0 if not math.isnan(shadow_equity) else math.nan
    blocked = "__ACCOUNT__" in (state.get("blocked_symbols") or {})
    checks = {
        "periods": (periods, min_periods, periods >= min_periods),
        "positive_periods": (positive, positive_periods, positive >= positive_periods),
        "shadow_pnl_pct": (pnl_pct, recovery_pnl_pct * 100.0,
                           not math.isnan(pnl_pct) and pnl_pct >= recovery_pnl_pct * 100.0),
        "account_not_halted": (not blocked, True, not blocked),
    }
    return {"state": state.get("state", ""), "checks": checks,
            "ready": all(ok for _, _, ok in checks.values()),
            "shadow_positions": state.get("shadow_positions") or {}}


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
        "funding_paid": -_num(state.get("total_funding")),
        "shadow_periods": state.get("shadow_periods"),
        "shadow_positive_periods": state.get("shadow_positive_periods"),
    }
