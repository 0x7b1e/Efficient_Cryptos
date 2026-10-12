"""Exchange-side safety guards for the live runner (all opt-in).

1. **Exchange risk controls** -- ``apply_exchange_risk_controls`` reads the
   *actual* Binance positions and account NAV (not the engine's estimate) and
   overrides targets: per-symbol stop/take from the exchange entry price, a
   same-day block after a take-profit or stop, and account-level daily stop,
   take-profit and drawdown halts reset at a configurable UTC offset.  It is the
   ``risk`` section of ``config/default.yaml``.  The functions below are moved
   verbatim from the former Predict_Cryptos testnet trader.

2. **Manual-change guard** -- after every successful sync the runner records a
   snapshot of the positions it manages.  If those positions change between
   syncs without the runner trading (a manual close in the Binance app, an
   external bot, a liquidation), the runner flattens everything it manages and
   halts until it is restarted with ``--reset-manual-halt``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable

def _execution_symbol_map(cfg) -> Dict[str, str]:
    raw = cfg.get("execution", {}).get("symbol_map", {}) if isinstance(cfg, dict) else {}
    return {str(k).upper(): str(v).upper() for k, v in raw.items() if k and v}

def _map_targets_for_execution(targets: Dict[str, float], cfg) -> Dict[str, float]:
    symbol_map = _execution_symbol_map(cfg)
    mapped: Dict[str, float] = {}
    for sym, notional in targets.items():
        trade_sym = symbol_map.get(str(sym).upper(), str(sym).upper())
        mapped[trade_sym] = mapped.get(trade_sym, 0.0) + float(notional)
    return mapped

def _risk_cfg(cfg) -> Dict[str, Any]:
    r = cfg.get("risk", {}) if isinstance(cfg, dict) else {}
    return {
        "enabled": bool(r.get("enabled", True)),
        "per_symbol_stop_loss_pct": float(
            r.get("per_symbol_stop_loss_pct", r.get("per_symbol_daily_loss_pct", 0.05))
        ),
        "take_profit_pct": float(r.get("take_profit_pct", 0.30)),
        "take_profit_block_for_day": bool(r.get("take_profit_block_for_day", True)),
        "daily_stop_loss_pct": float(r.get("daily_stop_loss_pct", 0.05)),
        "daily_take_profit_pct": float(r.get("daily_take_profit_pct", 0.30)),
        "total_drawdown_pct": float(r.get("total_drawdown_pct", 0.20)),
        # Account-level daily circuit breakers can be disabled explicitly for
        # testnet strategy iteration. Per-symbol stop/take controls remain on.
        "daily_halt_enabled": bool(r.get("daily_halt_enabled", True)),
        "daily_reset_utc_offset_hours": int(r.get("daily_reset_utc_offset_hours", 8)),
        "state_file": str(r.get("state_file", "data/live_risk_state.json")),
    }

def _risk_day(offset_hours: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() + offset_hours * 3600))

def _load_risk_state(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Failed to read risk state file; reinitializing: %s", path)
        return {}

def _save_risk_state(path: str, state: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

def _position_qty(row: Dict[str, Any]) -> float:
    return float(row.get("positionAmt", 0.0) or 0.0)

def _position_entry_price(row: Dict[str, Any]) -> float:
    return float(row.get("entryPrice", 0.0) or 0.0)

def _position_mark_price(row: Dict[str, Any]) -> float:
    return float(row.get("markPrice", 0.0) or 0.0)

def _position_unrealized_pnl(row: Dict[str, Any]) -> float:
    return float(row.get("unRealizedProfit", row.get("unrealizedProfit", 0.0)) or 0.0)

def _position_notional(row: Dict[str, Any]) -> float:
    notional = abs(float(row.get("notional", 0.0) or 0.0))
    if notional > 0:
        return notional
    return abs(_position_qty(row)) * _position_mark_price(row)

def _position_entry_notional(row: Dict[str, Any]) -> float:
    entry_notional = abs(_position_qty(row)) * _position_entry_price(row)
    return entry_notional if entry_notional > 0 else _position_notional(row)

def _position_return_from_entry(row: Dict[str, Any]) -> float | None:
    """Return PnL relative to the exchange-reported position opening notional."""
    entry_notional = _position_entry_notional(row)
    if entry_notional <= 0:
        return None
    return _position_unrealized_pnl(row) / entry_notional

def _position_side(row: Dict[str, Any]) -> str:
    return str(row.get("positionSide", "BOTH") or "BOTH").upper()

def _apply_live_risk_controls(
    targets: Dict[str, float],
    current_positions: list[Dict[str, Any]],
    equity: float | None,
    cfg,
) -> tuple[Dict[str, float], list[str]]:
    """Execution-layer hard risk control: override targets so the sync logic closes positions."""
    rcfg = _risk_cfg(cfg)
    if not rcfg["enabled"]:
        return targets, []

    today = _risk_day(rcfg["daily_reset_utc_offset_hours"])
    state = _load_risk_state(rcfg["state_file"])
    if state.get("day") != today:
        state = {
            "day": today,
            "blocked_symbols": [],
            "blocked_reasons": {},
            "day_open_nav": equity,
            "day_peak_nav": equity,
            "day_halt": False,
        }

    notes: list[str] = []
    blocked = set(str(s).upper() for s in state.get("blocked_symbols", []))
    blocked_reasons = {
        str(symbol).upper(): str(reason)
        for symbol, reason in (state.get("blocked_reasons", {}) or {}).items()
    }
    positions_by_symbol = {str(p.get("symbol", "")).upper(): p for p in current_positions}

    if not rcfg["daily_halt_enabled"]:
        # A previous testnet run may have persisted day_halt=true. Ignore and
        # clear that account-level state when the operator explicitly disables
        # the daily circuit breaker for testnet.
        if state.get("day_halt"):
            state["day_halt"] = False
            notes.append("Account-level daily halt disabled for testnet")

    if rcfg["daily_halt_enabled"] and equity is not None and equity > 0:
        day_open = state.get("day_open_nav")
        day_open = float(day_open) if day_open else equity
        state["day_open_nav"] = day_open
        daily_return = equity / day_open - 1.0 if day_open > 0 else 0.0
        if (
            rcfg["daily_stop_loss_pct"] > 0
            and daily_return <= -rcfg["daily_stop_loss_pct"]
        ):
            state["day_halt"] = True
            notes.append(
                f"Daily NAV return from opening node {daily_return:.2%} "
                f"<= -{rcfg['daily_stop_loss_pct']:.2%}; force flat and halt for the day"
            )
        if (
            rcfg["daily_take_profit_pct"] > 0
            and daily_return >= rcfg["daily_take_profit_pct"]
        ):
            state["day_halt"] = True
            notes.append(
                f"Daily NAV return from opening node {daily_return:.2%} "
                f">= {rcfg['daily_take_profit_pct']:.2%}; force flat and halt for the day"
            )
        peak = state.get("day_peak_nav", state.get("peak_equity"))
        peak = float(peak) if peak else equity
        peak = max(peak, equity)
        state["day_peak_nav"] = peak
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd >= rcfg["total_drawdown_pct"]:
            state["day_halt"] = True
            notes.append(f"Daily NAV drawdown {dd:.2%} >= {rcfg['total_drawdown_pct']:.2%}; force flat and stop new entries for the day")

    adjusted = dict(targets)
    if rcfg["daily_halt_enabled"] and state.get("day_halt"):
        adjusted = {}
        for sym in positions_by_symbol:
            adjusted[sym] = 0.0
        _save_risk_state(rcfg["state_file"], state)
        return adjusted, notes

    for sym, pos in positions_by_symbol.items():
        qty = _position_qty(pos)
        if abs(qty) <= 0:
            continue
        position_return = _position_return_from_entry(pos)
        if (
            rcfg["take_profit_pct"] > 0
            and position_return is not None
            and position_return >= rcfg["take_profit_pct"]
        ):
            adjusted[sym] = 0.0
            if rcfg["take_profit_block_for_day"]:
                blocked.add(sym)
                blocked_reasons[sym] = "take_profit"
            notes.append(
                f"{sym} take profit from entry {position_return:.2%} "
                f">= {rcfg['take_profit_pct']:.2%}; force close"
                + (" and block new entries for the day" if rcfg["take_profit_block_for_day"] else "")
            )
            continue
        if (
            rcfg["per_symbol_stop_loss_pct"] > 0
            and position_return is not None
            and position_return <= -rcfg["per_symbol_stop_loss_pct"]
        ):
            blocked.add(sym)
            blocked_reasons[sym] = "stop_loss"
            adjusted[sym] = 0.0
            notes.append(
                f"{sym} stop loss from entry {position_return:.2%} "
                f"<= -{rcfg['per_symbol_stop_loss_pct']:.2%}; "
                "force close and block new entries for the day"
            )

    for sym in blocked:
        if adjusted.get(sym, 0.0) != 0.0:
            adjusted[sym] = 0.0
            reason = blocked_reasons.get(sym, "unknown")
            notes.append(
                f"{sym} has triggered a daily risk block ({reason}); ignore new targets"
            )

    state["blocked_symbols"] = sorted(blocked)
    state["blocked_reasons"] = {
        symbol: blocked_reasons[symbol]
        for symbol in sorted(blocked)
        if symbol in blocked_reasons
    }
    _save_risk_state(rcfg["state_file"], state)
    return adjusted, notes


# Public name used by the runner.
apply_exchange_risk_controls = _apply_live_risk_controls


def _snapshot_changes(previous: dict[str, float], current: dict[str, float]) -> list[str]:
    changes = []
    for symbol in sorted(set(previous) | set(current)):
        before = float(previous.get(symbol, 0.0))
        after = float(current.get(symbol, 0.0))
        tolerance = max(1e-12, max(abs(before), abs(after)) * 1e-8)
        if abs(before - after) > tolerance:
            changes.append(f"{symbol}: {before:g} -> {after:g}")
    return changes


def position_snapshot(position_rows: Iterable[Dict[str, Any]],
                      managed_symbols: Iterable[str] | None = None) -> Dict[str, float]:
    """Non-zero position quantities keyed by symbol (``SYMBOL:SIDE`` in hedge mode)."""
    managed = None if managed_symbols is None else {str(s).upper() for s in managed_symbols}
    snapshot: Dict[str, float] = {}
    for row in position_rows:
        symbol = str(row.get("symbol", "")).upper()
        if not symbol or (managed is not None and symbol not in managed):
            continue
        qty = _position_qty(row)
        if abs(qty) <= 1e-12:
            continue
        side = _position_side(row)
        snapshot[symbol if side == "BOTH" else f"{symbol}:{side}"] = qty
    return snapshot


class ManualChangeGuard:
    """Detect position changes the runner did not make, then halt.

    State file fields: ``armed`` (a post-sync snapshot exists), ``halted``,
    ``positions`` (the snapshot) and ``reason``.
    """

    def __init__(self, state_file: Path):
        self.path = Path(state_file)
        self.state: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                logging.warning("Failed to read manual-change guard state: %s", self.path)

    @property
    def halted(self) -> bool:
        return bool(self.state.get("halted"))

    def _save(self, **state: Any) -> None:
        self.state = {**state, "updated_at_ms": int(time.time() * 1000)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, sort_keys=True),
                             encoding="utf-8")

    def reset(self) -> None:
        self._save(armed=False, halted=False, positions={}, reason="reset by operator")

    def check(self, current: Dict[str, float]) -> list[str]:
        """Return the detected changes; when non-empty the guard is now halted."""
        if not self.state.get("armed"):
            return []
        changes = _snapshot_changes(self.state.get("positions", {}), current)
        if changes:
            self._save(armed=False, halted=True, positions=current,
                       reason="External/manual position change detected: " + "; ".join(changes))
        return changes

    def arm(self, current: Dict[str, float]) -> None:
        self._save(armed=True, halted=False, positions=current)

    def disarm(self, reason: str) -> None:
        self._save(armed=False, halted=False, positions={}, reason=reason)
