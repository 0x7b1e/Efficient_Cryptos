"""Risk state machine shared by the offline replay and the live runner.

Accounting conventions
----------------------
* A position is a fixed *quantity*.  Its ``notional`` is re-marked every time
  a price is observed (``notional *= close / previous_close``), exactly like an
  exchange position whose quantity does not change between rebalances.
* PnL is the sum of simple dollar returns ``notional * (close / prev - 1)``.
  Summing per-asset log returns and exponentiating the sum is *not* a valid
  portfolio return; it systematically overstates short books (by roughly
  ``0.5 * w**2 * sigma**2`` per bar) and was removed.
* Every change of exposure pays ``_cost`` (fee + slippage + participation
  impact), including risk liquidations (stop/take, daily halt, SHADOW entry).
* Funding is applied in ``step`` when a bar carries ``funding_rate``: longs pay
  ``notional * rate`` and shorts receive it.
* Dollar PnL and costs are booked against ``self.equity`` in LIVE.  In SHADOW
  the same dollar amounts are expressed relative to the frozen live equity and
  compounded into the normalised ``shadow_equity``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import math
import json
from pathlib import Path


@dataclass
class EngineConfig:
    initial_equity: float = 10_000.0
    gross_leverage: float = 3.0
    margin_buffer_pct: float = 0.20
    rebalance_bars: int = 24
    per_symbol_stop_loss_pct: float = 0.05
    per_symbol_take_profit_pct: float = 0.30
    daily_stop_loss_pct: float = 0.05
    daily_take_profit_pct: float = 0.30
    portfolio_drawdown_pct: float = 0.10
    shadow_min_periods: int = 3
    shadow_positive_periods: int = 2
    shadow_recovery_pnl_pct: float = 0.01
    fee_bps: float = 5.0
    slippage_bps: float = 1.0
    funding_enabled: bool = True
    step_size: float = 0.0
    min_notional: float = 0.0


@dataclass
class Position:
    notional: float
    entry_price: float
    entry_time: str
    mark_price: float | None = None

    def __post_init__(self):
        if self.mark_price is None:
            self.mark_price = self.entry_price


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass
class Engine:
    config: EngineConfig
    equity: float = field(init=False)
    state: str = field(default="LIVE", init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    shadow_positions: dict[str, Position] = field(default_factory=dict, init=False)
    shadow_equity: float = field(default=1.0, init=False)
    shadow_period_return: float = field(default=1.0, init=False)
    shadow_periods: int = field(default=0, init=False)
    shadow_positive_periods: int = field(default=0, init=False)
    peak_equity: float = field(init=False)
    day: str | None = field(default=None, init=False)
    day_open_equity: float = field(default=None, init=False)
    day_peak_equity: float = field(default=None, init=False)
    blocked_symbols: dict[str, str] = field(default_factory=dict, init=False)
    bar_index: int = field(default=0, init=False)
    shadow_entries: int = field(default=0, init=False)
    shadow_exits: int = field(default=0, init=False)
    total_cost: float = field(default=0.0, init=False)
    total_funding: float = field(default=0.0, init=False)

    def __post_init__(self):
        if self.config.gross_leverage <= 0:
            raise ValueError("gross_leverage must be positive")
        self.equity = self.config.initial_equity
        self.peak_equity = self.equity

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _day(ts: str) -> str:
        value = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date().isoformat()

    def _reset_day(self, ts: str) -> None:
        day = self._day(ts)
        # Only move forward. The live runner marks with wall-clock time but
        # steps with the signal bar's open time, which can be the previous UTC
        # day just after midnight; flapping back would clear the day's blocks.
        if self.day is None or day > self.day:
            self.day = day
            self.day_open_equity = self.equity
            self.day_peak_equity = self.equity
            self.blocked_symbols.clear()

    @staticmethod
    def _position_return(pos: Position, close: float) -> float:
        if pos.entry_price <= 0:
            return 0.0
        return (close / pos.entry_price - 1.0) * (1.0 if pos.notional >= 0 else -1.0)

    def _cost(self, old: float, new: float, price: float, quote_volume: float) -> float:
        """Dollar cost of moving exposure from ``old`` to ``new`` notional."""
        delta = abs(new - old)
        if delta == 0:
            return 0.0
        participation = delta / max(quote_volume, 1e-12)
        slip = self.config.slippage_bps + min(80.0, 60.0 * participation)
        return delta * (self.config.fee_bps + slip) / 10_000.0

    def _market_volume(self, market: dict[str, dict[str, Any]], symbol: str) -> float:
        bar = market.get(symbol) or {}
        return _finite_float(bar.get("quote_volume")) or 0.0

    def _book_dollars(self, amount: float) -> float:
        """Book a dollar PnL/cost on the active book and return it as a fraction."""
        reference = max(self.equity, 1e-12)
        fraction = amount / reference
        if self.state == "LIVE":
            self.equity = max(0.0, self.equity + amount)
            self.peak_equity = max(self.peak_equity, self.equity)
        else:
            growth = max(0.0, 1.0 + fraction)
            self.shadow_equity *= growth
            self.shadow_period_return *= growth
        return fraction

    def _active(self) -> dict[str, Position]:
        return self.positions if self.state == "LIVE" else self.shadow_positions

    # ---------------------------------------------------------------- marking
    def _mark_positions(self, positions: dict[str, Position],
                        market: dict[str, dict[str, Any]]) -> tuple[float, list[str]]:
        """Re-mark positions at ``close``; return dollar PnL and missing symbols."""
        pnl = 0.0
        missing: list[str] = []
        for symbol, pos in positions.items():
            close = _finite_float((market.get(symbol) or {}).get("close"))
            if close is None or close <= 0:
                missing.append(symbol)
                continue
            previous = float(pos.mark_price or pos.entry_price)
            if previous > 0:
                ratio = close / previous
                pnl += pos.notional * (ratio - 1.0)
                pos.notional *= ratio
            pos.mark_price = close
        return pnl, missing

    def _funding(self, positions: dict[str, Position],
                 market: dict[str, dict[str, Any]]) -> float:
        if not self.config.funding_enabled:
            return 0.0
        pnl = 0.0
        for symbol, pos in positions.items():
            rate = _finite_float((market.get(symbol) or {}).get("funding_rate"))
            if rate:
                pnl -= pos.notional * rate
        return pnl

    # ------------------------------------------------------------------- risk
    def _liquidate(self, symbols: list[str], market: dict[str, dict[str, Any]]) -> float:
        """Close live positions and charge their exit cost. Returns dollars."""
        cost = 0.0
        for symbol in symbols:
            pos = self.positions.pop(symbol, None)
            if pos is None:
                continue
            price = float(pos.mark_price or pos.entry_price)
            cost += self._cost(pos.notional, 0.0, price, self._market_volume(market, symbol))
        return cost

    def _enter_shadow(self, events: list[str], reason: str,
                      market: dict[str, dict[str, Any]]) -> float:
        if self.state == "SHADOW":
            return 0.0
        cost = self._liquidate(list(self.positions), market)
        self.shadow_positions.clear()
        self.state = "SHADOW"
        self.shadow_equity = 1.0
        self.shadow_period_return = 1.0
        self.shadow_periods = self.shadow_positive_periods = 0
        self.shadow_entries += 1
        events.append(reason)
        return cost

    def _risk(self, ts: str, market: dict[str, dict[str, Any]]) -> tuple[list[str], float]:
        """Evaluate risk; liquidation costs are booked here. Returns (events, cost$)."""
        events: list[str] = []
        cost = 0.0
        self._reset_day(ts)
        self.day_peak_equity = max(self.day_peak_equity or self.equity, self.equity)
        if self.state == "LIVE":
            daily = self.equity / max(self.day_open_equity or self.equity, 1e-12) - 1.0
            day_dd = self.equity / max(self.day_peak_equity, 1e-12) - 1.0
            if (daily <= -self.config.daily_stop_loss_pct
                    or daily >= self.config.daily_take_profit_pct
                    or day_dd <= -self.config.portfolio_drawdown_pct):
                cost += self._liquidate(list(self.positions), market)
                self.blocked_symbols["__ACCOUNT__"] = "daily_risk"
                events.append("DAILY_ACCOUNT_HALT")
            for symbol, pos in list(self.positions.items()):
                bar = market.get(symbol)
                close = _finite_float((bar or {}).get("close"))
                if close is None:
                    continue
                r = self._position_return(pos, close)
                if r <= -self.config.per_symbol_stop_loss_pct:
                    cost += self._liquidate([symbol], market)
                    self.blocked_symbols[symbol] = "stop_loss"
                    events.append(f"STOP_LOSS:{symbol}")
                elif r >= self.config.per_symbol_take_profit_pct:
                    cost += self._liquidate([symbol], market)
                    self.blocked_symbols[symbol] = "take_profit"
                    events.append(f"TAKE_PROFIT:{symbol}")
            if any(x.startswith(("STOP_LOSS:", "TAKE_PROFIT:")) for x in events):
                cost += self._enter_shadow(events, "SYMBOL_RISK_TO_SHADOW", market)
            if self.state == "LIVE" and self.equity / max(self.peak_equity, 1e-12) - 1.0 <= -self.config.portfolio_drawdown_pct:
                cost += self._enter_shadow(events, "STOP_TO_SHADOW", market)
            elif self.state == "LIVE" and "__ACCOUNT__" in self.blocked_symbols:
                # The shadow engine keeps calculating while the execution account
                # is flat/blocked for the day. Only the counterfactual shadow
                # book observes recovery.
                cost += self._enter_shadow(events, "DAILY_HALT_TO_SHADOW", market)
        if cost:
            # Liquidations happen on the live book before/while switching state.
            self.equity = max(0.0, self.equity - cost)
            self.total_cost += cost
        return events, cost

    # ------------------------------------------------------------- rebalance
    def _targets(self, signals, market, account_equity):
        desired = {}
        if "__ACCOUNT__" in self.blocked_symbols and self.state == "LIVE":
            return desired
        cap = account_equity * (1.0 - self.config.margin_buffer_pct) * self.config.gross_leverage
        for row in sorted(signals, key=lambda x: float(x.get("signal", 0.0)), reverse=True):
            symbol = str(row["symbol"]).upper()
            if (symbol in self.blocked_symbols and self.state == "LIVE") or symbol not in market:
                continue
            value = _finite_float(row.get("target_notional", 0.0)) or 0.0
            if value:
                desired[symbol] = value
        gross = sum(abs(v) for v in desired.values())
        if gross > cap and gross > 0:
            scale = cap / gross
            desired = {k: v * scale for k, v in desired.items()}
        return desired

    def _sync_positions(self, positions, desired, market, ts) -> float:
        """Move ``positions`` to ``desired``; return the dollar trading cost.

        Symbols without a current price keep their position (it cannot be
        traded or valued this bar) instead of silently disappearing.
        """
        cost = 0.0
        for symbol in sorted(set(positions) | set(desired)):
            bar = market.get(symbol)
            close = _finite_float((bar or {}).get("close"))
            if close is None or close <= 0:
                continue
            old = positions.get(symbol)
            old_notional = old.notional if old else 0.0
            new_notional = float(desired.get(symbol, 0.0))
            cost += self._cost(old_notional, new_notional, close, self._market_volume(market, symbol))
            if not new_notional:
                positions.pop(symbol, None)
                continue
            if old and old_notional * new_notional > 0:
                # Same direction: keep the entry, or average it in when adding,
                # like the exchange's own entryPrice.
                entry, entry_time = old.entry_price, old.entry_time
                if abs(new_notional) > abs(old_notional) and old.entry_price > 0:
                    old_qty = abs(old_notional) / close
                    add_qty = (abs(new_notional) - abs(old_notional)) / close
                    entry = (old_qty * old.entry_price + add_qty * close) / (old_qty + add_qty)
                positions[symbol] = Position(new_notional, entry, entry_time, close)
            else:
                positions[symbol] = Position(new_notional, close, ts, close)
        return cost

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity, "state": self.state,
            "positions": {k: vars(v).copy() for k, v in self.positions.items()},
            "shadow_positions": {k: vars(v).copy() for k, v in self.shadow_positions.items()},
            "shadow_equity": self.shadow_equity,
            "shadow_period_return": self.shadow_period_return,
            "shadow_periods": self.shadow_periods,
            "shadow_positive_periods": self.shadow_positive_periods,
            "peak_equity": self.peak_equity, "day": self.day,
            "day_open_equity": self.day_open_equity,
            "day_peak_equity": self.day_peak_equity,
            "blocked_symbols": dict(self.blocked_symbols), "bar_index": self.bar_index,
            "shadow_entries": self.shadow_entries, "shadow_exits": self.shadow_exits,
            "total_cost": self.total_cost, "total_funding": self.total_funding,
        }

    def load_dict(self, payload: dict[str, Any]) -> None:
        """Restore state produced by ``to_dict`` (used for in-memory rollback)."""
        payload = json.loads(json.dumps(payload))  # deep copy, JSON-safe
        for name in ("equity", "state", "shadow_equity", "shadow_period_return",
                     "shadow_periods", "shadow_positive_periods", "peak_equity",
                     "day", "day_open_equity", "day_peak_equity", "blocked_symbols",
                     "bar_index", "shadow_entries", "shadow_exits",
                     "total_cost", "total_funding"):
            if name in payload:
                setattr(self, name, payload[name])
        self.positions = {k: Position(**v) for k, v in payload.get("positions", {}).items()}
        self.shadow_positions = {k: Position(**v) for k, v in payload.get("shadow_positions", {}).items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        self.load_dict(json.loads(path.read_text(encoding="utf-8")))
        return True

    # ------------------------------------------------------------ public API
    def _result(self, ts: str, *, net: float, cost_fraction: float,
                events: list[str], missing: list[str], funding: float = 0.0) -> dict[str, Any]:
        return {
            "timestamp": ts, "state": self.state, "equity": self.equity,
            "shadow_equity": self.shadow_equity,
            "net_return": net if self.state == "LIVE" else 0.0,
            "shadow_return": net if self.state == "SHADOW" else 0.0,
            "cost": cost_fraction, "funding": funding,
            "risk_events": ";".join(events),
            # Warnings never trigger an exchange sync in the runner.
            "warnings": ";".join(f"MISSING_PRICE:{s}" for s in missing),
            "positions": len(self.positions),
        }

    def mark(self, ts: str, bars: list[dict[str, Any]]) -> dict[str, Any]:
        """Mark risk between signal bars without advancing the rebalance clock."""
        market = {str(x["symbol"]).upper(): x for x in bars}
        start_equity = max(self.equity, 1e-12)
        pnl, missing = self._mark_positions(self._active(), market)
        net = self._book_dollars(pnl)
        events, risk_cost = self._risk(ts, market)
        cost_fraction = risk_cost / start_equity
        return self._result(ts, net=net - cost_fraction, cost_fraction=cost_fraction,
                            events=events, missing=missing)

    def step(self, ts: str, signals: list[dict[str, Any]], bars: list[dict[str, Any]],
             *, account_equity: float | None = None) -> dict[str, Any]:
        market = {str(x["symbol"]).upper(): x for x in bars}
        start_equity = max(self.equity, 1e-12)
        # First realise the move from the previous mark to this completed bar,
        # plus any funding settled inside it. Risk is evaluated after that.
        active_before = self._active()
        pnl, missing = self._mark_positions(active_before, market)
        funding = self._funding(active_before, market)
        if self.state == "LIVE":
            self.total_funding += funding
        net = self._book_dollars(pnl + funding)
        events, risk_cost = self._risk(ts, market)
        rebalance = self.bar_index % max(1, self.config.rebalance_bars) == 0
        active = self._active()
        # Shadow equity is normalised to 1.0 for recovery statistics; target
        # notionals stay on the real-account sizing basis so the shadow book uses
        # exactly the position sizes the live account would have used.
        sizing_equity = self.equity if account_equity is None else float(account_equity)
        if not math.isfinite(sizing_equity) or sizing_equity <= 0:
            raise ValueError("account_equity must be positive and finite")
        trade_cost = 0.0
        if rebalance:
            desired = self._targets(signals, market, sizing_equity)
            trade_cost = self._sync_positions(active, desired, market, ts)
            if self.state == "LIVE":
                self.total_cost += trade_cost
            net += self._book_dollars(-trade_cost)
        if self.state == "SHADOW":
            if (self.bar_index + 1) % max(1, self.config.rebalance_bars) == 0:
                self.shadow_periods += 1
                self.shadow_positive_periods += int(self.shadow_period_return > 1.0)
                self.shadow_period_return = 1.0
                if (self.shadow_periods >= self.config.shadow_min_periods
                        and self.shadow_positive_periods >= self.config.shadow_positive_periods
                        and self.shadow_equity - 1.0 >= self.config.shadow_recovery_pnl_pct
                        # A daily halt blocks the account for the rest of the UTC day.
                        and "__ACCOUNT__" not in self.blocked_symbols):
                    self.state = "LIVE"
                    self.shadow_exits += 1
                    self.shadow_positions.clear()
                    # Live equity was frozen while in SHADOW. Restart the
                    # drawdown baselines, otherwise the old high-water mark
                    # immediately sends the account back to SHADOW forever.
                    self.peak_equity = self.equity
                    self.day_peak_equity = self.equity
                    self.day_open_equity = self.equity
                    events.append("SHADOW_TO_LIVE")
        self.bar_index += 1
        cost_fraction = (risk_cost + trade_cost) / start_equity
        return self._result(ts, net=net - risk_cost / start_equity, cost_fraction=cost_fraction,
                            events=events, missing=missing,
                            funding=funding / start_equity)
