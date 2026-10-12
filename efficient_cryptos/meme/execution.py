"""Execution-neutral event strategy adapter.

It produces target notionals and risk actions but deliberately does not sign
orders.  A Binance/HYPERLIQUID adapter can consume the returned actions after
paper results pass review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .cooldown import CooldownBook


@dataclass(frozen=True)
class EventTarget:
    symbol: str
    side: int
    notional: float
    timestamp_ms: int
    reason: str = "event"


class MemeEventExecutor:
    def __init__(self, cooldown: CooldownBook, *, gross_leverage: float = 1.0,
                 max_symbol_weight: float = 0.10,
                 order_sink: Callable[[EventTarget], None] | None = None):
        self.cooldown = cooldown
        self.gross_leverage = float(gross_leverage)
        self.max_symbol_weight = float(max_symbol_weight)
        self.order_sink = order_sink

    def targets_from_events(self, events, *, equity: float, now_ms: int) -> list[EventTarget]:
        """Convert active event rows into equal-weight targets.

        Rows must contain symbol, event_active and event_side.  Cooldown is
        checked before allocation; execution remains paper/adaptor controlled.
        """
        active = [row for row in events if bool(row.get("event_active", False))
                  and int(row.get("event_side", 0)) in (-1, 1)
                  and not self.cooldown.is_blocked(str(row["symbol"]), now_ms)]
        if not active or equity <= 0:
            return []
        active = active[: max(1, int(1.0 / self.max_symbol_weight))]
        weight = min(self.max_symbol_weight, self.gross_leverage / len(active))
        targets = [EventTarget(str(row["symbol"]).upper(), int(row["event_side"]),
                               equity * weight * int(row["event_side"]), now_ms)
                   for row in active]
        if self.order_sink:
            for target in targets:
                self.order_sink(target)
        return targets
