"""Symbol-level and market-level cooldown state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CooldownReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MOMENTUM_DECAY = "momentum_decay"
    LIQUIDITY_DECAY = "liquidity_decay"
    MANUAL = "manual"


@dataclass(frozen=True)
class CooldownConfig:
    minutes: int = 15
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.30
    min_event_percentile: float = 0.95
    min_volume_shock: float = 0.20


@dataclass
class CooldownBook:
    config: CooldownConfig = field(default_factory=CooldownConfig)
    until_ms: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    market_halted: bool = False

    def is_blocked(self, symbol: str, now_ms: int) -> bool:
        return self.market_halted or int(self.until_ms.get(symbol.upper(), 0)) > int(now_ms)

    def trigger(self, symbol: str, now_ms: int, reason: CooldownReason | str) -> None:
        symbol = symbol.upper()
        self.until_ms[symbol] = int(now_ms) + self.config.minutes * 60_000
        self.reasons[symbol] = reason.value if isinstance(reason, CooldownReason) else str(reason)

    def clear_expired(self, now_ms: int) -> None:
        now_ms = int(now_ms)
        for symbol in list(self.until_ms):
            if self.until_ms[symbol] <= now_ms:
                self.until_ms.pop(symbol, None)
                self.reasons.pop(symbol, None)

    def evaluate_exit(self, *, symbol: str, entry_price: float, current_price: float,
                      side: int, now_ms: int, event_percentile: float | None = None,
                      volume_shock: float | None = None, momentum: float | None = None) -> str | None:
        if entry_price <= 0 or current_price <= 0 or side not in (-1, 1):
            return None
        pnl = side * (current_price / entry_price - 1.0)
        reason: str | None = None
        if pnl <= -self.config.stop_loss_pct:
            reason = CooldownReason.STOP_LOSS.value
        elif pnl >= self.config.take_profit_pct:
            reason = CooldownReason.TAKE_PROFIT.value
        elif momentum is not None and side * momentum <= 0:
            reason = CooldownReason.MOMENTUM_DECAY.value
        elif volume_shock is not None and volume_shock < self.config.min_volume_shock:
            reason = CooldownReason.LIQUIDITY_DECAY.value
        if reason:
            self.trigger(symbol, now_ms, reason)
        return reason

    def to_dict(self) -> dict[str, Any]:
        return {"market_halted": self.market_halted, "until_ms": self.until_ms, "reasons": self.reasons}
