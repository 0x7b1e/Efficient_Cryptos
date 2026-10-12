"""Cost model: fees, impact slippage, and funding.

Slippage is modeled as market impact instead of a fixed bps constant:
    slippage_bps = base + coef * participation_rate
where `participation_rate = order_notional / minute_quote_volume`.
The more market you consume, the more slippage you pay.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    taker_fee_bps: float = 5.0
    slippage_base_bps: float = 1.0
    slippage_impact_coef: float = 60.0
    max_slippage_bps: float = 80.0

    def slippage_bps(self, order_notional: float, bar_quote_volume: float) -> float:
        """Estimate one-way slippage in bps from order size and minute volume."""
        if bar_quote_volume <= 0:
            return self.max_slippage_bps
        participation = order_notional / bar_quote_volume
        bps = self.slippage_base_bps + self.slippage_impact_coef * participation
        return min(bps, self.max_slippage_bps)

    def fill_price(self, mid: float, side: str, order_notional: float,
                   bar_quote_volume: float) -> float:
        """Fill price = reference price adjusted for slippage."""
        slip = self.slippage_bps(order_notional, bar_quote_volume) / 10_000.0
        if side.upper() == "BUY":
            return mid * (1.0 + slip)
        return mid * (1.0 - slip)

    def fee(self, traded_notional: float) -> float:
        """One-way taker fee in USDT."""
        return abs(traded_notional) * self.taker_fee_bps / 10_000.0


def funding_cost(position_notional: float, side: str, funding_rate: float) -> float:
    """Cash impact of one funding event in USDT.

    Convention: when `funding_rate > 0`, longs pay shorts.
    """
    sign = -1.0 if side.upper() == "LONG" else 1.0
    return sign * abs(position_notional) * funding_rate
