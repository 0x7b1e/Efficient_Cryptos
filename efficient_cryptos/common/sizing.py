"""Position sizing via volatility targeting and inverse-vol weighting.

The idea:
  1. Weight each symbol by 1/volatility so high-vol names get less capital.
  2. Normalize long and short sides separately to stay dollar neutral.
  3. Scale gross exposure so expected portfolio volatility matches the target.
  4. Respect the leverage cap at the end.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


def inverse_vol_weights(vols: Dict[str, float]) -> Dict[str, float]:
    """Inverse-volatility weights normalized to sum to 1."""
    inv = {s: (1.0 / v if v and v > 0 else 0.0) for s, v in vols.items()}
    total = sum(inv.values())
    if total <= 0:
        n = len(vols) or 1
        return {s: 1.0 / n for s in vols}
    return {s: w / total for s, w in inv.items()}


def _apply_symbol_cap(weights: Dict[str, float], cap: float) -> Dict[str, float]:
    """Cap each absolute weight at `cap` and redistribute within the same side.

    Long and short sides are processed separately to preserve dollar neutrality.
    If the constraint is infeasible, the side is clipped and the total exposure shrinks.
    """
    if not cap or cap <= 0:
        return weights
    out: Dict[str, float] = {}
    for positive in (True, False):
        side = {s: abs(w) for s, w in weights.items() if (w > 0) == positive and w != 0}
        if not side:
            continue
        target = sum(side.values())          # Total exposure to preserve on this side (e.g. 0.5)
        free = set(side)
        remaining = target
        capped: Dict[str, float] = {}
        while free and remaining > 1e-12:
            total_free = sum(side[s] for s in free)
            if total_free <= 0:
                break
            over = [s for s in free if remaining * side[s] / total_free > cap]
            if not over:                      # No remaining names exceed the cap; split the rest proportionally
                for s in free:
                    capped[s] = remaining * side[s] / total_free
                remaining = 0.0
                break
            for s in over:                    # Cap the oversize names and keep redistributing the rest
                capped[s] = cap
                free.discard(s)
                remaining -= cap
        for s in free:                        # If infeasible, the rest also stay capped
            capped[s] = cap
        sign = 1.0 if positive else -1.0
        for s, w in capped.items():
            out[s] = sign * w
    return out


def target_positions(
    longs: List[str],
    shorts: List[str],
    vols: Dict[str, float],
    *,
    equity: float,
    leverage: float,
    target_vol_annual: float,
    gross_cap: float,
    max_symbol_weight: float = 0.0,
) -> Dict[str, float]:
    """Return target notionals in USDT per symbol (positive = long, negative = short)."""
    weights: Dict[str, float] = {}
    if longs:
        lw = inverse_vol_weights({s: vols.get(s, 0.0) for s in longs})
        for s, w in lw.items():
            weights[s] = weights.get(s, 0.0) + 0.5 * w   # Long side gets half the exposure
    if shorts:
        sw = inverse_vol_weights({s: vols.get(s, 0.0) for s in shorts})
        for s, w in sw.items():
            weights[s] = weights.get(s, 0.0) - 0.5 * w   # Short side gets the other half
    if not weights:
        return {}

    # Apply symbol cap before volatility scaling to preserve relative structure
    weights = _apply_symbol_cap(weights, max_symbol_weight)

    # Estimate portfolio volatility from inverse-vol weights, ignoring correlation
    port_var = sum((weights[s] * vols.get(s, 0.0)) ** 2 for s in weights)
    port_vol = math.sqrt(port_var) if port_var > 0 else target_vol_annual
    vol_scaler = target_vol_annual / port_vol if port_vol > 0 else 1.0

    gross = min(leverage, gross_cap) * vol_scaler
    gross = min(gross, gross_cap)                       # Keep the scaled gross exposure within the cap
    return {s: w * gross * equity for s, w in weights.items()}
