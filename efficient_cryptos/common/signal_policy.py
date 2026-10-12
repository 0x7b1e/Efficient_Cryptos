"""Shared signal scheduling and expiry rules for research and execution."""
from __future__ import annotations


def signal_is_due(signal_time_ms: int, bar_minutes: int, interval_bars: int) -> bool:
    """Return whether a closed bar is an allowed signal-generation bar."""
    bar_ms = max(1, int(bar_minutes)) * 60_000
    return (int(signal_time_ms) // bar_ms) % max(1, int(interval_bars)) == 0


def signal_is_expired(current_signal_ms: int, signal_time_ms: int,
                      bar_minutes: int, hold_bars: int) -> bool:
    """Return whether a signal has reached its maximum holding lifetime."""
    if signal_time_ms is None:
        return False
    bar_ms = max(1, int(bar_minutes)) * 60_000
    return int(current_signal_ms) >= int(signal_time_ms) + max(1, int(hold_bars)) * bar_ms


def policy_values(cfg, *, section: str | None = None,
                  interval_default: int = 1,
                  hold_default: int | None = None) -> tuple[int, int]:
    """Read a strategy override, then the shared signal_policy config."""
    shared = cfg.get("signal_policy", {}) if isinstance(cfg, dict) else {}
    local = cfg.get(section, {}) if section and isinstance(cfg, dict) else {}
    interval = local.get("signal_interval_bars", local.get("rebalance_bars", None))
    if interval is None:
        interval = shared.get("interval_bars", interval_default)
    hold = local.get("signal_hold_bars", local.get("hold_bars", None))
    if hold is None:
        hold = shared.get("hold_bars", hold_default if hold_default is not None else interval)
    return max(1, int(interval)), max(1, int(hold))
