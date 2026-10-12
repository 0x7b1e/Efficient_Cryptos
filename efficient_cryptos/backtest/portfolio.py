"""Cross-sectional portfolio backtest in Python (reference for the Rust core).

``lifecycle_pnl_costed`` runs the signal lifecycle with fees, impact slippage
and funding; ``factor_summary`` adds IC, decile spreads, Sharpe and drawdown.
The Rust evaluator in ``rust_backtest/`` implements the same portfolio rules
for large sweeps.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from efficient_cryptos.common.costs import CostModel
from efficient_cryptos.common.signal_policy import signal_is_due, signal_is_expired
from efficient_cryptos.backtest.metrics import summarize_metrics


def _weights_for_signal(part: pd.DataFrame, *, k_long: int, m_short: int,
                        long_percent: float | None, short_percent: float | None,
                        gross_leverage: float) -> dict[str, float]:
    ordered = part.dropna(subset=["prediction"]).sort_values("prediction", ascending=False)
    n = len(ordered)
    if n == 0:
        return {}
    def count(percent, fixed):
        if percent is None:
            return max(0, min(int(fixed), n))
        if not 0 <= float(percent) <= 100:
            raise ValueError("selection percentages must be between 0 and 100")
        return 0 if float(percent) == 0 else max(1, min(n, int(np.ceil(n * float(percent) / 100))))
    nl, ns = count(long_percent, k_long), count(short_percent, m_short)
    if nl + ns > n:
        raise ValueError(f"long/short selections overlap: {nl}+{ns}>{n}")
    longs = list(ordered.symbol.head(nl))
    shorts = [s for s in ordered.symbol.tail(ns) if s not in longs]
    weights: dict[str, float] = {}
    if longs:
        weights.update({s: gross_leverage / 2 / len(longs) for s in longs})
    if shorts:
        weights.update({s: -gross_leverage / 2 / len(shorts) for s in shorts})
    return weights


def lifecycle_pnl_costed(frame: pd.DataFrame, *, interval_bars: int = 1, hold_bars: int = 1,
                          k_long: int = 2, m_short: int = 1, long_percent: float | None = None,
                          short_percent: float | None = None, gross_leverage: float = 1.0,
                          initial_cash: float = 10000.0, cost_model: CostModel | None = None,
                          include_costs: bool = True, include_funding: bool = True) -> pd.DataFrame:
    """Run the signal lifecycle with fees, impact slippage and funding."""
    if gross_leverage <= 0:
        raise ValueError("gross leverage must be positive")
    cost_model = cost_model or CostModel()
    previous: dict[str, float] = {}
    # The first row of an explicitly bounded test window need not land on a
    # scheduled signal boundary (for example a 24h rebalance starting at
    # midnight UTC).  Keep an empty portfolio until the first due signal.
    current: dict[str, float] = {}
    signal_time = None
    equity = float(initial_cash)
    rows = []
    for ts, part in frame.groupby("timestamp", sort=True):
        starting_equity = equity
        ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
        if signal_is_due(ts_ms, 30, interval_bars):
            current = _weights_for_signal(part, k_long=k_long, m_short=m_short,
                                           long_percent=long_percent, short_percent=short_percent,
                                           gross_leverage=gross_leverage)
            signal_time = ts_ms
        elif signal_is_expired(ts_ms, signal_time, 30, hold_bars):
            current, signal_time = {}, None
        costs = 0.0
        turnover = 0.0
        if include_costs:
            qv = part.set_index("symbol").get("quote_volume", pd.Series(dtype=float))
            for symbol in set(previous) | set(current):
                delta = float(current.get(symbol, 0.0) - previous.get(symbol, 0.0))
                turnover += abs(delta)
                if not delta:
                    continue
                notional = abs(delta) * starting_equity
                volume = float(qv.get(symbol, 0.0) or 0.0)
                costs += cost_model.fee(notional)
                costs += notional * cost_model.slippage_bps(notional, volume) / 10_000
        # Portfolio return = sum of weight x *simple* asset return. Summing log
        # returns linearly understates longs and overstates shorts by ~sigma^2/2
        # per bar; ``raw_target`` (not demeaned) makes the return exact.
        indexed = part.set_index("symbol")
        log_returns = indexed["raw_target" if "raw_target" in indexed else "target"].astype(float)
        gross_return = sum(
            float(current.get(symbol, 0.0)) * float(np.expm1(value))
            for symbol, value in log_returns.items()
            if np.isfinite(value)
        )
        funding_cash = 0.0
        if include_funding and "funding_event_rate" in indexed:
            # Actual settlements inside this row's holding window.
            events = indexed["funding_event_rate"]
            funding_cash = sum(
                float(current.get(symbol, 0.0)) * starting_equity * float(rate)
                for symbol in current
                if np.isfinite(rate := events.get(symbol, 0.0))
            )
        elif include_funding and pd.Timestamp(ts).hour in (0, 8, 16) and pd.Timestamp(ts).minute == 0:
            funding = indexed.get("funding_rate", pd.Series(dtype=float))
            funding_cash = sum(
                float(current.get(symbol, 0.0)) * starting_equity * float(rate)
                for symbol in current
                if np.isfinite(rate := funding.get(symbol, 0.0))
            )
        cost_return = costs / max(starting_equity, 1e-12)
        funding_return = funding_cash / max(starting_equity, 1e-12)
        net_return = gross_return - cost_return - funding_return
        equity *= 1.0 + net_return
        rows.append({"timestamp": ts, "period_return": net_return, "gross_return": gross_return,
                     "cost": cost_return, "funding_cost": funding_return,
                     "turnover": turnover, "active_symbols": len(current), "equity": equity})
        previous = current
    return pd.DataFrame(rows)


def _split_frame(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if not start and not end:
        return frame
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= ts >= pd.Timestamp(start, tz="UTC")
    if end:
        mask &= ts < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return frame.loc[mask]


def factor_summary(predictions: pd.DataFrame, *, interval_bars: int, hold_bars: int,
                   k_long: int, m_short: int, long_percent: float | None,
                   short_percent: float | None, gross_leverage: float, initial_cash: float,
                   cost_model: CostModel, include_costs: bool, include_funding: bool,
                   start: str | None = None, end: str | None = None) -> dict[str, Any]:
    part = _split_frame(predictions, start, end)
    if part.empty:
        return {"sections": 0, "samples": 0}
    metrics = summarize_metrics(part)
    lifecycle = lifecycle_pnl_costed(
        part, interval_bars=interval_bars, hold_bars=hold_bars,
        k_long=k_long, m_short=m_short, long_percent=long_percent,
        short_percent=short_percent, gross_leverage=gross_leverage,
        initial_cash=initial_cash, cost_model=cost_model,
        include_costs=include_costs, include_funding=include_funding,
    )
    metrics.update({
        "factor_id": str(predictions.attrs.get("factor_id", "")),
        "final_equity": float(lifecycle.equity.iloc[-1]) if len(lifecycle) else None,
        "return_pct": float((lifecycle.equity.iloc[-1] / initial_cash - 1) * 100) if len(lifecycle) else None,
        "avg_turnover": float(lifecycle.turnover.mean()) if len(lifecycle) else None,
        "total_cost": float(lifecycle.cost.sum() + lifecycle.funding_cost.sum()) if len(lifecycle) else 0.0,
        "coverage": float(pd.to_numeric(part["prediction"], errors="coerce").notna().mean()),
    })
    profile = decile_profile(part, 10)
    if not profile.empty:
        decile_means = profile.groupby("decile")["mean_return"].mean()
        top = float(decile_means.get(10, np.nan))
        bottom = float(decile_means.get(1, np.nan))
        spread = top - bottom if np.isfinite(top) and np.isfinite(bottom) else np.nan
        spread_rows = profile[profile["decile"].isin([1, 10])].pivot(
            index="timestamp", columns="decile", values="mean_return"
        ).dropna()
        spread_series = (spread_rows[10] - spread_rows[1]) if not spread_rows.empty else pd.Series(dtype=float)
        metrics.update({
            "top_decile_return": top,
            "bottom_decile_return": bottom,
            "top_bottom_spread": spread,
            "top_bottom_ir": float(spread_series.mean() / spread_series.std(ddof=0)
                                    * np.sqrt(len(spread_series)))
                if len(spread_series) > 1 and spread_series.std(ddof=0) > 0 else np.nan,
        })
    period_returns = pd.to_numeric(lifecycle["period_return"], errors="coerce").dropna()
    if len(period_returns):
        curve = (1.0 + period_returns).cumprod()
        drawdown = curve / curve.cummax() - 1.0
        periods_per_year = 365.0 * 24.0 * 60.0 / 30.0
        mean_return = float(period_returns.mean())
        std_return = float(period_returns.std(ddof=0))
        metrics.update({
            "max_drawdown_pct": float(drawdown.min() * 100.0),
            "annualized_return_pct": float(((1.0 + mean_return) ** periods_per_year - 1.0) * 100.0)
                if mean_return > -1.0 else -100.0,
            "annualized_vol_pct": float(std_return * np.sqrt(periods_per_year) * 100.0),
            "sharpe": float(mean_return / std_return * np.sqrt(periods_per_year))
                if std_return > 0 else np.nan,
        })
    return metrics


def decile_profile(predictions: pd.DataFrame, deciles: int = 10) -> pd.DataFrame:
    """Return average next-bar return for prediction deciles.

    Decile 1 is the lowest predicted return and decile 10 is the highest.
    Ranking is recomputed independently at every timestamp.
    """
    rows = []
    for timestamp, part in predictions.groupby("timestamp", sort=True):
        valid = part[["prediction", "target"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 2:
            continue
        rank = valid["prediction"].rank(method="first").to_numpy()
        # Always reserve decile 1 and decile 10 for the cross-sectional
        # bottom/top, even when a smoke universe has fewer than ten assets.
        group = (1 + np.rint((rank - 1) / max(1, len(valid) - 1) * (deciles - 1)))
        group = pd.Series(group, index=valid.index).clip(1, deciles).astype(int)
        valid = valid.assign(decile=group.to_numpy())
        for decile, values in valid.groupby("decile"):
            rows.append({
                "timestamp": timestamp,
                "decile": int(decile),
                "mean_return": float(values["target"].mean()),
                "n": int(len(values)),
            })
    if not rows:
        return pd.DataFrame(columns=["timestamp", "decile", "mean_return", "n"])
    return pd.DataFrame(rows)
