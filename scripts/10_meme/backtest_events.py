#!/usr/bin/env python3
"""Backtest event entries with explicit stop/take/decay exits and MFE/MAE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def run(frame: pd.DataFrame, *, hold_minutes: int, stop_loss: float, take_profit: float,
        decay_volume_shock: float, fee_bps: float, slippage_bps: float) -> pd.DataFrame:
    frame = frame.sort_values(["symbol", "timestamp"]).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    trades = []
    for symbol, part in frame.groupby("symbol", sort=False):
        part = part.reset_index(drop=True)
        times = part["timestamp"].to_numpy()
        for i, row in part.iterrows():
            if not bool(row.get("event_active", False)) or int(row.get("event_side", 0)) == 0:
                continue
            side = int(row["event_side"])
            entry = float(row["close"])
            if not np.isfinite(entry) or entry <= 0:
                continue
            entry_time = pd.Timestamp(row["timestamp"])
            exit_idx = min(len(part) - 1, i + max(1, hold_minutes))
            reason = "time"
            mfe = -np.inf
            mae = np.inf
            for j in range(i + 1, exit_idx + 1):
                price = float(part.at[j, "close"])
                ret = side * (price / entry - 1.0)
                mfe = max(mfe, ret)
                mae = min(mae, ret)
                if ret <= -stop_loss:
                    exit_idx, reason = j, "stop_loss"
                    break
                if ret >= take_profit:
                    exit_idx, reason = j, "take_profit"
                    break
                if float(part.at[j, "volume_shock_5m"]) < decay_volume_shock:
                    exit_idx, reason = j, "liquidity_decay"
                    break
                if side * float(part.at[j, "ret_3m"]) <= 0:
                    exit_idx, reason = j, "momentum_decay"
                    break
            exit_price = float(part.at[exit_idx, "close"])
            gross = side * (exit_price / entry - 1.0)
            # Entry and exit are both charged. This is a conservative fixed
            # cost; the Rust engine can later replace it with participation slippage.
            cost = 2.0 * (fee_bps + slippage_bps) / 10_000.0
            net = gross - cost
            trades.append({
                "symbol": symbol, "entry_time": entry_time,
                "exit_time": pd.Timestamp(part.at[exit_idx, "timestamp"]),
                "side": side, "entry_price": entry, "exit_price": exit_price,
                "gross_return": gross, "cost_return": cost, "net_return": net,
                "mfe": 0.0 if mfe == -np.inf else mfe,
                "mae": 0.0 if mae == np.inf else mae,
                "hold_minutes": int((times[exit_idx] - times[i]) / np.timedelta64(1, "m")),
                "exit_reason": reason,
            })
            # One active position per symbol. The next event can only enter
            # after this trade's exit, preventing overlapping double counting.
            part.loc[i + 1:exit_idx, "event_active"] = False
    return pd.DataFrame(trades)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hold-minutes", type=int, default=30)
    ap.add_argument("--stop-loss", type=float, default=0.05)
    ap.add_argument("--take-profit", type=float, default=0.30)
    ap.add_argument("--decay-volume-shock", type=float, default=-0.20)
    ap.add_argument("--fee-bps", type=float, default=4.0)
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    args = ap.parse_args()
    frame = pd.read_parquet(args.features)
    trades = run(frame, hold_minutes=args.hold_minutes, stop_loss=args.stop_loss,
                 take_profit=args.take_profit, decay_volume_shock=args.decay_volume_shock,
                 fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    if trades.empty:
        summary = {"trades": 0}
    else:
        equity = (1.0 + trades["net_return"]).cumprod()
        summary = {
            "trades": int(len(trades)),
            "win_rate": float((trades["net_return"] > 0).mean()),
            "mean_net_return": float(trades["net_return"].mean()),
            "total_return_pct": float((equity.iloc[-1] - 1.0) * 100),
            "mean_mfe_pct": float(trades["mfe"].mean() * 100),
            "mean_mae_pct": float(trades["mae"].mean() * 100),
            "max_mfe_pct": float(trades["mfe"].max() * 100),
            "max_mae_pct": float(trades["mae"].min() * 100),
            "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
