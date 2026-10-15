# 05 Risk engine and shadow replay

`efficient_cryptos/engine/risk_engine.py` is used for both offline replay and live trading. The risk decisions in an offline replay are exactly the decisions live trading would make under the same market conditions.

## State machine

```text
            Per-symbol stop/take · intraday stop/take/drawdown · portfolio high-water-mark drawdown
   LIVE  ─────────────────────────────────────────────────────────────────►  SHADOW
    ▲     (close all real positions, charging closing costs)                     │
    │                                                                           │ Shadow book keeps simulating
    │   Recovery: ≥ shadow_min_periods periods,                                  │ trades from signals and costs
    └── of which ≥ shadow_positive_periods profitable, shadow return ≥ threshold ◄┘
        and no intraday circuit-breaker lock for the day
```

| Rule | Basis | Parameters (live / replay) |
|---|---|---|
| Per-symbol stop loss / take profit | Relative to entry price; same-direction rebalances keep the original entry price, additions are quantity-weighted (consistent with the exchange's `entryPrice`) | `--symbol-stop 0.05` / `--symbol-take 0.30` |
| Intraday stop loss / take profit | Relative to equity at the UTC day open | `--daily-stop 0.05` / `--daily-take 0.30` |
| Intraday drawdown, portfolio high-water-mark drawdown | Relative to the day's highest equity / all-time highest equity | `--portfolio-drawdown 0.10` |
| Recovery | See the diagram above | `--shadow-min-periods 3`, `--shadow-positive-periods 2`, `--shadow-recovery-pnl-pct 0.01` |

Note:

- **If any symbol hits its stop loss or take profit, the whole portfolio enters SHADOW**, rather than closing just that symbol.
- **Returning to LIVE resets the drawdown baselines**, including the high-water mark, the day-open equity and the day's highest equity; otherwise the equity frozen during SHADOW would retrigger the drawdown right after recovery.
- **No recovery while the day is locked**; recovery is allowed again the next (UTC) day.
- **The shadow book never produces real orders**; only `positions` in the LIVE state become exchange targets.

Accounting: positions are held by quantity and notional value is revalued with price; returns are simple returns; risk-triggered closes are also charged fees and slippage; funding is settled when a bar carries `funding_rate`. Symbols with missing prices are not skipped; `MISSING_PRICE:SYMBOL` is written to the `warnings` field.

## Batch replay of predictions

```bash
python scripts/05_shadow/run_selected_factor_shadow.py \
  --predictions-dir outputs/pred_test_selected \
  --data-dir data \
  --out-dir outputs/shadow_test \
  --rebalance-hours 1,2,4,8,12,24 \
  --initial-equity 10000 --gross-leverage 3 --margin-buffer-pct 0.20 \
  --long-pct 10 --short-pct 10 \
  --symbol-stop 0.10 --symbol-take 0.30 --daily-stop 0.05 --daily-take 0.30 --portfolio-drawdown 0.10 \
  --shadow-min-periods 3 --shadow-positive-periods 2 --shadow-recovery-pnl-pct 0.01 \
  --fee-bps 5 --slippage-bps 1 --mark-minutes 1
```

The timeline matches live trading:

```text
Signal bar T (open time)
  → T+30m: Engine.step — settle PnL and intra-bar funding at the 30m close, then risk checks and rebalancing
  → until the close of the next signal bar: every --mark-minutes minutes, call Engine.mark with
     (O+H+L+C)/4 of the completed 1m kline (the same price live risk control uses)
```

- A 30m bar with fewer than 30 1m klines is treated as incomplete and is not traded.
- `--mark-minutes 0` disables intraday marking; results will be optimistic and are only for comparison.
- Funding is read from `data/funding/<SYMBOL>.parquet`; `--no-funding` disables it.
- The rebalancing period must be a whole multiple of 0.5 hours.

Each period directory contains `<factor>.timeseries.csv` (every step, plus marks that triggered a risk event), `<factor>.metrics.json` (return, drawdown, Sharpe, costs, funding, shadow statistics) and `summary.csv` (sorted by Sharpe by default).

## Cross-period ranking and charts

```bash
python scripts/05_shadow/rank_across_periods.py --results-dir outputs/shadow_test --top 30 --out-csv outputs/shadow_rank.csv
python scripts/05_shadow/scan_shadow_engine.py --results-dir outputs/shadow_test --out-dir outputs/shadow_test_scan
```

`rank_across_periods.py` sorts by "number of profitable periods → number of periods staying LIVE → median Sharpe → mean return", and lists the best single-run combinations. Factors that stand out at only one period are usually less credible than factors that are stable across periods.

## Paper replay (bring your own signals and bars)

```bash
python scripts/05_shadow/paper_replay.py --signals signals.csv --bars bars.csv --output outputs/paper \
  --gross-leverage 3 --rebalance-bars 24 --fee-bps 5 --slippage-bps 1
```

- `signals.csv`: `timestamp,symbol,signal,target_notional`
- `bars.csv`: `timestamp,symbol,close,quote_volume[,funding_rate]`

`step` is called only at timestamps that have both signals and bars, and `--rebalance-bars` counts signal bars. Outputs `timeseries.csv`, `orders.csv` and `metrics.json`.

## Limitations of replay results

- Replay fills instantly at the bar close or the 1m average price; live trading waits for the next poll and fills with market orders.
- If predictions come from mainnet data while live trading runs on Testnet, factor inputs (especially volume) will differ noticeably.
- Historical replay results are in `docs/results/`.

## Dashboard

The replay's `timeseries.csv` also outputs a few display-only columns: `total_cost` and `total_funding` (the engine's exact cumulative values), long/short notionals and SHADOW period counts; `metrics.json` records the replay's parameters in `config`. None of these affect trading logic. The dashboard's "Shadow replay" page shows SHADOW switches, costs and margin usage; see [12 Dashboard](12_dashboard.md).
