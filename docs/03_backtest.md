# 03 Backtest

```text
factors.jsonl ──► export_factor_predictions.py ──► <factor>.csv (one row = one timestamp × one symbol)
                                                      │
                                     sweep_rust_rebalance.sh (1h 2h 4h 8h 12h 24h)
                                                      ▼
                              <period>/<factor>.metrics.json / layers.csv / timeseries.csv / plots/
```

Python executes the factor code; Rust handles layering, positions, costs, funding and NAV.

## Labels

| `--label-mode` | Definition | Use |
|---|---|---|
| `horizon` (default, `--horizon 1`) | Log return from this bar's close to the next bar's close | Ordinary 30m cross-sectional prediction, portfolio backtests |
| `utc_close` | From this bar's close to the close of the day's last UTC bar | Rest-of-day return |
| `funding_event` | From this bar's close to the symbol's next actual `funding_time` | Funding settlement window |

All labels are converted to market-neutral excess returns: `target = raw_target − cross-sectional mean of raw_target at that time`.

> **Portfolio backtests only support single-bar labels.** Rust books only one bar of return per row. `utc_close`, `funding_event` and `horizon > 1` labels span several bars, so summing them row by row double-counts, and portfolio return and Sharpe come out too high (a warning is printed on export). For such labels look at the layer statistics, not the portfolio return.

## Accounting

| Item | Rule |
|---|---|
| Portfolio return | `Σ wᵢ·(exp(raw_targetᵢ) − 1)`, weighting simple returns; error vs an exact ledger < 1e-12 |
| Fees | 5 bps per unit of turnover (taker, `--taker-fee-bps`) |
| Slippage | `min(80, 1 + 60 × traded notional / bar quote volume)` bps |
| Funding | Charged at actual `funding_time`: settlements within this row's holding period `(t+30m, t+60m]` are paid by this row's position; 1h / 4h / 8h settlement intervals are supported |
| Rebalancing | Select Top / Bottom every `--signal-interval-bars` bars, hold for at most `--signal-hold-bars` bars |

The old version computed portfolio return as `exp(Σ wᵢ·logrᵢ) − 1`. On zero-drift synthetic data, that overstated a 3x pure-random factor by about 131% per year, and "always short the most volatile symbol" by about 312%. `tests/test_backtest_accounting.py` uses the same data to verify that the new formula is unbiased.

## Exporting predictions

```bash
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,ZECUSDT,MOVRUSDT,NEARUSDT,QNTUSDT,XRPUSDT,DOGEUSDT,SUIUSDT,UNIUSDT,WLDUSDT,1000PEPEUSDT,BNBUSDT,LINKUSDT,ADAUSDT

# Training period
python -u scripts/03_backtest/export_factor_predictions.py \
  --factor-jsonl outputs/crypto_factors.jsonl --data-dir data --symbols $SYMBOLS \
  --interval 30m --horizon 1 --days 1100 \
  --train-start 2024-01-01 --train-end 2025-12-31 \
  --start 0 --limit 200 --jobs 2 --backend thread --resume \
  --out-dir outputs/pred_train_0_200

# Test period: change only the dates and the output directory
  --train-start 2026-01-01 --train-end 2026-10-01 --out-dir outputs/pred_test_0_200
```

- `--start` / `--limit`: export in batches. For example, `--start 200 --limit 200` exports records 201–400.
- `--resume`: skip CSVs that already exist and are non-empty.
- `--backend thread`: threads share one feature table, using far less memory. `process` is faster but needs more memory.
- `--train-start` / `--train-end`: only decide which date range is exported. Features (including rolling windows) are computed on the full `--days` history.
- Failed factors (including those rejected by the look-ahead checker) are recorded in `failures.json`.

CSV format:

```text
factor_id,timestamp,symbol,prediction,target,raw_target,quote_volume,funding_rate,funding_event_rate
```

`funding_event_rate` is used only for accounting; factor code cannot read this column. Old CSVs without `raw_target` / `funding_event_rate` can still be fed to Rust, but a warning is printed and an approximate basis is used; the `portfolio_return_basis` and `funding_basis` fields in `metrics.json` state which basis was actually used.

## Rust batch backtests

```bash
cargo build --release --manifest-path rust_backtest/Cargo.toml

bash scripts/03_backtest/sweep_rust_rebalance.sh \
  --predictions-dir outputs/pred_train_0_200 \
  --output-root outputs/sweep_train_0_200 \
  --jobs 3 --long-percent 10 --short-percent 10 --gross-leverage 1 \
  --plot-max-points 5000          # or --no-plots
```

The script runs every factor at six rebalancing periods: 1h, 2h, 4h, 8h, 12h and 24h (i.e. 2, 4, 8, 16, 24 and 48 30m bars). Factors with complete results are skipped; `--force` reruns them. `--gross-leverage` scales positions, returns, costs and funding together, but does not change the factor ranking.

A single factor:

```bash
./rust_backtest/target/release/crypto-factor-backtest \
  --input outputs/pred_train_0_200/<factor>.csv \
  --output out/<factor>.metrics.json --layers-output out/<factor>.layers.csv \
  --timeseries-output out/<factor>.timeseries.csv \
  --long-percent 10 --short-percent 10 --signal-interval-bars 8 --signal-hold-bars 8 \
  --gross-leverage 1 --include-costs --include-funding
```

Outputs:

| File | Content |
|---|---|
| `metrics.json` | `return_pct`, `sharpe`, `max_drawdown_pct`, `avg_turnover`, `total_cost`, `signed_q10_minus_q1`, `abs_q10_minus_q1`, `abs_q10_directional_return`, basis fields |
| `layers.csv` | For each timestamp: signed-rank deciles Q1–Q10, absolute-value-rank deciles, and direction-adjusted returns of the absolute-value deciles |
| `timeseries.csv` | Per-bar gross return, net return, PnL, NAV, drawdown, turnover, cost and funding |

Signed Q10 is the layer with the highest predictions. Absolute-value Q10 is the layer with the strongest signal regardless of direction, and must not be mixed up with the signed layers. The portfolio always goes long and short by the signed prediction.

Plotting:

```bash
python scripts/03_backtest/plot_rust_backtest.py --kind equity \
  --metrics out/<f>.metrics.json --layers out/<f>.layers.csv --timeseries out/<f>.timeseries.csv --out out/<f>.equity.png
python scripts/03_backtest/plot_rust_backtest.py --kind layers ...   # cumulative curves of the ten layers
```

## Python backtest (IC, deciles, lifecycle PnL)

```bash
python scripts/03_backtest/evaluate_generated_factors.py \
  --factor-jsonl outputs/crypto_factors.jsonl --data-dir data --symbols $SYMBOLS \
  --days 180 --train-start 2025-01-01 --train-end 2025-06-30 \
  --validation-start 2025-07-01 --validation-end 2025-09-30 \
  --long-percent 20 --short-percent 20 --signal-interval-bars 2 --signal-hold-bars 6 \
  --gross-leverage 1 --include-costs --jobs 4 --out outputs/crypto_factor_eval
```

Outputs `factor_metrics.csv` (training- and validation-period IC, RankIC, ICIR, coverage, turnover, return, cost), `selected.jsonl`, `summary.json`, `failures.json`, and `factors/<id>/report.png` for each factor (decile distribution, D10−D1, NAV, drawdown, turnover, cost).
Selection: first filter by IC / ICIR / coverage thresholds, then greedily dedupe by cross-sectional rank correlation (`--corr-threshold`, default 0.80). `--keep-all` only produces reports and does no selection.

## Re-backtesting selected factors

```bash
python scripts/03_backtest/backtest_selected_factors.py \
  --selection-csv outputs/selected_train.csv --predictions-dir outputs/pred_test_0_200 \
  --output-dir outputs/selected_test --jobs 2
```

For each (factor, rebalancing period) pair in the selection table, reruns the Rust backtest on the given prediction directory and plots it. This is typically used to validate training-period selections on the test period.
