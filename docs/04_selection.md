# 04 Selection and deduplication

## Scanning Rust results

```bash
python scripts/04_selection/scan_rust_factors.py \
  --results-dir outputs/sweep_train_0_200 \
  --factor-jsonl outputs/crypto_factors.jsonl \
  --return-min 5 --sharpe-min 1 --max-drawdown-pct-min -30 \
  --sort-by sharpe \
  --out-csv outputs/selected_train.csv --out-markdown outputs/selected_train.md
```

The script scans every `*.metrics.json` under `--results-dir` (including all rebalancing-period subdirectories). It outputs one row per (factor, period) pair, together with the factor's economic hypothesis, formula, required fields and lookback windows. The Markdown report lists this information factor by factor.

| Filter arguments | Fields |
|---|---|
| `--return-min`, `--sharpe-min`, `--max-drawdown-pct-min`, `--pnl-positive` | Portfolio return, Sharpe, maximum drawdown (drawdown is negative; `-30` means a drawdown of no more than 30%) |
| `--abs-q10-min`, `--abs-q1-max`, `--abs-q10-directional-min`, `--abs-q1-directional-max`, `--abs-q10-cum-pct-min` | Absolute-value layer statistics |

Points to keep in mind when using these filters:

- **Absolute-value layer metrics are only indirectly related to portfolio return.** They layer by |prediction|, looking at the symbols with the strongest or weakest signal regardless of direction, while the portfolio goes long and short by the signed prediction. They should not be the main thresholds.
- **Layer metrics are identical across the six rebalancing directories.** Layer statistics are based on the 30m label and do not depend on rebalancing frequency; only portfolio return, Sharpe and drawdown change with rebalancing frequency.
- **Filter only on training-period results.** For the test period only produce reports: point `--results-dir` at the test-period directory and pass no filters.

## Correlation deduplication

```bash
python scripts/04_selection/analyze_factor_correlation.py \
  --predictions-dir outputs/pred_train_0_200 \
  --metrics-dir outputs/sweep_train_0_200/12h --score-field sharpe \
  --method spearman --threshold 0.90 --max-rows 200000 --min-overlap 1000 \
  --out-dir outputs/factor_corr_0_200
```

Aligns all predictions on `timestamp + symbol` and computes correlations. Factors whose absolute correlation exceeds the threshold are grouped into one cluster, and each cluster keeps the factor with the highest `--score-field`.

Outputs: `spearman.csv` (full matrix), `high_correlation_pairs.csv`, `clusters.csv`, `selected_factors.txt`, `summary.json`.

- Do not correlate equity curves: that mixes rebalancing frequency, leverage and costs into factor similarity.
- `--max-rows` uses the same deterministic sample of rows for every factor, controlling memory without breaking alignment between factors.
- Run a first pass at 0.90, then a stricter one at 0.95.

## Multiple testing

10,000 factors × 6 rebalancing periods filtered with fixed thresholds will inevitably produce many false positives. On 2 years of 30m data, the standard error of a Sharpe estimate is about 0.7, and a pure-noise factor has roughly an 8% chance of reaching Sharpe ≥ 1.

Recommended practice:
1. Select on the training period;
2. Require the results to reproduce on a non-overlapping validation period (e.g. 2025-10-01 to 2025-12-31);
3. Correct for the number of factors tested, e.g. with the Deflated Sharpe ratio;
4. Factors that are stable across several rebalancing periods are more credible than factors that are best at just one period (see `rank_across_periods.py` in 05).
