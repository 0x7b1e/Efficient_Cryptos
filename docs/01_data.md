# 01 Data

All research and backtesting read only local parquet files. The live runner pulls completed klines directly from Binance and does not depend on local data.

## Storage layout

```text
data/
├── universe/top40.json          Liquidity universe snapshot
├── klines/<SYMBOL>.parquet      1m klines (deduplicated on open_time), with long/short ratio and other positioning columns
├── funding/<SYMBOL>.parquet     One row per funding settlement (funding_time, funding_rate)
└── features/*.parquet           Feature/label cache (see 03, 07)
```

Kline columns: `open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote`.
Positioning data (top-trader / global long-short ratios, taker buy/sell ratio) is stored as extra columns in the same kline file. Note that Binance only provides a short history of positioning data.

## Download

```bash
# First download: build a Top-N universe by 24h quote volume, 90-day lookback
python scripts/01_data/download_data.py --days 90 --top-n 40

# Specific symbols
python scripts/01_data/download_data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 90

# A UTC date range (both ends inclusive); cannot be combined with --incremental
python scripts/01_data/download_data.py --symbols BTCUSDT,ETHUSDT \
  --start-date 2025-01-01 --end-date 2025-06-30

# Klines and funding only, no positioning data
python scripts/01_data/download_data.py --symbols BTCUSDT --days 30 --skip-positioning

# Incrementally update the saved universe; --refresh-universe re-selects symbols
python scripts/01_data/download_data.py --incremental
python scripts/01_data/download_data.py --incremental --refresh-universe
```

On restricted networks add `--proxy http://127.0.0.1:1087` (`socks5h://` is also supported).
`--timeout` and `--retries` control the per-request timeout and number of retries.
Future or not-yet-closed klines are refused and never written.

Universe parameters (`universe.top_n`, `min_quote_volume_usdt`, exclusion list) are configured in `config/default.yaml`.

## Low local disk space: download per symbol and upload to a server

Runs "download → rsync upload → remote verification → (optionally) delete local file" one symbol at a time.
Progress is recorded in `data/download_queue.json`, so completed symbols are skipped after a restart; local files that already cover the target range are uploaded without being re-downloaded.

```bash
python scripts/01_data/stream_download_upload.py \
  --universe data/universe/top40.json \
  --start-date 2024-01-01 --end-date 2026-10-01 \
  --proxy http://127.0.0.1:1087 \
  --remote root@remote-host:/srv/data --remote-port 22 --control-persist 8h \
  --workers 2 --download-attempts 3 --download-backoff 20 \
  --delete-local --retry-failed
```

- An SSH key (`ssh-copy-id`) is recommended. With connection multiplexing, password login usually needs the password only once per run, and the password is never written to the command line or the queue file.
- Start `--workers` at 2 to avoid hitting Binance or proxy rate limits.
- Failed symbols are marked `failed` in the queue; once the network is fixed, retry them with `--retry-failed`.

## Filling missing dates on the server

When the parquet files on the server stop at some date, only the missing range is downloaded to a local temp directory, uploaded, deduplicated and merged on the server by `open_time` / `funding_time`, and the original file is replaced atomically:

```bash
python scripts/01_data/update_remote_data.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start-date 2026-10-02 --end-date 2026-10-10 \
  --proxy http://127.0.0.1:1087 \
  --remote root@host:/root/autodl-tmp/Efficient_Cryptos/data --remote-port 11402 \
  --remote-python /root/autodl-tmp/Efficient_Cryptos/.venv/bin/python
```

The old file is not overwritten until the merge succeeds. `--keep-local` keeps the local temporary downloads for troubleshooting.

## Checking data coverage

```bash
python scripts/01_data/check_parquet_times.py --data-root data
python scripts/01_data/check_parquet_times.py --data-root data --symbol BTCUSDT --output-csv outputs/coverage.csv
```

The script prints the first and last timestamp and row count of each file, plus the number of gaps longer than the expected interval (1 minute for klines, 8 hours for funding).

## 30m small sample

```bash
python scripts/01_data/download_30m_sample.py --help
```

This script downloads 30m klines directly for quick cross-sectional experiments, without resampling from 1m.

## Correlation analysis: choosing a trading pool

`kline_correlation.py` resamples local klines into complete bars (a bar with missing minutes is treated as missing) and computes two correlation matrices of log returns:

- **raw**: correlation of raw returns, including the common movement of the whole market.
- **resid**: correlation after removing the market. By default `--residual market` subtracts the equal-weighted mean return of all symbols from each bar, the same demeaning used for the backtest `target`; `--residual btc` instead regresses out the beta to `--reference`. Cross-sectional long-short strategies should care about resid: two symbols with high resid correlation are effectively the same bet.

```bash
# All local symbols, 30m returns, since 2026
python scripts/01_data/kline_correlation.py --data-dir data --interval 30m --start 2026-01-01 --out-dir outputs/kline_corr

# Only a given universe, always keeping BTC and ETH
python scripts/01_data/kline_correlation.py --symbols data/universe/top40.json --must-include BTCUSDT,ETHUSDT

# US stock perps: weekday US trading hours only (UTC 13–20)
python scripts/01_data/kline_correlation.py --symbols TSLAUSDT,NVDAUSDT,AAPLUSDT,METAUSDT,GOOGLUSDT \
  --weekdays-only --utc-hours 13-20 --interval 1h --out-dir outputs/kline_corr_stocks

# Mix several data sources (same-named files show up as data_hl/TSLAUSDT) to check whether one underlying is duplicated across two exchanges
python scripts/01_data/kline_correlation.py --data-dir data --data-dir data_hl --symbols TSLAUSDT,NVDAUSDT
```

Outputs (`--out-dir`):

| File | Content |
|---|---|
| `heatmap_raw.png` / `heatmap_resid.png` | Heatmaps ordered by hierarchical clustering, so similar symbols sit together |
| `corr_raw.csv` / `corr_resid.csv` / `overlap_bars.csv` | Correlation matrices and the number of bars each pair has in common |
| `high_corr_pairs.csv` / `high_corr_pairs_resid.csv` | Pairs with `|corr| ≥ --pair-threshold` |
| `symbol_stats.csv` | Coverage, annualized volatility, median daily quote volume, mean/max correlation and the matching symbol, correlation and beta to BTC, cluster id, volatility of the rolling mean correlation |
| `rolling_avg_corr.csv` | Every `--rolling-days` days, each symbol's mean correlation with the others (to check whether correlations are stable) |
| `pool_decisions.csv` | The greedy selection's decision for each symbol: `selected`, `correlated` (and with which selected symbol), `low_coverage`, `low_volume`, `pool_full` |
| `suggested_pool.json` | Suggested pool, `{"symbols": [...]}`, same format as `data/universe/top40.json` |

Greedy selection: walk symbols from highest to lowest by `--rank-by` (daily quote volume by default) and keep a symbol only if its `|corr|` with every already-selected symbol is below `--max-corr` (default 0.6, based on `--select-on resid`); symbols with coverage below `--min-coverage` or volume below `--min-daily-quote-volume` are skipped outright.

When interpreting the results:
- **resid has a negative bias**: after demeaning N symbols, even independent symbols show a negative correlation of about −1/(N−1). With few symbols the bias is noticeable.
- **Do not put US stocks and crypto in the same cross-section**: raw correlation between the two groups is low, so after demeaning they look strongly negatively correlated with each other (two blocks show up on the resid heatmap). Run them separately and build a pool for each.
- Weekend and after-hours prices for US stock perps are thin; use `--weekdays-only --utc-hours` to look only at regular trading hours. Recently listed symbols have low coverage; use `--start` to restrict the window to the period they all exist.
- `median_daily_quote_volume` is "median quote volume per bar × bars per day within the analysis window". With a session filter, it represents the volume within that session.
- Pearson is easily dominated by extreme moves; compare with `--method spearman`.
