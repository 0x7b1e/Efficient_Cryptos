# Efficient_Cryptos

A factor research and trading system for Binance USDⓈ-M perpetual futures: downloading data, mining factors with LLMs, backtesting and selection, through to risk-managed Testnet / mainnet live trading — all in one repository and one Python package.

This repository consolidates `Predict_Cryptos` (research) and `Developer_Crypto` (risk management and live trading). The backtesting, trading and factor-mining algorithms are identical to the originals: byte-for-byte comparison tests guarantee the live runner behaves exactly like the original version. See [docs/MIGRATION.md](docs/MIGRATION.md) for a mapping between old and new paths.

> This project is for research and engineering validation only; profits are not guaranteed. Testnet cannot reproduce real slippage, funding fees or fills. Review the code before going to mainnet, and start with a very small amount of capital.

## Pipeline

```text
 01 Data          Binance 1m klines / funding rates / long-short ratio  ──►  data/klines, data/funding
     │
 02 Mining        DeepSeek / OpenAI / Grok generate compute_factor(df) ──► factors.jsonl
     │            Look-ahead audit (static checks + sandboxed execution)
     │
 03 Backtest      Export prediction CSVs ──► Rust backtest (1h…24h rebalancing, costs, funding, decile layers)
     │
 04 Selection     Filter on the training period by Sharpe / drawdown / layering ──► cross-sectional correlation dedup ──► selected.jsonl
     │
 05 Shadow replay Same risk engine + 1m TWAP intraday risk control, replayed minute by minute over the test period
     │
 06 Live          dry-run ──► Testnet ──► mainnet (risk state machine, reconnects, exchange reconciliation)

 Side tracks: 07 ML models · 08 Alpha101/191 · 09 KOL social factors · 10 Meme events · 11 DEX
```

## Layout

```text
efficient_cryptos/            Python package (dependencies flow top to bottom)
├── common/                   Config, local parquet storage, Binance REST client, cost model, rebalancing rules
├── data/                     Liquidity-based universe selection
├── features/                 Look-ahead-free features and labels, layered cache (no torch dependency)
├── mining/                   Factor generators, expression checks, factor registry; prompts/{crypto,dex,meme}
├── factors/                  Factor code sandbox (validation + execution), Alpha101/191 formulas
├── backtest/                 Python portfolio backtest, IC metrics, funding settlement mapping, report plots
├── engine/                   LIVE/SHADOW risk state machine (shared by replay and live)
├── execution/                Binance adapter, live runner, signal sources, exchange-side risk control
├── ml/                       ResNet / GRU / MLP / LightGBM (optional torch dependency)
├── kol/  meme/               KOL social factors, Meme event strategies (optional)
├── dashboard/                Data access layer for the dashboard (read-only)
app/                          Streamlit dashboard: factor leaderboard, live monitor (docs/12)
rust_backtest/                Rust cross-sectional backtest core
scripts/01_data … 11_dex/     Command-line entry points for each stage
config/                       default.yaml, KOL/Meme configs
docs/                         Per-stage documentation
tests/                        Unit, comparison and end-to-end tests
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # data, backtesting, live trading
pip install -e ".[mining]"       # + LLM factor generation (openai client)
pip install -e ".[ml]"           # + torch / tensorboard / lightgbm
pip install -e ".[kol]"          # + twscrape
pip install -e ".[test]"         # + pytest

cargo build --release --manifest-path rust_backtest/Cargo.toml   # Rust backtest core
```

Run all commands from the repository root. Scripts add the repository root to `sys.path` automatically, so they work even without `pip install`.

## 10-minute quickstart

```bash
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,LINKUSDT,ADAUSDT

# 1. Download 60 days of data
python scripts/01_data/download_data.py --symbols $SYMBOLS --days 60 --proxy http://127.0.0.1:1087

# 2. Audit the example factors and export predictions
python scripts/02_mining/audit_factor_lookahead.py \
  --factor-jsonl examples/crypto_factors.example.jsonl --out-csv outputs/audit.csv
python scripts/03_backtest/export_factor_predictions.py \
  --factor-jsonl examples/crypto_factors.example.jsonl --data-dir data --symbols $SYMBOLS \
  --interval 30m --horizon 1 --days 60 --out-dir outputs/pred

# 3. Rust backtest across six rebalancing periods (1h…24h), then select
bash scripts/03_backtest/sweep_rust_rebalance.sh --predictions-dir outputs/pred --output-root outputs/sweep --no-plots
python scripts/04_selection/scan_rust_factors.py --results-dir outputs/sweep --out-csv outputs/scan.csv --sort-by sharpe

# 4. Live dry-run (places no orders, no API key needed)
python scripts/06_live/run_live.py \
  --realtime-factor-id liquidity_crypto_templates_10k_v1_liquidity_0041_5 \
  --symbols $SYMBOLS --min-assets 6 --network mainnet --output outputs/dry_run
```

## Documentation

| Stage | Doc | Main entry points |
|---|---|---|
| Data | [docs/01_data.md](docs/01_data.md) | `scripts/01_data/download_data.py`, `kline_correlation.py` (universe selection) |
| Factor mining | [docs/02_factor_mining.md](docs/02_factor_mining.md) | `scripts/02_mining/generate_factors.py`, `audit_factor_lookahead.py` |
| Backtest | [docs/03_backtest.md](docs/03_backtest.md) | `export_factor_predictions.py`, `sweep_rust_rebalance.sh` |
| Selection | [docs/04_selection.md](docs/04_selection.md) | `scan_rust_factors.py`, `analyze_factor_correlation.py` |
| Shadow replay | [docs/05_shadow_replay.md](docs/05_shadow_replay.md) | `run_selected_factor_shadow.py`, `rank_across_periods.py` |
| Live trading | [docs/06_live_trading.md](docs/06_live_trading.md) | `scripts/06_live/run_live.py` |
| ML models | [docs/07_ml.md](docs/07_ml.md) | `train_crypto_t1.py` |
| Alpha101/191 | [docs/08_alpha101_191.md](docs/08_alpha101_191.md) | `sweep_alpha_test.py` |
| KOL social factors | [docs/09_kol.md](docs/09_kol.md) | `scripts/09_kol/` |
| Meme events | [docs/10_meme_event.md](docs/10_meme_event.md) | `scripts/10_meme/` |
| DEX | [docs/11_dex.md](docs/11_dex.md) | `generate_dex_factors.py` |
| Dashboard | [docs/12_dashboard.md](docs/12_dashboard.md) | `streamlit run app/streamlit_app.py` |
| Legacy repo mapping | [docs/MIGRATION.md](docs/MIGRATION.md) | |

## Research discipline

1. **Time splits**: mine, select, dedupe and fix the rebalancing period on the training set; confirm the results reproduce on the validation set; look at the test set only once. Never go back and change factors, thresholds or rebalancing frequency based on test-set results.
2. **Multiple testing**: picking the best out of tens of thousands of factors × 6 rebalancing periods inevitably yields many false positives. On 2 years of 30m data the standard error of a Sharpe estimate is about 0.7, and roughly 8% of pure-noise factors reach Sharpe ≥ 1.
3. **Backtest ≠ achievable returns**: backtests fill instantly at the close or the 1m average price; live trading has polling latency, book impact, minimum order sizes and rejections.
4. **Rollout order**: dry-run → sustained Testnet observation → small-capital mainnet → scale up gradually only after long-term monitoring.

## Tests

```bash
pytest -q
```

The tests cover:
- Byte-for-byte comparison of the old and new live runners (dry run, live orders, crash-triggered stop loss, rollback on disconnect during order placement, CSV / JSONL signal sources);
- Paired checks of the return formula against exact position-quantity accounting;
- Look-ahead blocking;
- Exchange-side risk control, manual position-change detection, one-click flatten, graceful shutdown;
- End-to-end tests of shadow replay.

Rust tests require building `rust_backtest` first; ML tests require torch and are skipped automatically otherwise.
