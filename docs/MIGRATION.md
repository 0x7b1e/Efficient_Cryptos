# Migrating from Predict_Cryptos / Developer_Crypto

This repository was merged from:
- `Predict_Cryptos`: branch `claude/backtest-accounting-fix`, i.e. `main` plus the backtest accounting fix from PR #1;
- `Developer_Crypto`: commit `d59ce9e` on the `main` branch.

Backtesting, trading and factor-mining algorithms are identical to those two versions. All changes are listed at the end of this page.

## Code location mapping

| Old location | New location |
|---|---|
| `crypto_quant/config.py`, `storage.py`, `binance_client.py`, `costs.py`, `signal_policy.py`, `sizing.py` | `efficient_cryptos/common/` |
| `crypto_quant/universe.py` | `efficient_cryptos/data/universe.py` |
| `crypto_quant/ml/features.py` (features and labels) | `efficient_cryptos/features/frame.py` (no longer depends on torch) |
| `crypto_quant/ml/features.py` (`Section`, `build_sections`, `select_features`) | `efficient_cryptos/ml/sections.py` |
| `crypto_quant/ml/cache_layers.py` | `efficient_cryptos/features/cache.py` |
| `crypto_quant/ml/metrics.py` | `efficient_cryptos/backtest/metrics.py` |
| `crypto_quant/ml/models.py`, `losses.py`, `lgbm.py` | `efficient_cryptos/ml/` |
| `crypto_quant/backtest/evaluator.py` (factor validation and execution) | `efficient_cryptos/factors/sandbox.py` |
| `crypto_quant/backtest/evaluator.py` (`lifecycle_pnl_costed`, `factor_summary`, `decile_profile`) | `efficient_cryptos/backtest/portfolio.py` |
| `crypto_quant/backtest/funding.py`, `report.py` | `efficient_cryptos/backtest/` |
| `crypto_quant/alpha.py`, `external/popbo_alphas/` | `efficient_cryptos/factors/alpha.py`, `factors/popbo_alphas/` |
| `crypto_quant/live/alpha_scanner.py` | `efficient_cryptos/execution/sources/alpha_scanner.py` |
| `crypto_quant/live/ml_signal.py` | `efficient_cryptos/execution/sources/ml_model.py` |
| `crypto_quant/live/testnet_trader.py` (risk functions) | `efficient_cryptos/execution/guards.py` (moved verbatim) |
| `crypto_quant/live/runner.py` (manual position-change detection) | `ManualChangeGuard` in `efficient_cryptos/execution/guards.py` |
| `dpsk_factor/mining_factors/codex_factor_generator.py`, `factor_dsl.py`, `factor_registry.py`, `factor_generation_status.py` | `efficient_cryptos/mining/generator.py`, `dsl.py`, `registry.py`, `status.py` |
| `dpsk_factor/crypto_prompts/`, `dex_prompts/`, `meme_event/prompts/` | `efficient_cryptos/mining/prompts/{crypto,dex,meme}/` |
| `kol_factor/`, `kol_factor/config/` | `efficient_cryptos/kol/`, `config/kol/` |
| `meme_event/`, `meme_event/scripts/` | `efficient_cryptos/meme/`, `scripts/10_meme/` |
| `shadow_testnet/engine.py` | `efficient_cryptos/engine/risk_engine.py` |
| `shadow_testnet/binance_testnet_adapter.py` | `efficient_cryptos/execution/binance_adapter.py` (class renamed to `BinanceFuturesAdapter`; the old name `BinanceTestnetAdapter` is kept as an alias) |
| `shadow_testnet/live_runner.py` | `efficient_cryptos/execution/runner.py` (run logic) + `cli.py` (command line) + `sources/` (signal sources) |
| `shadow_testnet/realtime_signal.py`, `generic_realtime_signal.py` | `efficient_cryptos/execution/sources/liquidity_factor.py`, `jsonl_factor.py` |
| `shadow_testnet/cli.py` | `efficient_cryptos/engine/paper_replay.py`, entry point `scripts/05_shadow/paper_replay.py` |
| `scripts/*.py` (both repositories) | `scripts/<stage>/`, see the documentation table in the README |
| `rust_backtest/` | Unchanged |

## Command mapping

| Old command | New command |
|---|---|
| `python -m shadow_testnet.live_runner ...` | `python scripts/06_live/run_live.py ...` (same arguments) |
| `python scripts/run_strategy.py` (Alpha live) | `run_live.py --alpha-config config/default.yaml ...` |
| `python scripts/run_ml_strategy.py --model-dir D` (ML live) | `run_live.py --ml-model D ...` |
| `python scripts/scan_alpha.py` | `run_live.py --alpha-config ...` (dry-run by default, no orders) |
| `python scripts/backtest.py linear\|generated` | `scripts/07_ml/evaluate_cross_section.py` / `scripts/03_backtest/evaluate_generated_factors.py` |
| `python dpsk_factor/mining_factors/codex_factor_generator.py --prompt-root dpsk_factor/crypto_prompts` | `python scripts/02_mining/generate_factors.py` (crypto prompts by default) |
| `python -m kol_factor.collect_once` | `python scripts/09_kol/collect_once.py` |
| Old Alpha/ML live `--mainnet` plus typing `YES` interactively | `--network mainnet --live-orders --confirm-mainnet-orders` |
| Mainnet environment variables `BINANCE_API_KEY` / `BINANCE_API_SECRET` | `BINANCE_MAINNET_API_KEY` / `BINANCE_MAINNET_API_SECRET` |
| Risk and manual-change state files kept in `data/`, separated by `--state-prefix` | Kept in each run's own `--output` directory, isolated automatically |

## Not migrated

| Content | Reason |
|---|---|
| `data_utils`, `run_factor_mining`, the T0 backtest, the numba wide-table engine, `select_factors`, `prepare_generated_factors` and the generic `prompts/` in `dpsk_factor/mining_factors/` | A-share intraday factor pipeline with hard-coded A-share server paths such as `/data/nvme0/...`; does not handle crypto data |
| The order-placement parts of `crypto_quant/live/runner.py` and `testnet_trader.py`, `target_printer.py`, `scripts/run_strategy.py`, `run_ml_strategy.py` | Live trading now uses a single engine; their unique risk controls, manual-change detection and close functions were merged into the new runner |
| Managing only positions the strategy opened itself (managed symbols), `execution.symbol_map` mapping wiring, `position_side_mode` validation | The new runner manages the whole account; position mode is detected automatically by the adapter. The `_map_targets_for_execution` function is still kept in `guards.py` |
| `scripts/download_dex_data.py`, `find_dex_pools.py`, `stream_download_upload_dex.py` | The `dex_data` module they depend on was never committed to the original repository, so the scripts cannot run |

## Fixes and changes made during consolidation

- **New runner**: split into "account → signal → risk → targets → sync → audit" steps; `tests/test_runner_golden.py` guarantees byte-identical output with the original.
- **New optional features**: Alpha and ML live signal sources, `--exchange-risk-config`, `--manual-guard`, `--close-on-exit`, `--close-all`, graceful SIGINT/SIGTERM shutdown. All off by default.
- **Shadow replay timestamp format**: intraday mark rows and step rows now use the same timestamp format; mixing two formats previously broke plotting in `scan_shadow_engine.py`.
- **Plotting interpreter**: `backtest_selected_factors.py` and `sweep_rust_rebalance.sh` no longer hard-code `.venv/bin/python`.
- **Anycast script**: node-number validation now matches the whole string (the old check only looked at the first character, and the script runs as root); the install script no longer hard-codes the user's home directory.
- **Export warning**: for labels spanning multiple bars — `utc_close`, `funding_event` and `horizon > 1` — export now warns that the Rust portfolio return will be double-counted.
- **Dependency cleanup**: `features/frame.py` no longer imports torch, so backtesting and live trading alone do not require torch.
- **Monitoring fields**: every runner audit record gains a read-only `engine_totals` field; Shadow replay outputs cumulative cost and position columns. These are used only by the dashboard and do not affect trading logic. The golden test strips `engine_totals` first and still requires everything else to be byte-identical to the original.
