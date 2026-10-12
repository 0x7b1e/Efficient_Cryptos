# 从 Predict_Cryptos / Developer_Crypto 迁移

本仓库合并自：
- `Predict_Cryptos`：分支 `claude/backtest-accounting-fix`，即 `main` 加上 PR #1 的回测口径修复；
- `Developer_Crypto`：`main` 分支的 `d59ce9e`。

回测、交易和因子挖掘的算法与这两个版本一致。所有改动都列在本页末尾。

## 代码位置对照

| 旧位置 | 新位置 |
|---|---|
| `crypto_quant/config.py`、`storage.py`、`binance_client.py`、`costs.py`、`signal_policy.py`、`sizing.py` | `efficient_cryptos/common/` |
| `crypto_quant/universe.py` | `efficient_cryptos/data/universe.py` |
| `crypto_quant/ml/features.py`（特征与标签部分） | `efficient_cryptos/features/frame.py`（不再依赖 torch） |
| `crypto_quant/ml/features.py`（`Section`、`build_sections`、`select_features`） | `efficient_cryptos/ml/sections.py` |
| `crypto_quant/ml/cache_layers.py` | `efficient_cryptos/features/cache.py` |
| `crypto_quant/ml/metrics.py` | `efficient_cryptos/backtest/metrics.py` |
| `crypto_quant/ml/models.py`、`losses.py`、`lgbm.py` | `efficient_cryptos/ml/` |
| `crypto_quant/backtest/evaluator.py`（因子校验与执行） | `efficient_cryptos/factors/sandbox.py` |
| `crypto_quant/backtest/evaluator.py`（`lifecycle_pnl_costed`、`factor_summary`、`decile_profile`） | `efficient_cryptos/backtest/portfolio.py` |
| `crypto_quant/backtest/funding.py`、`report.py` | `efficient_cryptos/backtest/` |
| `crypto_quant/alpha.py`、`external/popbo_alphas/` | `efficient_cryptos/factors/alpha.py`、`factors/popbo_alphas/` |
| `crypto_quant/live/alpha_scanner.py` | `efficient_cryptos/execution/sources/alpha_scanner.py` |
| `crypto_quant/live/ml_signal.py` | `efficient_cryptos/execution/sources/ml_model.py` |
| `crypto_quant/live/testnet_trader.py`（风控函数） | `efficient_cryptos/execution/guards.py`（原样搬迁） |
| `crypto_quant/live/runner.py`（人工改仓检测） | `efficient_cryptos/execution/guards.py` 中的 `ManualChangeGuard` |
| `dpsk_factor/mining_factors/codex_factor_generator.py`、`factor_dsl.py`、`factor_registry.py`、`factor_generation_status.py` | `efficient_cryptos/mining/generator.py`、`dsl.py`、`registry.py`、`status.py` |
| `dpsk_factor/crypto_prompts/`、`dex_prompts/`、`meme_event/prompts/` | `efficient_cryptos/mining/prompts/{crypto,dex,meme}/` |
| `kol_factor/`、`kol_factor/config/` | `efficient_cryptos/kol/`、`config/kol/` |
| `meme_event/`、`meme_event/scripts/` | `efficient_cryptos/meme/`、`scripts/10_meme/` |
| `shadow_testnet/engine.py` | `efficient_cryptos/engine/risk_engine.py` |
| `shadow_testnet/binance_testnet_adapter.py` | `efficient_cryptos/execution/binance_adapter.py`（类名改为 `BinanceFuturesAdapter`，保留旧名 `BinanceTestnetAdapter` 作为别名） |
| `shadow_testnet/live_runner.py` | `efficient_cryptos/execution/runner.py`（运行逻辑）+ `cli.py`（命令行）+ `sources/`（信号源） |
| `shadow_testnet/realtime_signal.py`、`generic_realtime_signal.py` | `efficient_cryptos/execution/sources/liquidity_factor.py`、`jsonl_factor.py` |
| `shadow_testnet/cli.py` | `efficient_cryptos/engine/paper_replay.py`，入口 `scripts/05_shadow/paper_replay.py` |
| `scripts/*.py`（两个仓库） | `scripts/<阶段>/`，见 README 中的文档导航 |
| `rust_backtest/` | 不变 |

## 命令对照

| 旧命令 | 新命令 |
|---|---|
| `python -m shadow_testnet.live_runner ...` | `python scripts/06_live/run_live.py ...`（参数相同） |
| `python scripts/run_strategy.py`（Alpha 实盘） | `run_live.py --alpha-config config/default.yaml ...` |
| `python scripts/run_ml_strategy.py --model-dir D`（ML 实盘） | `run_live.py --ml-model D ...` |
| `python scripts/scan_alpha.py` | `run_live.py --alpha-config ...`（默认就是 dry-run，不下单） |
| `python scripts/backtest.py linear\|generated` | `scripts/07_ml/evaluate_cross_section.py` / `scripts/03_backtest/evaluate_generated_factors.py` |
| `python dpsk_factor/mining_factors/codex_factor_generator.py --prompt-root dpsk_factor/crypto_prompts` | `python scripts/02_mining/generate_factors.py`（默认就是 crypto 提示词） |
| `python -m kol_factor.collect_once` | `python scripts/09_kol/collect_once.py` |
| 旧 Alpha/ML 实盘的 `--mainnet` 加交互输入 `YES` | `--network mainnet --live-orders --confirm-mainnet-orders` |
| 主网环境变量 `BINANCE_API_KEY` / `BINANCE_API_SECRET` | `BINANCE_MAINNET_API_KEY` / `BINANCE_MAINNET_API_SECRET` |
| 风控和人工改仓的状态文件放在 `data/`，用 `--state-prefix` 区分 | 放在各自的 `--output` 目录，自动隔离 |

## 没有迁移的内容

| 内容 | 原因 |
|---|---|
| `dpsk_factor/mining_factors/` 中的 `data_utils`、`run_factor_mining`、T0 回测、numba 宽表引擎、`select_factors`、`prepare_generated_factors`，以及通用 `prompts/` | A 股日内因子流程，写死了 `/data/nvme0/...` 等 A 股服务器路径，不处理加密货币数据 |
| `crypto_quant/live/runner.py`、`testnet_trader.py` 的下单部分、`target_printer.py`、`scripts/run_strategy.py`、`run_ml_strategy.py` | 实盘统一使用一套引擎；它们独有的风控、人工改仓检测和平仓功能已合并进新 runner |
| 只管理策略自己建的仓位（managed symbols）、`execution.symbol_map` 映射接线、`position_side_mode` 校验 | 新 runner 管理整个账户；持仓模式由适配器自动识别。`_map_targets_for_execution` 函数仍保留在 `guards.py` 中 |
| `scripts/download_dex_data.py`、`find_dex_pools.py`、`stream_download_upload_dex.py` | 依赖的 `dex_data` 模块从未提交到原仓库，脚本无法运行 |

## 整理过程中的修复和改动

- **新 runner**：拆分成"账户 → 信号 → 风控 → 目标 → 同步 → 审计"几个步骤；`tests/test_runner_golden.py` 保证与原版输出逐字节一致。
- **新增可选功能**：Alpha 和 ML 实时信号源、`--exchange-risk-config`、`--manual-guard`、`--close-on-exit`、`--close-all`、SIGINT/SIGTERM 优雅退出。默认全部关闭。
- **Shadow 回放的时间格式**：盘中标记行与 step 行的时间戳改为同一格式，之前混用两种格式会导致 `scan_shadow_engine.py` 画图失败。
- **画图解释器**：`backtest_selected_factors.py` 和 `sweep_rust_rebalance.sh` 不再写死 `.venv/bin/python`。
- **Anycast 脚本**：节点号校验改为整串匹配（原来的写法只检查第一个字符，而这个脚本以 root 运行）；安装脚本不再写死用户主目录。
- **导出警告**：`utc_close`、`funding_event`、`horizon > 1` 这几种跨多根 bar 的标签，导出时会提示 Rust 组合收益会被重复计算。
- **依赖整理**：`features/frame.py` 不再导入 torch，所以只做回测和实盘不需要安装 torch。
