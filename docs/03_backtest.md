# 03 回测

```text
factors.jsonl ──► export_factor_predictions.py ──► <factor>.csv（每行 = 一个时间点 × 一个币种）
                                                      │
                                     sweep_rust_rebalance.sh（1h 2h 4h 8h 12h 24h）
                                                      ▼
                              <周期>/<factor>.metrics.json / layers.csv / timeseries.csv / plots/
```

Python 负责执行因子代码；Rust 负责分层、持仓、成本、资金费和净值计算。

## 标签

| `--label-mode` | 定义 | 用途 |
|---|---|---|
| `horizon`（默认，`--horizon 1`） | 当前 bar 收盘到下一根 bar 收盘的对数收益 | 普通 30m 截面预测、组合回测 |
| `utc_close` | 当前 bar 收盘到当日 UTC 最后一根收盘 | 日内剩余时间收益 |
| `funding_event` | 当前 bar 收盘到该币下一次真实 `funding_time` | 资金费结算窗口 |

所有标签都会转成市场中性的超额收益：`target = raw_target − 当期截面的 raw_target 均值`。

> **组合回测只支持单根 bar 的标签。** Rust 每一行只记一根 bar 的收益。`utc_close`、`funding_event` 和 `horizon > 1` 的标签跨越多根 bar，逐行累加会重复计算，因此组合收益和 Sharpe 会偏高（导出时会打印警告）。这类标签请看分层统计，不要看组合收益。

## 记账口径

| 项目 | 规则 |
|---|---|
| 组合收益 | `Σ wᵢ·(exp(raw_targetᵢ) − 1)`，按简单收益加权，与精确账本误差 < 1e-12 |
| 手续费 | 每单位换手 5bps（taker，`--taker-fee-bps`） |
| 滑点 | `min(80, 1 + 60 × 成交额 / 该 bar 成交额)` bps |
| 资金费 | 按真实 `funding_time` 收取：本行持有期 `(t+30m, t+60m]` 内的结算由这一行的仓位支付，支持 1h / 4h / 8h 结算 |
| 调仓 | 每 `--signal-interval-bars` 根 bar 选一次 Top / Bottom，最长持有 `--signal-hold-bars` 根 |

旧版本用 `exp(Σ wᵢ·logrᵢ) − 1` 计算组合收益。在零漂移的合成数据上，3x 纯随机因子每年会多算约 131%，"永远做空波动最大的币"会多算约 312%。`tests/test_backtest_accounting.py` 用同样的数据检验新公式没有偏差。

## 导出 prediction

```bash
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,ZECUSDT,MOVRUSDT,NEARUSDT,QNTUSDT,XRPUSDT,DOGEUSDT,SUIUSDT,UNIUSDT,WLDUSDT,1000PEPEUSDT,BNBUSDT,LINKUSDT,ADAUSDT

# 训练期
python -u scripts/03_backtest/export_factor_predictions.py \
  --factor-jsonl outputs/crypto_factors.jsonl --data-dir data --symbols $SYMBOLS \
  --interval 30m --horizon 1 --days 1100 \
  --train-start 2024-01-01 --train-end 2025-12-31 \
  --start 0 --limit 200 --jobs 2 --backend thread --resume \
  --out-dir outputs/pred_train_0_200

# 测试期：只改日期和输出目录
  --train-start 2026-01-01 --train-end 2026-10-01 --out-dir outputs/pred_test_0_200
```

- `--start` / `--limit`：分批导出。例如 `--start 200 --limit 200` 导出第 201–400 条。
- `--resume`：跳过已经存在且非空的 CSV。
- `--backend thread`：多个线程共享同一份特征表，内存占用小很多。`process` 更快，但需要更多内存。
- `--train-start` / `--train-end`：只决定导出哪段日期。特征（包括滚动窗口）基于 `--days` 天的完整历史计算。
- 失败的因子（包括被未来函数检查拒绝的）记录在 `failures.json`。

CSV 格式：

```text
factor_id,timestamp,symbol,prediction,target,raw_target,quote_volume,funding_rate,funding_event_rate
```

`funding_event_rate` 只用于记账，因子代码读不到这一列。没有 `raw_target` / `funding_event_rate` 的旧 CSV 仍然可以输入 Rust，但会打印警告并使用近似口径；`metrics.json` 的 `portfolio_return_basis`、`funding_basis` 字段会标明实际使用的口径。

## Rust 批量回测

```bash
cargo build --release --manifest-path rust_backtest/Cargo.toml

bash scripts/03_backtest/sweep_rust_rebalance.sh \
  --predictions-dir outputs/pred_train_0_200 \
  --output-root outputs/sweep_train_0_200 \
  --jobs 3 --long-percent 10 --short-percent 10 --gross-leverage 1 \
  --plot-max-points 5000          # 或 --no-plots
```

这个脚本对每个因子跑 1h、2h、4h、8h、12h、24h 六种调仓周期（即 2、4、8、16、24、48 根 30m bar）；已有完整结果的会跳过，`--force` 强制重跑。`--gross-leverage` 会同时放大仓位、收益、成本和资金费，但不改变因子的排序。

单个因子：

```bash
./rust_backtest/target/release/crypto-factor-backtest \
  --input outputs/pred_train_0_200/<factor>.csv \
  --output out/<factor>.metrics.json --layers-output out/<factor>.layers.csv \
  --timeseries-output out/<factor>.timeseries.csv \
  --long-percent 10 --short-percent 10 --signal-interval-bars 8 --signal-hold-bars 8 \
  --gross-leverage 1 --include-costs --include-funding
```

输出：

| 文件 | 内容 |
|---|---|
| `metrics.json` | `return_pct`、`sharpe`、`max_drawdown_pct`、`avg_turnover`、`total_cost`、`signed_q10_minus_q1`、`abs_q10_minus_q1`、`abs_q10_directional_return`、口径字段 |
| `layers.csv` | 每个时间点：带符号排序的十层 Q1–Q10、按绝对值排序的十层、按绝对值分层后的方向调整收益 |
| `timeseries.csv` | 每根 bar 的毛收益、净收益、PnL、净值、回撤、换手、成本、资金费 |

带符号的 Q10 是预测值最高的一层。按绝对值排序的 Q10 是信号最强的一层，不区分方向，不能和带符号的分层混用。组合始终按带符号的 prediction 做多做空。

画图：

```bash
python scripts/03_backtest/plot_rust_backtest.py --kind equity \
  --metrics out/<f>.metrics.json --layers out/<f>.layers.csv --timeseries out/<f>.timeseries.csv --out out/<f>.equity.png
python scripts/03_backtest/plot_rust_backtest.py --kind layers ...   # 十层累计曲线
```

## Python 回测（IC、十分位、生命周期 PnL）

```bash
python scripts/03_backtest/evaluate_generated_factors.py \
  --factor-jsonl outputs/crypto_factors.jsonl --data-dir data --symbols $SYMBOLS \
  --days 180 --train-start 2025-01-01 --train-end 2025-06-30 \
  --validation-start 2025-07-01 --validation-end 2025-09-30 \
  --long-percent 20 --short-percent 20 --signal-interval-bars 2 --signal-hold-bars 6 \
  --gross-leverage 1 --include-costs --jobs 4 --out outputs/crypto_factor_eval
```

输出 `factor_metrics.csv`（训练期和验证期的 IC、RankIC、ICIR、覆盖率、换手、收益、成本）、`selected.jsonl`、`summary.json`、`failures.json`，以及每个因子的 `factors/<id>/report.png`（十分位分布、D10−D1、净值、回撤、换手、成本）。
筛选条件：先按 IC / ICIR / 覆盖率门槛过滤，再按截面秩相关做贪心去重（`--corr-threshold` 默认 0.80）。`--keep-all` 只出报告、不筛选。

## 重新回测入选的因子

```bash
python scripts/03_backtest/backtest_selected_factors.py \
  --selection-csv outputs/selected_train.csv --predictions-dir outputs/pred_test_0_200 \
  --output-dir outputs/selected_test --jobs 2
```

按筛选表中每行的（因子, 调仓周期）组合，在指定的 prediction 目录上用 Rust 重新回测并画图，通常用于在测试期验证训练期的筛选结果。
