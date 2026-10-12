# 05 风控引擎与 Shadow 回放

`efficient_cryptos/engine/risk_engine.py` 同时用于离线回放和实盘。离线回放的风控结果，就是实盘遇到同样行情时会做出的决定。

## 状态机

```text
            单币止损/止盈 · 日内止损/止盈/回撤 · 组合高水位回撤
   LIVE  ─────────────────────────────────────────────────►  SHADOW
    ▲     （平掉全部真实仓位，收取平仓成本）                       │
    │                                                         │ 影子账本继续按信号
    │   满足恢复条件：≥ shadow_min_periods 个周期，              │ 和成本模拟交易
    └── 其中 ≥ shadow_positive_periods 个盈利，影子收益 ≥ 阈值 ◄──┘
        且当天没有被日内熔断封锁
```

| 规则 | 依据 | 参数（实盘 / 回放） |
|---|---|---|
| 单币止损 / 止盈 | 相对入场价；同向调仓时保留原入场价，加仓时按数量加权（与交易所 `entryPrice` 一致） | `--symbol-stop 0.05` / `--symbol-take 0.30` |
| 日内止损 / 止盈 | 相对 UTC 当日开盘时的权益 | `--daily-stop 0.05` / `--daily-take 0.30` |
| 日内回撤、组合高水位回撤 | 相对当日最高权益 / 历史最高权益 | `--portfolio-drawdown 0.10` |
| 恢复 | 见上图 | `--shadow-min-periods 3`、`--shadow-positive-periods 2`、`--shadow-recovery-pnl-pct 0.01` |

需要注意：

- **任一币种触发止损或止盈，整个组合都会进入 SHADOW**，而不是只平掉这一个币。
- **恢复 LIVE 时会重置回撤基准。** 包括高水位、当日开盘权益和当日最高权益，否则 SHADOW 期间冻结的权益会让账户刚恢复就再次触发回撤。
- **当日封锁期间不恢复**，第二天（UTC）才允许恢复。
- **影子账本永远不会产生真实订单**；只有 LIVE 状态下的 `positions` 才会成为交易所目标。

记账方式：持仓按数量计，名义价值随价格重估；收益按简单收益计算；风控平仓也收取手续费和滑点；bar 上带 `funding_rate` 时结算资金费。行情缺失的币种不会被跳过，会在 `warnings` 字段写入 `MISSING_PRICE:SYMBOL`。

## 批量回放 prediction

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

时间线与实盘一致：

```text
信号 bar T（开盘时间）
  → T+30m：Engine.step —— 用 30m 收盘价结算 PnL 和 bar 内的资金费，然后风控、调仓
  → 到下一根信号 bar 收盘前：每 --mark-minutes 分钟用已完成 1m K 线的 (O+H+L+C)/4
     调用 Engine.mark（和实盘风控用的价格相同）
```

- 不足 30 根 1m 数据的 30m bar 视为未完成，不参与交易。
- `--mark-minutes 0` 关闭盘中标记，结果会偏乐观，只用于对比。
- 资金费从 `data/funding/<SYMBOL>.parquet` 读取；`--no-funding` 关闭。
- 调仓周期必须是 0.5 小时的整数倍。

每个周期目录下输出 `<factor>.timeseries.csv`（每次 step 以及触发风险事件的 mark）、`<factor>.metrics.json`（收益、回撤、Sharpe、成本、资金费、shadow 统计）、`summary.csv`（默认按 Sharpe 排序）。

## 跨周期排名与图表

```bash
python scripts/05_shadow/rank_across_periods.py --results-dir outputs/shadow_test --top 30 --out-csv outputs/shadow_rank.csv
python scripts/05_shadow/scan_shadow_engine.py --results-dir outputs/shadow_test --out-dir outputs/shadow_test_scan
```

`rank_across_periods.py` 按"盈利周期数 → 保持 LIVE 的周期数 → Sharpe 中位数 → 平均收益"排序，并列出单次表现最好的组合。只在单一周期表现突出的因子，通常不如跨周期稳定的因子可信。

## 纸面回放（自备信号和 bar）

```bash
python scripts/05_shadow/paper_replay.py --signals signals.csv --bars bars.csv --output outputs/paper \
  --gross-leverage 3 --rebalance-bars 24 --fee-bps 5 --slippage-bps 1
```

- `signals.csv`：`timestamp,symbol,signal,target_notional`
- `bars.csv`：`timestamp,symbol,close,quote_volume[,funding_rate]`

只在同时有信号和 bar 的时间点调用 `step`，`--rebalance-bars` 按信号 bar 计数。输出 `timeseries.csv`、`orders.csv`、`metrics.json`。

## 回放结果的局限

- 回放按 bar 收盘价或 1m 均价瞬间成交；实盘要等下一次轮询，并且用市价单成交。
- 如果 prediction 来自主网数据，而实盘跑在 Testnet，因子输入（尤其是成交量）会有明显差异。
- 历史回放结果见 `docs/results/`。
