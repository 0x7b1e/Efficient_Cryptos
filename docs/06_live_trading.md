# 06 实盘：dry-run → Testnet → 主网

入口：`python scripts/06_live/run_live.py`（等价于 `python -m efficient_cryptos.execution.cli`）。**默认是 dry-run**：只拉行情、打印下单计划，不下任何订单，也不需要 API key。

## 每一轮做什么

```text
1 账户    读 Binance 权益（NAV）和可用保证金；NAV 用来校准引擎权益，
          新仓位按 min(NAV, 可用保证金) 定仓
2 信号    信号源返回最近一根已完成 bar 的截面
3 风控    用已完成的 1m K 线均价标记持仓（每次轮询都检查风控）；
          出现新的信号时间戳时再执行 step（结算 PnL、风控、调仓）
4 目标    只有 LIVE 账本才产生交易所目标；SHADOW 时目标为空，也就是平仓；
          可选的交易所侧风控可以覆盖目标
5 同步    首轮、新信号或出现风险事件时：设置杠杆 → 按数量差额生成对账计划
          （先减仓后开仓）→ 市价单
6 审计    写 latest.json，追加 audit.jsonl，原子地保存引擎和 runner 状态
```

异常处理：

| 情况 | 行为 |
|---|---|
| 网络错误（`BinanceNetworkError`） | 引擎回滚到本轮开始前的状态，记录到 `runner_errors.jsonl`，退避重试（5s 起翻倍，最长 120s）。适配器本身对 GET 请求重试 4 次并在需要时重连代理；**下单（POST）从不自动重发**，下一轮直接与交易所真实持仓对账 |
| 截面不足、单笔下单被拒等（`ValueError` / `RuntimeError`） | 同样回滚并退避，记录为 `cycle_error` |
| 418 / 429 限频 | 按 `Retry-After` 等待，不重连代理 |
| 签名请求 | 启动时校准交易所服务器时间，每次重试都重新生成时间戳和签名 |
| SIGINT / SIGTERM | 完成当前这一轮后退出；带 `--close-on-exit` 时先平仓再退出 |

## 信号源（五选一）

| 参数 | 信号 | 选币和定仓 |
|---|---|---|
| `--realtime-factor-id liquidity_crypto_templates_10k_v1_liquidity_0041_5` | 内置流动性因子，用最近已完成的 30m K 线计算 | 按 `--long-percent`/`--short-percent` 选 Top/Bottom，等权 |
| `--realtime-factor-jsonl F.jsonl [--realtime-factor-record-id ID]` | 任意生成因子；与回测使用同一个沙箱 | 同上 |
| `--signals F.csv` | CSV 中最新一个时间截面（`prediction`，或 `signal,target_notional`） | `prediction` 列按百分比等权；`target_notional` 列直接使用 |
| `--alpha-config config/default.yaml` | Alpha101/191 实时扫描（YAML 的 `alpha_live` 段） | 沿用 Alpha 自己的规则（默认按波动率倒数定仓），每根 bar 只扫描一次 |
| `--ml-model outputs/crypto_t1/<run>` | 训练好的 ML 模型 | 沿用 ML 自己的 `--ml-k-long`/`--ml-m-short` 等权定仓 |

所有信号源最终都经过同一个风控引擎：引擎会把总仓位限制在 `权益 × (1 − margin_buffer) × gross_leverage` 以内，再执行止损、日内熔断和 SHADOW。

**定仓公式**（百分比类信号源）：每边预算 = `sizing_equity × (1 − margin_buffer_pct) × gross_leverage / 2`，在这一边选中的币之间等分。例如权益 10,000、3 倍杠杆、缓冲 20%、多空各 2 个币：每个多头 +6,000，每个空头 −6,000。`--gross-leverage` 同时也作为交易所每个币种的杠杆设置，必须是 1–125 的整数。

## 调仓周期与轮询

- `--rebalance-hours 4` 或 `--rebalance-minutes 240`：必须是 30 分钟的整数倍，否则直接报错，不会静默取整；`--rebalance-bars` 直接按 bar 数指定。
- `--poll-seconds 60`：只决定多久检查一次风控和行情，不改变调仓周期。`0` 表示只运行一轮。
- 状态保存在 `--output` 目录：`engine_state.json`（仓位、权益、SHADOW 状态）和 `runner_state.json`（最后处理的信号时间戳）。重启时自动恢复，同一个信号时间戳不会处理两次。`--reset-state` 只清除本地状态，**不会改动交易所上的仓位**。

## 运行

```bash
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,ZECUSDT,MOVRUSDT,NEARUSDT,QNTUSDT,XRPUSDT,DOGEUSDT,SUIUSDT,UNIUSDT,WLDUSDT,1000PEPEUSDT,BNBUSDT,LINKUSDT,ADAUSDT
COMMON="--realtime-factor-id liquidity_crypto_templates_10k_v1_liquidity_0041_5 --symbols $SYMBOLS \
  --poll-seconds 60 --rebalance-hours 4 --gross-leverage 3 --margin-buffer-pct 0.20 \
  --long-percent 10 --short-percent 10 --symbol-stop 0.10 --symbol-take 0.30 \
  --daily-stop 0.05 --daily-take 0.30 --portfolio-drawdown 0.10 --proxy http://127.0.0.1:1087"

# 1) dry-run：用主网行情，只打印下单计划
python -u scripts/06_live/run_live.py $COMMON --network mainnet --output outputs/dry_run

# 2) Testnet 下单；用主网行情计算信号，避免 Testnet 行情失真
export BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=...
python -u scripts/06_live/run_live.py $COMMON --network testnet --market-network mainnet \
  --output outputs/testnet_live --manual-guard --exchange-risk-config config/default.yaml \
  --live-orders --confirm-testnet-orders

# 3) 主网：先用 1 倍杠杆、小资金，使用单独的 output 目录
export BINANCE_MAINNET_API_KEY=... BINANCE_MAINNET_API_SECRET=...
python -u scripts/06_live/run_live.py $COMMON --gross-leverage 1 --network mainnet \
  --output outputs/mainnet_live --manual-guard --exchange-risk-config config/default.yaml \
  --close-on-exit --live-orders --confirm-mainnet-orders
```

- 下单需要同时满足四个条件：带 `--live-orders`、带当前网络对应的确认参数（`--confirm-testnet-orders` 或 `--confirm-mainnet-orders`）、设置了该网络的 API key、两个确认参数没有混用（例如 Testnet 网络上不能出现 `--confirm-mainnet-orders`）。
- `--network` 决定在哪里下单；`--market-network` 决定行情从哪里取，默认与 `--network` 相同。Testnet 的 K 线和成交量与主网差异很大，用它计算成交量类因子会失真。

## 交易所侧保护（可选，建议实盘开启）

| 参数 | 作用 |
|---|---|
| `--exchange-risk-config config/default.yaml` | 读取 YAML 的 `risk` 段，基于**交易所真实持仓和账户权益**再做一层风控：按交易所入场价计算单币止损/止盈，触发后当日不再开仓；按账户权益做日内止损、止盈和回撤熔断；按 `daily_reset_utc_offset_hours`（默认 UTC+8）重置。它与引擎风控互相独立，覆盖了引擎估算不到的情况（真实滑点、资金费、强平）。状态保存在 `<output>/exchange_risk_state.json` |
| `--manual-guard` | 每次同步后记录持仓快照。如果持仓在两次同步之间被外部改动（在 Binance App 手动平仓、其他程序、强平），runner 会平掉全部仓位并停止。之后重启会被拒绝，确认账户安全后加 `--reset-manual-halt` 才能继续。状态保存在 `<output>/manual_guard_state.json` |
| `--close-on-exit` | 收到 SIGINT/SIGTERM 时，完成当前这一轮，然后平掉全部仓位再退出 |
| `--close-all` | 立即平掉全部仓位并退出（同样需要 `--live-orders` 和确认参数） |

runner 把整个合约账户视为自己管理：不在目标里的持仓都会在对账时被平掉。**不要让其他策略和它共用同一个账户。**

## 代理断线重连

用 `--proxy` 时，GET 请求遇到连接失败或 5xx 错误，会先执行重连命令，再检查代理端口是否恢复。默认重连命令是 `sudo -n /usr/local/sbin/shadow-reconnect-anycast`，也可以用 `--proxy-reconnect-command` 自定义。

首次安装（把脚本和 Anycast 运行目录安装成 root 所有，避免免密 sudo 被普通用户可写的文件利用）：

```bash
sudo ANYCAST_SRC=$HOME/Anycast bash scripts/06_live/install_anycast_reconnect.sh
sudo visudo -f /etc/sudoers.d/shadow-anycast
#   <用户名> ALL=(root) NOPASSWD: /usr/local/sbin/shadow-reconnect-anycast
sudo -n /usr/local/sbin/shadow-reconnect-anycast 30     # 测试，30 是节点号
```

`sudo -n` 不会弹出密码提示；如果 sudoers 没有配置好，会立即报错，而不是卡住等待输入。

## 输出与监控

| 文件（在 `--output` 下） | 内容 |
|---|---|
| `latest.json` / `audit.jsonl` | 每一轮的模式、信号时间戳、Binance 权益、可用保证金、定仓权益、引擎结果、目标仓位、订单计划、下单回报、交易所真实持仓；开启交易所风控时还有 `exchange_risk_notes` |
| `runner_errors.jsonl` | 网络错误和单轮错误 |
| `engine_state.json` / `runner_state.json` | 用于重启恢复的状态 |
| `exchange_risk_state.json` / `manual_guard_state.json` | 交易所侧保护的状态 |

建议用 tmux 运行并保存日志：

```bash
tmux new -s live
python -u scripts/06_live/run_live.py ... 2>&1 | tee -a outputs/testnet_live/runner.log
# Ctrl-b d 离开会话；tmux attach -t live 重新进入
```

停止进程用 Ctrl-C 或 `kill <pid>`（SIGTERM），不要用 `kill -9`：它会跳过收尾和退出平仓。

## 故障与恢复

- **`-2019 Margin is insufficient`**：可用保证金不足。新仓位本来就按可用保证金定仓，可以把 `--margin-buffer-pct` 调高到 0.3。单笔下单被拒会记录在 `submitted` 的 `order_error` 里，不会中断这一轮。
- **人工改仓熔断后**：先登录 Binance，确认仓位、挂单和持仓模式；再用 dry-run 确认新目标；最后加 `--reset-manual-halt` 重启。不要直接删除状态文件来绕过检查。
- **网络长时间中断**：软件止损、交易所侧风控和退出平仓都无法执行，必须人工检查账户。

## 上线检查清单

- [ ] dry-run 中的多空币种、数量、方向都正确
- [ ] 每个网络、每个策略使用单独的 `--output` 目录
- [ ] Testnet 用 `--market-network mainnet`，并连续观察足够长时间
- [ ] 主网 API key 只开合约交易权限、禁止提现、设置了 IP 白名单（白名单要包含代理的出口 IP）
- [ ] 开启 `--manual-guard` 和 `--exchange-risk-config`
- [ ] 没有其他进程在操作同一个账户
- [ ] 准备好人工紧急平仓方案（`--close-all` 或 Binance App）
- [ ] 主网首次使用 1 倍杠杆、极小资金

## 已知限制

1. **没有交易所端的条件止损单。** 止损止盈都是软件风控，只在进程和网络正常时生效。真实资金上线前，建议另外预挂 `STOP_MARKET` 等条件单作为兜底。
2. **整点附近的截面可能不完整。** 每个币的 K 线是逐个请求的，跨过整点时可能只拿到部分币种的新 bar。内置流动性因子会取覆盖币种最多的时间戳，可能用不完整的截面出信号，并且同一个时间戳不会重算。时间戳倒退不会被当成新信号。
3. **实盘 JSONL 因子的输入与回测不完全一致。** 回测时派生特征（`log_ret_1`、`funding_rate` 等）已按截面做过 z-score；实盘的 JSONL 信号源用原始 K 线计算，没有这一步，也没有资金费和持仓数据。只用原始 OHLCV / `quote_volume` 的因子（例如内置流动性因子）不受影响；用到派生特征的因子，上线前要先比对实盘与回测的预测值。
4. **Testnet 不能代表真实成交质量**：流动性、滑点和资金费都与主网不同。
