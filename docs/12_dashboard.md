# 12 可视化看板

一个只读的 Streamlit 网站，直接读取 `outputs/` 下已有的结果文件，有两个页面：

- **因子排行榜**：读取 Rust 回测结果，可以筛选、排序，对比训练期和测试期，查看单个因子的净值、回撤和十分位收益。
- **实盘监控**：读取实盘 runner 的输出目录，显示权益、持仓、多空敞口、风控状态、订单和断线重连记录。

看板不保存 API key，不连接 Binance，也不能下单，所有操作都在命令行完成。

## 启动

```bash
pip install -e ".[ui]"          # streamlit + plotly
streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

在 AutoDL 这类远程机器上，用 SSH 隧道在本地浏览器打开（端口和主机换成你实例的 SSH 信息）：

```bash
# 在你自己的电脑上执行，然后访问 http://localhost:8501
ssh -N -L 8501:127.0.0.1:8501 -p <SSH端口> root@<SSH主机>
```

Streamlit 没有登录功能。不要用 `--server.address 0.0.0.0` 或平台的公开端口把看板直接暴露到公网，否则任何人都能看到你的持仓和权益。

## 因子排行榜

数据来自 `scripts/03_backtest/sweep_rust_rebalance.sh` 的输出：

```text
outputs/sweep_train/
├── 1h/<factor>.metrics.json   .timeseries.csv   .layers.csv
├── 2h/ ...
└── 24h/ ...
```

左侧栏的设置：
- **回测结果目录**：自动列出 `outputs/` 下所有包含回测结果的目录，最近更新的排在最前面。
- **对照期目录**：选择另一个时间段的回测结果（例如 2026 测试期），会增加测试期指标列、训练期 vs 测试期的散点图，详情图里也会叠加测试期的净值和十分位收益。
- **因子 JSONL**：显示公式、分类、方向和经济假设，并可以按分类筛选。

页面内容：

| 区域 | 内容 |
|---|---|
| 筛选 | 调仓周期、Sharpe、收益、最大回撤、搜索、分类；"每个因子只看最佳周期"；"测试期也盈利" |
| 排行榜 | 点击一行，下方的详情会切换到这个因子和这个周期；可以下载筛选后的 CSV |
| 跨周期稳定性 | 每个因子在几个周期上盈利、Sharpe 中位数、最差 Sharpe、最差回撤 |
| 训练 vs 测试 | 两期 Sharpe 散点图、两期都为正的比例、Sharpe 排名的 Spearman 相关系数 |
| 因子详情 | 各周期指标；净值和回撤（叠加测试期）；十分位平均超额收益；Q1、Q10 和 Q10−Q1 的累计曲线 |

训练 vs 测试页里的 **Spearman 相关系数** 是判断过拟合最直接的指标：
- 接近 0：训练期排名对测试期几乎没有预测力，排行榜靠前的因子大多是运气；
- 明显为正（例如 0.2 以上）：训练期的排序在测试期仍然部分有效。

## 实盘监控

数据来自 `run_live.py --output DIR`：

| 文件 | 用途 |
|---|---|
| `latest.json` | 最近一个周期：权益、持仓、目标仓位、订单 |
| `audit.jsonl` | 每个周期一行。文件会越来越大，看板只读取末尾 N 行（左侧栏可调） |
| `engine_state.json` | 风控状态：LIVE/SHADOW、高点、当日开盘权益、被屏蔽的币 |
| `runner_errors.jsonl` | 网络错误和周期错误，以及重试等待时间 |
| `exchange_risk_state.json`、`manual_guard_state.json` | 可选的交易所风控和人工改仓检测状态 |

页面内容：
- **状态栏**：因子、模式（TESTNET/MAINNET，LIVE 或 DRY_RUN）、LIVE/SHADOW 状态、最近更新时间。`latest.json` 超过设定的分钟数没有更新时显示红色警告，这个阈值建议设为 poll 间隔的 2–3 倍。
- **指标**：账户权益、可用保证金、今日盈亏、距高点回撤、总杠杆、净敞口 / 总敞口（美元中性时接近 0）、累计成本和资金费。
- **持仓**：签名运行时显示交易所的实际持仓，dry-run 时显示引擎的模拟账本；右侧是多空名义价值的柱状图。
- **目标 vs 实际**：每个币目标仓位和实际仓位的差额，用来发现取整、最小下单额或拒单造成的偏差。
- **权益与敞口**：权益曲线，SHADOW 时段用橙色背景标出，同时显示影子账本；下方是多头、空头和净敞口。
- **风控事件 / 最近订单 / 错误与重连 / 状态文件**。

页面默认每 30 秒自动刷新，左侧栏可以关闭或修改间隔。

## 代码结构

| 文件 | 作用 |
|---|---|
| `efficient_cryptos/dashboard/data.py` | 所有文件读取和计算，不依赖 streamlit，由 `tests/test_dashboard_data.py` 覆盖（实盘部分用真实 runner 配合假交易所生成数据） |
| `app/streamlit_app.py` | 入口和页面导航 |
| `app/pages/factors.py`、`app/pages/live.py` | 两个页面 |
| `app/_shared.py` | 路径选择、格式化、配色 |

新增页面时，把数据处理写进 `dashboard/data.py` 并补测试，页面文件只负责展示。
