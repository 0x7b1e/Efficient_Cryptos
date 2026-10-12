# 10 Meme 事件策略

这是一个独立的短周期事件策略，与 30m 截面流程互不影响：

```text
动态 Meme 币池 → 1m/5m 动量、成交额冲击、交易强度、主动买卖流 → 事件分数和截面分位数
  → DPSK 事件因子 → 持仓、止损止盈、流动性衰减退出、冷却
```

| 模块（`efficient_cryptos/meme/`） | 作用 |
|---|---|
| `universe.py` | 用 24h ticker 构建动态的非主流 USDT 合约池 |
| `features.py` | 1m、3m、5m、10m 动量，成交额冲击，交易强度，订单流失衡（OFI） |
| `detector.py` | 按截面分位数触发事件 |
| `cooldown.py` | 单币冷却；止损、止盈、动量衰减和流动性衰减退出 |
| `execution.py` | 把事件转换成目标名义仓位；只输出目标，不签名、不下单 |
| `social_attention.py` | 读取已导出、带时间戳的社交数据 |

## 运行

```bash
# 币池快照（应按天或按小时生成，回测时使用当时可见的快照，避免幸存者偏差）
python scripts/10_meme/make_universe.py --tickers-json data/universe/binance_futures_24h.json \
  --top-n 200 --min-quote-volume 1000000 --out data_meme/universe_snapshot.json

# 下载 1m 数据到单独目录（只用公开接口，可断点续传）
python scripts/10_meme/download_futures_1m.py --universe data_meme/universe_snapshot.json \
  --start-date 2026-01-01 --end-date 2026-10-01 --out-dir data_meme --proxy http://127.0.0.1:1087 --workers 2

# 特征和未来收益标签（future_ret_1m/5m/10m/30m）
python scripts/10_meme/build_features.py --kline-dir data_meme/klines --universe data_meme/universe_snapshot.json \
  --start-date 2026-01-01 --end-date 2026-10-01 --with-labels --out data/features/meme_event_1m.parquet

# 事件窗口回测：统计 MFE/MAE 和退出原因
python scripts/10_meme/backtest_events.py --features data/features/meme_event_1m.parquet --out-dir outputs/meme_bt \
  --hold-minutes 30 --stop-loss 0.05 --take-profit 0.30 --decay-volume-shock -0.20 --fee-bps 4 --slippage-bps 8

# 用 meme 提示词生成事件因子
python scripts/10_meme/generate_dpsk_factors.py --provider deepseek --model deepseek-chat --total-candidates 100 --dry-run
```

- **默认触发条件**：`event_percentile ≥ 95%`、`volume_shock_5m ≥ 0.5`、`|ret_5m| ≥ 1%`。阈值应该在训练期内扫描确定。
- **回测规则**：同一个币同一时间只允许一笔事件仓位。`trades.csv` 记录每笔交易的 MFE、MAE、持仓时间和退出原因。
- **输入要求**：至少需要 `open_time`（或 `timestamp`）、`open,high,low,close,quote_volume`；`volume,trades,taker_buy_quote` 可选。
- **数据来源**：不要把现货的 1s K 线混进合约回测。需要更高频率时，优先用合约的 aggTrade 或 bookTicker 聚合到 1s/5s，再生成 1m/5m 特征。

`MemeEventExecutor` 目前只生成 `EventTarget`。接入实盘之前，必须先确认扣除手续费和滑点后的事件窗口结果仍然为正。
