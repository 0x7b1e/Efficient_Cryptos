# 01 数据

所有研究和回测都只读本地 parquet。实盘 runner 直接从 Binance 拉取已完成的 K 线，不依赖本地数据。

## 存储布局

```text
data/
├── universe/top40.json          流动性币池快照
├── klines/<SYMBOL>.parquet      1m K线（open_time 去重），附带多空比等持仓数据列
├── funding/<SYMBOL>.parquet     每次资金费结算一行（funding_time, funding_rate）
└── features/*.parquet           特征/标签缓存（见 03、07）
```

K 线列：`open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote`。
持仓数据（大户 / 全市场多空比、主动买卖比）作为额外列保存在同一个 K 线文件里。注意 Binance 只提供较短的持仓数据历史。

## 下载

```bash
# 首次下载：按 24h 成交额自动选 Top-N 建立币池，回看 90 天
python scripts/01_data/download_data.py --days 90 --top-n 40

# 指定币种
python scripts/01_data/download_data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 90

# 指定 UTC 日期区间（首尾两天都包含）；不能和 --incremental 同时使用
python scripts/01_data/download_data.py --symbols BTCUSDT,ETHUSDT \
  --start-date 2025-01-01 --end-date 2025-06-30

# 只要 K 线和资金费，不要持仓数据
python scripts/01_data/download_data.py --symbols BTCUSDT --days 30 --skip-positioning

# 增量更新已保存的币池；--refresh-universe 会重新选币
python scripts/01_data/download_data.py --incremental
python scripts/01_data/download_data.py --incremental --refresh-universe
```

网络受限时加 `--proxy http://127.0.0.1:1087`（也支持 `socks5h://`）。
`--timeout` 和 `--retries` 控制单个请求的超时和重试次数。
未来的或尚未收盘的 K 线会被拒绝写入。

币池参数（`universe.top_n`、`min_quote_volume_usdt`、排除列表）在 `config/default.yaml` 中配置。

## 本机空间不足：逐币下载并上传到服务器

逐个币种执行"下载 → rsync 上传 → 远端校验 →（可选）删除本地文件"。
进度记录在 `data/download_queue.json`，重启后会跳过已完成的币种；本地已覆盖目标区间的文件只上传、不重新下载。

```bash
python scripts/01_data/stream_download_upload.py \
  --universe data/universe/top40.json \
  --start-date 2024-01-01 --end-date 2026-10-01 \
  --proxy http://127.0.0.1:1087 \
  --remote root@remote-host:/srv/data --remote-port 22 --control-persist 8h \
  --workers 2 --download-attempts 3 --download-backoff 20 \
  --delete-local --retry-failed
```

- 推荐用 SSH key（`ssh-copy-id`）。连接复用后，密码登录每次运行一般只需输入一次，密码不会写进命令行或队列文件。
- `--workers` 从 2 开始，避免触发 Binance 或代理的限频。
- 失败的币种在队列中标记为 `failed`，修复网络后加 `--retry-failed` 重试。

## 补齐服务器上缺失的日期

服务器上的 parquet 停在某一天时，只下载缺失区间到本地临时目录，上传后在服务器端按 `open_time` / `funding_time` 去重合并，并原子替换原文件：

```bash
python scripts/01_data/update_remote_data.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start-date 2026-10-02 --end-date 2026-10-10 \
  --proxy http://127.0.0.1:1087 \
  --remote root@host:/root/autodl-tmp/Efficient_Cryptos/data --remote-port 11402 \
  --remote-python /root/autodl-tmp/Efficient_Cryptos/.venv/bin/python
```

合并成功之前不会覆盖旧文件。`--keep-local` 会保留本地临时下载，用于排查问题。

## 检查数据覆盖范围

```bash
python scripts/01_data/check_parquet_times.py --data-root data
python scripts/01_data/check_parquet_times.py --data-root data --symbol BTCUSDT --output-csv outputs/coverage.csv
```

脚本输出每个文件的首尾时间、行数，以及超过预期间隔的缺口数量（K 线 1 分钟，资金费 8 小时）。

## 30m 小样本

```bash
python scripts/01_data/download_30m_sample.py --help
```

这个脚本直接下载 30m K 线，用于快速做截面实验，不经过 1m 重采样。

## 相关性分析：选择交易币池

`kline_correlation.py` 把本地 K 线重采样成完整的 bar（缺分钟的 bar 记为缺失），计算对数收益的两种相关矩阵：

- **raw**：原始收益的相关性，包含整体行情的共同波动。
- **resid**：去掉市场后的相关性。默认 `--residual market` 每根 bar 减去所有币的等权平均收益，与回测 `target` 的去均值方式相同；`--residual btc` 则回归掉 `--reference` 的 beta。截面多空策略要关注的是 resid：两个币 resid 相关性很高，就等于在下同一个注。

```bash
# 全部本地币种，30m 收益，2026 年以来
python scripts/01_data/kline_correlation.py --data-dir data --interval 30m --start 2026-01-01 --out-dir outputs/kline_corr

# 只看某个币池，并强制保留 BTC、ETH
python scripts/01_data/kline_correlation.py --symbols data/universe/top40.json --must-include BTCUSDT,ETHUSDT

# 美股合约：只看工作日美股盘中（UTC 13–20 点）
python scripts/01_data/kline_correlation.py --symbols TSLAUSDT,NVDAUSDT,AAPLUSDT,METAUSDT,GOOGLUSDT \
  --weekdays-only --utc-hours 13-20 --interval 1h --out-dir outputs/kline_corr_stocks

# 混合多个数据源（同名文件会显示为 data_hl/TSLAUSDT），检查同一标的在两个交易所是否重复
python scripts/01_data/kline_correlation.py --data-dir data --data-dir data_hl --symbols TSLAUSDT,NVDAUSDT
```

输出（`--out-dir`）：

| 文件 | 内容 |
|---|---|
| `heatmap_raw.png` / `heatmap_resid.png` | 热力图，按层次聚类排序，相似的币排在一起 |
| `corr_raw.csv` / `corr_resid.csv` / `overlap_bars.csv` | 相关矩阵，以及每对币共同拥有的 bar 数 |
| `high_corr_pairs.csv` / `high_corr_pairs_resid.csv` | `|corr| ≥ --pair-threshold` 的币对 |
| `symbol_stats.csv` | 覆盖率、年化波动、日成交额中位数、平均/最大相关性及对应的币、对 BTC 的相关性和 beta、聚类编号、滚动平均相关性的波动 |
| `rolling_avg_corr.csv` | 每 `--rolling-days` 天，每个币与其他币的平均相关性（检查相关性是否稳定） |
| `pool_decisions.csv` | 贪心选币对每个币的决定：`selected`、`correlated`（与哪个已选币相关）、`low_coverage`、`low_volume`、`pool_full` |
| `suggested_pool.json` | 建议币池，`{"symbols": [...]}`，格式与 `data/universe/top40.json` 相同 |

贪心选币：按 `--rank-by`（默认日成交额）从高到低遍历，一个币与已选的每个币 `|corr|` 都低于 `--max-corr`（默认 0.6，基于 `--select-on resid`）才保留；覆盖率低于 `--min-coverage` 或成交额低于 `--min-daily-quote-volume` 的直接跳过。

解读时注意：
- **resid 有负偏**：N 个币去均值后，独立的币之间也会出现约 −1/(N−1) 的负相关。币少的时候这个偏差很明显。
- **不要把美股和加密货币放进同一个截面**：两组之间 raw 相关性很低，去均值后会互相表现为明显的负相关（resid 热力图上会出现两个色块）。分开跑，各自建池。
- 美股合约的周末和盘后价格很薄，用 `--weekdays-only --utc-hours` 只看正常交易时段。上市时间短的币，覆盖率会低，可以用 `--start` 把窗口限制在共同存在的时期。
- `median_daily_quote_volume` 是"每根 bar 成交额中位数 × 分析窗口内每天的 bar 数"。加了时段过滤后，它表示的是该时段内的成交额。
- Pearson 容易被极端行情主导，可以用 `--method spearman` 对照。
