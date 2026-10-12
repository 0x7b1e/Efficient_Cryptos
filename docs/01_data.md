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
