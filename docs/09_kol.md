# 09 KOL 社交因子

把 X/Twitter 白名单账号的推文转成一个 30m 截面因子。这个模块与行情和 DPSK 因子流程完全隔离：

```text
twscrape 账号池 → 白名单增量采集 → SQLite 原始推文库 → 保守的币种识别（BTC/ETH/SOL…）
  → DeepSeek 打标（JSON）→ KOL 权重 × 互动量 × 时间衰减 → 30m 因子 parquet → 截面回测
```

安装：`pip install -e ".[kol]"`（twscrape 0.20.x）。

## 配置

- `config/kol/kol_whitelist.yaml`：白名单账号、权重和标签。
- `config/kol/entity_map.yaml`：币种别名。只添加不会产生歧义的别名。

账号用 cookie 登录：

```python
from twscrape import API
api = API("data/kol_factor/twscrape_accounts.db")
await api.pool.add_account_cookies("account_1", "auth_token=...; ct0=...")
```

不要把 cookie、密码、API key 或账号数据库提交到 Git。

## 运行

```bash
# 每 5–30 分钟运行一次；增量采集，重复运行不会产生重复数据
python scripts/09_kol/collect_once.py \
  --accounts-db data/kol_factor/twscrape_accounts.db --db data/kol_factor/tweets.sqlite \
  --config config/kol/kol_whitelist.yaml --entity-map config/kol/entity_map.yaml

# --db 必须与采集时相同（两个脚本的默认值不同，所以显式传入）
python scripts/09_kol/build_factor.py --db data/kol_factor/tweets.sqlite --config config/kol/kol_whitelist.yaml --out data/features/kol_30m.parquet
```

采集游标保存在原始推文库的 `metadata` 表里，推文按 tweet id 去重写入。查询形如 `from:handle -is:retweet since:...`；twscrape 相关的细节都封装在 `efficient_cryptos/kol/collector.py` 中。

## 打标与输出

每个（推文, 币种）组合，由大模型返回有界的 JSON：

```json
{"sentiment": -1.0, "conviction": 0.0, "novelty": 0.0, "hype": 0.0, "relevance": 0.0, "rationale": "..."}
```

返回值会先校验并截断到合法范围。只有 `relevance` 达到阈值的记录才会计入因子。

输出列：`timestamp,symbol,kol_factor,kol_abs_factor,mention_count,positive_score,negative_score`。
- `kol_factor` 带符号，用于决定做多还是做空；
- `kol_abs_factor` 表示信号强度，用于按绝对值分层的诊断。

这是一个研究用的适配器，不负责账号管理，也不保证不被限频。请遵守 X 的条款，使用规模小、经过授权的账号池。
