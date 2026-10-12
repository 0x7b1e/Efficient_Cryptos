# 02 因子挖掘

大模型只拿到提示词和字段说明，**不会收到任何行情数据**。它返回 JSON 格式的因子记录，其中包括经济假设、公式和一段 `compute_factor(df)` 代码。这段代码在本地经过静态检查后，才会在沙箱里执行。

```text
提示词（mining/prompts/crypto）──► 大模型 ──► factor 记录 ──► 去重 / 表达式检查 ──► factors.jsonl
                                                                    └─► rejected.jsonl（审计）
factors.jsonl ──► 未来函数审计 ──► 导出 prediction（03）
```

## 生成

```bash
export DEEPSEEK_API_KEY='...'

# 先 dry-run，只看提示词，不调用 API
python scripts/02_mining/generate_factors.py --dry-run --blocks momentum reversal --candidate-count 2

python -u scripts/02_mining/generate_factors.py \
  --provider deepseek --model deepseek-chat \
  --blocks momentum reversal volatility volume liquidity price_volume \
  --fold-id crypto_2024_2025 \
  --train-start 20240101 --train-end 20251231 \
  --validation-start 20260101 --validation-end 20261001 \
  --total-candidates 10000 --batches 100 --concurrency 4 \
  --output-jsonl outputs/crypto_factors.jsonl \
  --rejected-jsonl outputs/crypto_factors.rejected.jsonl \
  --registry-md outputs/crypto_factor_registry.md \
  --result-root outputs/crypto_factor_results
```

| 提供方 | `--provider` | 环境变量 |
|---|---|---|
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY`，可选 `DEEPSEEK_BASE_URL` |
| OpenAI | `openai` | `OPENAI_API_KEY`，可选 `OPENAI_BASE_URL` |
| Grok / 兼容 OpenAI 接口的中转 | `grok` | `GROK_API_KEY`、`GROK_BASE_URL`（需支持 `POST /chat/completions` 和 JSON 输出） |

- **区块**（`--blocks`）：`momentum reversal volatility volume liquidity price_volume microstructure intraday nonlinear style sentiment flow`，另有专用于资金费结算窗口标签的 `funding_event`。
- **提示词目录**（`--prompt-root`）：默认是 `efficient_cryptos/mining/prompts/crypto`。DEX 和 Meme 的提示词在同级的 `dex/`、`meme/` 目录。
- **断点续跑**：重复运行同一条命令会跳过已完成的 batch。`--existing-jsonl` 可以传入以前生成过的文件，避免重复生成相同的公式。
- **日期参数的作用**：`--train-*`、`--validation-*` 只作为文本写进提示词，生成器不会读取数据，也不会把任何回测结果反馈给模型。`direction`（因子方向）由模型在生成时给定，不会根据回测结果修改。

查看各 batch 的完成情况：

```bash
python scripts/02_mining/generation_status.py --help
```

## 因子记录格式

```json
{"factor_id": "example_momentum_12", "block": "momentum", "direction": "positive",
 "hypothesis": "...", "formula": "close/lag(close,12)-1", "normalized_formula": "pct_change(close,12)",
 "required_columns": ["close", "symbol"], "lookback_bars": [12],
 "code": "def compute_factor(df):\n    return df['close'].groupby(df['symbol'], sort=False).pct_change(12)\n"}
```

`direction: negative` 表示因子值越高越看空，导出时 prediction 会乘以 -1。完整示例见 `examples/crypto_factors.example.jsonl`。

## 沙箱与未来函数检查（`efficient_cryptos/factors/sandbox.py`）

因子代码执行时看到的是一张按 `[timestamp, symbol]` 排序的 30m 截面表，其中：
- 不包含标签（`target`、`raw_target`）和任何未来才知道的字段（下一次资金费结算时间、本行持有期内的资金费）；
- 除原始的 OHLCV / `quote_volume` 外，派生特征（如 `log_ret_1`、`funding_rate`、`realized_vol_12`）都已按时间截面做过 z-score。

静态检查会拒绝以下写法：

| 类别 | 例子 |
|---|---|
| 导入、文件、网络、循环、异常处理 | `import os`、`open()`、`for`、`while`、`try` |
| 负向位移 | `shift(-1)`、`diff(-1)`、`pct_change(-1)` |
| 居中窗口 | `rolling(..., center=True)` |
| 按币种 / 日期分组后对全历史聚合 | `df.groupby("symbol")[x].transform("mean")`、`groupby("symbol").rank()`、`transform(lambda s: (s - s.mean()) / s.std())` |
| 整列全样本统计 | `df[x] - df[x].mean()`、`df[x].rank()` |
| 倒序切片、向后填充、插值 | `iloc[::-1]`、`fillna(method="bfill")`、`interpolate()` |
| 不存在的列 | `df["target"]`、`df["funding_event_rate"]` |

允许的写法：按币种做滚动或累积计算（`rolling`、`expanding`、`ewm`、`cum*`），以及按 `timestamp` 做截面运算（`groupby("timestamp")`）。

内置辅助函数：`rank(x)`、`zscore(x)` 在每个时间截面内计算；`rolling_*`、`ddb_rolling_*` 按币种滚动；`group_shift(df, col, n)` 按币种位移；`safe_div` 把 ±inf 替换为 NaN。

## 审计已有因子库

```bash
python scripts/02_mining/audit_factor_lookahead.py \
  --factor-jsonl outputs/crypto_factors.jsonl --out-csv outputs/factor_lookahead_audit.csv
```

| `status` | 含义 |
|---|---|
| `leak` | 检查器拒绝（`leak_reason` 写明原因）；以前的回测含有未来数据，导出时会失败 |
| `changed` | 用到了 `share`、`rank()`、`zscore()`、`ddb_rolling_*`；这些辅助函数修正过，预测值与旧版不同 |
| `warning` | 对 `df[列]` 直接做 `rolling/shift/diff/pct_change` 而没有按币种分组，会在不同币种之间计算 |
| `ok` | 以上都不是 |

## 从筛选结果提取因子记录

```bash
python scripts/02_mining/extract_selected_factor_records.py \
  --selected-dir factor_0_200 --selected-dir factor_200_400 \
  --factor-jsonl outputs/crypto_factors.jsonl --out-jsonl outputs/selected.jsonl
```
