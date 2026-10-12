# 11 DEX 因子

DEX 因子目前只完成到"生成公式"这一步：

```bash
python scripts/11_dex/generate_dex_factors.py --dry-run --blocks flow liquidity microstructure volume momentum --total-candidates 20

export DEEPSEEK_API_KEY=...
python scripts/11_dex/generate_dex_factors.py --provider deepseek --model deepseek-chat \
  --blocks flow liquidity microstructure volume momentum \
  --fold-id dex_30m_v1 --train-start 20240101 --train-end 20251231 \
  --validation-start 20260101 --validation-end 20261001 \
  --total-candidates 500 --batches 10 --concurrency 2 \
  --output-jsonl outputs/dex_factors.jsonl --rejected-jsonl outputs/dex_factors.rejected.jsonl \
  --registry-md outputs/dex_factor_registry.md --result-root outputs/dex_factor_results
```

DEX 提示词（`efficient_cryptos/mining/prompts/dex/`）只允许使用 DEX 字段，要求按 `pool_id` 分组、使用因果滚动，目标为 `next_30min`。

## 尚未完成的部分

- **数据下载**：原 Predict_Cryptos 中的 `download_dex_data.py`、`find_dex_pools.py`、`stream_download_upload_dex.py` 依赖 `dex_data.gecko_terminal` 模块，但这个模块从未提交到仓库，脚本无法运行，所以没有迁移过来。如果你本地有 `dex_data/`，补充提交后可以恢复这三个脚本。
- **接入回测**：生成的 DEX 因子还不能直接用 Binance 的回测流程。需要先把各个池子聚合到同一时间粒度，建立池子级别的特征缓存，并确认每个时间点有足够多的池子有有效数据，再接入 Rust 截面回测。
