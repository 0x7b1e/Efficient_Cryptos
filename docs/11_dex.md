# 11 DEX factors

DEX factors currently only go as far as "generating formulas":

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

The DEX prompts (`efficient_cryptos/mining/prompts/dex/`) only allow DEX fields, require grouping by `pool_id` and causal rolling windows, and target `next_30min`.

## Not yet done

- **Data download**: `download_dex_data.py`, `find_dex_pools.py` and `stream_download_upload_dex.py` in the original Predict_Cryptos depend on the `dex_data.gecko_terminal` module, which was never committed to that repository, so the scripts could not run and were not migrated. If you have `dex_data/` locally, committing it would let these three scripts be restored.
- **Backtest integration**: generated DEX factors cannot yet use the Binance backtest pipeline directly. Pools first need to be aggregated to a common time granularity, a pool-level feature cache built, and enough pools with valid data at each timestamp confirmed, before plugging into the Rust cross-sectional backtest.
