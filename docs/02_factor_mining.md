# 02 Factor mining

The LLM only receives the prompt and the field descriptions; it **never receives any market data**. It returns factor records in JSON, each with an economic hypothesis, a formula and a piece of `compute_factor(df)` code. That code is statically checked locally before it is executed in a sandbox.

```text
prompts (mining/prompts/crypto) ──► LLM ──► factor records ──► dedup / expression checks ──► factors.jsonl
                                                                         └─► rejected.jsonl (audit)
factors.jsonl ──► look-ahead audit ──► export predictions (03)
```

## Generation

```bash
export DEEPSEEK_API_KEY='...'

# Dry-run first: only show the prompts, no API calls
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

| Provider | `--provider` | Environment variables |
|---|---|---|
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY`, optional `DEEPSEEK_BASE_URL` |
| OpenAI | `openai` | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` |
| Grok / OpenAI-compatible relay | `grok` | `GROK_API_KEY`, `GROK_BASE_URL` (must support `POST /chat/completions` and JSON output) |

- **Blocks** (`--blocks`): `momentum reversal volatility volume liquidity price_volume microstructure intraday nonlinear style sentiment flow`, plus `funding_event`, which is dedicated to the funding-settlement-window label.
- **Prompt directory** (`--prompt-root`): defaults to `efficient_cryptos/mining/prompts/crypto`. DEX and Meme prompts are in the sibling `dex/` and `meme/` directories.
- **Resuming**: rerunning the same command skips completed batches. `--existing-jsonl` accepts previously generated files so the same formulas are not generated again.
- **What the date arguments do**: `--train-*` and `--validation-*` are only written into the prompt as text. The generator reads no data and never feeds any backtest results back to the model. `direction` (the factor's sign) is set by the model at generation time and is never changed based on backtest results.

Check the completion status of each batch:

```bash
python scripts/02_mining/generation_status.py --help
```

## Factor record format

```json
{"factor_id": "example_momentum_12", "block": "momentum", "direction": "positive",
 "hypothesis": "...", "formula": "close/lag(close,12)-1", "normalized_formula": "pct_change(close,12)",
 "required_columns": ["close", "symbol"], "lookback_bars": [12],
 "code": "def compute_factor(df):\n    return df['close'].groupby(df['symbol'], sort=False).pct_change(12)\n"}
```

`direction: negative` means a higher factor value is more bearish, so the prediction is multiplied by -1 on export. See `examples/crypto_factors.example.jsonl` for complete examples.

## Sandbox and look-ahead checks (`efficient_cryptos/factors/sandbox.py`)

When factor code runs, it sees a 30m cross-sectional table sorted by `[timestamp, symbol]` in which:
- labels (`target`, `raw_target`) and any field only known in the future (the next funding settlement time, funding within this row's holding period) are absent;
- apart from the raw OHLCV / `quote_volume`, derived features (such as `log_ret_1`, `funding_rate`, `realized_vol_12`) have already been z-scored per timestamp cross-section.

The static checker rejects the following:

| Category | Examples |
|---|---|
| Imports, files, network, loops, exception handling | `import os`, `open()`, `for`, `while`, `try` |
| Negative shifts | `shift(-1)`, `diff(-1)`, `pct_change(-1)` |
| Centered windows | `rolling(..., center=True)` |
| Full-history aggregation after grouping by symbol / date | `df.groupby("symbol")[x].transform("mean")`, `groupby("symbol").rank()`, `transform(lambda s: (s - s.mean()) / s.std())` |
| Whole-column full-sample statistics | `df[x] - df[x].mean()`, `df[x].rank()` |
| Reversed slices, backward fill, interpolation | `iloc[::-1]`, `fillna(method="bfill")`, `interpolate()` |
| Nonexistent columns | `df["target"]`, `df["funding_event_rate"]` |

Allowed: rolling or cumulative computations per symbol (`rolling`, `expanding`, `ewm`, `cum*`), and cross-sectional operations by `timestamp` (`groupby("timestamp")`).

Built-in helpers: `rank(x)` and `zscore(x)` are computed within each timestamp cross-section; `rolling_*` and `ddb_rolling_*` roll per symbol; `group_shift(df, col, n)` shifts per symbol; `safe_div` replaces ±inf with NaN.

## Auditing an existing factor library

```bash
python scripts/02_mining/audit_factor_lookahead.py \
  --factor-jsonl outputs/crypto_factors.jsonl --out-csv outputs/factor_lookahead_audit.csv
```

| `status` | Meaning |
|---|---|
| `leak` | Rejected by the checker (`leak_reason` gives the reason); earlier backtests contained future data, and export will fail |
| `changed` | Uses `share`, `rank()`, `zscore()` or `ddb_rolling_*`; these helpers were fixed, so predictions differ from the old version |
| `warning` | Applies `rolling/shift/diff/pct_change` directly to `df[col]` without grouping by symbol, so the computation crosses symbols |
| `ok` | None of the above |

## Extracting factor records from selection results

```bash
python scripts/02_mining/extract_selected_factor_records.py \
  --selected-dir factor_0_200 --selected-dir factor_200_400 \
  --factor-jsonl outputs/crypto_factors.jsonl --out-jsonl outputs/selected.jsonl
```
