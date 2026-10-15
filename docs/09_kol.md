# 09 KOL social factor

Turns tweets from allow-listed X/Twitter accounts into a 30m cross-sectional factor. This module is fully isolated from the market data and DPSK factor pipelines:

```text
twscrape account pool → incremental allow-list collection → SQLite raw tweet store → conservative symbol recognition (BTC/ETH/SOL…)
  → DeepSeek labeling (JSON) → KOL weight × engagement × time decay → 30m factor parquet → cross-sectional backtest
```

Install: `pip install -e ".[kol]"` (twscrape 0.20.x).

## Configuration

- `config/kol/kol_whitelist.yaml`: allow-listed accounts, weights and tags.
- `config/kol/entity_map.yaml`: symbol aliases. Only add aliases that cannot be ambiguous.

Accounts log in with cookies:

```python
from twscrape import API
api = API("data/kol_factor/twscrape_accounts.db")
await api.pool.add_account_cookies("account_1", "auth_token=...; ct0=...")
```

Never commit cookies, passwords, API keys or the account database to Git.

## Running

```bash
# Run every 5–30 minutes; collection is incremental, so reruns produce no duplicates
python scripts/09_kol/collect_once.py \
  --accounts-db data/kol_factor/twscrape_accounts.db --db data/kol_factor/tweets.sqlite \
  --config config/kol/kol_whitelist.yaml --entity-map config/kol/entity_map.yaml

# --db must match the one used for collection (the two scripts have different defaults, so pass it explicitly)
python scripts/09_kol/build_factor.py --db data/kol_factor/tweets.sqlite --config config/kol/kol_whitelist.yaml --out data/features/kol_30m.parquet
```

The collection cursor is stored in the `metadata` table of the raw tweet store, and tweets are deduplicated by tweet id on write. Queries look like `from:handle -is:retweet since:...`; all twscrape details are encapsulated in `efficient_cryptos/kol/collector.py`.

## Labeling and output

For each (tweet, symbol) pair the LLM returns bounded JSON:

```json
{"sentiment": -1.0, "conviction": 0.0, "novelty": 0.0, "hype": 0.0, "relevance": 0.0, "rationale": "..."}
```

Returned values are validated and clipped to their legal ranges first. Only records whose `relevance` reaches the threshold count toward the factor.

Output columns: `timestamp,symbol,kol_factor,kol_abs_factor,mention_count,positive_score,negative_score`.
- `kol_factor` is signed and decides long vs short;
- `kol_abs_factor` is signal strength, used for absolute-value layer diagnostics.

This is a research adapter: it does not manage accounts and makes no guarantee about avoiding rate limits. Follow X's terms and use a small, authorized account pool.
