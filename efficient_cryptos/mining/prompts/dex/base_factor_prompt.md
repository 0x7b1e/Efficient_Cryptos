# DEX cross-sectional factor contract

You are generating executable Python/pandas factors for DEX pool data.  Each
factor must predict the next 30-minute cross-sectional return of a pool.  The
input is a long DataFrame sorted by `timestamp`, then `pool_id`; all rolling
operations must be grouped by `pool_id` and use current or historical rows
only.  Never use future rows, centered windows, network calls, imports, or
fitting on the target.

The canonical columns are:

`timestamp`, `chain`, `dex`, `pool_id`, `pool_address`, `base_token_address`,
`quote_token_address`, `base_symbol`, `quote_symbol`, `open`, `high`, `low`,
`close`, `volume_usd`, `liquidity_usd`, `tx_count`, `buy_volume_usd`,
`sell_volume_usd`, `buy_tx_count`, `sell_tx_count`.

`buy_*` and `sell_*` may be missing for OHLCV-only pools.  A factor using
optional flow fields must handle all-NaN inputs and return NaN rather than
inventing values.  A valid implementation has the exact shape and signature:

```python
def compute_factor(df):
    # return one numeric Series aligned with df.index
    ...
```

Use pandas/numpy vectorized operations only.  Prefer `groupby(...).transform`
or grouped rolling helpers.  Cross-sectional ranking is performed later by
the backtester, so the factor itself should not use future cross-sectional
information.
