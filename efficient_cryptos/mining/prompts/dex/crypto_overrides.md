# DEX-specific overrides

Use `pool_id` as the grouping key, not `symbol` or `ticker`.  The data is
already sampled at the requested interval.  Allowed price fields are
`open`, `high`, `low`, `close`; allowed activity fields are `volume_usd`,
`tx_count`, `buy_volume_usd`, `sell_volume_usd`, `buy_tx_count`, and
`sell_tx_count`; `liquidity_usd` is the pool liquidity snapshot.  Do not
silently substitute CEX fields.  If a field is optional, guard division by
zero and return NaN where the denominator or history is unavailable.

Do not generate a factor that merely repeats `close.pct_change(n)` unless it
combines it with a DEX-specific activity or liquidity state.
