# Binance perpetual-futures override

This is a crypto cross-sectional factor task, not an A-share intraday task.

Use only columns that exist in `Predict_Cryptos`:

`symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`,
`funding_rate`, `log_ret_1`, and the downloaded Binance positioning columns.

Every ordinary candidate must predict `next_30min` market-neutral excess return.
The `funding_event` block is the only exception and must predict
`next_funding_event_excess_return`, as specified by its block override. It must
define exactly one `compute_factor(df)` and return a Series with the same row
index. Compute rolling values independently by `symbol`; never use future
values, `shift(-n)`, centered windows, random data, file access, or network
calls. Cross-sectional normalization may be done at each `timestamp`.

The generated code may use the following aliases for compatibility with the
original Dpsk validator:

```text
closePrice=open/close/high/low mapped to close/open/high/low
totalVolume=volume
totalValue=quote_volume
chgPct=log_ret_1
ticker=symbol
```

Do not use A-share-only concepts such as复权因子、中证成分、交易日、14:55 平仓 or
`turnoverRate` as if they were native crypto fields. If a liquidity proxy is
needed, use `quote_volume` explicitly and document the proxy in the hypothesis.
