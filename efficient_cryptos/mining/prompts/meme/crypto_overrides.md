# Meme event columns

Allowed causal columns include:
`symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`,
`quote_volume`, `trades`, `taker_buy_quote`, `funding_rate`, `log_ret_1m`,
`ret_3m`, `ret_5m`, `ret_10m`, `momentum_accel_1m`, `realized_vol_10m`,
`quote_volume_5m`, `quote_volume_10m`, `quote_volume_baseline_60m`,
`volume_shock_5m`, `volume_shock_10m`, `trade_count_5m`,
`trade_count_baseline_60m`, `trade_intensity_shock`, `taker_buy_ratio`,
`order_flow_imbalance`, `ofi_5m`, `event_score`, `event_percentile`,
and timestamp-safe social attention fields.

All factors target an event-conditioned forward return. Never use
`future_ret_*`, future event flags, or data collected after the factor time.
