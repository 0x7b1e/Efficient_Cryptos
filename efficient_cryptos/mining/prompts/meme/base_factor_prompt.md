# Meme event factor contract

Generate causal, auditable short-horizon factors for Binance USD-M meme-event
rows. Return one JSON object with a `factors` array. Every factor must contain:
`factor_id`, `block`, `name`, `hypothesis`, `target_horizon`, `direction`,
`required_columns`, `lookback_bars`, `formula`, `normalized_formula`,
`complexity`, `reusable_nodes`, `leakage_check`, `code`, `status`.

`code` must define exactly `compute_factor(df)`, accept one pandas DataFrame,
and return one Series with the same index. Use only causal rolling/groupby
operations. No imports, loops, file/network access, negative shifts or future
labels. The factor is evaluated after an event detector, so it may use event
features but not `future_ret_*` columns.
