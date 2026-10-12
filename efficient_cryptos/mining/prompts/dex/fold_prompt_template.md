# Candidate schema and validation

Return JSON only, matching the requested schema.  `target_horizon` must be
`next_30min`.  `required_columns` must be a subset of the canonical DEX
columns.  `lookback_bars` must list positive integers.  Formula and code must
describe the same causal computation.  Reject a candidate if it needs a
field outside the schema, uses a negative shift, uses a centered rolling
window, or cannot remain finite on sparse pools.
