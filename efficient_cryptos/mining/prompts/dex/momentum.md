# DEX momentum block

Explore short-horizon price continuation/reversal only when conditioned on
DEX activity or liquidity state.  Candidate formulas must be causal and
grouped by `pool_id`; avoid generating a large family of exact momentum
duplicates.
