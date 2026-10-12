# Liquidity Prompt

只研究 Binance 合约的交易成本、价格冲击和流动性状态对未来 30 分钟收益预测和可交易性的影响。

必须把流动性指标解释为交易成本或市场深度的代理，并同时报告因子在高/低流动性合约中的稳定性和容量含义。

优先方向：Amihud 代理、单位 quote volume 价格变动、流动性恶化、成交额冲击、波动/成交额。

只允许使用实际存在的 Binance OHLCV、`quote_volume`、资金费和 positioning 字段。不得虚构 bid/ask/spread、换手率或市值。必须区分 liquidity 与 volume，除 IC 外还要报告换手和极端值比例。

每个候选必须计算与已有因子的训练区间相关性；绝对相关性超过 0.80 默认拒绝，并记录淘汰原因。
