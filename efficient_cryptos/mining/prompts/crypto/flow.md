# Flow Prompt

只使用 OHLCV 构造资金流代理，并预测未来 30 分钟收益或横截面收益排名。

优先方向：close location × volume、signed-volume proxy、money flow、资金流累计、流入/流出冲击、资金流与价格背离、流向持续性。

必须解释买卖方向如何从 OHLC 推断，明确这不是真实逐笔主动买卖数据。不得虚构资金流字段，不得和 volume、price_volume 因子重复。优先使用可缓存的 signed volume 基础节点，并给出宽表批量计算方案。

必须计算与已有 flow、volume、price_volume 因子的相关性，绝对相关性超过 0.80 默认拒绝。
