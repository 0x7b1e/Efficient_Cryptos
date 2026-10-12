# Sentiment Prompt

当前数据不包含新闻、公告、文本或社交媒体，只允许生成基于 OHLCV 的 sentiment proxy，并预测未来 30 分钟收益。

优先方向：连续上涨/下跌情绪、极端收益频率、正负收益成交量不对称、涨跌停接近程度、情绪恢复/恶化、正负 bar 的非对称性。

不得声称使用文本情绪或 NLP 得分，不得把普通 momentum 简单改名为 sentiment。必须在元数据中标注 proxy 局限，并与 momentum、reversal、flow 做严格结构和相关性去重。

