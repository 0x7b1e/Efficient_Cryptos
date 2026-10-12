# Microstructure Prompt

当前没有真实盘口或逐笔数据，只能使用 OHLCV 构造分钟级微观结构代理，并预测未来 30 分钟收益。

优先围绕“订单流失衡导致短期价格冲击、冲击斜率受流动性约束”的机制构造代理；不得把代理的统计相关性描述成真实盘口因果。

优先方向：close location、实体/振幅、上下影线、连续 bar、收益自相关、价格冲击后的成交量响应、跳跃代理。

必须明确这是 OHLCV proxy，禁止声称是真实 bid/ask 或主动买卖方向。避免与 price_volume、flow 的公式重复。优先生成可以在宽表上通过少量 rolling 和基础节点复用的表达式。

必须计算与已有 microstructure、price_volume、flow 因子的相关性，绝对相关性超过 0.80 时默认不保留。
