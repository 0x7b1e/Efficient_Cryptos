# Codex 加密货币因子生成 Prompt 集

本目录专门用于 Binance 永续合约的横截面因子挖掘，不是股票日内 T0 Prompt。
这是 `scripts/02_mining/generate_factors.py` 的默认提示词目录（`--prompt-root` 可改为同级的 `dex/` 或 `meme/`）。

## 使用顺序

1. 读取 `base_factor_prompt.md`，作为每次任务的总控约束。
2. 读取对应板块 Prompt，例如 `momentum.md`。
3. 读取 `economic_principles.md`，确保每个候选有可证伪的交易机制。
4. 读取 `fold_prompt_template.md`，填入当前 Fold 的训练区间、验证区间和已有因子摘要。
5. 每批只要求生成固定数量的候选因子，并提交结构签名、字段依赖和验证结果。

计算引擎使用 Predict 的 Binance 长表（`symbol × timestamp`）和共享回测器；因子计算按 `symbol` 独立滚动，并在每个 `timestamp` 做横截面排序。

所有候选表达式（包括最终淘汰者）应通过根目录的 `factor_registry.py` 写入一个 Markdown 注册表，并为每个 `factor_id` 分配独立结果文件夹。

通过筛选的信号直接交给 `scripts/03_backtest/evaluate_generated_factors.py`，不依赖股票 PreLoader。

## 硬性目标

- 普通板块预测下一个约 30 分钟收益/相对收益；`funding_event` 预测到下一次真实 Binance `funding_time` 的市场中性超额收益。
- 任何因子都不得使用未来数据。
- 不允许只通过改变窗口或正负号重复已有因子。
- 生成前必须检查公式签名和历史因子相关性。
- 因子必须可以在 `ticker × barTime` 宽表上向量化计算。
- 优先生成低复杂度、可批量执行、可缓存中间结果的表达式。
- 训练、验证、最终 OOS 必须按时间切分，不能用验证/OOS 反向修改公式。
- 普通板块的预测目标固定为下一个约 30 分钟的市场中性超额收益；`funding_event` 使用下一次真实资金费率结算事件作为标签终点。
- 只允许使用 Binance 数据中的 `symbol`、`timestamp`、OHLCV、`quote_volume`、`funding_rate` 和已下载的持仓字段。
