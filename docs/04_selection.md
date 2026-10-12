# 04 筛选与去重

## 扫描 Rust 结果

```bash
python scripts/04_selection/scan_rust_factors.py \
  --results-dir outputs/sweep_train_0_200 \
  --factor-jsonl outputs/crypto_factors.jsonl \
  --return-min 5 --sharpe-min 1 --max-drawdown-pct-min -30 \
  --sort-by sharpe \
  --out-csv outputs/selected_train.csv --out-markdown outputs/selected_train.md
```

脚本扫描 `--results-dir` 下所有 `*.metrics.json`（包括全部调仓周期子目录）。每个（因子, 周期）组合输出一行，并附上因子的经济假设、公式、所需字段和回看窗口。Markdown 报告逐个因子列出这些信息。

| 条件参数 | 字段 |
|---|---|
| `--return-min`、`--sharpe-min`、`--max-drawdown-pct-min`、`--pnl-positive` | 组合收益、Sharpe、最大回撤（回撤是负数，`-30` 表示回撤不超过 30%） |
| `--abs-q10-min`、`--abs-q1-max`、`--abs-q10-directional-min`、`--abs-q1-directional-max`、`--abs-q10-cum-pct-min` | 按绝对值分层的统计 |

使用这些条件时要注意：

- **按绝对值分层的指标与组合收益只是间接相关。** 这类指标按 |prediction| 分层，看的是信号最强或最弱的那组币，不区分方向；而组合按带符号的 prediction 做多做空。它们不宜作为主要门槛。
- **分层指标在六个调仓目录里完全相同。** 分层统计基于 30m 标签，与调仓频率无关；只有组合收益、Sharpe、回撤会随调仓频率变化。
- **只在训练期结果上筛选。** 测试期只出报告：把 `--results-dir` 换成测试期目录，不加任何条件。

## 相关性去重

```bash
python scripts/04_selection/analyze_factor_correlation.py \
  --predictions-dir outputs/pred_train_0_200 \
  --metrics-dir outputs/sweep_train_0_200/12h --score-field sharpe \
  --method spearman --threshold 0.90 --max-rows 200000 --min-overlap 1000 \
  --out-dir outputs/factor_corr_0_200
```

把所有 prediction 按 `timestamp + symbol` 对齐后计算相关系数，相关系数绝对值超过阈值的因子归为同一簇，每簇保留 `--score-field` 最高的那个。

输出：`spearman.csv`（完整矩阵）、`high_correlation_pairs.csv`、`clusters.csv`、`selected_factors.txt`、`summary.json`。

- 不要对净值曲线算相关：那样会把调仓频率、杠杆和成本也混进因子相似度里。
- `--max-rows` 对所有因子使用同一组确定性抽样的行，既控制内存，又不破坏因子之间的对齐。
- 建议先用 0.90 做一轮，再用 0.95 做更严格的去重。

## 多重检验

1 万个因子 × 6 个调仓周期，用固定门槛筛选必然产生大量假阳性。2 年 30m 数据上，Sharpe 估计的标准误约为 0.7，纯噪声因子约有 8% 的概率达到 Sharpe ≥ 1。

建议的做法：
1. 在训练期筛选；
2. 要求结果在不重叠的验证期（例如 2025-10-01 ~ 2025-12-31）上复现；
3. 结合 Deflated Sharpe 等方法，按测试的因子数量做校正；
4. 跨多个调仓周期都稳定的因子，比只在某一个周期表现最好的因子更可信（见 05 的 `rank_across_periods.py`）。
