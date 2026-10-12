# Fold Prompt：时间序列 Fold 任务模板

当前任务必须遵守 `base_factor_prompt.md` 和指定板块 Prompt。

## 当前 Fold

```text
fold_id: <fold_id>
train_start: <YYYYMMDD>
train_end: <YYYYMMDD>
validation_start: <YYYYMMDD>
validation_end: <YYYYMMDD>
test_policy: 2025 仅最终 OOS，不得用于公式修改
target_horizon: next_30min（`funding_event` 板块使用 `next_funding_event_excess_return`）
```

## 本批生成配置

```text
block: <block>
batch_id: <batch_id>
candidate_count: <N>
max_complexity: 4
max_operator_count: 8
correlation_reject_threshold: 0.80
```

## 已有因子信息

下面的信息用于避免重复，不得用于对验证集做反向拟合：

```text
existing_factor_ids: <...>
existing_normalized_formulas: <...>
existing_similarity_groups: <...>
high_correlation_pairs: <...>
```

## 任务要求

1. 只使用训练区间设计和解释因子。
2. 普通板块针对未来 30 分钟收益；`funding_event` 针对当前 bar 到下一次真实 funding_time 的收益或横截面收益排名。
3. 对每个候选因子先做公式结构去重，再做训练区间相关性筛选。
4. 禁止生成仅改变窗口、正负号或常数的变体。
5. 每个因子必须给出宽表向量化计算方案和可复用中间节点。
6. 验证区间只能用于评估，不能用于修改公式。
7. 最终按 JSON 输出，不要输出无法解析的自然语言包装。
