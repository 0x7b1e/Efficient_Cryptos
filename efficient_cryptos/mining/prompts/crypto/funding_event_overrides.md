# Funding-event target override

本批候选不是普通的 `next_30min` 标签。所有记录必须填写：

```text
target_horizon: next_funding_event_excess_return
```

标签定义为：

```text
raw_target(t, symbol) = log(close(next_funding_time) / close(t))
target(t, symbol) = raw_target(t, symbol)
                       - cross_sectional_mean(raw_target(t, *))
```

`next_funding_time` 只用于本地标签构建，不得出现在 `compute_factor` 的输入或
`required_columns` 中。因子只能使用 t 及之前已发布的 `funding_rate` 和市场字段。
下一次资金费率事件通过本地 funding Parquet 的 `funding_time` 对齐，不得用未来
事件的数值构造当前因子。
