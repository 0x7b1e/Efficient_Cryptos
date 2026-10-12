"""Factor leaderboard: Rust sweep metrics, cross-period stability, train vs test, factor detail."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from _shared import DOWN, OUTPUTS, UP, fmt_num, path_picker, rel
from efficient_cryptos.dashboard import data as d


# ------------------------------------------------------------------ cache
@st.cache_data(show_spinner="读取回测结果…", max_entries=8)
def _scan(root: str, signature: tuple) -> pd.DataFrame:
    return d.scan_sweep(root)


@st.cache_data(show_spinner=False, max_entries=4)
def _meta(path: str, mtime: float) -> pd.DataFrame:
    return d.load_factor_meta(path)


@st.cache_data(show_spinner=False, ttl=600)
def _roots() -> list[Path]:
    # Most recently updated first, so a fresh sweep is selected by default.
    return sorted(d.find_sweep_roots(OUTPUTS), key=lambda p: d.dir_signature(p, "*")[1], reverse=True)


@st.cache_data(show_spinner=False, max_entries=64)
def _ts(path: str, mtime: float) -> pd.DataFrame:
    return d.load_timeseries(path)


@st.cache_data(show_spinner=False, max_entries=64)
def _layers(path: str, mtime: float):
    return d.layer_summary(path)


def _height(rows: int) -> int:
    return int(min(440, 36 * (min(rows, 2000) + 1) + 4))


def _mtime(path) -> float:
    try:
        return Path(path).stat().st_mtime
    except (OSError, TypeError):
        return 0.0


def _load(root: Path | None) -> pd.DataFrame:
    if root is None or not root.exists():
        return pd.DataFrame()
    return _scan(str(root), d.dir_signature(root, "*"))


# ------------------------------------------------------------------ sidebar
st.title("因子排行榜")
if st.sidebar.button("重新扫描目录", help="新的回测跑完后点一下"):
    st.cache_data.clear()
roots = _roots()
train_root = path_picker("回测结果目录", roots, "train",
                         help="sweep_rust_rebalance.sh 的 --output-root，里面是 1h/2h/… 子目录")
test_root = path_picker("对照期目录（可选）", [r for r in roots if r != train_root], "test", optional=True,
                        help="另一个时间段（例如 2026 测试期）的 sweep，用来检查过拟合")
jsonl_candidates = sorted(OUTPUTS.glob("**/*.jsonl"))[:200] if OUTPUTS.exists() else []
jsonl_candidates = [p for p in jsonl_candidates if "audit" not in p.name and "error" not in p.name]
meta_path = path_picker("因子 JSONL（可选）", jsonl_candidates, "meta", optional=True,
                        help="显示公式、分类和经济假设")

train = _load(train_root)
if train.empty:
    st.info(
        "没有找到回测结果。先跑完 Rust 回测：\n\n"
        "```bash\nbash scripts/03_backtest/sweep_rust_rebalance.sh \\\n"
        "  --predictions-dir outputs/pred_train_0_4000 --output-root outputs/sweep_train --jobs 8 --no-plots\n```\n"
        "然后在左侧选择 `outputs/sweep_train`。"
    )
    st.stop()
test = _load(test_root) if test_root else pd.DataFrame()
has_test = not test.empty
frame = d.join_periods(train, test if has_test else None)
meta = _meta(str(meta_path), _mtime(meta_path)) if meta_path else pd.DataFrame()
if not meta.empty:
    frame = frame.merge(meta[["factor_id", "block", "direction"]], on="factor_id", how="left")

# ------------------------------------------------------------------ filters
rebalances = d.sort_rebalances(frame["rebalance"])
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
    chosen = c1.multiselect("调仓周期", rebalances, default=rebalances)
    min_sharpe = c2.number_input("Sharpe ≥", value=0.0, step=0.5)
    min_return = c3.number_input("收益% ≥", value=0.0, step=5.0)
    min_dd = c4.number_input("最大回撤% ≥", value=-100.0, step=5.0, help="回撤是负数，例如 -30 表示不超过 30%")
    search = c5.text_input("搜索因子 / 分类", placeholder="volume, liquidity_0041 …")
    t1, t2, t3 = st.columns(3)
    best_only = t1.toggle("每个因子只看最佳周期", value=True)
    need_test = t2.toggle("测试期也盈利", value=False, disabled=not has_test)
    blocks = sorted(x for x in frame.get("block", pd.Series(dtype=str)).dropna().unique() if x)
    chosen_blocks = t3.multiselect("分类", blocks) if blocks else []

view = frame[frame["rebalance"].isin(chosen)]
view = view[(view["sharpe"].fillna(-np.inf) >= min_sharpe) & (view["return_pct"].fillna(-np.inf) >= min_return)
            & (view["max_drawdown_pct"].fillna(-np.inf) >= min_dd)]
if search.strip():
    needle = search.strip().lower()
    hay = view["factor_id"].str.lower()
    if "block" in view:
        hay = hay + " " + view["block"].fillna("").str.lower()
    view = view[hay.str.contains(needle, regex=False)]
if chosen_blocks:
    view = view[view["block"].isin(chosen_blocks)]
if need_test and has_test:
    view = view[view["test_return_pct"] > 0]
if best_only:
    view = d.best_per_factor(view)

total_factors = frame["factor_id"].nunique()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("因子数", f"{total_factors:,}")
k2.metric("回测组合（因子×周期）", f"{len(frame):,}")
k3.metric("盈利组合占比", fmt_num((frame["return_pct"] > 0).mean() * 100, 1, "%"))
k4.metric("Sharpe 中位数", fmt_num(frame["sharpe"].median(), 2))
k5.metric("筛选后", f"{len(view):,}")

tabs = ["排行榜", "跨周期稳定性"] + (["训练 vs 测试"] if has_test else [])
tab_objs = st.tabs(tabs)

# ------------------------------------------------------------------ leaderboard
with tab_objs[0]:
    columns = ["factor_id", "rebalance", "return_pct", "sharpe", "max_drawdown_pct", "avg_turnover",
               "total_cost", "cost_pct_of_equity", "cost_share_of_gross_pct", "signed_q10_minus_q1"]
    if has_test:
        columns += ["test_return_pct", "test_sharpe", "test_max_drawdown_pct"]
    if "block" in view:
        columns.insert(1, "block")
    table = view[columns].reset_index(drop=True)
    config = {
        "factor_id": st.column_config.TextColumn("因子", width="medium"),
        "block": st.column_config.TextColumn("分类", width="small"),
        "rebalance": st.column_config.TextColumn("周期", width="small"),
        "return_pct": st.column_config.NumberColumn("收益%", format="%.2f"),
        "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "max_drawdown_pct": st.column_config.NumberColumn("最大回撤%", format="%.2f"),
        "avg_turnover": st.column_config.NumberColumn("平均换手", format="%.3f"),
        "total_cost": st.column_config.NumberColumn("总成本", format="%.1f"),
        "cost_pct_of_equity": st.column_config.NumberColumn("成本占本金%", format="%.2f",
                                                            help="手续费 + 滑点 / 初始资金"),
        "cost_share_of_gross_pct": st.column_config.NumberColumn(
            "成本吃掉毛收益%", format="%.0f",
            help="手续费 + 滑点 / 扣成本前的收益。毛收益为负时留空。超过 50% 说明换手太高，换成更慢的调仓周期试试"),
        "signed_q10_minus_q1": st.column_config.NumberColumn("Q10−Q1 (每bar)", format="%.2e"),
        "test_return_pct": st.column_config.NumberColumn("测试收益%", format="%.2f"),
        "test_sharpe": st.column_config.NumberColumn("测试 Sharpe", format="%.2f"),
        "test_max_drawdown_pct": st.column_config.NumberColumn("测试回撤%", format="%.2f"),
    }
    st.caption("点击一行查看下方的因子详情。表格最多显示 2,000 行，可下载完整结果。")
    event = st.dataframe(table.head(2000), column_config=config, hide_index=True, width="stretch",
                         height=_height(len(table)), on_select="rerun", selection_mode="single-row", key="board")
    st.download_button("下载筛选结果 CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name="factor_leaderboard.csv", mime="text/csv")
    picked_rows = event.selection.rows if event and event.selection else []
    if picked_rows:
        pick = (table.iloc[picked_rows[0]]["factor_id"], table.iloc[picked_rows[0]]["rebalance"])
        if st.session_state.get("last_pick") != pick:
            # Drive the detail widgets below (they keep their own state otherwise).
            st.session_state["last_pick"] = pick
            st.session_state["factor_pick"], st.session_state["rebalance_pick"] = pick
            st.session_state["detail_factor"], st.session_state["detail_reb"] = pick

# ------------------------------------------------------------------ stability
with tab_objs[1]:
    stab = d.stability_table(frame[frame["rebalance"].isin(chosen)])
    if not stab.empty:
        st.caption("同一个因子在不同调仓周期下是否都赚钱。周期之间结论一致的因子，比只在某个周期表现好的因子更可信。")
        st.dataframe(stab.head(2000), hide_index=True, width="stretch", height=_height(len(stab)), column_config={
            "factor_id": st.column_config.TextColumn("因子", width="large"),
            "periods": "周期数", "positive_periods": "盈利周期数",
            "mean_return_pct": st.column_config.NumberColumn("平均收益%", format="%.2f"),
            "median_sharpe": st.column_config.NumberColumn("Sharpe 中位数", format="%.2f"),
            "min_sharpe": st.column_config.NumberColumn("最差 Sharpe", format="%.2f"),
            "worst_drawdown_pct": st.column_config.NumberColumn("最差回撤%", format="%.2f"),
        })

# ------------------------------------------------------------------ train vs test
if has_test:
    with tab_objs[2]:
        both = frame[frame["rebalance"].isin(chosen)].dropna(subset=["sharpe", "test_sharpe"])
        if both.empty:
            st.info("两个目录里没有同名的因子和周期。")
        else:
            agree = ((both["sharpe"] > 0) & (both["test_sharpe"] > 0)).mean() * 100
            corr = both[["sharpe", "test_sharpe"]].corr(method="spearman").iloc[0, 1]
            m1, m2, m3 = st.columns(3)
            m1.metric("组合数", f"{len(both):,}")
            m2.metric("两期 Sharpe 都为正", fmt_num(agree, 1, "%"))
            m3.metric("Sharpe 排名相关（Spearman）", fmt_num(corr, 2),
                      help="接近 0 说明训练期排名对测试期几乎没有预测力，排行榜靠前的多半是过拟合")
            sample = both.sample(min(len(both), 6000), random_state=0) if len(both) > 6000 else both
            fig = go.Figure()
            for reb in d.sort_rebalances(sample["rebalance"]):
                part = sample[sample["rebalance"] == reb]
                fig.add_trace(go.Scattergl(x=part["sharpe"], y=part["test_sharpe"], mode="markers", name=reb,
                                           marker={"size": 5, "opacity": 0.6}, text=part["factor_id"],
                                           hovertemplate="%{text}<br>训练 %{x:.2f} / 测试 %{y:.2f}<extra>" + reb + "</extra>"))
            lo = float(np.nanmin([sample["sharpe"].min(), sample["test_sharpe"].min()]))
            hi = float(np.nanmax([sample["sharpe"].max(), sample["test_sharpe"].max()]))
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dot", "color": "gray"})
            fig.add_hline(y=0, line_color="gray", line_width=1)
            fig.add_vline(x=0, line_color="gray", line_width=1)
            fig.update_layout(height=480, xaxis_title="训练期 Sharpe", yaxis_title="测试期 Sharpe",
                              margin={"l": 10, "r": 10, "t": 30, "b": 10})
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ detail
st.divider()
st.subheader("因子详情")
factor_ids = view["factor_id"].drop_duplicates().tolist() or frame["factor_id"].drop_duplicates().tolist()
default = st.session_state.get("factor_pick")
if default not in factor_ids:
    default = factor_ids[0]
if st.session_state.get("detail_factor") not in factor_ids:
    st.session_state["detail_factor"] = default
factor = st.selectbox("因子", factor_ids, key="detail_factor")
rows = frame[frame["factor_id"] == factor].copy()
rows = rows.assign(_order=rows["rebalance"].map(d.rebalance_hours)).sort_values("_order")

if not meta.empty and factor in set(meta["factor_id"]):
    info = meta.set_index("factor_id").loc[factor]
    cols = st.columns(3)
    cols[0].markdown(f"**分类**　{info.get('block') or '—'}")
    cols[1].markdown(f"**方向**　{info.get('direction') or '—'}")
    cols[2].markdown(f"**名称**　{info.get('name') or '—'}")
    if info.get("hypothesis"):
        st.markdown(f"**经济假设**　{info['hypothesis']}")
    formula = info.get("normalized_formula") or info.get("formula")
    if formula:
        st.code(formula, language="python")

per = rows[["rebalance", "return_pct", "sharpe", "max_drawdown_pct", "avg_turnover"]
           + (["test_return_pct", "test_sharpe", "test_max_drawdown_pct"] if has_test else [])]
st.dataframe(per, hide_index=True, width="stretch", column_config={
    "rebalance": "周期",
    "return_pct": st.column_config.NumberColumn("收益%", format="%.2f"),
    "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
    "max_drawdown_pct": st.column_config.NumberColumn("最大回撤%", format="%.2f"),
    "avg_turnover": st.column_config.NumberColumn("平均换手", format="%.3f"),
    "test_return_pct": st.column_config.NumberColumn("测试收益%", format="%.2f"),
    "test_sharpe": st.column_config.NumberColumn("测试 Sharpe", format="%.2f"),
    "test_max_drawdown_pct": st.column_config.NumberColumn("测试回撤%", format="%.2f"),
})

reb_options = rows["rebalance"].tolist()
reb_default = st.session_state.get("rebalance_pick")
if st.session_state.get("detail_reb") not in reb_options:
    st.session_state["detail_reb"] = reb_default if reb_default in reb_options else reb_options[0]
reb = st.radio("查看周期", reb_options, horizontal=True, key="detail_reb")
row = rows[rows["rebalance"] == reb].iloc[0]

left, right = st.columns([3, 2])
with left:
    ts = _ts(row["timeseries_path"], _mtime(row["timeseries_path"]))
    test_ts = (_ts(row["test_timeseries_path"], _mtime(row["test_timeseries_path"]))
               if has_test and isinstance(row.get("test_timeseries_path"), str) else pd.DataFrame())
    if ts.empty:
        st.info("没有 timeseries.csv（回测时是否用了 --timeseries-output？）")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=ts["timestamp"], y=ts["equity"], name="训练期净值", line={"width": 1.5}), 1, 1)
        fig.add_trace(go.Scatter(x=ts["timestamp"], y=ts["drawdown"] * 100, name="训练期回撤%",
                                 fill="tozeroy", line={"width": 0.8, "color": DOWN}), 2, 1)
        if not test_ts.empty:
            fig.add_trace(go.Scatter(x=test_ts["timestamp"], y=test_ts["equity"], name="测试期净值",
                                     line={"width": 1.5}), 1, 1)
            fig.add_trace(go.Scatter(x=test_ts["timestamp"], y=test_ts["drawdown"] * 100, name="测试期回撤%",
                                     fill="tozeroy", line={"width": 0.8}), 2, 1)
        fig.update_yaxes(title_text="净值", row=1, col=1)
        fig.update_yaxes(title_text="回撤%", row=2, col=1)
        fig.update_layout(height=460, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                          title=f"{factor} · {reb} 调仓", legend={"orientation": "h", "y": -0.12})
        st.plotly_chart(fig, width="stretch")

with right:
    bars, cumulative = _layers(row["layers_path"], _mtime(row["layers_path"]))
    test_bars = pd.DataFrame()
    if has_test and isinstance(row.get("test_layers_path"), str):
        test_bars, _ = _layers(row["test_layers_path"], _mtime(row["test_layers_path"]))
    if bars.empty:
        st.info("没有 layers.csv")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bars["decile"], y=bars["mean_bps"], name="训练期",
                             marker_color=[UP if v > 0 else DOWN for v in bars["mean_bps"]]))
        if not test_bars.empty:
            fig.add_trace(go.Bar(x=test_bars["decile"], y=test_bars["mean_bps"], name="测试期",
                                 marker_color="rgba(120,120,120,0.55)"))
        fig.update_layout(height=230, barmode="group", margin={"l": 10, "r": 10, "t": 30, "b": 10},
                          title="十分位平均超额收益（bps / bar）", yaxis_title="bps")
        st.plotly_chart(fig, width="stretch")
        st.caption("Q1 是因子值最低的 10%，Q10 最高。单调递增说明因子排序有效；只有两端有差异也可以用。")
        thin = cumulative.iloc[:: max(1, len(cumulative) // 3000)]
        fig = go.Figure()
        for column, color in (("Q1", DOWN), ("Q10", UP)):
            if column in thin:
                fig.add_trace(go.Scatter(x=thin["timestamp"], y=thin[column], name=column,
                                         line={"color": color, "width": 1.3}))
        if {"Q1", "Q10"} <= set(thin.columns):
            fig.add_trace(go.Scatter(x=thin["timestamp"], y=thin["Q10"] - thin["Q1"], name="Q10−Q1",
                                     line={"color": "gray", "dash": "dot"}))
        fig.update_layout(height=230, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                          title="Q1 / Q10 累计超额收益 %", legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ cost breakdown
st.markdown("##### 成本拆解")
curves, costs = d.rust_costs(_ts(row["timeseries_path"], _mtime(row["timeseries_path"])))
test_costs = {}
if has_test and isinstance(row.get("test_timeseries_path"), str):
    _, test_costs = d.rust_costs(_ts(row["test_timeseries_path"], _mtime(row["test_timeseries_path"])))
if not costs:
    st.caption("没有 timeseries.csv，无法拆解成本")
else:
    c = st.columns(6)
    c[0].metric("扣成本前收益", fmt_num(costs["gross_return_pct"], 2, "%"))
    c[1].metric("手续费 + 滑点", fmt_num(-costs["cost_pct_of_equity"], 2, "%"))
    c[2].metric("资金费", fmt_num(-costs["funding_pct_of_equity"], 2, "%"), help="负数表示净支付资金费")
    c[3].metric("净收益", fmt_num(costs["net_return_pct"], 2, "%"))
    c[4].metric("成本吃掉毛收益", fmt_num(costs["cost_share_of_gross_pct"], 0, "%"),
                help="(手续费 + 滑点 + 资金费) / 扣成本前收益；毛收益为负时不显示")
    c[5].metric("年化换手", fmt_num(costs.get("annual_turnover"), 0, "x"),
                help="每年交易的名义金额是组合规模的多少倍")
    if test_costs:
        st.caption(f"测试期：扣成本前 {fmt_num(test_costs['gross_return_pct'], 2, '%')}，"
                   f"成本 {fmt_num(-test_costs['cost_pct_of_equity'], 2, '%')}，"
                   f"资金费 {fmt_num(-test_costs['funding_pct_of_equity'], 2, '%')}，"
                   f"净 {fmt_num(test_costs['net_return_pct'], 2, '%')}")
    thin = curves.iloc[:: max(1, len(curves) // 3000)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thin["timestamp"], y=thin["gross_pnl"], name="扣成本前", line={"color": "#2563eb"}))
    fig.add_trace(go.Scatter(x=thin["timestamp"], y=thin["net_pnl"], name="净收益", line={"color": "#111827", "width": 2}))
    fig.add_trace(go.Scatter(x=thin["timestamp"], y=-thin["cost"], name="累计手续费+滑点", fill="tozeroy",
                             line={"color": DOWN, "width": 1}))
    fig.add_trace(go.Scatter(x=thin["timestamp"], y=-thin["funding_paid"], name="累计资金费",
                             line={"color": "#f59e0b", "width": 1}))
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="USDT",
                      title="累计盈亏拆解（训练期）", legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(fig, width="stretch")
    st.caption("Rust 回测按固定杠杆持仓，没有风控状态机，所以这里没有 SHADOW。"
               "带 SHADOW 的回测结果见左侧的「Shadow 回放」页。")

st.caption(f"数据来源：{rel(train_root)}" + (f"，对照期：{rel(test_root)}" if has_test else ""))
