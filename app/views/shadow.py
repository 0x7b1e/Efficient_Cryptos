"""Shadow replay: backtests run through the same LIVE/SHADOW risk engine as the live runner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from _shared import DOWN, OUTPUTS, SHADOW, UP, fmt_num, path_picker, rel
from efficient_cryptos.dashboard import data as d

st.title("Shadow 回放")
st.caption("用和实盘完全相同的风控引擎回放因子预测：单币止损止盈、日内止损、组合回撤、进入 SHADOW 和恢复 LIVE 都会生效，"
           "并按 1m TWAP 做盘中风控。结果来自 `scripts/05_shadow/run_selected_factor_shadow.py`。")


@st.cache_data(show_spinner=False, ttl=600)
def _roots() -> list[Path]:
    return sorted(d.find_sweep_roots(OUTPUTS, kind="shadow"), key=lambda p: d.dir_signature(p, "*")[1], reverse=True)


@st.cache_data(show_spinner="读取回放结果…", max_entries=8)
def _scan(root: str, signature: tuple) -> pd.DataFrame:
    return d.scan_shadow(root)


@st.cache_data(show_spinner=False, max_entries=64)
def _series(path: str, mtime: float, initial: float, leverage: float):
    return d.shadow_timeseries(path, initial, leverage)


def _mtime(path) -> float:
    try:
        return Path(path).stat().st_mtime
    except (OSError, TypeError):
        return 0.0


if st.sidebar.button("重新扫描目录"):
    st.cache_data.clear()
root = path_picker("回放结果目录", _roots(), "shadow_root",
                   help="run_selected_factor_shadow.py 的 --out-dir，里面是 1h/2h/… 子目录")
if root is None or not root.exists():
    st.info(
        "没有找到 Shadow 回放结果。例如：\n\n"
        "```bash\npython scripts/05_shadow/run_selected_factor_shadow.py \\\n"
        "  --predictions-dir outputs/pred_test_selected --data-dir data \\\n"
        "  --out-dir outputs/shadow_test --rebalance-hours 1,2,4,8,12,24\n```"
    )
    st.stop()

frame = _scan(str(root), d.dir_signature(root, "*"))
if frame.empty:
    st.warning(f"{rel(root)} 里没有回放的 metrics.json")
    st.stop()

# ------------------------------------------------------------------ filters
rebalances = d.sort_rebalances(frame["rebalance"])
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    chosen = c1.multiselect("调仓周期", rebalances, default=rebalances)
    states = c2.multiselect("最终状态", ["LIVE", "SHADOW"], default=["LIVE", "SHADOW"])
    min_sharpe = c3.number_input("Sharpe ≥", value=-100.0, step=0.5)
    search = c4.text_input("搜索因子")
    best_only = st.toggle("每个因子只看最佳周期", value=False)

view = frame[frame["rebalance"].isin(chosen) & frame["final_state"].isin(states)
             & (frame["sharpe"].fillna(-np.inf) >= min_sharpe)]
if search.strip():
    view = view[view["factor_id"].str.lower().str.contains(search.strip().lower(), regex=False)]
if best_only:
    view = d.best_per_factor(view)

k = st.columns(5)
k[0].metric("组合（因子×周期）", f"{len(frame):,}")
k[1].metric("最终 LIVE 占比", fmt_num((frame["final_state"] == "LIVE").mean() * 100, 1, "%"))
k[2].metric("平均进入 SHADOW 次数", fmt_num(frame["shadow_entries"].mean(), 1))
k[3].metric("成本占本金（中位数）", fmt_num(frame["cost_pct_of_equity"].median(), 2, "%"))
k[4].metric("筛选后", f"{len(view):,}")

tab_board, tab_stab = st.tabs(["排行榜", "跨周期稳定性"])
with tab_board:
    columns = ["factor_id", "rebalance", "return_pct", "sharpe", "max_drawdown_pct", "final_state",
               "shadow_entries", "shadow_exits", "cost_pct_of_equity", "funding_paid", "cost_share_of_gross_pct"]
    table = view[columns].reset_index(drop=True)
    event = st.dataframe(
        table.head(2000), hide_index=True, width="stretch", height=int(min(440, 36 * (min(len(table), 2000) + 1) + 4)),
        on_select="rerun", selection_mode="single-row", key="shadow_board",
        column_config={
            "factor_id": st.column_config.TextColumn("因子", width="medium"),
            "rebalance": st.column_config.TextColumn("周期", width="small"),
            "return_pct": st.column_config.NumberColumn("收益%", format="%.2f"),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "max_drawdown_pct": st.column_config.NumberColumn("最大回撤%", format="%.2f"),
            "final_state": st.column_config.TextColumn("最终状态", width="small"),
            "shadow_entries": st.column_config.NumberColumn("进入 SHADOW", format="%d"),
            "shadow_exits": st.column_config.NumberColumn("恢复 LIVE", format="%d"),
            "cost_pct_of_equity": st.column_config.NumberColumn("成本占本金%", format="%.2f"),
            "funding_paid": st.column_config.NumberColumn("资金费支出", format="%.2f", help="负数表示净收到"),
            "cost_share_of_gross_pct": st.column_config.NumberColumn("成本吃掉毛收益%", format="%.0f"),
        })
    st.download_button("下载 CSV", table.to_csv(index=False).encode("utf-8"), "shadow_replay.csv", "text/csv")
    picked = event.selection.rows if event and event.selection else []
    if picked:
        pick = (table.iloc[picked[0]]["factor_id"], table.iloc[picked[0]]["rebalance"])
        if st.session_state.get("shadow_last_pick") != pick:
            st.session_state["shadow_last_pick"] = pick
            st.session_state["shadow_factor"], st.session_state["shadow_reb"] = pick
with tab_stab:
    st.dataframe(d.stability_table(frame[frame["rebalance"].isin(chosen)]).head(2000), hide_index=True,
                 width="stretch", column_config={
                     "factor_id": st.column_config.TextColumn("因子", width="large"),
                     "periods": "周期数", "positive_periods": "盈利周期数",
                     "mean_return_pct": st.column_config.NumberColumn("平均收益%", format="%.2f"),
                     "median_sharpe": st.column_config.NumberColumn("Sharpe 中位数", format="%.2f"),
                     "min_sharpe": st.column_config.NumberColumn("最差 Sharpe", format="%.2f"),
                     "worst_drawdown_pct": st.column_config.NumberColumn("最差回撤%", format="%.2f"),
                 })

# ------------------------------------------------------------------ detail
st.divider()
st.subheader("回放详情")
factor_ids = view["factor_id"].drop_duplicates().tolist() or frame["factor_id"].drop_duplicates().tolist()
if st.session_state.get("shadow_factor") not in factor_ids:
    st.session_state["shadow_factor"] = factor_ids[0]
factor = st.selectbox("因子", factor_ids, key="shadow_factor")
rows = frame[frame["factor_id"] == factor].sort_values("rebalance_hours")
reb_options = rows["rebalance"].tolist()
if st.session_state.get("shadow_reb") not in reb_options:
    st.session_state["shadow_reb"] = reb_options[0]
reb = st.radio("查看周期", reb_options, horizontal=True, key="shadow_reb")
row = rows[rows["rebalance"] == reb].iloc[0]

metrics_path = Path(row["timeseries_path"]).with_name(Path(row["timeseries_path"]).name.replace(".timeseries.csv", ".metrics.json"))
meta = d.read_json(metrics_path) or {}
config = meta.get("config") or {}
if config.get("initial_equity"):
    initial = float(config["initial_equity"])
elif meta.get("final_equity") is not None and meta.get("return_pct") is not None:
    initial = float(meta["final_equity"]) / (1 + float(meta["return_pct"]) / 100)
else:
    initial = 10_000.0
leverage = float(config.get("gross_leverage") or 0) or None
steps, events = _series(row["timeseries_path"], _mtime(row["timeseries_path"]), initial, leverage or 0.0)
if steps.empty:
    st.info("没有 timeseries.csv")
    st.stop()

shadow_share = (steps["state"] == "SHADOW").mean() * 100
last = steps.iloc[-1]
gross_final = float(last["gross_pnl"])
c = st.columns(7)
state_color = SHADOW if row["final_state"] == "SHADOW" else UP
c[0].markdown(f"<div style='font-size:14px'>最终状态</div><div style='margin-top:10px'><span style='background:{state_color};"
              f"color:white;padding:4px 12px;border-radius:12px;font-weight:600'>{row['final_state']}</span></div>",
              unsafe_allow_html=True)
c[1].metric("SHADOW 时间占比", fmt_num(shadow_share, 1, "%"))
c[2].metric("进入 / 恢复", f"{int(row['shadow_entries'] or 0)} / {int(row['shadow_exits'] or 0)}")
c[3].metric("扣成本前", fmt_num(gross_final / initial * 100, 2, "%"))
c[4].metric("手续费 + 滑点", fmt_num(-float(last["cum_cost"]) / initial * 100, 2, "%"))
c[5].metric("资金费", fmt_num(-float(last["cum_funding_paid"]) / initial * 100, 2, "%"))
c[6].metric("净收益", fmt_num(float(last["net_pnl"]) / initial * 100, 2, "%"))
if not bool(steps["cost_exact"].iloc[0]):
    st.caption("这份回放是旧版本生成的，成本曲线由每步的成本比例重建（近似）。重新运行回放脚本即可得到精确值和仓位占用。")

spans = d.frame_state_spans(steps)


def _shade(fig, row_index=None):
    for start, end, state in spans:
        if state == "SHADOW":
            kwargs = {"row": row_index, "col": 1} if row_index else {}
            fig.add_vrect(x0=start, x1=end, fillcolor=SHADOW, opacity=0.15, line_width=0, **kwargs)


has_exposure = "gross_notional" in steps
fig = make_subplots(rows=3 if has_exposure else 2, cols=1, shared_xaxes=True,
                    row_heights=[0.45, 0.3, 0.25] if has_exposure else [0.6, 0.4], vertical_spacing=0.04,
                    specs=[[{"secondary_y": True}], [{}]] + ([[{"secondary_y": True}]] if has_exposure else []))
fig.add_trace(go.Scatter(x=steps["timestamp"], y=steps["equity"], name="实盘账本权益", line={"width": 1.6}), 1, 1)
fig.add_trace(go.Scatter(x=steps["timestamp"], y=steps["shadow_equity"], name="影子账本（相对）",
                         line={"width": 1, "dash": "dot", "color": SHADOW}), 1, 1, secondary_y=True)
fig.add_trace(go.Scatter(x=steps["timestamp"], y=steps["gross_pnl"], name="扣成本前盈亏", line={"color": "#2563eb"}), 2, 1)
fig.add_trace(go.Scatter(x=steps["timestamp"], y=steps["net_pnl"], name="净盈亏", line={"color": "#111827", "width": 2}), 2, 1)
fig.add_trace(go.Scatter(x=steps["timestamp"], y=-steps["cum_cost"], name="累计手续费+滑点", fill="tozeroy",
                         line={"color": DOWN, "width": 1}), 2, 1)
fig.add_trace(go.Scatter(x=steps["timestamp"], y=-steps["cum_funding_paid"], name="累计资金费",
                         line={"color": "#a16207", "width": 1}), 2, 1)
if has_exposure:
    fig.add_trace(go.Scatter(x=steps["timestamp"], y=steps["long_notional"], name="多头", line={"color": UP, "width": 1}), 3, 1)
    fig.add_trace(go.Scatter(x=steps["timestamp"], y=-steps["short_notional"], name="空头", line={"color": DOWN, "width": 1}), 3, 1)
    if "margin_used_pct" in steps:
        fig.add_trace(go.Scatter(x=steps["timestamp"], y=steps["margin_used_pct"], name="保证金占用%",
                                 line={"color": "gray", "dash": "dot"}), 3, 1, secondary_y=True)
        fig.update_yaxes(title_text="保证金%", row=3, col=1, secondary_y=True, showgrid=False, rangemode="tozero")
    fig.update_yaxes(title_text="名义价值", row=3, col=1, secondary_y=False)
for i in range(1, (3 if has_exposure else 2) + 1):
    _shade(fig, i)
fig.update_yaxes(title_text="权益", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="影子", row=1, col=1, secondary_y=True, showgrid=False)
fig.update_yaxes(title_text="USDT", row=2, col=1)
fig.update_layout(height=720 if has_exposure else 560, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                  title=f"{factor} · {reb} 调仓", legend={"orientation": "h", "y": -0.08})
st.plotly_chart(fig, width="stretch")
st.caption("橙色背景为 SHADOW：实盘账本空仓、权益冻结，影子账本继续按信号模拟，满足恢复条件后回到 LIVE。"
           "成本只计实盘账本；影子账本的交易成本体现在影子净值里。")

if has_exposure:
    live_rows = steps[steps["state"] == "LIVE"]
    m = st.columns(4)
    m[0].metric("平均总杠杆（LIVE）", fmt_num(live_rows["gross_leverage"].mean(), 2, "x"))
    m[1].metric("最大总杠杆", fmt_num(steps["gross_leverage"].max(), 2, "x"))
    if "margin_used_pct" in steps:
        m[2].metric("平均保证金占用（LIVE）", fmt_num(live_rows["margin_used_pct"].mean(), 1, "%"))
        m[3].metric("最大保证金占用", fmt_num(steps["margin_used_pct"].max(), 1, "%"))

left, right = st.columns([3, 2])
with left:
    st.markdown("##### 风控事件")
    if events.empty:
        st.caption("没有风控事件")
    else:
        st.dataframe(events.sort_values("timestamp", ascending=False), hide_index=True, width="stretch",
                     column_config={
                         "timestamp": st.column_config.DatetimeColumn("时间 (UTC)", format="YYYY-MM-DD HH:mm"),
                         "state": "之后状态", "risk_events": st.column_config.TextColumn("事件", width="large"),
                         "equity": st.column_config.NumberColumn("权益", format="%.2f"),
                         "shadow_equity": st.column_config.NumberColumn("影子净值", format="%.4f"),
                     })
with right:
    st.markdown("##### 回放参数")
    if config:
        labels = {
            "gross_leverage": "杠杆", "margin_buffer_pct": "保证金缓冲", "long_pct": "做多%", "short_pct": "做空%",
            "symbol_stop": "单币止损", "symbol_take": "单币止盈", "daily_stop": "日内止损", "daily_take": "日内止盈",
            "portfolio_drawdown": "组合回撤上限", "shadow_min_periods": "恢复：最少周期",
            "shadow_positive_periods": "恢复：盈利周期", "shadow_recovery_pnl_pct": "恢复：影子收益",
            "fee_bps": "手续费 bps", "slippage_bps": "滑点 bps", "mark_minutes": "盘中风控间隔(分)", "funding": "计资金费",
        }
        st.dataframe(pd.DataFrame([{"参数": labels.get(k, k), "值": str(v)} for k, v in config.items()
                                   if k in labels]), hide_index=True, width="stretch")
    else:
        st.caption("旧版本回放没有记录参数")

st.caption(f"数据来源：{rel(root)}")
