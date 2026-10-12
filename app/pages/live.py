"""Live monitor: reads the live runner's output directory (latest.json, audit.jsonl, state files)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from _shared import DOWN, LIVE, OUTPUTS, SHADOW, UP, fmt_age, fmt_num, path_picker, rel
from efficient_cryptos.dashboard import data as d

st.title("实盘监控")


@st.cache_data(show_spinner=False, ttl=600)
def _dirs() -> list[Path]:
    return d.find_live_dirs(OUTPUTS)


@st.cache_data(show_spinner=False, max_entries=8)
def _records(path: str, max_lines: int, size: int, mtime: float) -> list[dict]:
    return d.read_jsonl_tail(path, max_lines)


def _stat(path: Path) -> tuple[int, float]:
    try:
        st_ = path.stat()
        return st_.st_size, st_.st_mtime
    except OSError:
        return 0, 0.0


live_dir = path_picker("实盘输出目录", _dirs(), "live",
                       help="run_live.py 的 --output，默认 outputs/testnet_shadow_run")
max_lines = st.sidebar.select_slider("读取最近多少个周期", options=[500, 1000, 3000, 10000, 30000], value=3000,
                                     help="audit.jsonl 会越来越大，只读取末尾这部分")
refresh = st.sidebar.selectbox("自动刷新", [0, 15, 30, 60, 300], index=2,
                               format_func=lambda s: "关闭" if s == 0 else f"每 {s} 秒")
stale_minutes = st.sidebar.number_input("超过多少分钟没更新算异常", value=10, min_value=1,
                                        help="设成 poll 间隔的 2–3 倍")
if st.sidebar.button("重新查找目录"):
    st.cache_data.clear()

if live_dir is None or not live_dir.exists():
    st.info("没有找到实盘输出目录。启动 runner 后（`python scripts/06_live/run_live.py ... --output outputs/xxx`），"
            "在左侧选择对应目录。")
    st.stop()


def _state_badge(state: str) -> str:
    color = LIVE if state == "LIVE" else SHADOW if state == "SHADOW" else "gray"
    return f"<span style='background:{color};color:white;padding:2px 10px;border-radius:10px;font-weight:600'>{state or '—'}</span>"


def render() -> None:
    latest_path = live_dir / "latest.json"
    audit_path = live_dir / "audit.jsonl"
    latest = d.read_json(latest_path)
    engine_state = d.read_json(live_dir / "engine_state.json")
    size, mtime = _stat(audit_path)
    records = _records(str(audit_path), max_lines, size, mtime)
    audit = d.audit_frame(records)
    summary = d.live_summary(latest, engine_state, _stat(latest_path)[1] or None)

    # ---------------------------------------------------------- status line
    age = summary["seconds_since_update"]
    head = st.columns([3, 2, 2, 2])
    head[0].markdown(f"**{summary['factor_id'] or '—'}**<br><span style='opacity:.7'>{summary['mode'] or '—'} · "
                     f"{rel(live_dir)}</span>", unsafe_allow_html=True)
    head[1].markdown(f"状态<br>{_state_badge(summary['state'])}", unsafe_allow_html=True)
    head[2].markdown(f"最近更新<br>**{fmt_age(age)}**", unsafe_allow_html=True)
    head[3].markdown(f"最新信号 bar<br>**{summary['signal_timestamp'] or '—'}**", unsafe_allow_html=True)
    if math.isfinite(age) and age > stale_minutes * 60:
        st.error(f"latest.json 已经 {fmt_age(age)} 没有更新：runner 可能已停止、卡在重试，或机器掉线。"
                 "查看下方的错误日志。")
    if summary["state"] == "SHADOW":
        st.warning("当前处于 SHADOW：风控已触发，实盘账户应为空仓，引擎只在模拟账本上跟踪，等待恢复条件。")
    if summary["blocked_symbols"]:
        st.warning("被风控屏蔽的币：" + "、".join(f"{k}（{v}）" for k, v in summary["blocked_symbols"].items()))

    # ---------------------------------------------------------- KPIs
    k = st.columns(7)
    k[0].metric("账户权益", fmt_num(summary["equity"], 2))
    k[1].metric("可用保证金", fmt_num(summary["available"], 2))
    k[2].metric("今日盈亏", fmt_num(summary["day_pnl_pct"], 2, "%"))
    k[3].metric("距高点回撤", fmt_num(summary["drawdown_pct"], 2, "%"))
    k[4].metric("总杠杆", fmt_num(summary["gross_leverage"], 2, "x"))
    k[5].metric("净敞口 / 总敞口", fmt_num(summary["net_exposure_pct"], 1, "%"),
                help="(多头 − 空头) / (多头 + 空头)。美元中性时接近 0；单边止损后会偏离")
    k[6].metric("累计成本 / 资金费", f"{fmt_num(summary['total_cost'], 1)} / {fmt_num(summary['total_funding'], 1)}")

    # ---------------------------------------------------------- positions
    positions = d.positions_frame(latest, engine_state)
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### 当前持仓")
        if positions.empty:
            st.caption("空仓")
        else:
            src = positions["source"].iloc[0]
            st.caption("来自交易所持仓" if src == "exchange" else "dry-run：来自引擎的模拟账本")
            st.dataframe(positions.drop(columns="source"), hide_index=True, width="stretch", column_config={
                "symbol": "币种", "side": "方向",
                "notional": st.column_config.NumberColumn("名义价值", format="%.2f"),
                "quantity": st.column_config.NumberColumn("数量", format="%.4f"),
                "entry_price": st.column_config.NumberColumn("开仓价", format="%.6g"),
                "mark_price": st.column_config.NumberColumn("标记价", format="%.6g"),
                "unrealized_pnl": st.column_config.NumberColumn("未实现盈亏", format="%.2f"),
                "leverage": st.column_config.NumberColumn("杠杆", format="%.0f"),
            })
    with right:
        if not positions.empty:
            ordered = positions.sort_values("notional")
            fig = go.Figure(go.Bar(x=ordered["notional"], y=ordered["symbol"], orientation="h",
                                   marker_color=[UP if v > 0 else DOWN for v in ordered["notional"]],
                                   hovertemplate="%{y}: %{x:,.2f}<extra></extra>"))
            fig.update_layout(height=max(220, 26 * len(ordered) + 60), margin={"l": 10, "r": 10, "t": 30, "b": 10},
                              title=f"多 {fmt_num(summary['long_notional'], 0)} / 空 {fmt_num(summary['short_notional'], 0)}")
            st.plotly_chart(fig, width="stretch")

    tva = d.targets_vs_actual(latest)
    if not tva.empty and (latest or {}).get("exchange_positions") is not None:
        with st.expander("目标仓位 vs 实际仓位", expanded=bool((tva["diff"].abs() > 1).any())):
            st.dataframe(tva, hide_index=True, width="stretch", column_config={
                "symbol": "币种",
                "target": st.column_config.NumberColumn("目标", format="%.2f"),
                "actual": st.column_config.NumberColumn("实际", format="%.2f"),
                "diff": st.column_config.NumberColumn("差额", format="%.2f"),
                "diff_pct_of_target": st.column_config.NumberColumn("差额 % 目标", format="%.1f"),
            })
            st.caption("差额来自下单数量的精度取整、最小下单额，或者被拒绝的订单。持续存在较大差额时检查订单记录。")

    # ---------------------------------------------------------- equity
    st.markdown("##### 权益与敞口")
    if audit.empty:
        st.caption("audit.jsonl 里还没有周期记录")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.05,
                            specs=[[{"secondary_y": True}], [{}]])
        fig.add_trace(go.Scatter(x=audit["time"], y=audit["equity"], name="账户权益", line={"width": 1.6}),
                      1, 1, secondary_y=False)
        if audit["shadow_equity"].nunique(dropna=True) > 1:
            fig.add_trace(go.Scatter(x=audit["time"], y=audit["shadow_equity"], name="影子账本（相对）",
                                     line={"width": 1, "dash": "dot"}), 1, 1, secondary_y=True)
        for start, end, state in d.state_spans(audit):
            if state == "SHADOW":
                fig.add_vrect(x0=start, x1=end, fillcolor=SHADOW, opacity=0.15, line_width=0, row=1, col=1)
        fig.add_trace(go.Scatter(x=audit["time"], y=audit["long_notional"], name="多头", stackgroup=None,
                                 line={"color": UP, "width": 1}), 2, 1)
        fig.add_trace(go.Scatter(x=audit["time"], y=-audit["short_notional"], name="空头",
                                 line={"color": DOWN, "width": 1}), 2, 1)
        fig.add_trace(go.Scatter(x=audit["time"], y=audit["net_notional"], name="净敞口",
                                 line={"color": "gray", "dash": "dot"}), 2, 1)
        fig.update_yaxes(title_text="权益", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="影子", row=1, col=1, secondary_y=True, showgrid=False)
        fig.update_yaxes(title_text="名义价值", row=2, col=1)
        fig.update_layout(height=480, margin={"l": 10, "r": 10, "t": 20, "b": 10},
                          legend={"orientation": "h", "y": -0.12})
        st.plotly_chart(fig, width="stretch")
        st.caption("橙色背景为 SHADOW 时段。影子账本是风控触发后的模拟净值，用来判断何时恢复实盘。")

    # ---------------------------------------------------------- events / orders / errors
    events = d.event_frame(audit, records)
    orders = d.orders_frame(records)
    errors = d.errors_frame(live_dir / "runner_errors.jsonl")
    t_events, t_orders, t_errors, t_raw = st.tabs([
        f"风控事件 ({len(events)})", f"最近订单 ({len(orders)})", f"错误与重连 ({len(errors)})", "状态文件",
    ])
    with t_events:
        if events.empty:
            st.caption("最近没有风控事件")
        else:
            st.dataframe(events, hide_index=True, width="stretch", column_config={
                "time": st.column_config.DatetimeColumn("时间 (UTC)", format="YYYY-MM-DD HH:mm:ss"),
                "kind": "类型", "state": "状态", "detail": st.column_config.TextColumn("内容", width="large"),
            })
    with t_orders:
        if orders.empty:
            st.caption("最近没有订单")
        else:
            rejected = int(orders["rejected"].sum())
            if rejected:
                st.warning(f"{rejected} 笔订单在本地被拒绝（数量低于最小下单量等），见 reason 列")
            failed = int((orders["status"] == "ERROR").sum())
            if failed:
                st.error(f"{failed} 笔订单被交易所拒绝，见 error 列")
            st.dataframe(orders, hide_index=True, width="stretch", column_config={
                "time": st.column_config.DatetimeColumn("时间 (UTC)", format="YYYY-MM-DD HH:mm:ss"),
                "symbol": "币种", "side": "方向", "quantity": st.column_config.NumberColumn("数量", format="%.6g"),
                "reduce_only": "只减仓", "rejected": "本地拒绝", "reason": "原因", "status": "状态",
                "executed_qty": st.column_config.NumberColumn("成交数量", format="%.6g"),
                "error": st.column_config.TextColumn("错误", width="medium"),
                "target_notional": st.column_config.NumberColumn("目标名义", format="%.2f"),
                "price": st.column_config.NumberColumn("价格", format="%.6g"),
                "avg_price": st.column_config.NumberColumn("成交均价", format="%.6g"),
            })
    with t_errors:
        if errors.empty:
            st.caption("没有错误记录")
        else:
            recent = errors[errors["timestamp"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)]
            counts = recent["type"].value_counts()
            cols = st.columns(max(1, len(counts)) + 1)
            cols[0].metric("近 24 小时错误", len(recent))
            for i, (kind, n) in enumerate(counts.items(), 1):
                cols[i].metric(kind, int(n))
            st.dataframe(errors, hide_index=True, width="stretch", column_config={
                "timestamp": st.column_config.DatetimeColumn("时间 (UTC)", format="YYYY-MM-DD HH:mm:ss"),
                "type": "类型", "retry_in_seconds": st.column_config.NumberColumn("重试等待 (秒)", format="%.0f"),
                "phase": "阶段",
                "error": st.column_config.TextColumn("错误", width="large"),
            })
            st.caption("network_error 会自动回滚本周期并按 5s→120s 退避重试；POST 下单不会重放，下个周期按实际持仓对账。")
    with t_raw:
        for name in ("engine_state.json", "runner_state.json", "exchange_risk_state.json",
                     "manual_guard_state.json", "latest.json"):
            payload = d.read_json(live_dir / name)
            if payload is not None:
                with st.expander(name):
                    st.code(json.dumps(payload, indent=2, ensure_ascii=False)[:20000], language="json")


if refresh:
    st.fragment(run_every=refresh)(render)()
else:
    render()
