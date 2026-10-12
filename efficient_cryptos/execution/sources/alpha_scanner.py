"""Live scanner for Alpha101/191: pull market data, compute the chosen Alpha, and output target positions."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from efficient_cryptos.common.storage import LocalStore
from efficient_cryptos.common.sizing import target_positions
from efficient_cryptos.data.universe import liquidity_universe, load_universe_snapshot
from efficient_cryptos.factors.alpha import alpha_class

try:
    from efficient_cryptos.data.universe import live_liquidity_universe
except ImportError:  # Training branches may only expose the canonical helper.
    live_liquidity_universe = liquidity_universe

INTERVAL_MS = 60_000


def _last_closed_minute(now_ms: int, bar_min: int) -> int:
    bar_ms = max(1, int(bar_min)) * INTERVAL_MS
    return (now_ms // bar_ms) * bar_ms - bar_ms


def _alpha_cfg(cfg) -> Dict:
    a = cfg.get("alpha_live", {})
    p = cfg.get("portfolio", {})
    return {
        "family": str(a.get("family", "alpha101")),
        "alpha": str(a.get("alpha", "alpha020")),
        "direction_sign": int(a.get("direction_sign", -1)),
        "bar_min": int(a.get("bar_min", p.get("bar_min", 15))),
        "rebalance_bars": int(a.get("rebalance_bars", 96)),
        "hold_bars": int(a.get("hold_bars", a.get("rebalance_bars", 96))),
        "n_long": int(a.get("n_long", 10)),
        "n_short": int(a.get("n_short", 10)),
        "min_assets_per_bar": int(a.get("min_assets_per_bar", 20)),
        "history_bars": int(a.get("history_bars", 320)),
        "generated_factor_id": str(a.get("generated_factor_id", "")).strip(),
        "symbols": [str(s).upper() for s in a.get("symbols", [])],
        "sizing": str(a.get("sizing", "inverse_vol")).strip().lower(),
    }


def _generated_factor(wide: Dict[str, pd.DataFrame], factor_id: str) -> pd.DataFrame:
    """Compute generated factors explicitly approved for live/paper execution.

    Generated factor JSONL is deliberately not executed in the live process.  A
    factor must be implemented here so its online calculation can be reviewed
    and kept identical to the research formula.
    """
    if factor_id == "momentum_crypto_templates_10k_v1_momentum_0065_0005":
        # rolling_sum(close.pct_change() * volume, 48) / rolling_sum(volume, 48)
        close = wide["close"].astype(float)
        volume = wide["volume"].astype(float)
        numerator = (close.pct_change() * volume).rolling(48, min_periods=48).sum()
        denominator = volume.rolling(48, min_periods=48).sum().replace(0, np.nan)
        return numerator / denominator
    raise ValueError(f"generated factor is not approved for live execution: {factor_id}")


def _resample_klines(df: pd.DataFrame, bar_min: int) -> pd.DataFrame:
    src = df.sort_values("open_time").drop_duplicates("open_time").set_index("open_time")
    idx = pd.to_datetime(src.index.astype("int64"), unit="ms", utc=True)
    tmp = src.copy()
    tmp.index = idx
    rule = f"{bar_min}min"
    out = pd.DataFrame({
        "open": tmp["open"].resample(rule).first(),
        "high": tmp["high"].resample(rule).max(),
        "low": tmp["low"].resample(rule).min(),
        "close": tmp["close"].resample(rule).last(),
        "volume": tmp["volume"].resample(rule).sum(min_count=1),
        "quote_volume": tmp["quote_volume"].resample(rule).sum(min_count=1),
    }).dropna(subset=["open", "high", "low", "close"])
    out["open_time"] = [int(ts.timestamp() * 1000) for ts in out.index]
    return out.reset_index(drop=True)


def _build_wide_from_store(store: LocalStore, symbols: List[str], *, bar_min: int,
                           history_bars: int, end_ms: int) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    start_ms = end_ms - max(history_bars + 20, 60) * bar_min * INTERVAL_MS
    fields: Dict[str, Dict[str, pd.Series]] = {
        "open": {}, "high": {}, "low": {}, "close": {},
        "volume": {}, "amount": {}, "quote_volume": {}, "vwap": {},
    }
    for sym in symbols:
        raw = store.load_klines(sym, start_ms=start_ms, end_ms=end_ms)
        if raw.empty:
            continue
        df = _resample_klines(raw, bar_min).tail(history_bars)
        if len(df) < max(30, min(history_bars // 3, history_bars)):
            continue
        idx = df["open_time"].astype("int64")
        vol = df["volume"].astype(float).replace(0, np.nan)
        vwap = (df["quote_volume"].astype(float) / vol).replace([np.inf, -np.inf], np.nan)
        vwap = vwap.fillna(df["close"].astype(float))
        fields["open"][sym] = pd.Series(df["open"].astype(float).values, index=idx)
        fields["high"][sym] = pd.Series(df["high"].astype(float).values, index=idx)
        fields["low"][sym] = pd.Series(df["low"].astype(float).values, index=idx)
        fields["close"][sym] = pd.Series(df["close"].astype(float).values, index=idx)
        fields["volume"][sym] = pd.Series(df["volume"].astype(float).values, index=idx)
        fields["amount"][sym] = pd.Series(df["quote_volume"].astype(float).values, index=idx)
        fields["quote_volume"][sym] = pd.Series(df["quote_volume"].astype(float).values, index=idx)
        fields["vwap"][sym] = pd.Series(vwap.values, index=idx)

    wide = {k: pd.DataFrame(v).sort_index() for k, v in fields.items()}
    if wide["close"].empty:
        raise ValueError("Not enough local candles to compute alpha")
    if "BTCUSDT" in wide["close"].columns:
        bench_open = wide["open"]["BTCUSDT"]
        bench_close = wide["close"]["BTCUSDT"]
    else:
        bench_open = wide["open"].mean(axis=1)
        bench_close = wide["close"].mean(axis=1)
    wide["benchmark_open"] = pd.DataFrame({s: bench_open for s in wide["close"].columns}, index=wide["close"].index)
    wide["benchmark_close"] = pd.DataFrame({s: bench_close for s in wide["close"].columns}, index=wide["close"].index)
    popbo_input = pd.concat(
        {k: wide[k] for k in [
            "open", "close", "high", "low", "volume", "amount",
            "vwap", "benchmark_open", "benchmark_close",
        ]},
        axis=1,
    )
    return popbo_input, wide


def _select_symbols(factor_row: pd.Series, qv_row: pd.Series, cfg, acfg: Dict) -> tuple[List[str], List[str]]:
    u = cfg["universe"]
    exclude = {s.upper() for s in u.get("exclude", [])}
    rows = []
    for sym, val in factor_row.items():
        if sym.upper() in exclude or not np.isfinite(val):
            continue
        qv = qv_row.get(sym, np.nan)
        if not np.isfinite(qv) or qv < float(u["min_quote_volume_usdt"]):
            continue
        rows.append((sym, float(qv), float(val) * acfg["direction_sign"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:int(u["top_n"])]
    rows.sort(key=lambda x: x[2], reverse=True)
    syms = [s for s, _, _ in rows]
    nl = min(acfg["n_long"], len(syms))
    ns = min(acfg["n_short"], max(0, len(syms) - nl))
    return syms[:nl], syms[len(syms) - ns:] if ns else []


def scan_alpha_once(client, store: LocalStore, cfg, *, equity: float, workers: int = 8) -> Dict[str, float]:
    """Run one live Alpha scan and return target notionals as {symbol: target_usdt}."""
    acfg = _alpha_cfg(cfg)
    g, u = cfg["general"], cfg["universe"]
    universe = list(acfg["symbols"])
    if not universe:
        universe = load_universe_snapshot(g["data_dir"])
    if not universe:
        tickers = client.ticker_24hr(futures=True)
        universe = live_liquidity_universe(
            tickers, quote_asset=g["quote_asset"], top_n=int(u["top_n"]),
            min_quote_volume_usdt=float(u["min_quote_volume_usdt"]), exclude=u.get("exclude", []),
        )
    universe = universe[:int(u["top_n"])]
    if not universe:
        logging.warning("No tradable symbols after Alpha liquidity filtering")
        return {}

    end = _last_closed_minute(int(time.time() * 1000), acfg["bar_min"])
    need_1m = max(acfg["history_bars"] * acfg["bar_min"] + 10, 100)
    fetch_end = end + max(1, acfg["bar_min"]) * INTERVAL_MS - 1

    def fetch(sym: str):
        cursor = max(0, fetch_end - need_1m * INTERVAL_MS)
        rows_all = []
        while cursor <= fetch_end and len(rows_all) < need_1m:
            rows = client.klines(sym, "1m", limit=min(1500, need_1m - len(rows_all)),
                                 start_ms=cursor, end_ms=fetch_end, futures=True)
            if not rows:
                break
            rows_all.extend(rows)
            last_open = int(rows[-1][0])
            if last_open < cursor:
                break
            cursor = last_open + INTERVAL_MS
        return sym, store.raw_klines_to_df(rows_all)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fetch, s) for s in universe]):
            try:
                sym, df = fut.result()
                if len(df):
                    store.upsert_klines(sym, df)
            except Exception as exc:
                logging.warning("Alpha fetch failed: %r", exc)

    popbo_input, wide = _build_wide_from_store(
        store, universe, bar_min=acfg["bar_min"],
        history_bars=acfg["history_bars"], end_ms=end,
    )
    # Never turn stale cached candles into a live order.  This matters when a
    # public-data fetch fails but older research data remains in LocalStore.
    usable_now = int(wide["close"].loc[end].notna().sum()) if end in wide["close"].index else 0
    if usable_now < acfg["min_assets_per_bar"]:
        latest = int(wide["close"].index.max()) if len(wide["close"].index) else None
        logging.warning(
            "Alpha has too few fresh symbols at expected bar %s: usable=%s latest=%s",
            end, usable_now, latest,
        )
        return {}

    if acfg["generated_factor_id"]:
        factor = _generated_factor(wide, acfg["generated_factor_id"])
    else:
        cls = alpha_class(acfg["family"])
        alpha_obj = cls(popbo_input)
        factor = getattr(alpha_obj, acfg["alpha"])()
    factor = factor.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ts = factor.index[-1]
    bars_per_day = max(1, int(round(1440 / max(acfg["bar_min"], 1))))
    qv_row = wide["quote_volume"].rolling(bars_per_day, min_periods=max(2, bars_per_day // 24)).sum().loc[ts]
    longs, shorts = _select_symbols(factor.loc[ts], qv_row, cfg, acfg)
    close = wide["close"]
    ret = np.log(close / close.shift(1))
    vol_window = max(2, int(round(60 / max(acfg["bar_min"], 1))))
    vol_ann = ret.rolling(vol_window, min_periods=max(2, vol_window // 2)).std() * np.sqrt(525_600 / acfg["bar_min"])
    vols = {s: float(vol_ann.at[ts, s]) if s in vol_ann.columns and np.isfinite(vol_ann.at[ts, s]) else 0.0
            for s in longs + shorts}
    if acfg["sizing"] == "equal_weight":
        gross = min(float(g.get("leverage", 1)), float(cfg["portfolio"]["gross_leverage_cap"]))
        targets: Dict[str, float] = {}
        if longs:
            long_notional = equity * gross * 0.5 / len(longs)
            targets.update({symbol: long_notional for symbol in longs})
        if shorts:
            short_notional = -equity * gross * 0.5 / len(shorts)
            targets.update({symbol: short_notional for symbol in shorts})
        return targets
    if acfg["sizing"] != "inverse_vol":
        raise ValueError(f"unknown alpha_live sizing: {acfg['sizing']}")
    return target_positions(
        longs, shorts, vols,
        equity=equity,
        leverage=float(g.get("leverage", 3)),
        target_vol_annual=float(cfg["portfolio"]["target_vol_annual"]),
        gross_cap=float(cfg["portfolio"]["gross_leverage_cap"]),
        max_symbol_weight=float(cfg["portfolio"].get("max_symbol_weight", 0.0)),
    )
