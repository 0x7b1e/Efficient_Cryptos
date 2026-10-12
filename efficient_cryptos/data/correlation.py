"""Return-correlation analysis for choosing a trading pool.

Loads local kline parquet files, resamples them to complete bars, and builds
log-return panels. From those it computes two correlation matrices:

* ``raw``: correlation of plain log returns, i.e. how much two contracts move
  together, including the common market move.
* ``resid``: correlation after removing the market move. ``market`` subtracts
  the equal-weight cross-sectional mean at each bar, the same demeaning the
  backtest ``target`` uses. ``btc`` regresses out a reference symbol's beta.
  For a cross-sectional long/short strategy this is the redundancy that
  matters, because two contracts with high residual correlation are effectively
  the same bet.

The greedy selector walks the candidates by liquidity and keeps a symbol only
if its correlation to every symbol already kept is below a threshold.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MS_PER_MINUTE = 60_000


def interval_minutes(interval: str) -> int:
    value, unit = int(interval[:-1]), interval[-1].lower()
    return value * {"m": 1, "h": 60, "d": 1440}[unit]


def discover_kline_files(data_dirs: Iterable[Path], symbols: list[str] | None = None) -> dict[str, Path]:
    """Map a display label to each kline parquet.

    The label is the file stem. If the same stem appears in two data
    directories, the second one is prefixed with its directory name, so a
    Binance ``TSLAUSDT`` and a Hyperliquid ``TSLAUSDT`` can be compared.
    """
    wanted = {s.upper() for s in symbols} if symbols else None
    found: dict[str, Path] = {}
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        kline_dir = data_dir / "klines" if (data_dir / "klines").is_dir() else data_dir
        for path in sorted(kline_dir.glob("*.parquet")):
            stem = path.stem
            if wanted is not None and stem.upper() not in wanted:
                continue
            label = stem if stem not in found else f"{data_dir.name}/{stem}"
            found[label] = path
    return found


def resample_close(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample klines to complete ``minutes`` bars.

    Returns a frame indexed by bar open time (UTC) with ``close`` (NaN when
    the bar is incomplete) and ``quote_volume`` columns. The native bar
    length is inferred from the data, so 1m and 30m source files both work.
    """
    if frame.empty:
        return pd.DataFrame(columns=["close", "quote_volume"], dtype=float)
    frame = frame.drop_duplicates("open_time", keep="last").sort_values("open_time")
    open_ms = pd.to_numeric(frame["open_time"], errors="coerce").astype("int64")
    diffs = np.diff(open_ms.to_numpy())
    native = max(1, int(round(float(np.median(diffs)) / MS_PER_MINUTE))) if len(diffs) else minutes
    if native > minutes or minutes % native:
        raise ValueError(f"cannot resample {native}m bars to {minutes}m")
    expected = minutes // native
    ts = pd.to_datetime(open_ms, unit="ms", utc=True)
    bucket = ts.dt.floor(f"{minutes}min")
    close = pd.to_numeric(frame["close"], errors="coerce")
    qv = pd.to_numeric(frame.get("quote_volume", pd.Series(np.nan, index=frame.index)), errors="coerce")
    grouped = pd.DataFrame({"bucket": bucket.to_numpy(), "close": close.to_numpy(),
                            "quote_volume": qv.to_numpy()}).groupby("bucket", sort=True)
    out = pd.DataFrame({
        "close": grouped["close"].last(),
        "quote_volume": grouped["quote_volume"].sum(min_count=1),
        "count": grouped["close"].count(),
    })
    out.loc[out["count"] < expected, "close"] = np.nan
    out.loc[~(out["close"] > 0), "close"] = np.nan
    full = pd.date_range(out.index.min(), out.index.max(), freq=f"{minutes}min", tz="UTC")
    return out.drop(columns="count").reindex(full)


@dataclass
class Panel:
    returns: pd.DataFrame          # bar open time x symbol, log returns
    quote_volume: pd.DataFrame     # bar open time x symbol, quote volume per bar
    first_bar: dict[str, pd.Timestamp] = field(default_factory=dict)
    last_bar: dict[str, pd.Timestamp] = field(default_factory=dict)


def load_panel(files: dict[str, Path], minutes: int, start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None) -> Panel:
    rets, vols, first, last = {}, {}, {}, {}
    for label, path in files.items():
        cols = ["open_time", "close", "quote_volume"]
        try:
            frame = pd.read_parquet(path, columns=cols)
        except Exception:
            frame = pd.read_parquet(path)
            frame = frame[[c for c in cols if c in frame.columns]]
        if start is not None:
            frame = frame[frame["open_time"] >= int(start.value // 1_000_000) - minutes * MS_PER_MINUTE]
        if end is not None:
            frame = frame[frame["open_time"] < int(end.value // 1_000_000)]
        bars = resample_close(frame, minutes)
        if bars["close"].notna().sum() < 2:
            continue
        ret = np.log(bars["close"]).diff()
        rets[label] = ret
        vols[label] = bars["quote_volume"]
        valid = bars.index[bars["close"].notna()]
        first[label], last[label] = valid.min(), valid.max()
    returns = pd.DataFrame(rets).sort_index()
    quote_volume = pd.DataFrame(vols).reindex(returns.index)
    if start is not None:
        returns, quote_volume = returns[returns.index >= start], quote_volume[quote_volume.index >= start]
    if end is not None:
        returns, quote_volume = returns[returns.index < end], quote_volume[quote_volume.index < end]
    return Panel(returns, quote_volume, first, last)


def session_mask(index: pd.DatetimeIndex, weekdays_only: bool = False,
                 utc_hours: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.ones(len(index), dtype=bool)
    if weekdays_only:
        mask &= index.dayofweek < 5
    if utc_hours is not None:
        lo, hi = utc_hours
        hours = index.hour
        mask &= (hours >= lo) & (hours < hi) if lo < hi else (hours >= lo) | (hours < hi)
    return mask


def residualize(returns: pd.DataFrame, mode: str, reference: str | None = None) -> pd.DataFrame:
    if mode == "none":
        return returns
    if mode == "market":
        # Same demeaning as the backtest target: subtract the equal-weight mean
        # of the symbols that have a return at that bar.
        return returns.sub(returns.mean(axis=1), axis=0)
    if mode == "btc":
        if not reference or reference not in returns:
            raise ValueError(f"reference symbol {reference!r} not in panel")
        x = returns[reference]
        out = {}
        for col in returns:
            if col == reference:
                continue
            y = returns[col]
            ok = x.notna() & y.notna()
            if ok.sum() < 3:
                out[col] = y * np.nan
                continue
            beta = np.cov(y[ok], x[ok])[0, 1] / np.var(x[ok], ddof=1)
            out[col] = y - beta * x
        return pd.DataFrame(out, index=returns.index)
    raise ValueError(f"unknown residual mode {mode!r}")


def overlap_counts(returns: pd.DataFrame) -> pd.DataFrame:
    valid = returns.notna().to_numpy(dtype=np.int64)
    return pd.DataFrame(valid.T @ valid, index=returns.columns, columns=returns.columns)


def correlation(returns: pd.DataFrame, method: str = "pearson", min_overlap: int = 100) -> pd.DataFrame:
    corr = returns.corr(method=method, min_periods=max(3, min_overlap))
    return corr


def cluster_order(corr: pd.DataFrame, cluster_corr: float) -> tuple[list[str], dict[str, int]]:
    """Average-linkage clustering on the correlation distance sqrt((1-rho)/2)."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform

    names = list(corr.columns)
    if len(names) < 3:
        return names, {name: i + 1 for i, name in enumerate(names)}
    rho = corr.to_numpy(dtype=float, copy=True)
    rho = np.where(np.isfinite(rho), rho, 0.0)
    np.fill_diagonal(rho, 1.0)
    dist = np.sqrt(np.clip((1.0 - rho) / 2.0, 0.0, 1.0))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")
    order = [names[i] for i in leaves_list(link)]
    threshold = float(np.sqrt(max(0.0, (1.0 - cluster_corr) / 2.0)))
    labels = fcluster(link, t=threshold, criterion="distance")
    return order, {name: int(label) for name, label in zip(names, labels)}


def high_corr_pairs(corr: pd.DataFrame, overlap: pd.DataFrame, threshold: float,
                    other: pd.DataFrame | None = None, other_name: str = "other") -> pd.DataFrame:
    rows = []
    names = list(corr.columns)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            value = corr.loc[left, right]
            if np.isfinite(value) and abs(value) >= threshold:
                row = {"left": left, "right": right, "corr": float(value),
                       "overlap_bars": int(overlap.loc[left, right])}
                if other is not None and left in other.index and right in other.columns:
                    row[f"corr_{other_name}"] = float(other.loc[left, right])
                rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("corr", ascending=False, key=np.abs).reset_index(drop=True) if rows else frame


def rolling_average_corr(returns: pd.DataFrame, window_bars: int, method: str,
                         min_overlap: int) -> pd.DataFrame:
    """Mean correlation of each symbol to the others in non-overlapping windows."""
    rows = {}
    for begin in range(0, len(returns), window_bars):
        chunk = returns.iloc[begin:begin + window_bars]
        if len(chunk) < max(3, min_overlap):
            continue
        corr = chunk.corr(method=method, min_periods=max(3, min_overlap)).to_numpy(dtype=float, copy=True)
        np.fill_diagonal(corr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
            avg = np.nanmean(corr, axis=1)
        rows[chunk.index[0]] = pd.Series(avg, index=returns.columns)
    return pd.DataFrame(rows).T


def symbol_stats(panel: Panel, raw: pd.DataFrame, resid: pd.DataFrame, analysis: pd.DataFrame,
                 reference: str | None, rolling: pd.DataFrame | None) -> pd.DataFrame:
    idx = analysis.index
    total_bars = len(idx)
    step = pd.Series(idx).diff().min() if total_bars > 1 else pd.Timedelta(0)
    span_years = max((idx.max() - idx.min() + step) / pd.Timedelta(days=365), 1e-9) if total_bars else np.nan
    bars_per_year = total_bars / span_years if total_bars else np.nan
    bars_per_day = bars_per_year / 365.0 if total_bars else np.nan
    rows = []
    for name in analysis.columns:
        r = analysis[name]
        raw_off = raw[name].drop(index=name, errors="ignore")
        resid_off = resid[name].drop(index=name, errors="ignore") if name in resid else pd.Series(dtype=float)
        qv = panel.quote_volume[name].reindex(idx)
        peer = raw_off.abs().idxmax() if raw_off.notna().any() else None
        row = {
            "symbol": name,
            "first_bar": panel.first_bar.get(name),
            "last_bar": panel.last_bar.get(name),
            "bars": int(r.notna().sum()),
            "coverage": float(r.notna().sum() / total_bars) if total_bars else np.nan,
            "ann_vol": float(r.std() * np.sqrt(bars_per_year)) if r.notna().sum() > 2 else np.nan,
            "median_daily_quote_volume": float(qv.median() * bars_per_day) if qv.notna().any() else np.nan,
            "avg_corr_raw": float(raw_off.mean()) if raw_off.notna().any() else np.nan,
            "avg_corr_resid": float(resid_off.mean()) if resid_off.notna().any() else np.nan,
            "max_corr_peer": peer,
            "max_corr_raw": float(raw_off.loc[peer]) if peer is not None else np.nan,
        }
        if resid_off.notna().any():
            rpeer = resid_off.abs().idxmax()
            row["max_resid_peer"], row["max_corr_resid"] = rpeer, float(resid_off.loc[rpeer])
        if reference and reference in analysis and name != reference:
            x = analysis[reference]
            ok = x.notna() & r.notna()
            if ok.sum() > 2:
                row["corr_to_ref"] = float(np.corrcoef(r[ok], x[ok])[0, 1])
                row["beta_to_ref"] = float(np.cov(r[ok], x[ok])[0, 1] / np.var(x[ok], ddof=1))
        if rolling is not None and name in rolling and rolling[name].notna().sum() > 1:
            row["rolling_avg_corr_min"] = float(rolling[name].min())
            row["rolling_avg_corr_max"] = float(rolling[name].max())
            row["rolling_avg_corr_std"] = float(rolling[name].std())
        rows.append(row)
    return pd.DataFrame(rows)


def greedy_pool(stats: pd.DataFrame, corr: pd.DataFrame, max_corr: float, *, rank_by: str = "median_daily_quote_volume",
                min_coverage: float = 0.0, min_daily_quote_volume: float = 0.0, pool_size: int = 0,
                must_include: list[str] | None = None) -> pd.DataFrame:
    """Keep the most liquid symbols whose |corr| to every kept symbol is below ``max_corr``.

    Pairs without enough overlap (NaN correlation) do not block each other.
    Symbols in ``must_include`` are kept first and are never filtered.
    """
    must = [s for s in (must_include or []) if s in set(stats["symbol"])]
    ascending = rank_by in {"avg_corr_raw", "avg_corr_resid"}
    ordered = stats.sort_values(rank_by, ascending=ascending, na_position="last")["symbol"].tolist()
    ordered = must + [s for s in ordered if s not in must]
    info = stats.set_index("symbol")
    kept: list[str] = []
    rows = []
    for name in ordered:
        decision, peer, value = "selected", None, np.nan
        forced = name in must
        if not forced and info.loc[name, "coverage"] < min_coverage:
            decision = "low_coverage"
        elif (not forced and min_daily_quote_volume > 0
              and not info.loc[name, "median_daily_quote_volume"] >= min_daily_quote_volume):
            decision = "low_volume"
        if decision == "selected" and kept and name in corr:
            against = corr.loc[name, kept].abs() if name in corr.index else pd.Series(dtype=float)
            if against.notna().any():
                peer = against.idxmax()
                value = float(corr.loc[name, peer])
                if not forced and abs(value) >= max_corr:
                    decision = "correlated"
        if decision == "selected" and pool_size and len(kept) >= pool_size and not forced:
            decision = "pool_full"
        if decision == "selected":
            kept.append(name)
        rows.append({"symbol": name, "decision": decision, "closest_kept": peer, "corr_to_closest": value,
                     "coverage": info.loc[name, "coverage"],
                     "median_daily_quote_volume": info.loc[name, "median_daily_quote_volume"]})
    return pd.DataFrame(rows)


def plot_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(corr)
    size = min(30, max(6, 0.32 * n + 3))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    image = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    font = max(4, min(10, int(220 / max(n, 1))))
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=90, fontsize=font)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.index, fontsize=font)
    if n <= 30:
        values = corr.to_numpy(dtype=float)
        for i in range(n):
            for j in range(n):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=max(4, font - 2),
                            color="white" if abs(values[i, j]) > 0.6 else "black")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
