"""Validate and execute generated factor code (``compute_factor(df)``).

Factor records come from the DPSK generator (``efficient_cryptos.mining``).
Their code is restricted to a small, auditable pandas/numpy subset, statically
checked for look-ahead, and executed against a frame that never exposes labels
or future accounting columns.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd



FORBIDDEN_NODES = (
    ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.For,
    ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
    ast.Delete, ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await,
)
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "__import__", "os", "sys",
    "subprocess", "socket", "requests", "httpx", "pathlib",
}
FORBIDDEN_ATTRIBUTES = {
    "to_csv", "to_parquet", "to_pickle", "to_excel", "to_sql",
    "read_csv", "read_parquet", "read_pickle", "read_excel", "read_sql",
    "eval", "query", "pipe", "bfill", "backfill",
}


# Aggregations that, applied over a per-symbol (or per-day / per-clock-time)
# group, use every bar of that group -- including bars after ``t``.
FULL_SAMPLE_AGGREGATIONS = {
    "mean", "std", "var", "sum", "max", "min", "median", "quantile", "skew",
    "kurt", "count", "nunique", "first", "last", "zscore", "rank",
}
# Grouping keys whose groups span time.  Only ``timestamp`` (one cross-section
# per bar) is safe for full-group aggregation.
TIME_SPANNING_KEYS = {"symbol", "ticker", "dataDate", "barTime"}
CAUSAL_WINDOW_METHODS = {"rolling", "expanding", "ewm", "cummax", "cummin", "cumsum", "cumprod", "resample"}


def _receiver_chain(node: ast.AST) -> list[ast.AST]:
    """Nodes from a method call back to its root (``df[...].groupby(...).x``)."""
    chain = []
    current = node
    while True:
        chain.append(current)
        if isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        else:
            return chain


def _key_names(node: ast.AST) -> set[str]:
    """String column names referenced by a groupby key expression."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            names.add(child.value)
    return names


def _lookahead_reason(node: ast.Call) -> str | None:
    """Return why a call reads bars after ``t`` (or None)."""
    if not isinstance(node.func, ast.Attribute):
        return None
    method = node.func.attr
    chain = _receiver_chain(node.func.value)
    methods = [c.func.attr for c in chain if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
    if any(m in CAUSAL_WINDOW_METHODS for m in methods):
        return None
    groupbys = [c for c in chain if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "groupby"]
    if groupbys and method in {"transform", "apply"} and node.args and isinstance(node.args[0], ast.Lambda):
        keys = set()
        for call in groupbys:
            for arg in list(call.args) + [k.value for k in call.keywords if k.arg in (None, "by")]:
                keys |= _key_names(arg)
        if keys & TIME_SPANNING_KEYS:
            params = {a.arg for a in node.args[0].args.args}
            for inner in ast.walk(node.args[0].body):
                if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr in FULL_SAMPLE_AGGREGATIONS):
                    continue
                inner_chain = _receiver_chain(inner.func.value)
                inner_methods = {c.func.attr for c in inner_chain
                                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
                root = inner_chain[-1]
                if (isinstance(root, ast.Name) and root.id in params
                        and not inner_methods & CAUSAL_WINDOW_METHODS
                        and not any(k.arg == "axis" for k in inner.keywords)):
                    return (f"groupby({','.join(sorted(keys))}).{method}(lambda: .{inner.func.attr}()) "
                            "aggregates each group's whole history (future bars)")
    aggregation = None
    if method in {"transform", "agg", "aggregate"} and node.args and isinstance(node.args[0], ast.Constant):
        aggregation = str(node.args[0].value)
    elif method in FULL_SAMPLE_AGGREGATIONS:
        aggregation = method
    if aggregation not in FULL_SAMPLE_AGGREGATIONS:
        return None
    if any(isinstance(k, ast.keyword) and k.arg == "axis" and isinstance(k.value, ast.Constant)
           and k.value.value in (1, "columns") for k in node.keywords):
        return None  # row-wise across columns of the same bar
    if groupbys:
        keys = set()
        for call in groupbys:
            for arg in list(call.args) + [k.value for k in call.keywords if k.arg in (None, "by")]:
                keys |= _key_names(arg)
        if keys & TIME_SPANNING_KEYS:
            call = f".{method}('{aggregation}')" if method != aggregation else f".{method}()"
            return (f"groupby({','.join(sorted(keys))}){call} aggregates over the "
                    "whole history of each group (future bars); use rolling/expanding or group by timestamp")
        return None
    root = chain[-1]
    reads_column = any(isinstance(c, ast.Subscript) and isinstance(c.value, ast.Name) and c.value.id == "df"
                       for c in chain)
    if reads_column or (isinstance(root, ast.Name) and root.id == "df"):
        return f"full-sample .{aggregation}() over the whole column uses future bars"
    return None


def validate_factor_code(code: str, allowed_columns: Iterable[str] | None = None) -> str | None:
    """Validate generated code without executing it."""
    if not str(code).strip():
        return "empty code"
    try:
        tree = ast.parse(str(code))
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} line={exc.lineno}"
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "compute_factor":
        return "code must define exactly one compute_factor function"
    fn = functions[0]
    if len(fn.args.args) != 1 or fn.args.args[0].arg != "df":
        return "compute_factor must accept exactly one df argument"
    allowed = set(allowed_columns or ())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
            if not names.issubset({"numpy", "pandas", "np", "pd"}):
                return f"forbidden import: {sorted(names)}"
        if isinstance(node, FORBIDDEN_NODES):
            return f"forbidden syntax: {type(node).__name__}"
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_NAMES):
            return f"forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES):
            return f"forbidden attribute: {node.attr}"
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "df":
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if allowed and isinstance(key, str) and key not in allowed:
                return f"unsupported column: {key}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in {"shift", "diff", "pct_change"}:
                periods = node.args[0] if node.args else None
                for keyword in node.keywords:
                    if keyword.arg == "periods":
                        periods = keyword.value
                if isinstance(periods, ast.UnaryOp) and isinstance(periods.op, ast.USub):
                    return f"future-looking negative {method} is forbidden"
            if method == "rolling":
                for keyword in node.keywords:
                    if keyword.arg == "center" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                        return "rolling(center=True) is forbidden"
            if method == "interpolate":
                return "interpolate() fills from future values"
            if method == "fillna":
                for keyword in node.keywords:
                    if keyword.arg == "method" and isinstance(keyword.value, ast.Constant) \
                            and str(keyword.value.value) in {"bfill", "backfill"}:
                        return "fillna(method='bfill') fills from future values"
            reason = _lookahead_reason(node)
            if reason:
                return reason
        if isinstance(node, ast.Slice) and node.step is not None:
            step = node.step
            if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
                return "reversed slicing ([::-1]) lets later bars flow into earlier ones"
    return None


def _cross_sectional(data: pd.DataFrame, values, how: str) -> pd.Series:
    """Rank or z-score ``values`` within each timestamp (one cross-section)."""
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values), index=data.index)
    series = pd.to_numeric(series, errors="coerce")
    keys = data["timestamp"] if series.index.equals(data.index) else data["timestamp"].reindex(series.index)
    grouped = series.groupby(keys.to_numpy(), sort=False)
    if how == "rank":
        return grouped.rank(pct=True)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (series - mean) / std


def _per_symbol_rolling(data: pd.DataFrame, values, window, method: str) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values), index=data.index)
    keys = data["symbol"] if series.index.equals(data.index) else data["symbol"].reindex(series.index)
    grouped = series.groupby(keys.to_numpy(), sort=False)
    return grouped.transform(lambda s: getattr(s.rolling(int(window), min_periods=1), method)())


def _bind_data_helpers(namespace: dict[str, Any], data: pd.DataFrame) -> None:
    """Replace helpers whose old versions used the whole sample or mixed symbols.

    * ``rank``/``zscore`` used to rank/normalise over *all* rows (every
      timestamp, i.e. including the future).  They are now cross-sectional.
    * ``ddb_rolling_*`` used ``Series.rolling`` on a frame sorted by
      ``[timestamp, symbol]``, mixing different coins.  They now roll per symbol.
    """
    namespace["rank"] = lambda values: _cross_sectional(data, values, "rank")
    namespace["zscore"] = lambda values: _cross_sectional(data, values, "zscore")
    for method in ("mean", "sum", "std", "min", "max"):
        namespace[f"ddb_rolling_{method}"] = (
            lambda values, window, m=method: _per_symbol_rolling(data, values, window, m)
        )


def _safe_div(left, right):
    result = left / right
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    return pd.Series(result).replace([np.inf, -np.inf], np.nan)


def _zscore(values):
    """Numerically safe fallback for legacy generated ``zscore(x)`` calls."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    std = series.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def _group_zscore(series: pd.Series) -> pd.Series:
    """Callable replacement for legacy groupby.transform('zscore')."""
    std = series.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def _rank(values):
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.rank(pct=True)


def _group_shift(df, values, periods=1):
    if isinstance(values, str):
        values = df[values]
    return values.groupby(df["symbol"], sort=False).shift(periods)


def _group_rolling(df, values, window, method):
    if isinstance(values, str):
        values = df[values]
    grouped = values.groupby(df["symbol"], sort=False)
    result = getattr(grouped.rolling(int(window), min_periods=1), method)()
    result = result.reset_index(level=0, drop=True).sort_index()
    counts = grouped.cumcount()
    return result.where(counts >= int(window) - 1)


def _safe_globals() -> dict[str, Any]:
    builtins = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len,
                "float": float, "int": int, "bool": bool, "round": round}
    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "np": np, "numpy": np, "pd": pd, "pandas": pd,
        "safe_div": _safe_div,
        "zscore": _zscore,
        "rank": _rank,
        "_group_zscore": _group_zscore,
        "group_shift": _group_shift,
    }
    for method in ("mean", "sum", "std", "min", "max"):
        namespace[f"rolling_{method}"] = lambda df, values, window, m=method: _group_rolling(df, values, window, m)
        namespace[f"ddb_rolling_{method}"] = lambda values, window, m=method: getattr(values.rolling(int(window), min_periods=1), m)()
    return namespace


class _RemoveAllowedImports(ast.NodeTransformer):
    """Remove generated numpy/pandas imports; those names are pre-bound safely."""

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        return None


class _RewriteLegacyCalls(ast.NodeTransformer):
    """Translate generated pandas spellings that are not native APIs."""

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "transform"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "zscore"
        ):
            node.args[0] = ast.Name(id="_group_zscore", ctx=ast.Load())
        return node


def _factor_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose both Predict and Dpsk-style aliases to generated code."""
    # ``next_funding_time`` is used only to construct the event label. Never
    # expose it to generated code: doing so would leak the future event clock.
    # Labels are constructed from future closes.  They may be present on the
    # prepared research frame for evaluation/reporting, but must never be
    # visible to generated factor code.  The event clock is future metadata
    # for label construction and is excluded for the same reason.
    out = frame.drop(
        columns=[
            "target", "raw_target", "next_funding_time", "funding_time",
            # Settlement paid during the *next* holding window: accounting only.
            "funding_event_rate",
        ],
        errors="ignore",
    ).copy().reset_index(drop=True)
    out["ticker"] = out["symbol"]
    out["dataDate"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    out["barTime"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%H:%M:%S")
    aliases = {
        "openPrice": "open", "highPrice": "high", "lowPrice": "low",
        "closePrice": "close", "totalVolume": "volume", "totalValue": "quote_volume",
        "chgPct": "log_ret_1", "turnoverRate": "quote_volume",
    }
    for dst, src in aliases.items():
        if dst not in out and src in out:
            out[dst] = out[src]
    # Compatibility aliases used by earlier generated factor batches.  Keep
    # their meanings explicit and causal so old JSONL files can be replayed
    # without regenerating them against a different schema.
    grouped = out.groupby("symbol", sort=False)
    if "ret" not in out and "log_ret_1" in out:
        out["ret"] = out["log_ret_1"]
    if "ret_1" not in out and "log_ret_1" in out:
        out["ret_1"] = out["log_ret_1"]
    if "lag1" not in out and "close" in out:
        out["lag1"] = grouped["close"].shift(1)
    if "signed_vol" not in out:
        # Quote-volume direction proxy, matching the historical crypto
        # factor convention: positive candle => positive flow, negative candle
        # => negative flow.  It remains NaN when either input is unavailable.
        if "quote_volume" in out and "log_ret_1" in out:
            out["signed_vol"] = out["quote_volume"] * np.sign(out["log_ret_1"])
        elif "volume" in out and "log_ret_1" in out:
            out["signed_vol"] = out["volume"] * np.sign(out["log_ret_1"])
    if "ret_vol" not in out and "ret" in out:
        if "log_volume" in out:
            out["ret_vol"] = out["ret"] * out["log_volume"]
        elif "volume" in out:
            out["ret_vol"] = out["ret"] * np.log1p(pd.to_numeric(out["volume"], errors="coerce"))
    if "dollar_volume" not in out:
        if "quote_volume" in out:
            out["dollar_volume"] = out["quote_volume"]
        elif "volume" in out and "close" in out:
            out["dollar_volume"] = out["volume"] * out["close"]
    if "vol" not in out and "volume" in out:
        out["vol"] = out["volume"]
    # Names emitted by the older liquidity/price-volume prompt templates.
    # They are deliberately transparent proxies, not additional information.
    if "log_qv" not in out and "quote_volume" in out:
        out["log_qv"] = np.log1p(pd.to_numeric(out["quote_volume"], errors="coerce"))
    if "log_vol" not in out and "volume" in out:
        out["log_vol"] = np.log1p(pd.to_numeric(out["volume"], errors="coerce"))
    if "raw" not in out and "quote_volume" in out:
        out["raw"] = pd.to_numeric(out["quote_volume"], errors="coerce")
    if "amihud" not in out and "ret" in out and "quote_volume" in out:
        out["amihud"] = out["ret"].abs() / out["quote_volume"].replace(0, np.nan)
    if "impact" not in out and "ret" in out and "quote_volume" in out:
        out["impact"] = out["ret"].abs() / np.sqrt(out["quote_volume"].replace(0, np.nan))
    if "share" not in out and "quote_volume" in out:
        # Cross-sectional market share at each bar.  (It used to divide by the
        # symbol's whole-history volume, which leaked future volume.)
        out["share"] = out["quote_volume"] / out.groupby("timestamp")["quote_volume"].transform("sum").replace(0, np.nan)
    if "liq" not in out and "quote_volume" in out:
        out["liq"] = out["quote_volume"]
    # Additional precomputed names emitted by the old liquidity templates.
    # All are causal per-symbol rolling/proxy values.
    symbol_key = out["symbol"]
    def rolling(series: pd.Series, window: int, method: str) -> pd.Series:
        grouped_series = series.groupby(symbol_key, sort=False)
        return grouped_series.transform(
            lambda values: getattr(values.rolling(window, min_periods=window), method)()
        )
    quote = pd.to_numeric(out.get("quote_volume"), errors="coerce")
    ret = pd.to_numeric(out.get("ret"), errors="coerce")
    if quote is not None:
        out["quote_vol_ma288"] = rolling(quote, 288, "mean")
        out["quote_vol_std48"] = rolling(quote, 48, "std")
        out["totalValue_ma30"] = rolling(quote, 30, "mean")
        out["totalValue_ma60"] = rolling(quote, 60, "mean")
        out["totalValue_std30"] = rolling(quote, 30, "std")
    if ret is not None:
        out["abs_ret"] = ret.abs()
        out["abs_ret_1"] = ret.abs()
        out["abs_log_ret"] = ret.abs()
        out["log_ret_std48"] = rolling(ret, 48, "std")
        out["_neg_ret"] = ret.clip(upper=0)
        if quote is not None:
            denominator = quote.replace(0, np.nan)
            out["funding_rate_value"] = pd.to_numeric(
                out.get("funding_rate"), errors="coerce"
            ) * quote
            out["ret_value"] = ret * quote
            out["weighted_ret"] = ret * quote
            out["amihud_raw"] = ret.abs() / denominator
            out["_amihud"] = out["amihud_raw"]
            out["impact_ratio"] = ret.abs() / np.sqrt(denominator)
            out["vol_up"] = quote.where(ret > 0, 0.0)
    if "high" in out and "low" in out and "close" in out:
        out["range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    if "volume" in out:
        out["vol_diff"] = pd.to_numeric(out["volume"], errors="coerce").groupby(
            symbol_key, sort=False
        ).diff()
    return out


def evaluate_factor_record(
    frame: pd.DataFrame,
    record: dict[str, Any],
    *,
    prepared_input: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run one generated factor and return prediction/target rows."""
    data = prepared_input if prepared_input is not None else _factor_input(frame)
    allowed = set(data.columns)
    code = str(record.get("code", ""))
    reason = validate_factor_code(code, allowed)
    if reason:
        raise ValueError(f"{record.get('factor_id', '<unknown>')}: {reason}")
    namespace = _safe_globals()
    _bind_data_helpers(namespace, data)
    # Imports are already supplied by the evaluator and are removed to keep
    # generated code unable to import anything at runtime.
    tree = ast.parse(code)
    # Imports are allowed only for numpy/pandas by validate_factor_code. Remove
    # them at every nesting level because generated factors sometimes put an
    # import inside compute_factor; the restricted builtins intentionally omit
    # __import__ to prevent arbitrary module access.
    tree = _RemoveAllowedImports().visit(tree)
    tree = _RewriteLegacyCalls().visit(tree)
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<generated-crypto-factor>", "exec"), namespace)
    # Older generated candidates frequently use groupby.apply.  It is valid
    # for the current evaluator but pandas emits a deprecation warning for
    # every factor, which can flood multiprocessing workers and obscure real
    # failures.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        raw = namespace["compute_factor"](data)
    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] != 1:
            raise ValueError("compute_factor must return one Series")
        raw = raw.iloc[:, 0]
    values = raw if isinstance(raw, pd.Series) else pd.Series(np.asarray(raw))
    if len(values) != len(data):
        raise ValueError(
            f"factor length mismatch: {len(values)} != {len(data)} "
            f"(factor_id={record.get('factor_id', '<unknown>')})"
        )
    if not values.index.equals(data.index):
        if values.index.is_unique:
            values = values.reindex(data.index)
        else:
            values = pd.Series(values.to_numpy(), index=data.index)
    sign = -1.0 if str(record.get("direction", "positive")).lower() == "negative" else 1.0
    result = frame[["timestamp", "symbol", "target", "raw_target", "quote_volume"]].copy()
    if "funding_event_rate" in frame:
        result["funding_event_rate"] = pd.to_numeric(frame["funding_event_rate"], errors="coerce").fillna(0.0)
    result["funding_rate"] = pd.to_numeric(
        frame["funding_rate_raw"] if "funding_rate_raw" in frame else frame["funding_rate"],
        errors="coerce",
    )
    result["prediction"] = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) * sign
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def load_factor_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if str(record.get("status", "")).lower() not in {"rejected", "train_rejected"}:
                    records.append(record)
    return records
