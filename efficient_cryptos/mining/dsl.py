"""Translate the restricted generated-factor Python subset to Numba ASTs.

The translator intentionally rejects arbitrary/groupby/apply code. Rejected
records remain available for a slower compatibility path, while only this
restricted DSL is sent to the wide-table Numba engine.
"""

from __future__ import annotations

import ast
from typing import Any


FIELD_NAMES = {
    "closePrice", "openPrice", "highPrice", "lowPrice", "chgPct",
    "totalVolume", "totalValue", "turnoverRate",
}
FIELD_ALIASES = {
    "close": "closePrice", "open": "openPrice", "high": "highPrice",
    "low": "lowPrice", "volume": "totalVolume", "amount": "totalValue",
}


def _field(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "df":
        key = node.slice.value if isinstance(node.slice, ast.Constant) else None
        if isinstance(key, str):
            if key in FIELD_NAMES:
                return key
            if key in FIELD_ALIASES:
                return FIELD_ALIASES[key]
    return None


def _is_frame_wrapper(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "df":
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"sort_values", "sort_index", "copy", "reset_index", "set_index"}:
            return _is_frame_wrapper(node.func.value)
    return False


def _wrapped_field(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    key = node.slice.value if isinstance(node.slice, ast.Constant) else None
    if isinstance(key, str) and _is_frame_wrapper(node.value):
        return key if key in FIELD_NAMES else FIELD_ALIASES.get(key)
    return None


def _number(node: ast.AST) -> float | int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number(node.operand)
        return -value if value is not None else None
    return None


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return f"{node.func.value.id}.{node.func.attr}"
    return None


def _groupby_field(node: ast.AST) -> str | None:
    """Recognize df.groupby('ticker')['field'] without materializing a groupby."""
    if not isinstance(node, ast.Subscript):
        return None
    key = node.slice.value if isinstance(node.slice, ast.Constant) else None
    base = node.value
    direct_group = (
        isinstance(base, ast.Call)
        and isinstance(base.func, ast.Attribute)
        and isinstance(base.func.value, ast.Name)
        and base.func.value.id == "df"
        and base.func.attr == "groupby"
        and base.args
        and isinstance(base.args[0], ast.Constant)
        and base.args[0].value == "ticker"
        and isinstance(key, str)
        and (key in FIELD_NAMES or key in FIELD_ALIASES)
    )
    named_group = (
        isinstance(base, ast.Name)
        and base.id.startswith("__groupby_ticker_")
    )
    if direct_group or named_group:
        return key if key in FIELD_NAMES else FIELD_ALIASES.get(key)
    return None


def _ticker_groupby_series(node: ast.AST, env: dict[str, Any]) -> Any | None:
    """Translate df.groupby('ticker')['field'] into the field matrix."""
    return _groupby_field(node)


def _keyword_number(node: ast.Call, name: str) -> float | int | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return _number(keyword.value)
    return None


def _translate(node: ast.AST, env: dict[str, Any]) -> Any:
    direct = _field(node) or _wrapped_field(node)
    if direct:
        return direct
    grouped = _groupby_field(node)
    if grouped:
        return grouped
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {"op": "constant", "value": float(node.value)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return {"op": "mul", "left": {"op": "constant", "value": -1.0}, "right": _translate(node.operand, env)}
    if isinstance(node, ast.BinOp):
        op_map = {
            ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
            ast.BitAnd: "and", ast.BitOr: "or",
        }
        if isinstance(node.op, ast.Pow):
            exponent = _number(node.right)
            if exponent is None:
                raise ValueError("power exponent must be constant")
            if exponent == 2:
                base = _translate(node.left, env)
                return {"op": "mul", "left": base, "right": base}
            return {"op": "pow", "input": _translate(node.left, env), "exponent": float(exponent)}
        op = next((name for cls, name in op_map.items() if isinstance(node.op, cls)), None)
        if op is None:
            raise ValueError(f"unsupported binary operator: {type(node.op).__name__}")
        return {"op": op, "left": _translate(node.left, env), "right": _translate(node.right, env)}
    if isinstance(node, ast.BoolOp) and len(node.values) >= 2:
        op = "and" if isinstance(node.op, ast.And) else "or"
        value = _translate(node.values[0], env)
        for child in node.values[1:]:
            value = {"op": op, "left": value, "right": _translate(child, env)}
        return value
    if isinstance(node, ast.IfExp):
        return {
            "op": "where",
            "condition": _translate(node.test, env),
            "left": _translate(node.body, env),
            "right": _translate(node.orelse, env),
        }
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        op_map = {
            ast.Gt: "gt", ast.GtE: "ge", ast.Lt: "lt",
            ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "ne",
        }
        op = next((name for cls, name in op_map.items() if isinstance(node.ops[0], cls)), None)
        if op is None:
            raise ValueError(f"unsupported comparison: {type(node.ops[0]).__name__}")
        return {"op": op, "left": _translate(node.left, env), "right": _translate(node.comparators[0], env)}
    if isinstance(node, ast.Call):
        name = _call_name(node)
        args = list(node.args)
        if isinstance(node.func, ast.Attribute) and not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"np", "numpy"}
        ):
            attr = node.func.attr
            base_node = node.func.value
            # Sorting/copying/index resets do not change numeric values. They
            # are common Pandas wrappers around an otherwise executable series.
            if attr in {"sort_values", "sort_index", "copy", "reset_index", "set_index"}:
                return _translate(base_node, env)
            if attr == "astype":
                return _translate(base_node, env)
            if attr == "fillna" and len(args) == 1:
                return {"op": "fillna", "input": _translate(base_node, env), "replacement": _translate(args[0], env)}
            # Series/grouped-series transform(lambda x: x.rolling(w).mean())
            if attr == "transform" and args and isinstance(args[0], ast.Lambda):
                field = _groupby_field(base_node)
                if field is None and isinstance(base_node, ast.Subscript):
                    key = base_node.slice.value if isinstance(base_node.slice, ast.Constant) else None
                    group_name = base_node.value.id if isinstance(base_node.value, ast.Name) else None
                    if isinstance(key, str) and group_name and str(env.get(group_name, "")).startswith("__groupby_ticker_"):
                        field = key if key in FIELD_NAMES else FIELD_ALIASES.get(key)
                if field and len(args[0].args.args) == 1:
                    local_env = dict(env)
                    local_env[args[0].args.args[0].arg] = field
                    return _translate(args[0].body, local_env)
            # Chained rolling syntax: x.rolling(w).mean()/std()/sum()/min()/max().
            if attr in {"mean", "std", "sum", "min", "max"} and isinstance(base_node, ast.Call):
                if isinstance(base_node.func, ast.Attribute) and base_node.func.attr == "rolling":
                    window = _number(base_node.args[0]) if base_node.args else _keyword_number(base_node, "window")
                    if window is None:
                        raise ValueError("rolling window must be constant")
                    op = f"rolling_{attr}"
                    return {"op": op, "input": _translate(base_node.func.value, env), "window": int(window)}
            # Direct Series.rank(), z-score helpers and fillna().
            if attr == "rank":
                return {"op": "rank", "input": _translate(base_node, env)}
            if attr in {"abs", "log", "log1p", "sqrt", "exp"} and not args:
                unary = {"abs": "abs", "log": "safe_log", "log1p": "signed_log1p", "sqrt": "signed_sqrt", "exp": "safe_exp"}
                return {"op": unary[attr], "input": _translate(base_node, env)}
            if attr == "clip":
                lower_node = next((keyword.value for keyword in node.keywords if keyword.arg == "lower"), None)
                upper_node = next((keyword.value for keyword in node.keywords if keyword.arg == "upper"), None)
                if lower_node is None and args:
                    lower_node = args[0]
                if upper_node is None and len(args) > 1:
                    upper_node = args[1]
                if lower_node is None or upper_node is None:
                    raise ValueError("clip requires lower and upper")
                lower = _number(lower_node)
                upper = _number(upper_node)
                return {
                    "op": "clip",
                    "input": _translate(base_node, env),
                    "lower": float(lower) if lower is not None else _translate(lower_node, env),
                    "upper": float(upper) if upper is not None else _translate(upper_node, env),
                }
            if attr == "where" and args:
                condition = _translate(args[0], env)
                replacement = _translate(args[1], env) if len(args) > 1 else {"op": "constant", "value": float("nan")}
                return {"op": "where", "condition": condition, "left": _translate(base_node, env), "right": replacement}
        if name in {"rolling_mean", "ddb_rolling_mean"} and len(args) >= 2:
            value = _translate(args[-2] if name == "rolling_mean" else args[0], env)
            window = _number(args[-1] if name == "rolling_mean" else args[1])
            if window is None:
                raise ValueError("rolling window must be constant")
            return {"op": "rolling_mean", "input": value, "window": int(window)}
        if name in {"rolling_std", "ddb_rolling_std"} and len(args) >= 2:
            value = _translate(args[-2] if name == "rolling_std" else args[0], env)
            window = _number(args[-1] if name == "rolling_std" else args[1])
            if window is None:
                raise ValueError("rolling window must be constant")
            return {"op": "rolling_std", "input": value, "window": int(window)}
        if name in {"rolling_sum", "ddb_rolling_sum", "rolling_min", "ddb_rolling_min", "rolling_max", "ddb_rolling_max"} and len(args) >= 2:
            base = name.replace("ddb_", "")
            if name.startswith("rolling_") and not name.startswith("ddb_"):
                value = _translate(args[-2], env)
                window_node = args[-1]
            else:
                value = _translate(args[0], env)
                window_node = args[1]
            window = _number(window_node)
            if window is None:
                raise ValueError("rolling window must be constant")
            return {"op": base, "input": value, "window": int(window)}
        unary = {
            "abs": "abs", "np.abs": "abs", "numpy.abs": "abs",
            "signed_log1p": "signed_log1p", "np.log1p": "signed_log1p",
            "signed_sqrt": "signed_sqrt", "np.sqrt": "signed_sqrt",
            "safe_log": "safe_log", "np.log": "safe_log", "numpy.log": "safe_log",
            "safe_exp": "safe_exp", "np.exp": "safe_exp", "numpy.exp": "safe_exp",
            "sign": "sign", "np.sign": "sign", "numpy.sign": "sign",
            "tanh": "tanh", "np.tanh": "tanh", "numpy.tanh": "tanh",
        }.get(name)
        if unary and len(args) == 1:
            return {"op": unary, "input": _translate(args[0], env)}
        if name in {"safe_div", "div"} and len(args) == 2:
            return {"op": "div", "left": _translate(args[0], env), "right": _translate(args[1], env)}
        if name in {"rank", "cs_rank", "cross_sectional_rank"} and len(args) == 1:
            return {"op": "rank", "input": _translate(args[0], env)}
        if name in {"zscore", "cs_zscore", "cross_sectional_zscore"} and len(args) == 1:
            return {"op": "zscore", "input": _translate(args[0], env)}
        if name in {"np.clip", "numpy.clip"} and len(args) >= 3:
            lower = _number(args[1])
            upper = _number(args[2])
            return {
                "op": "clip",
                "input": _translate(args[0], env),
                "lower": float(lower) if lower is not None else _translate(args[1], env),
                "upper": float(upper) if upper is not None else _translate(args[2], env),
            }
        if name in {"clip", "winsorize"} and len(args) >= 3:
            lower = _number(args[1])
            upper = _number(args[2])
            return {
                "op": "clip",
                "input": _translate(args[0], env),
                "lower": float(lower) if lower is not None else _translate(args[1], env),
                "upper": float(upper) if upper is not None else _translate(args[2], env),
            }
        if name in {"np.maximum", "numpy.maximum", "maximum", "max"} and len(args) == 2:
            return {"op": "max", "left": _translate(args[0], env), "right": _translate(args[1], env)}
        if name in {"np.minimum", "numpy.minimum", "minimum", "min"} and len(args) == 2:
            return {"op": "min", "left": _translate(args[0], env), "right": _translate(args[1], env)}
        if name in {"np.where", "numpy.where"} and len(args) == 3:
            return {
                "op": "where",
                "condition": _translate(args[0], env),
                "left": _translate(args[1], env),
                "right": _translate(args[2], env),
            }
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"shift", "diff", "pct_change"}:
            periods = _number(args[0]) if args else 1
            if periods is None or periods < 0:
                raise ValueError("future-looking shift/diff is unsupported")
            base = _translate(node.func.value, env)
            if node.func.attr == "shift":
                return {"op": "shift", "input": base, "periods": int(periods)}
            if node.func.attr == "diff":
                return {"op": "sub", "left": base, "right": {"op": "shift", "input": base, "periods": int(periods)}}
            if node.func.attr == "pct_change":
                previous = {"op": "shift", "input": base, "periods": int(periods)}
                return {
                    "op": "sub",
                    "left": {"op": "div", "left": base, "right": previous},
                    "right": {"op": "constant", "value": 1.0},
                }
        raise ValueError(f"unsupported call: {name or ast.dump(node.func)}")
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def compile_factor_code(code: str) -> Any:
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_factor"]
    if len(functions) != 1:
        raise ValueError("exactly one compute_factor is required")
    env: dict[str, Any] = {}
    function = functions[0]
    for statement in function.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target = statement.targets[0].id
            if target == "df" and _is_frame_wrapper(statement.value):
                continue
            if (
                isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and isinstance(statement.value.func.value, ast.Name)
                and statement.value.func.value.id == "df"
                and statement.value.func.attr == "groupby"
                and statement.value.args
                and isinstance(statement.value.args[0], ast.Constant)
                and statement.value.args[0].value == "ticker"
            ):
                env[target] = f"__groupby_ticker_{target}"
                continue
            env[target] = _translate(statement.value, env)
        elif isinstance(statement, ast.Return):
            return _translate(statement.value, env)
        elif isinstance(statement, ast.If):
            if len(statement.body) == 1 and isinstance(statement.body[0], ast.Return) and len(statement.orelse) == 1 and isinstance(statement.orelse[0], ast.Return):
                return {
                    "op": "where",
                    "condition": _translate(statement.test, env),
                    "left": _translate(statement.body[0].value, env),
                    "right": _translate(statement.orelse[0].value, env),
                }
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        else:
            raise ValueError(f"unsupported compute_factor statement: {type(statement).__name__}")
    raise ValueError("compute_factor has no return")


def compile_factor_record(record: dict[str, Any]) -> tuple[Any | None, str | None]:
    try:
        if record.get("expression_ast") is not None:
            return record["expression_ast"], None
        return compile_factor_code(str(record.get("code", ""))), None
    except Exception as exc:
        return None, f"DSL compile failed: {type(exc).__name__}: {exc}"
