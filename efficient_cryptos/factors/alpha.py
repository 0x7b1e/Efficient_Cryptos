"""Self-contained Alpha101/191 formula loading and evaluation helpers."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
import numpy as np


def _install_imports() -> None:
    root = Path(__file__).resolve().parent / "popbo_alphas"  # vendored popbo Alpha101/191
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "datas" not in sys.modules:
        sys.modules["datas"] = types.ModuleType("datas")


def _fast_rank(arr: np.ndarray) -> float:
    valid = arr[np.isfinite(arr)]
    return float((valid <= arr[-1]).sum()) if len(valid) and np.isfinite(arr[-1]) else np.nan


def _fast_decay(arr: np.ndarray) -> float:
    mask = np.isfinite(arr)
    if not mask.any():
        return np.nan
    weights = np.where(mask, np.arange(1, len(arr) + 1, dtype=float), 0.0)
    return float(np.dot(np.where(mask, arr, 0.0), weights) / weights.sum()) if weights.sum() else np.nan


def alpha_class(family: str):
    _install_imports()
    family = family.lower()
    module_name = "alphas101" if family == "alpha101" else "alphas191" if family == "alpha191" else ""
    if not module_name:
        raise ValueError(f"Unknown Alpha family: {family}")
    mod = importlib.import_module(module_name)
    if family == "alpha101":
        mod.returns = lambda df: df.pct_change()
        cls_name = "Alphas101"
    else:
        mod.Returns = lambda df: df.pct_change()
        cls_name = "Alphas191"
    def rank(df, window=10): return df.rolling(int(window)).apply(_fast_rank, raw=True)
    def argmax(df, window=10): return df.rolling(int(window)).apply(lambda x: float(np.nanargmax(x) + 1) if np.isfinite(x).any() else np.nan, raw=True)
    def argmin(df, window=10): return df.rolling(int(window)).apply(lambda x: float(np.nanargmin(x) + 1) if np.isfinite(x).any() else np.nan, raw=True)
    def decay(df, period=10): return df.rolling(int(period)).apply(_fast_decay, raw=True)
    for name in ("ts_rank", "Tsrank"): setattr(mod, name, rank)
    for name in ("ts_argmax", "Tsargmax"): setattr(mod, name, argmax)
    for name in ("ts_argmin", "Tsargmin"): setattr(mod, name, argmin)
    for name in ("decay_linear", "Decaylinear"): setattr(mod, name, decay)
    return getattr(mod, cls_name)


def alpha_names(family: str) -> list[str]:
    cls = alpha_class(family)
    return sorted(name for name in dir(cls) if name.startswith("alpha") and callable(getattr(cls, name)))

