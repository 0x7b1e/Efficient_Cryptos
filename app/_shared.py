"""Small UI helpers shared by the dashboard pages."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    """Make ``efficient_cryptos`` importable however streamlit was launched.

    Started from the repo's parent directory, the repo folder itself can be
    picked up as an empty namespace package named ``efficient_cryptos`` and
    shadow the real one; drop it and put the repo root first.
    """
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))
    module = sys.modules.get("efficient_cryptos")
    if module is not None and getattr(module, "__file__", None) is None:
        for name in [n for n in sys.modules if n == "efficient_cryptos" or n.startswith("efficient_cryptos.")]:
            del sys.modules[name]


_ensure_repo_on_path()

import streamlit as st  # noqa: E402
OUTPUTS = ROOT / "outputs"

UP = "#16a34a"      # long / gain
DOWN = "#dc2626"    # short / loss
LIVE = "#16a34a"
SHADOW = "#f59e0b"


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def path_picker(label: str, candidates: list[Path], key: str, *, optional: bool = False,
                help: str | None = None) -> Path | None:
    """Selectbox of discovered directories plus a free-text override."""
    options = (["（不使用）"] if optional else []) + [rel(p) for p in candidates] + ["手动输入…"]
    choice = st.sidebar.selectbox(label, options, key=f"{key}_select", help=help)
    if choice == "手动输入…":
        typed = st.sidebar.text_input(f"{label}路径", key=f"{key}_text", placeholder="outputs/...")
        return resolve(typed) if typed.strip() else None
    if choice == "（不使用）":
        return None
    return resolve(choice)


def fmt_num(value, digits: int = 2, suffix: str = "") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}{suffix}"


def fmt_age(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    if seconds < 90:
        return f"{seconds:.0f} 秒前"
    if seconds < 5400:
        return f"{seconds / 60:.0f} 分钟前"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} 小时前"
    return f"{seconds / 86400:.1f} 天前"
