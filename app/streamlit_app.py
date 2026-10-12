"""Efficient Cryptos dashboard (read-only).

    pip install -e ".[ui]"
    streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1

The app only reads files under outputs/ (or any directory you type in);
it never holds API keys and never sends orders.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

import _shared  # noqa: E402,F401  (puts the repo root on sys.path)
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Efficient Cryptos", page_icon="📈", layout="wide")

pages = [
    st.Page(str(ROOT / "app" / "pages" / "factors.py"), title="因子排行榜", icon="🏆", default=True),
    st.Page(str(ROOT / "app" / "pages" / "live.py"), title="实盘监控", icon="📡"),
]
st.navigation(pages).run()
