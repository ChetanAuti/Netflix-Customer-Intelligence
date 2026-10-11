"""Shared Streamlit helpers (page setup, cached SQL, small chart utilities)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.database.connection import query_df  # noqa: E402
from src.llm.client import get_llm  # noqa: E402

RED = "#E50914"


def setup(title: str, icon: str = "🎬"):
    st.set_page_config(page_title=f"{title} | Customer Intelligence", page_icon=icon, layout="wide")
    st.markdown(f"<style>h1, h2, h3 {{color:{RED};}} div[data-testid='stMetricValue'] {{color:{RED};}}</style>", unsafe_allow_html=True)
    st.title(f"{icon} {title}")
    


@st.cache_data(ttl=600, show_spinner=False)
def sql(q: str) -> pd.DataFrame:
    return query_df(q)


def bar(df: pd.DataFrame, x: str, y: str, horizontal: bool = False):
    if df.empty:
        st.caption("No data.")
        return
    st.bar_chart(df.set_index(x)[y], horizontal=horizontal, color=RED)
