"""Shared helpers for the Streamlit pages: paths, cached loaders, chart styling, formatting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets the app import couponroi without installing it

APP_DIR = ROOT / "artifacts" / "app"
FIG_DIR = ROOT / "reports" / "figures"
REPO_URL = "https://github.com/<your-username>/coupon-roi-targeting"

INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
NEUTRAL = "#c3c2b7"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

pio.templates["coupon"] = go.layout.Template(
    layout={
        "font": {"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "color": INK, "size": 13},
        "colorway": SERIES,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis": {
            "gridcolor": GRID,
            "linecolor": AXIS,
            "zeroline": False,
            "ticks": "",
            "tickfont": {"color": INK_2},
        },
        "yaxis": {
            "gridcolor": GRID,
            "linecolor": AXIS,
            "zeroline": False,
            "ticks": "",
            "tickfont": {"color": INK_2},
        },
        "margin": {"l": 10, "r": 10, "t": 50, "b": 10},
        "hoverlabel": {"bgcolor": "white", "font": {"color": INK}},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "font": {"color": INK_2}},
        "title": {"x": 0, "xanchor": "left", "font": {"size": 16}},
    }
)
pio.templates.default = "coupon"


def page(title: str, icon: str = "🎯") -> None:
    st.set_page_config(page_title=f"{title} | Coupon ROI", page_icon=icon, layout="wide")
    st.markdown(
        """<style>
        .block-container {padding-top: 2.2rem; max-width: 1200px;}
        [data-testid="stMetricValue"] {font-size: 1.7rem;}
        .small-note {color: #52514e; font-size: 0.85rem;}
        </style>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def results(name: str) -> dict:
    path = APP_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_resource(show_spinner=False)
def db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(
        str(APP_DIR / "app.duckdb"), read_only=True, config={"enable_external_access": False}
    )


@st.cache_data(show_spinner=False)
def query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    con = db().cursor()
    return con.execute(sql, list(params) if params else None).df()


def money(x: float, decimals: int = 0) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.1f}M"
    if x >= 1e4:
        return f"{sign}${x / 1e3:.1f}K"
    return f"{sign}${x:,.{decimals}f}"


def bundle_ready() -> bool:
    if (APP_DIR / "app.duckdb").exists():
        return True
    st.error("The app bundle is missing. Run `couponroi all` (or `make all`) first, then refresh.")
    st.stop()
    return False


def footer() -> None:
    st.divider()
    st.markdown(
        '<p class="small-note">Data: 84.51° The Complete Journey 2.0 (2,469 households, 2017), via the CC0 '
        "completejourney package. Money figures marked simulated use assumed margins and costs; the data has neither.</p>",
        unsafe_allow_html=True,
    )
