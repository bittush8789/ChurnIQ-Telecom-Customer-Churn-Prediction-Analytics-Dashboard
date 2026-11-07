"""
Telecom Customer Churn Prediction & Analytics Dashboard
=======================================================

A Streamlit application for exploring synthetic telecom customer data,
comparing churn models, scoring individual customers, and viewing
rule-based retention recommendations.

Run with::

    streamlit run app.py

IMPORTANT: The dataset is synthetic. Every metric and prediction shown here is
for demonstration and learning purposes only.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import (
    DATA_PATH,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    ensure_dataset,
    ensure_model,
    predict_customer,
    score_dataframe,
)

# ---------------------------------------------------------------------------
# Page configuration and design tokens
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnIQ · Telecom Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#4f46e5"
PRIMARY_LIGHT = "#eef2ff"
NAVY = "#0f172a"
SLATE = "#64748b"
SUCCESS = "#10b981"
DANGER = "#ef4444"
WARNING = "#f59e0b"
CHURN_COLORS = {"No": SUCCESS, "Yes": DANGER}
RISK_COLORS = {"High": DANGER, "Medium": WARNING, "Low": SUCCESS}
RISK_LIGHT = {"High": "#fef2f2", "Medium": "#fffbeb", "Low": "#ecfdf5"}

PAGES = [
    "Dashboard Overview",
    "Customer Analytics",
    "Model Performance",
    "Churn Prediction",
    "Retention Insights",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 88% 0%, #e0e7ff 0%, rgba(224,231,255,0) 34%),
        linear-gradient(180deg, #f8fafc 0%, #f4f6fb 100%);
}
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.2rem;
    max-width: 1450px;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

/* ----------------------------- Sidebar ----------------------------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #111827 58%, #1e1b4b 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.3rem; }
[data-testid="stSidebar"] * { color: #cbd5e1; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #f8fafc; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] .stRadio label {
    padding: 9px 12px;
    border-radius: 10px;
    font-size: 14px;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.07); }
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(99,102,241,0.28);
    box-shadow: inset 0 0 0 1px rgba(129,140,248,0.55);
}
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.08);
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 10px;
    font-weight: 600;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.14);
    border-color: rgba(255,255,255,0.28);
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }

/* ----------------------------- Brand ------------------------------- */
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.brand-logo {
    width: 44px; height: 44px; border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    font-size: 22px; box-shadow: 0 8px 20px rgba(79,70,229,0.35);
}
.brand-name { font-size: 19px; font-weight: 800; color: #ffffff; letter-spacing: -0.3px; }
.brand-sub { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.3px; }
.sidebar-tag {
    display: inline-block; margin: 10px 0 4px 0; padding: 4px 10px;
    border-radius: 999px; background: rgba(245,158,11,0.14);
    border: 1px solid rgba(245,158,11,0.35);
    color: #fbbf24 !important; font-size: 10px; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase;
}
.sidebar-note {
    margin-top: 12px; padding: 12px 14px; border-radius: 12px;
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.28);
    color: #fca5a5 !important; font-size: 11.5px; line-height: 1.55;
}
.sidebar-stat {
    display: flex; justify-content: space-between; gap: 8px;
    padding: 7px 0; font-size: 12.5px; border-bottom: 1px dashed rgba(255,255,255,0.08);
}
.sidebar-stat span { color: #94a3b8 !important; }
.sidebar-stat b { color: #f1f5f9 !important; font-weight: 600; text-align: right; }

/* ----------------------------- Hero -------------------------------- */
.page-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #0f172a 0%, #312e81 62%, #4f46e5 100%);
    border-radius: 22px; padding: 26px 30px; color: #ffffff;
    box-shadow: 0 18px 40px rgba(49,46,129,0.22);
    margin-bottom: 18px;
}
.page-hero::after {
    content: ""; position: absolute; width: 280px; height: 280px;
    border-radius: 50%; background: rgba(255,255,255,0.07);
    top: -140px; right: -60px;
}
.hero-badge {
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #c7d2fe; margin-bottom: 10px;
}
.page-hero h1 {
    margin: 0 0 6px 0; font-size: 30px; font-weight: 800;
    letter-spacing: -0.7px; color: #ffffff;
}
.page-hero p {
    margin: 0; max-width: 780px; color: #cbd5e1;
    font-size: 13.5px; line-height: 1.6;
}
.page-hero .hero-meta { margin-top: 12px; font-size: 11.5px; color: #a5b4fc; }

/* ----------------------------- Sections ---------------------------- */
.section-title {
    font-size: 19px; font-weight: 700; color: #0f172a;
    letter-spacing: -0.3px; margin: 4px 0 2px 0;
}
.section-sub { font-size: 12.5px; color: #64748b; margin-bottom: 10px; }
.section-divider { height: 1px; background: #e2e8f0; margin: 10px 0 18px 0; }

/* ----------------------------- KPI cards --------------------------- */
.kpi-card {
    position: relative; overflow: hidden;
    background: #ffffff; border: 1px solid #e5e9f2;
    border-radius: 17px; padding: 17px 18px 15px 20px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.045);
    min-height: 124px; margin-bottom: 8px;
}
.kpi-card::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0;
    width: 4px; background: var(--accent);
}
.kpi-top { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.kpi-icon {
    width: 30px; height: 30px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; background: var(--accent-soft);
}
.kpi-label {
    font-size: 10.5px; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.9px;
}
.kpi-value {
    font-size: 27px; font-weight: 800; color: #0f172a;
    letter-spacing: -0.8px; line-height: 1.1;
}
.kpi-sub { font-size: 11px; color: #94a3b8; margin-top: 5px; }

/* ----------------------------- Callouts ---------------------------- */
.callout {
    background: #eef2ff; border: 1px solid #c7d2fe;
    border-left: 4px solid #4f46e5; border-radius: 12px;
    padding: 14px 16px; color: #312e81; font-size: 13px; line-height: 1.6;
}
.callout-warn {
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b; border-radius: 12px;
    padding: 13px 16px; color: #92400e; font-size: 12.5px; line-height: 1.6;
}
.callout h4 { margin: 0 0 6px 0; font-size: 14px; color: inherit; }
.callout ul { margin: 4px 0 0 0; padding-left: 18px; }
.callout li { margin-bottom: 5px; }

/* ----------------------------- Badges ------------------------------ */
.badge {
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.2px;
}
.badge-high { background: #fee2e2; color: #b91c1c; }
.badge-medium { background: #fef3c7; color: #b45309; }
.badge-low { background: #d1fae5; color: #047857; }
.badge-info { background: #e0e7ff; color: #3730a3; }
.badge-neutral { background: #f1f5f9; color: #475569; }

/* ----------------------------- Results ----------------------------- */
.result-card {
    background: #ffffff; border: 1px solid #e5e9f2; border-radius: 16px;
    padding: 18px 18px 14px 18px; box-shadow: 0 8px 22px rgba(15,23,42,0.045);
    margin-bottom: 12px;
}
.result-label {
    font-size: 10.5px; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.9px;
}
.result-status { font-size: 21px; font-weight: 800; color: #0f172a; margin: 2px 0 6px 0; }
.result-meta { font-size: 11.5px; color: #94a3b8; margin-top: 8px; }
.empty-state {
    background: #f8fafc; border: 1.5px dashed #cbd5e1; border-radius: 16px;
    padding: 34px 20px; text-align: center; color: #94a3b8; font-size: 13px;
    line-height: 1.6;
}
.empty-state b { display: block; color: #475569; font-size: 15px; margin-bottom: 4px; }

/* ----------------------------- Profile ----------------------------- */
.profile-grid {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 9px 16px; margin-top: 6px;
}
.profile-item {
    background: #f8fafc; border: 1px solid #eef2f7; border-radius: 10px;
    padding: 9px 11px;
}
.profile-item span {
    display: block; font-size: 10px; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;
}
.profile-item b { font-size: 12.5px; color: #1e293b; font-weight: 600; }

/* ----------------------------- Recommendations --------------------- */
.rec-card {
    background: #ffffff; border: 1px solid #e5e9f2; border-radius: 13px;
    padding: 13px 15px; margin-bottom: 10px;
    border-left: 4px solid #cbd5e1;
}
.rec-high { border-left-color: #ef4444; }
.rec-medium { border-left-color: #f59e0b; }
.rec-low { border-left-color: #10b981; }
.rec-head { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
.rec-icon { font-size: 16px; }
.rec-title { font-size: 13.5px; font-weight: 700; color: #0f172a; flex: 1; }
.rec-detail { font-size: 12px; color: #64748b; line-height: 1.55; margin-top: 5px; }
.rec-tag {
    font-size: 9.5px; font-weight: 700; letter-spacing: 0.6px;
    text-transform: uppercase; color: #94a3b8; margin-left: 4px;
}

/* ----------------------------- Footer ------------------------------ */
.footer {
    margin-top: 30px; padding: 18px 20px; border-radius: 14px;
    background: #ffffff; border: 1px solid #e5e9f2;
    color: #94a3b8; font-size: 11.5px; line-height: 1.6; text-align: center;
}
.footer b { color: #475569; }

/* ----------------------------- Streamlit widgets ------------------- */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 8px 14px; font-weight: 600; font-size: 13px;
}
.stButton button, .stFormSubmitButton button {
    border-radius: 10px; font-weight: 600;
}
.stFormSubmitButton button {
    background: #4f46e5; color: #ffffff; border: none;
}
.stFormSubmitButton button:hover { background: #4338ca; color: #ffffff; }
div[data-testid="stExpander"] {
    border: 1px solid #e5e9f2; border-radius: 12px; background: #ffffff;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary *,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li {
    color: #334155 !important;
}
div[data-testid="stExpander"] summary svg { fill: #334155; }

@media (max-width: 900px) {
    .page-hero { padding: 20px; }
    .page-hero h1 { font-size: 23px; }
    .kpi-value { font-size: 23px; }
    .profile-grid { grid-template-columns: 1fr; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Cached data/model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load (or generate) the synthetic dataset once per session."""
    return ensure_dataset()


@st.cache_resource(show_spinner=False)
def load_bundle(data_stamp: float) -> dict[str, Any]:
    """Load the trained model bundle, training on first run if necessary."""
    return ensure_model()


def with_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add a numeric churn flag used by aggregations and charts."""
    out = df.copy()
    out["ChurnFlag"] = (
        out["Churn"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)
    return out


def get_scored_data(data: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Batch-score the dataset and memoise the result in session state."""
    key = (
        f"{bundle.get('trained_at', '')}|{len(data)}|"
        f"{int(data['ChurnFlag'].sum())}"
    )
    if st.session_state.get("scored_key") != key:
        st.session_state["scored"] = score_dataframe(bundle, data)
        st.session_state["scored_key"] = key
    return st.session_state["scored"]


# ---------------------------------------------------------------------------
# Reusable UI helpers
# ---------------------------------------------------------------------------
def render_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_hero(title: str, subtitle: str, badge: str = "Synthetic data demo") -> None:
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="hero-badge">{html.escape(badge)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
            <div class="hero-meta">📡 ChurnIQ · Telecom retention intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>'
        f'<div class="section-sub">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    sub: str,
    icon: str,
    accent: str,
    soft: str,
) -> None:
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent}; --accent-soft:{soft};">
            <div class="kpi-top">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{html.escape(label)}</div>
            </div>
            <div class="kpi-value">{html.escape(value)}</div>
            <div class="kpi-sub">{html.escape(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk: str) -> str:
    css_class = {
        "High": "badge-high",
        "Medium": "badge-medium",
        "Low": "badge-low",
    }.get(risk, "badge-neutral")
    return f'<span class="badge {css_class}">{html.escape(risk)} risk</span>'


def style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply a consistent, modern visual style to every Plotly chart."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            size=12,
            color="#475569",
        ),
        margin=dict(l=8, r=8, t=46, b=8),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, Segoe UI, sans-serif",
        ),
        title=dict(font=dict(size=14, color="#0f172a")),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#e2e8f0",
        tickfont=dict(color="#64748b", size=11),
        title_font=dict(size=11, color="#64748b"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eef2f7",
        zeroline=False,
        tickfont=dict(color="#64748b", size=11),
        title_font=dict(size=11, color="#64748b"),
    )
    return fig


def show_chart(fig: go.Figure, height: int = 360) -> None:
    st.plotly_chart(
        style_figure(fig, height=height),
        width="stretch",
        theme=None,
        config={"displayModeBar": False, "responsive": True},
    )


def churn_rate_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    work = df.dropna(subset=[column])
    table = (
        work.groupby(column, dropna=False)
        .agg(Customers=("ChurnFlag", "size"), Churned=("ChurnFlag", "sum"))
        .reset_index()
    )
    table["ChurnRate"] = table["Churned"] / table["Customers"].replace(0, np.nan)
    return table


def tenure_band(tenure: pd.Series) -> pd.Series:
    bins = [-1, 12, 24, 48, 72]
    labels = ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
    return pd.cut(tenure, bins=bins, labels=labels, include_lowest=True)


def histogram_axis_ranges(
    df: pd.DataFrame,
    column: str,
    bins: int = 45,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute stable axis ranges for a percentage histogram.

    Plotly can occasionally auto-range a histogram to its annotation shapes
    instead of the bars. Explicit ranges keep the chart readable.
    """
    values = df[column].dropna()
    if values.empty:
        return (0.0, 1.0), (0.0, 1.0)

    edges = np.linspace(float(values.min()), float(values.max()), bins + 1)
    max_percent = 0.0
    for flag in (0, 1):
        subset = df.loc[df["ChurnFlag"] == flag, column].dropna()
        counts, _ = np.histogram(subset, bins=edges)
        if len(df):
            max_percent = max(max_percent, float((counts / len(df) * 100).max()))

    x_range = (0.0, float(values.max()) * 1.05)
    y_range = (0.0, max_percent * 1.25 + 0.5)
    return x_range, y_range


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="brand">
            <div class="brand-logo">📡</div>
            <div>
                <div class="brand-name">ChurnIQ</div>
                <div class="brand-sub">Telecom Analytics</div>
            </div>
        </div>
        <div class="sidebar-tag">Synthetic data · demo</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        key="page",
    )
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    return page


def render_sidebar_status(data: pd.DataFrame, bundle: dict[str, Any]) -> None:
    best_f1 = float(bundle["comparison"].iloc[0]["CV F1"])
    st.sidebar.markdown("#### System status")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-stat"><span>Dataset</span><b>Synthetic</b></div>
        <div class="sidebar-stat"><span>Customers</span><b>{len(data):,}</b></div>
        <div class="sidebar-stat"><span>Best model</span><b>{html.escape(bundle['best_model_name'])}</b></div>
        <div class="sidebar-stat"><span>CV F1</span><b>{best_f1:.3f}</b></div>
        <div class="sidebar-stat"><span>Trained</span><b>{html.escape(bundle['trained_at'])}</b></div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button(
        "🔄 Retrain all models",
        width="stretch",
        help="Re-run 5-fold stratified cross-validation and refit the best model.",
    ):
        try:
            with st.spinner("Retraining four models with 5-fold CV…"):
                fresh_data = ensure_dataset()
                ensure_model(force=True, df=fresh_data)
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in ("scored", "scored_key", "last_prediction"):
                st.session_state.pop(key, None)
            st.sidebar.success("Models retrained successfully.")
            st.rerun()
        except Exception as exc:  # pragma: no cover - UI safety net
            st.sidebar.error(f"Retraining failed: {exc}")

    st.sidebar.markdown(
        """
        <div class="sidebar-note">
            ⚠️ <b>Demo only.</b> All customers are synthetically generated.
            Metrics and predictions do not represent real telecom business
            performance and must not be used for real decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <b>ChurnIQ</b> · Built with Streamlit, scikit-learn, XGBoost, Plotly
            and Joblib.<br>
            Dataset is synthetic and generated for demonstration. Model results
            are illustrative only — not real telecom business performance.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Dashboard Overview
# ---------------------------------------------------------------------------
def render_dashboard(
    data: pd.DataFrame,
    scored: pd.DataFrame,
    bundle: dict[str, Any],
) -> None:
    total = len(data)
    churned = int(data["ChurnFlag"].sum())
    active = total - churned
    churn_rate = churned / total if total else 0.0
    high_risk = int((scored["RiskLevel"] == "High").sum())
    avg_charge = float(data["MonthlyCharges"].mean())

    page_hero(
        "Telecom Churn Intelligence",
        "Monitor portfolio health, identify at-risk customers, and turn model "
        "signals into retention action. Explore the synthetic customer base, "
        "compare models, and score a customer in seconds.",
    )

    cols = st.columns(5)
    with cols[0]:
        kpi_card("Total customers", f"{total:,}", "Synthetic portfolio", "👥", PRIMARY, PRIMARY_LIGHT)
    with cols[1]:
        kpi_card("Active customers", f"{active:,}", f"{(1 - churn_rate):.1%} of portfolio", "✅", SUCCESS, "#ecfdf5")
    with cols[2]:
        kpi_card("Churned customers", f"{churned:,}", "Historical label", "📉", DANGER, "#fef2f2")
    with cols[3]:
        kpi_card("Overall churn rate", f"{churn_rate:.1%}", f"Avg. bill ${avg_charge:.0f}/mo", "🎯", WARNING, "#fffbeb")
    with cols[4]:
        kpi_card(
            "High-risk customers",
            f"{high_risk:,}",
            f"ML probability ≥ {HIGH_RISK_THRESHOLD:.0%}",
            "🚨",
            "#7c3aed",
            "#f5f3ff",
        )

    st.write("")
    left, right = st.columns([1, 1.65])

    with left:
        section_header("Churn distribution", "Share of customers who left")
        counts = data["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
        fig = go.Figure(
            go.Pie(
                labels=["Retained", "Churned"],
                values=[counts.get("No", 0), counts.get("Yes", 0)],
                hole=0.64,
                marker=dict(colors=[SUCCESS, DANGER], line=dict(color="white", width=3)),
                textinfo="percent",
                textfont=dict(size=12, color="white"),
                hovertemplate="%{label}: %{value:,} customers<extra></extra>",
                sort=False,
            )
        )
        fig.add_annotation(
            text=(
                f"<b>{churn_rate:.1%}</b><br>"
                "<span style='font-size:10px;color:#64748b'>churn rate</span>"
            ),
            showarrow=False,
            font=dict(size=22, color=NAVY),
        )
        show_chart(fig, height=330)

    with right:
        section_header("Churn by contract type", "Churn rate and customer volume by agreement")
        contract_table = churn_rate_table(data, "ContractType")
        order = ["Month-to-month", "One year", "Two year"]
        contract_table["ContractType"] = pd.Categorical(
            contract_table["ContractType"], categories=order, ordered=True
        )
        contract_table = contract_table.sort_values("ContractType")
        colors = [DANGER, WARNING, SUCCESS]
        fig = go.Figure(
            go.Bar(
                x=contract_table["ContractType"].astype(str),
                y=contract_table["ChurnRate"],
                marker_color=colors,
                marker_line_width=0,
                text=[f"{r:.1%}" for r in contract_table["ChurnRate"]],
                textposition="outside",
                customdata=contract_table["Customers"],
                hovertemplate=(
                    "<b>%{x}</b><br>Churn rate: %{y:.1%}"
                    "<br>Customers: %{customdata:,}<extra></extra>"
                ),
                width=0.55,
            )
        )
        fig.add_hline(
            y=churn_rate,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text=f"Portfolio {churn_rate:.1%}",
            annotation_position="top right",
            annotation_font_size=10,
        )
        fig.update_yaxes(tickformat=".0%", range=[0, min(1, contract_table["ChurnRate"].max() * 1.28 + 0.05)])
        fig.update_layout(showlegend=False)
        show_chart(fig, height=330)

    st.write("")
    left, right = st.columns(2)

    with left:
        section_header("Churn by tenure", "Newer customers are more likely to leave")
        tenure_table = churn_rate_table(
            data.assign(TenureBand=tenure_band(data["Tenure"])), "TenureBand"
        )
        fig = go.Figure(
            go.Bar(
                x=tenure_table["TenureBand"].astype(str),
                y=tenure_table["ChurnRate"],
                marker_color=["#ef4444", "#f97316", "#f59e0b", "#10b981"],
                text=[
                    f"{rate:.0%} · {count:,}"
                    for rate, count in zip(
                        tenure_table["ChurnRate"], tenure_table["Customers"]
                    )
                ],
                textposition="outside",
                customdata=tenure_table["Customers"],
                hovertemplate=(
                    "<b>%{x}</b><br>Churn rate: %{y:.1%}"
                    "<br>Customers: %{customdata:,}<extra></extra>"
                ),
                width=0.58,
            )
        )
        fig.add_hline(y=churn_rate, line_dash="dot", line_color="#94a3b8")
        fig.update_yaxes(tickformat=".0%", range=[0, min(1, tenure_table["ChurnRate"].max() * 1.3 + 0.05)])
        fig.update_layout(showlegend=False)
        show_chart(fig, height=330)

    with right:
        section_header("Churn by payment method", "Electronic check is a consistent risk signal")
        payment_table = churn_rate_table(data, "PaymentMethod").sort_values("ChurnRate")
        fig = go.Figure(
            go.Bar(
                x=payment_table["ChurnRate"],
                y=payment_table["PaymentMethod"],
                orientation="h",
                marker_color=[
                    DANGER if rate > churn_rate else "#94a3b8"
                    for rate in payment_table["ChurnRate"]
                ],
                text=[f"{r:.1%}" for r in payment_table["ChurnRate"]],
                textposition="outside",
                customdata=payment_table["Customers"],
                hovertemplate=(
                    "<b>%{y}</b><br>Churn rate: %{x:.1%}"
                    "<br>Customers: %{customdata:,}<extra></extra>"
                ),
                width=0.6,
            )
        )
        fig.update_xaxes(tickformat=".0%", range=[0, min(1, payment_table["ChurnRate"].max() * 1.3 + 0.05)])
        fig.update_layout(showlegend=False)
        show_chart(fig, height=330)

    st.write("")
    left, right = st.columns([1.65, 1])

    with left:
        section_header("Monthly charges analysis", "Billing distribution for retained vs churned customers")
        fig = px.histogram(
            data,
            x="MonthlyCharges",
            color="Churn",
            nbins=45,
            barmode="overlay",
            histnorm="percent",
            color_discrete_map=CHURN_COLORS,
            labels={"MonthlyCharges": "Monthly charges ($)", "Churn": "Churn"},
        )
        fig.update_traces(opacity=0.72, marker_line_width=0)
        x_range, y_range = histogram_axis_ranges(data, "MonthlyCharges", bins=45)
        mean_retained = data.loc[data["ChurnFlag"] == 0, "MonthlyCharges"].mean()
        mean_churned = data.loc[data["ChurnFlag"] == 1, "MonthlyCharges"].mean()
        fig.add_vline(
            x=mean_retained,
            line_dash="dash",
            line_color=SUCCESS,
            annotation_text=f"Retained ${mean_retained:.0f}",
            annotation_position="top left",
            annotation_font_size=10,
        )
        fig.add_vline(
            x=mean_churned,
            line_dash="dash",
            line_color=DANGER,
            annotation_text=f"Churned ${mean_churned:.0f}",
            annotation_position="top right",
            annotation_font_size=10,
        )
        fig.update_xaxes(range=list(x_range))
        fig.update_yaxes(title="Share of customers (%)", range=list(y_range))
        show_chart(fig, height=360)

    with right:
        section_header("What stands out", "Quick observations from the synthetic portfolio")
        worst_contract = contract_table.sort_values("ChurnRate", ascending=False).iloc[0]
        worst_payment = payment_table.sort_values("ChurnRate", ascending=False).iloc[0]
        youngest = tenure_table.sort_values("ChurnRate", ascending=False).iloc[0]
        high_risk_share = high_risk / total if total else 0

        st.markdown(
            f"""
            <div class="callout">
                <h4>Portfolio signals</h4>
                <ul>
                    <li><b>{html.escape(str(worst_contract['ContractType']))}</b> customers
                        churn at <b>{worst_contract['ChurnRate']:.1%}</b>, the highest of any
                        contract type.</li>
                    <li><b>{html.escape(str(worst_payment['PaymentMethod']))}</b> is the
                        riskiest payment method at <b>{worst_payment['ChurnRate']:.1%}</b>.</li>
                    <li>Customers in the <b>{html.escape(str(youngest['TenureBand']))}</b>
                        band churn at <b>{youngest['ChurnRate']:.1%}</b>.</li>
                    <li><b>{high_risk:,}</b> customers ({high_risk_share:.1%}) are flagged
                        <b>high risk</b> by the selected model.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            <div class="callout-warn">
                <b>Synthetic dataset.</b> These patterns were generated for a
                machine-learning demo. They are not evidence about a real
                telecom market.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    section_header(
        "Priority customers",
        "Highest predicted churn probabilities from the selected model",
    )
    top = scored.sort_values("ChurnProbability", ascending=False).head(10)
    display = top[
        [
            "CustomerID",
            "ContractType",
            "Tenure",
            "MonthlyCharges",
            "PaymentMethod",
            "ChurnProbability",
            "RiskLevel",
        ]
    ].copy()
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "CustomerID": st.column_config.TextColumn("Customer", width="small"),
            "ContractType": st.column_config.TextColumn("Contract"),
            "Tenure": st.column_config.NumberColumn("Tenure (mo)", format="%d"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
            "PaymentMethod": st.column_config.TextColumn("Payment method"),
            "ChurnProbability": st.column_config.ProgressColumn(
                "Churn probability",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "RiskLevel": st.column_config.TextColumn("Risk", width="small"),
        },
    )


# ---------------------------------------------------------------------------
# Page: Customer Analytics
# ---------------------------------------------------------------------------
def render_analytics(data: pd.DataFrame, scored: pd.DataFrame, bundle: dict[str, Any]) -> None:
    page_hero(
        "Customer Analytics",
        "Slice the synthetic customer base by contract, service, tenure and "
        "billing behaviour to understand where churn concentrates.",
        badge="Exploratory analysis",
    )

    section_header("Filters", "Charts update instantly with your selection")
    f1, f2, f3 = st.columns(3)
    contract_options = sorted(data["ContractType"].dropna().unique().tolist())
    internet_options = sorted(data["InternetService"].dropna().unique().tolist())
    gender_options = sorted(data["Gender"].dropna().unique().tolist())

    contracts = f1.multiselect("Contract type", contract_options, default=contract_options)
    internets = f2.multiselect("Internet service", internet_options, default=internet_options)
    genders = f3.multiselect("Gender", gender_options, default=gender_options)

    if not contracts or not internets or not genders:
        st.warning("Select at least one value in each filter to view analytics.")
        return

    mask = (
        data["ContractType"].isin(contracts)
        & data["InternetService"].isin(internets)
        & data["Gender"].isin(genders)
    )
    filtered = data.loc[mask].copy()

    if filtered.empty:
        st.warning("No customers match the selected filters.")
        return

    total = len(filtered)
    churned = int(filtered["ChurnFlag"].sum())
    churn_rate = churned / total if total else 0.0
    avg_tenure = float(filtered["Tenure"].mean())
    avg_charge = float(filtered["MonthlyCharges"].mean())

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Customers in view", f"{total:,}", "Filtered segment", "👥", PRIMARY, PRIMARY_LIGHT)
    with cols[1]:
        kpi_card("Segment churn rate", f"{churn_rate:.1%}", f"{churned:,} churned", "📉", DANGER, "#fef2f2")
    with cols[2]:
        kpi_card("Avg. tenure", f"{avg_tenure:.0f} mo", "Time with company", "⏳", WARNING, "#fffbeb")
    with cols[3]:
        kpi_card("Avg. monthly bill", f"${avg_charge:.0f}", "Per customer", "💳", SUCCESS, "#ecfdf5")

    st.write("")
    left, right = st.columns(2)

    with left:
        section_header("Churn distribution", "Retained vs churned in the current view")
        counts = filtered["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
        fig = go.Figure(
            go.Pie(
                labels=["Retained", "Churned"],
                values=[counts.get("No", 0), counts.get("Yes", 0)],
                hole=0.62,
                marker=dict(colors=[SUCCESS, DANGER], line=dict(color="white", width=3)),
                textinfo="percent",
                textfont=dict(color="white", size=12),
                hovertemplate="%{label}: %{value:,}<extra></extra>",
                sort=False,
            )
        )
        fig.add_annotation(
            text=f"<b>{churn_rate:.1%}</b><br><span style='font-size:10px;color:#64748b'>churn</span>",
            showarrow=False,
            font=dict(size=21, color=NAVY),
        )
        show_chart(fig, height=340)

    with right:
        section_header("Churn by contract", "Rate with customer counts")
        table = churn_rate_table(filtered, "ContractType")
        order = [c for c in ["Month-to-month", "One year", "Two year"] if c in table["ContractType"].values]
        table["ContractType"] = pd.Categorical(table["ContractType"], categories=order, ordered=True)
        table = table.sort_values("ContractType")
        fig = go.Figure(
            go.Bar(
                x=table["ContractType"].astype(str),
                y=table["ChurnRate"],
                marker_color=[DANGER, WARNING, SUCCESS][: len(table)],
                text=[f"{r:.1%}" for r in table["ChurnRate"]],
                textposition="outside",
                customdata=table["Customers"],
                hovertemplate="<b>%{x}</b><br>Churn: %{y:.1%}<br>Customers: %{customdata:,}<extra></extra>",
                width=0.55,
            )
        )
        fig.add_hline(y=churn_rate, line_dash="dot", line_color="#94a3b8")
        fig.update_yaxes(tickformat=".0%", range=[0, min(1, table["ChurnRate"].max() * 1.3 + 0.05)])
        fig.update_layout(showlegend=False)
        show_chart(fig, height=340)

    left, right = st.columns(2)

    with left:
        section_header("Churn by tenure", "Tenure bands in months")
        table = churn_rate_table(filtered.assign(TenureBand=tenure_band(filtered["Tenure"])), "TenureBand")
        fig = go.Figure(
            go.Bar(
                x=table["TenureBand"].astype(str),
                y=table["ChurnRate"],
                marker_color=["#ef4444", "#f97316", "#f59e0b", "#10b981"][: len(table)],
                text=[f"{r:.0%} · {c:,}" for r, c in zip(table["ChurnRate"], table["Customers"])],
                textposition="outside",
                customdata=table["Customers"],
                hovertemplate="<b>%{x}</b><br>Churn: %{y:.1%}<br>Customers: %{customdata:,}<extra></extra>",
                width=0.58,
            )
        )
        fig.add_hline(y=churn_rate, line_dash="dot", line_color="#94a3b8")
        fig.update_yaxes(tickformat=".0%", range=[0, min(1, table["ChurnRate"].max() * 1.3 + 0.05)])
        fig.update_layout(showlegend=False)
        show_chart(fig, height=340)

    with right:
        section_header("Churn by payment method", "Sorted by churn rate")
        table = churn_rate_table(filtered, "PaymentMethod").sort_values("ChurnRate")
        fig = go.Figure(
            go.Bar(
                x=table["ChurnRate"],
                y=table["PaymentMethod"],
                orientation="h",
                marker_color=[DANGER if r > churn_rate else "#94a3b8" for r in table["ChurnRate"]],
                text=[f"{r:.1%}" for r in table["ChurnRate"]],
                textposition="outside",
                customdata=table["Customers"],
                hovertemplate="<b>%{y}</b><br>Churn: %{x:.1%}<br>Customers: %{customdata:,}<extra></extra>",
                width=0.6,
            )
        )
        fig.update_xaxes(tickformat=".0%", range=[0, min(1, table["ChurnRate"].max() * 1.3 + 0.05)])
        fig.update_layout(showlegend=False)
        show_chart(fig, height=340)

    left, right = st.columns([1.4, 1])

    with left:
        section_header("Monthly charges analysis", "Billing distribution by churn status")
        fig = px.histogram(
            filtered,
            x="MonthlyCharges",
            color="Churn",
            nbins=40,
            barmode="overlay",
            histnorm="percent",
            color_discrete_map=CHURN_COLORS,
            labels={"MonthlyCharges": "Monthly charges ($)", "Churn": "Churn"},
        )
        fig.update_traces(opacity=0.72, marker_line_width=0)
        x_range, y_range = histogram_axis_ranges(filtered, "MonthlyCharges", bins=40)
        fig.update_xaxes(range=list(x_range))
        fig.update_yaxes(title="Share of customers (%)", range=list(y_range))
        show_chart(fig, height=350)

    with right:
        section_header("Monthly charges by churn", "Median, quartiles and outliers")
        fig = px.box(
            filtered,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map=CHURN_COLORS,
            labels={"Churn": "Churn", "MonthlyCharges": "Monthly charges ($)"},
            points="outliers",
        )
        fig.update_layout(showlegend=False)
        show_chart(fig, height=350)

    section_header("Churn heatmap", "Churn rate by contract type and internet service")
    pivot = filtered.pivot_table(
        index="ContractType",
        columns="InternetService",
        values="ChurnFlag",
        aggfunc="mean",
    )
    if not pivot.empty:
        pivot = pivot.reindex([c for c in ["Month-to-month", "One year", "Two year"] if c in pivot.index])
        pivot = pivot[[c for c in ["DSL", "Fiber optic", "No"] if c in pivot.columns]]
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[[0, "#ecfdf5"], [0.5, "#fef3c7"], [1, "#ef4444"]],
                text=[[f"{v:.0%}" if pd.notna(v) else "–" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=13, color="#1e293b"),
                showscale=False,
                hovertemplate="Contract: %{y}<br>Internet: %{x}<br>Churn rate: %{z:.1%}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        show_chart(fig, height=300)

    st.write("")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered synthetic data (CSV)",
        data=csv,
        file_name="filtered_synthetic_customers.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
def render_model_performance(
    data: pd.DataFrame,
    scored: pd.DataFrame,
    bundle: dict[str, Any],
) -> None:
    page_hero(
        "Model Performance",
        "Four classifiers were compared with 5-fold stratified cross-validation. "
        "The best validation F1 model was refit and evaluated once on the "
        "untouched 20% test set.",
        badge="Validation & test metrics",
    )

    metrics = bundle["test_metrics"]
    best_name = bundle["best_model_name"]

    st.markdown(
        f'<span class="badge badge-info">Selected model</span> '
        f'<span style="font-size:15px;font-weight:700;color:#0f172a;margin-left:6px;">'
        f'{html.escape(best_name)}</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Trained on {bundle['train_rows']:,} customers · tested on "
        f"{bundle['test_rows']:,} held-out customers · "
        f"{bundle['cv_folds']}-fold stratified CV · threshold "
        f"{bundle['prediction_threshold']:.0%}"
    )

    cols = st.columns(5)
    metric_cards = [
        ("Accuracy", metrics["accuracy"], "Overall correctness", "✅", SUCCESS, "#ecfdf5"),
        ("Precision", metrics["precision"], "Of flagged, how many churned", "🎯", PRIMARY, PRIMARY_LIGHT),
        ("Recall", metrics["recall"], "Of churners, how many found", "🔍", WARNING, "#fffbeb"),
        ("F1-score", metrics["f1"], "Balance of precision & recall", "⚖️", "#7c3aed", "#f5f3ff"),
        ("ROC-AUC", metrics["roc_auc"], "Ranking quality", "📈", "#0891b2", "#ecfeff"),
    ]
    for col, (label, value, sub, icon, accent, soft) in zip(cols, metric_cards):
        with col:
            kpi_card(label, f"{value:.3f}", sub, icon, accent, soft)

    st.write("")
    left, right = st.columns([1.35, 1])

    with left:
        section_header("Model comparison", "Cross-validation means on the training set")
        comparison = bundle["comparison"].copy()
        comparison.insert(
            0,
            "Selected",
            np.where(comparison["Model"] == best_name, "★ Best", ""),
        )
        display = comparison.rename(
            columns={
                "CV F1": "CV F1",
                "CV F1 Std": "F1 std",
                "CV Accuracy": "CV accuracy",
                "CV Precision": "CV precision",
                "CV Recall": "CV recall",
                "CV ROC-AUC": "CV ROC-AUC",
            }
        )
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "Selected": st.column_config.TextColumn("", width="small"),
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "CV F1": st.column_config.NumberColumn(format="%.3f"),
                "F1 std": st.column_config.NumberColumn(format="%.3f"),
                "CV accuracy": st.column_config.NumberColumn(format="%.3f"),
                "CV precision": st.column_config.NumberColumn(format="%.3f"),
                "CV recall": st.column_config.NumberColumn(format="%.3f"),
                "CV ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(
            "Selection rule: highest mean CV F1-score. The test set is not "
            "used for model selection."
        )

    with right:
        section_header("Cross-validation F1", "Error bars show one standard deviation")
        comp = bundle["comparison"].sort_values("CV F1")
        fig = go.Figure(
            go.Bar(
                x=comp["CV F1"],
                y=comp["Model"],
                orientation="h",
                marker_color=[
                    PRIMARY if name == best_name else "#c7d2fe" for name in comp["Model"]
                ],
                error_x=dict(
                    type="data",
                    array=comp["CV F1 Std"],
                    visible=True,
                    color="#64748b",
                    thickness=1.2,
                    width=5,
                ),
                text=[f"{v:.3f}" for v in comp["CV F1"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>CV F1: %{x:.3f}<extra></extra>",
            )
        )
        fig.update_xaxes(range=[0, min(1, comp["CV F1"].max() + 0.12)])
        fig.update_layout(showlegend=False)
        show_chart(fig, height=340)

    st.write("")
    left, right = st.columns(2)

    with left:
        section_header("Confusion matrix", "Untouched test-set predictions")
        cm = np.array(bundle["confusion_matrix"])
        if cm.shape == (2, 2):
            labels = [["TN", "FP"], ["FN", "TP"]]
            text = [
                [f"{labels[i][j]}<br><b>{cm[i, j]:,}</b>" for j in range(2)]
                for i in range(2)
            ]
            fig = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=["Predicted No", "Predicted Yes"],
                    y=["Actual No", "Actual Yes"],
                    colorscale=[[0, "#eef2ff"], [1, "#4f46e5"]],
                    text=text,
                    texttemplate="%{text}",
                    textfont=dict(size=14, color="#0f172a"),
                    showscale=False,
                    hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>",
                )
            )
            fig.update_yaxes(autorange="reversed")
            show_chart(fig, height=340)

    with right:
        section_header("ROC curve", "True positive rate vs false positive rate")
        fpr = bundle["roc_curve"]["fpr"]
        tpr = bundle["roc_curve"]["tpr"]
        auc = metrics["roc_auc"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Model (AUC = {auc:.3f})",
                line=dict(color=PRIMARY, width=3),
                fill="tozeroy",
                fillcolor="rgba(79,70,229,0.10)",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random baseline",
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
                hoverinfo="skip",
            )
        )
        fig.add_annotation(
            x=0.58,
            y=0.16,
            text=f"<b>AUC {auc:.3f}</b>",
            showarrow=False,
            font=dict(size=14, color=PRIMARY),
            bgcolor="rgba(238,242,255,0.9)",
            borderpad=6,
        )
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1.02])
        show_chart(fig, height=340)

    st.write("")
    section_header("Feature importance", "Top signals used by the selected model")
    importance = bundle.get("feature_importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        imp = importance.head(15).sort_values("Importance")
        fig = go.Figure(
            go.Bar(
                x=imp["Importance"],
                y=imp["Feature"],
                orientation="h",
                marker_color=[
                    PRIMARY if i >= len(imp) - 3 else "#a5b4fc" for i in range(len(imp))
                ],
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(showlegend=False)
        show_chart(fig, height=max(360, 26 * len(imp)))
        st.caption(
            "Importance is model-specific. For logistic regression it is the "
            "absolute coefficient on the encoded features."
        )
    else:
        st.info("Feature importance is not available for this model type.")

    with st.expander("How the evaluation avoids data leakage"):
        st.markdown(
            f"""
            - The synthetic dataset is split **80/20 with stratification**
              (`random_state={bundle['random_state']}`), so the test set keeps the
              same churn ratio.
            - **All preprocessing** (median/most-frequent imputation,
              one-hot encoding and scaling) lives inside the scikit-learn
              `Pipeline`. It is fitted on training folds only.
            - **5-fold stratified cross-validation** runs on the training split
              for all four models.
            - The winner is chosen by **mean validation F1**, then refit on the
              full training split.
            - The **test set is touched exactly once**, after model selection.
            - Missing values in `TotalCharges`, `MonthlyUsage`,
              `PaymentMethod` and `CustomerSupportCalls` are imputed inside the
              pipeline.
            """
        )

    st.markdown(
        """
        <div class="callout-warn">
            <b>Synthetic results disclaimer.</b> These metrics come from
            randomly generated data. They demonstrate a correct ML workflow,
            not real telecom business performance, and should not be quoted as
            evidence of production accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Churn Prediction
# ---------------------------------------------------------------------------
def probability_gauge(probability: float, risk: str) -> go.Figure:
    color = RISK_COLORS.get(risk, PRIMARY)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36, "color": NAVY}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#cbd5e1",
                    "tickfont": {"size": 10, "color": "#94a3b8"},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#ecfdf5"},
                    {"range": [30, 60], "color": "#fffbeb"},
                    {"range": [60, 100], "color": "#fef2f2"},
                ],
                "threshold": {
                    "line": {"color": NAVY, "width": 2},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    return fig


def render_prediction_result(
    result: dict[str, Any],
    customer: dict[str, Any],
    bundle: dict[str, Any],
) -> None:
    probability = result["churn_probability"]
    risk = result["risk_level"]
    will_churn = result["prediction"] == "Yes"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Model prediction</div>
            <div class="result-status">
                {"⚠️ Churn likely" if will_churn else "✅ Likely to stay"}
            </div>
            <div>{risk_badge(risk)}
                <span class="badge badge-neutral" style="margin-left:4px;">
                    {probability:.1%} probability
                </span>
            </div>
            <div class="result-meta">
                {html.escape(result['model_name'])} · decision threshold
                {result['threshold']:.0%} · score is a model estimate, not a fact.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        probability_gauge(probability, risk),
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
    )

    recs = generate_recommendations(customer)[:2]
    st.markdown(
        '<div class="rec-tag" style="margin:4px 0 8px 2px;">'
        "Rule-based next steps · not ML output</div>",
        unsafe_allow_html=True,
    )
    render_recommendation_cards(recs, compact=True)


def render_prediction(data: pd.DataFrame, scored: pd.DataFrame, bundle: dict[str, Any]) -> None:
    page_hero(
        "Churn Prediction",
        "Enter a customer profile to estimate churn probability, get a risk "
        "band, and see rule-based retention suggestions. The model is trained "
        "on synthetic data.",
        badge="Single-customer scoring",
    )

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        section_header("Customer profile", "All fields are required")
        with st.form("churn_prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.checkbox("Senior citizen", value=False)
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                contract = st.selectbox(
                    "Contract type", ["Month-to-month", "One year", "Two year"]
                )
                internet = st.selectbox(
                    "Internet service", ["DSL", "Fiber optic", "No"]
                )

            with c2:
                phone = st.selectbox("Phone service", ["Yes", "No"])
                multiple = st.selectbox(
                    "Multiple lines", ["Yes", "No", "No phone service"]
                )
                security = st.selectbox(
                    "Online security", ["Yes", "No", "No internet service"], index=1
                )
                tech = st.selectbox(
                    "Tech support", ["Yes", "No", "No internet service"], index=1
                )
                payment = st.selectbox(
                    "Payment method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )

            with c3:
                paperless = st.selectbox("Paperless billing", ["Yes", "No"])
                monthly = st.number_input(
                    "Monthly charges ($)", min_value=1.0, max_value=300.0,
                    value=79.5, step=0.5,
                )
                total = st.number_input(
                    "Total charges ($)", min_value=0.0, max_value=50_000.0,
                    value=954.0, step=10.0,
                )
                usage = st.slider(
                    "Monthly usage (GB)", 0.0, 800.0, 180.0, 5.0
                )
                calls = st.slider(
                    "Support calls (last 6 months)", 0, 15, 2
                )

            st.caption(
                "Service-dependent fields are normalised automatically "
                "(e.g. no internet → “No internet service”)."
            )
            submitted = st.form_submit_button(
                "Predict churn risk", type="primary", width="stretch"
            )

        if submitted:
            # Keep categorical values logically consistent with the services.
            if phone == "No":
                multiple = "No phone service"
            if internet == "No":
                security = "No internet service"
                tech = "No internet service"

            customer = {
                "Gender": gender,
                "SeniorCitizen": int(senior),
                "Tenure": int(tenure),
                "ContractType": contract,
                "MonthlyCharges": float(monthly),
                "TotalCharges": float(total),
                "InternetService": internet,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "OnlineSecurity": security,
                "TechSupport": tech,
                "PaymentMethod": payment,
                "PaperlessBilling": paperless,
                "MonthlyUsage": float(usage),
                "CustomerSupportCalls": int(calls),
            }

            try:
                result = predict_customer(bundle, customer)
                st.session_state["last_prediction"] = {
                    "customer": customer,
                    "result": result,
                }
            except ValueError as exc:
                st.error(f"Please check the input: {exc}")
            except Exception as exc:  # pragma: no cover - UI safety net
                st.error("The prediction could not be completed.")
                st.exception(exc)

    with right:
        section_header("Prediction result", "Probability, status and risk band")
        saved = st.session_state.get("last_prediction")
        if not saved:
            st.markdown(
                """
                <div class="empty-state">
                    <b>No prediction yet</b>
                    Fill in the customer profile and click
                    “Predict churn risk” to see the model estimate,
                    risk band and suggested next steps.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_prediction_result(
                saved["result"], saved["customer"], bundle
            )

    st.write("")
    left, right = st.columns([1.6, 1], gap="large")

    with left:
        section_header("Input snapshot", "Values sent to the model")
        if st.session_state.get("last_prediction"):
            customer = st.session_state["last_prediction"]["customer"]
            snapshot = pd.DataFrame(
                {
                    "Field": list(customer.keys()),
                    "Value": [str(v) for v in customer.values()],
                }
            )
            st.dataframe(
                snapshot,
                width="stretch",
                hide_index=True,
                height=330,
            )
        else:
            st.caption("The submitted profile will appear here.")

    with right:
        section_header("How to read this", "Prediction guidance")
        st.markdown(
            f"""
            <div class="callout">
                <h4>Risk bands</h4>
                <ul>
                    <li><b>High:</b> probability ≥ {HIGH_RISK_THRESHOLD:.0%}</li>
                    <li><b>Medium:</b> {MEDIUM_RISK_THRESHOLD:.0%}–{HIGH_RISK_THRESHOLD:.0%}</li>
                    <li><b>Low:</b> below {MEDIUM_RISK_THRESHOLD:.0%}</li>
                </ul>
                <div style="margin-top:8px;">
                    The <b>Yes/No</b> status uses a
                    {bundle['prediction_threshold']:.0%} decision threshold.
                    Probabilities are model estimates from synthetic data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
            <div class="callout-warn">
                A high score is a prompt for a conversation, not a reason to
                treat a customer unfairly. Retention teams should combine model
                output with context and human judgement.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Rule-based retention logic (deliberately separate from the ML model)
# ---------------------------------------------------------------------------
def generate_recommendations(
    customer: dict[str, Any] | pd.Series,
) -> list[dict[str, str]]:
    """Return deterministic, rule-based retention recommendations.

    These rules are business heuristics. They are NOT machine-learning
    predictions and do not use the model score to decide which rule fires.
    """
    get = customer.get
    contract = str(get("ContractType", ""))
    tenure = float(get("Tenure", 0) or 0)
    monthly = float(get("MonthlyCharges", 0) or 0)
    calls = float(get("CustomerSupportCalls", 0) or 0)
    internet = str(get("InternetService", ""))
    security = str(get("OnlineSecurity", ""))
    tech = str(get("TechSupport", ""))
    payment = str(get("PaymentMethod", ""))

    recommendations: list[dict[str, str]] = []

    if contract == "Month-to-month":
        recommendations.append(
            {
                "title": "Offer a better contract plan",
                "detail": (
                    "Month-to-month customers show the highest churn. Offer a "
                    "12- or 24-month plan with a clear monthly saving and no "
                    "hidden exit penalty."
                ),
                "priority": "High",
                "icon": "📄",
            }
        )

    if calls >= 4:
        recommendations.append(
            {
                "title": "Provide customer support",
                "detail": (
                    f"This customer contacted support {int(calls)} times in six "
                    "months. Assign a named agent, investigate unresolved "
                    "issues and schedule a follow-up call."
                ),
                "priority": "High",
                "icon": "🎧",
            }
        )

    if tenure <= 12:
        recommendations.append(
            {
                "title": "Strengthen onboarding",
                "detail": (
                    "The customer is in the first year, a high-risk window. "
                    "Send a setup check-in, usage tips and a 30-day service "
                    "review to build early habit."
                ),
                "priority": "Medium",
                "icon": "🚀",
            }
        )

    if monthly >= 85:
        recommendations.append(
            {
                "title": "Review pricing concerns",
                "detail": (
                    f"At ${monthly:.2f} per month, this is a premium bill. "
                    "Review the plan for unused services and offer a "
                    "right-sized package or a temporary credit."
                ),
                "priority": "Medium",
                "icon": "💳",
            }
        )

    if internet in ("DSL", "Fiber optic") and (
        security == "No" or tech == "No"
    ):
        recommendations.append(
            {
                "title": "Bundle security and support",
                "detail": (
                    "The customer has internet but lacks online security or "
                    "tech support. A free 3-month support bundle can increase "
                    "perceived value and reduce friction."
                ),
                "priority": "Medium",
                "icon": "🛡️",
            }
        )

    if payment == "Electronic check":
        recommendations.append(
            {
                "title": "Incentivise automatic payments",
                "detail": (
                    "Electronic check is associated with higher churn. Offer a "
                    "small bill credit for switching to autopay by card or "
                    "bank transfer."
                ),
                "priority": "Low",
                "icon": "🔁",
            }
        )

    if tenure >= 48:
        recommendations.append(
            {
                "title": "Offer a loyalty discount",
                "detail": (
                    f"The customer has stayed {int(tenure)} months. A loyalty "
                    "discount or device upgrade protects a long relationship "
                    "and rewards tenure."
                ),
                "priority": "Low",
                "icon": "🏅",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "title": "Maintain engagement",
                "detail": (
                    "No strong rule-based risk factor was found. Continue "
                    "regular value communication and monitor for changes in "
                    "usage or support activity."
                ),
                "priority": "Low",
                "icon": "✅",
            }
        )

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recommendations.sort(key=lambda rec: priority_order.get(rec["priority"], 3))
    return recommendations


def render_recommendation_cards(
    recommendations: list[dict[str, str]],
    compact: bool = False,
) -> None:
    for rec in recommendations:
        priority = rec["priority"]
        css_priority = priority.lower()
        detail = rec["detail"]
        if compact and len(detail) > 150:
            detail = detail[:147].rstrip() + "…"
        st.markdown(
            f"""
            <div class="rec-card rec-{css_priority}">
                <div class="rec-head">
                    <span class="rec-icon">{rec['icon']}</span>
                    <span class="rec-title">{html.escape(rec['title'])}</span>
                    <span class="badge badge-{css_priority}">{priority}</span>
                </div>
                <div class="rec-detail">{html.escape(detail)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page: Retention Insights
# ---------------------------------------------------------------------------
def render_retention(data: pd.DataFrame, scored: pd.DataFrame, bundle: dict[str, Any]) -> None:
    page_hero(
        "Customer Retention Insights",
        "Combine the model's risk estimate with transparent business rules. "
        "The two are shown separately so teams always know what came from ML "
        "and what came from a deterministic playbook.",
        badge="Rules + ML, clearly separated",
    )

    sorted_scored = scored.sort_values("ChurnProbability", ascending=False)
    id_to_row = sorted_scored.set_index("CustomerID")

    section_header(
        "Choose a customer",
        "Pick from the highest-risk customers or search any synthetic CustomerID",
    )
    s1, s2 = st.columns([1, 1.4])
    with s1:
        search = st.text_input(
            "Search CustomerID",
            placeholder="e.g. TC-000123",
            help="Leave blank to choose from the top 100 highest-risk customers.",
        )
    with s2:
        top_options = sorted_scored.head(100)["CustomerID"].tolist()

        def format_customer(cid: str) -> str:
            row = id_to_row.loc[cid]
            return (
                f"{cid} · {row['ChurnProbability']:.0%} risk · "
                f"{row['ContractType']} · {int(row['Tenure'])} mo"
            )

        selected = st.selectbox(
            "Top 100 highest-risk customers",
            top_options,
            format_func=format_customer,
        )

    if search.strip():
        match = scored[
            scored["CustomerID"].str.upper() == search.strip().upper()
        ]
        if match.empty:
            st.warning(
                f"No customer found with ID “{search.strip()}”. "
                "Showing the selected high-risk customer instead."
            )
        else:
            selected = match.iloc[0]["CustomerID"]

    row = id_to_row.loc[selected]
    customer = row.to_dict()
    probability = float(row["ChurnProbability"])
    risk = str(row["RiskLevel"])
    recs = generate_recommendations(customer)

    total_display = (
        f"${row['TotalCharges']:.2f}"
        if pd.notna(row["TotalCharges"])
        else "Unknown"
    )
    calls_display = (
        f"{int(row['CustomerSupportCalls'])}"
        if pd.notna(row["CustomerSupportCalls"])
        else "Unknown"
    )
    usage_display = (
        f"{row['MonthlyUsage']:.0f} GB"
        if pd.notna(row["MonthlyUsage"])
        else "Unknown"
    )

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        section_header("Customer profile", f"Customer {selected}")
        st.markdown(
            f"""
            <div class="result-card">
                <div class="profile-grid">
                    <div class="profile-item"><span>Contract</span><b>{html.escape(str(row['ContractType']))}</b></div>
                    <div class="profile-item"><span>Tenure</span><b>{int(row['Tenure'])} months</b></div>
                    <div class="profile-item"><span>Monthly charges</span><b>${row['MonthlyCharges']:.2f}</b></div>
                    <div class="profile-item"><span>Total charges</span><b>{total_display}</b></div>
                    <div class="profile-item"><span>Internet</span><b>{html.escape(str(row['InternetService']))}</b></div>
                    <div class="profile-item"><span>Payment</span><b>{html.escape(str(row['PaymentMethod']))}</b></div>
                    <div class="profile-item"><span>Online security</span><b>{html.escape(str(row['OnlineSecurity']))}</b></div>
                    <div class="profile-item"><span>Tech support</span><b>{html.escape(str(row['TechSupport']))}</b></div>
                    <div class="profile-item"><span>Support calls</span><b>{calls_display}</b></div>
                    <div class="profile-item"><span>Monthly usage</span><b>{usage_display}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        section_header("ML assessment", "Model output — statistical estimate")
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted churn probability</div>
                <div class="result-status">{probability:.1%}</div>
                <div>{risk_badge(risk)}
                    <span class="badge badge-neutral" style="margin-left:4px;">
                        threshold {bundle['prediction_threshold']:.0%}
                    </span>
                </div>
                <div class="result-meta">
                    {html.escape(bundle['best_model_name'])} ·
                    trained on synthetic data
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            probability_gauge(probability, risk),
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
        )
        if risk == "High":
            st.markdown(
                '<div class="callout-warn"><b>Model-triggered action:</b> '
                "prioritise this customer for a retention call. This action is "
                "driven by the ML score, not by the rule engine below.</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([1.5, 1], gap="large")

    with left:
        section_header(
            "Rule-based recommendations",
            "Deterministic business rules based on customer characteristics — NOT ML",
        )
        st.markdown(
            '<span class="badge badge-neutral">Rule engine</span> '
            '<span style="font-size:11.5px;color:#94a3b8;">'
            "Same inputs always produce the same recommendations."
            "</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        render_recommendation_cards(recs)

    with right:
        section_header("How to use this page", "Keep the two signals distinct")
        st.markdown(
            """
            <div class="callout">
                <h4>Two separate signals</h4>
                <ul>
                    <li><b>ML assessment:</b> a probability learned from
                        historical patterns. It ranks risk but does not explain
                        cause.</li>
                    <li><b>Rule-based recommendations:</b> transparent if/then
                        heuristics written by the business. They suggest an
                        action but are not predictions.</li>
                </ul>
                <div style="margin-top:8px;">
                    Use the model to <b>prioritise</b> and the rules to
                    <b>decide the next best action</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    section_header(
        "Portfolio retention playbook",
        "How many synthetic customers each rule or model flag touches",
    )

    rules = [
        (
            "Month-to-month contract",
            "Business rule",
            scored["ContractType"] == "Month-to-month",
            "Offer a better contract plan",
        ),
        (
            "4+ support calls",
            "Business rule",
            scored["CustomerSupportCalls"] >= 4,
            "Provide customer support",
        ),
        (
            "Tenure ≤ 12 months",
            "Business rule",
            scored["Tenure"] <= 12,
            "Strengthen onboarding",
        ),
        (
            "Monthly charges ≥ $85",
            "Business rule",
            scored["MonthlyCharges"] >= 85,
            "Review pricing concerns",
        ),
        (
            "No security or tech support",
            "Business rule",
            scored["InternetService"].isin(["DSL", "Fiber optic"])
            & ((scored["OnlineSecurity"] == "No") | (scored["TechSupport"] == "No")),
            "Bundle security and support",
        ),
        (
            "Electronic check payment",
            "Business rule",
            scored["PaymentMethod"] == "Electronic check",
            "Incentivise automatic payments",
        ),
        (
            "Tenure ≥ 48 months",
            "Business rule",
            scored["Tenure"] >= 48,
            "Offer a loyalty discount",
        ),
        (
            f"High ML risk (≥ {HIGH_RISK_THRESHOLD:.0%})",
            "ML model",
            scored["RiskLevel"] == "High",
            "Priority retention outreach",
        ),
    ]

    rows = []
    for segment, kind, mask, action in rules:
        subset = scored.loc[mask]
        rows.append(
            {
                "Segment": segment,
                "Type": kind,
                "Customers": int(len(subset)),
                "Share": len(subset) / len(scored) if len(scored) else 0,
                "Avg. churn probability": (
                    float(subset["ChurnProbability"].mean()) if len(subset) else 0.0
                ),
                "Actual churn rate": (
                    float(subset["ChurnFlag"].mean()) if len(subset) else 0.0
                ),
                "Recommended action": action,
            }
        )
    playbook = pd.DataFrame(rows)

    st.dataframe(
        playbook,
        width="stretch",
        hide_index=True,
        column_config={
            "Segment": st.column_config.TextColumn("Segment", width="medium"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Customers": st.column_config.NumberColumn(format="%d"),
            "Share": st.column_config.ProgressColumn(
                "Share of base", min_value=0.0, max_value=1.0, format="percent"
            ),
            "Avg. churn probability": st.column_config.ProgressColumn(
                "Avg. churn probability", min_value=0.0, max_value=1.0, format="percent"
            ),
            "Actual churn rate": st.column_config.NumberColumn(
                "Actual churn", format="percent"
            ),
            "Recommended action": st.column_config.TextColumn(
                "Recommended action", width="medium"
            ),
        },
    )
    st.caption(
        "Segments overlap: a customer can appear in several rows. “Actual churn "
        "rate” uses the synthetic historical label for reference only."
    )

    st.write("")
    section_header(
        "High-risk customer list",
        "Download the full list for a retention campaign",
    )
    high_risk = scored[scored["RiskLevel"] == "High"].sort_values(
        "ChurnProbability", ascending=False
    )
    export_columns = [
        "CustomerID",
        "Gender",
        "SeniorCitizen",
        "Tenure",
        "ContractType",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "CustomerSupportCalls",
        "ChurnProbability",
        "RiskLevel",
    ]
    st.dataframe(
        high_risk[export_columns].head(25),
        width="stretch",
        hide_index=True,
        column_config={
            "CustomerID": st.column_config.TextColumn("Customer", width="small"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
            "TotalCharges": st.column_config.NumberColumn("Total $", format="$%.2f"),
            "ChurnProbability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )
    st.download_button(
        "⬇️ Download all high-risk customers (CSV)",
        data=high_risk[export_columns].to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )
    st.caption(
        f"{len(high_risk):,} of {len(scored):,} customers are currently flagged "
        f"high risk (probability ≥ {HIGH_RISK_THRESHOLD:.0%})."
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------
def main() -> None:
    render_css()
    page = render_sidebar()

    try:
        with st.spinner("Loading synthetic data and trained models…"):
            data = with_flags(load_data())
            bundle = load_bundle(DATA_PATH.stat().st_mtime)
    except Exception as exc:  # pragma: no cover - startup safety net
        st.error(
            "The dashboard could not start. Make sure all dependencies are "
            "installed and the project folder is writable."
        )
        with st.expander("Error details", expanded=True):
            st.exception(exc)
        st.stop()

    scored = with_flags(get_scored_data(data, bundle))
    render_sidebar_status(data, bundle)

    if page == "Dashboard Overview":
        render_dashboard(data, scored, bundle)
    elif page == "Customer Analytics":
        render_analytics(data, scored, bundle)
    elif page == "Model Performance":
        render_model_performance(data, scored, bundle)
    elif page == "Churn Prediction":
        render_prediction(data, scored, bundle)
    elif page == "Retention Insights":
        render_retention(data, scored, bundle)

    render_footer()


if __name__ == "__main__":
    main()
