"""
dashboard/app.py
-----------------
Meridian Financial Intelligence Dashboard — Main Entry Point

Multi-page Streamlit app.

Run:
    streamlit run dashboard/app.py

Environment variables:
    API_BASE_URL        FastAPI backend URL (default: http://localhost:8000)
    ENVIRONMENT         development | staging | production
    REQUEST_TIMEOUT     seconds (default: 30)

HuggingFace Spaces:
    Set API_BASE_URL as a HF Space secret pointing to your AWS backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure dashboard/ package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.utils.config import CONFIG
from dashboard.utils.api_client import health_check, is_backend_online
from dashboard.components.cards import status_banner

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=CONFIG.dashboard_title,
    page_icon=CONFIG.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/bottyash/MeridianFinancial",
        "Report a bug": "https://github.com/bottyash/MeridianFinancial/issues",
        "About": "Meridian Financial Customer Intelligence Platform",
    },
)

# ---------------------------------------------------------------------------
# Inject global CSS
# ---------------------------------------------------------------------------
_CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
if _CSS_PATH.exists():
    with open(_CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — backend status + navigation info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 Meridian Financial")
    st.markdown("*Customer Intelligence Platform*")
    st.divider()

    # Backend health
    with st.spinner("Checking backend…"):
        health = health_check()

    healthy = not health.get("_error") and health.get("status") == "ok"
    model_version = health.get("model_version", "")
    status_banner(healthy, model_version=model_version, environment=CONFIG.environment)

    st.markdown(f"**API:** `{CONFIG.api_base_url}`")
    st.markdown(f"**Env:** `{CONFIG.environment}`")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("""
- 🤖 **ML Predictions** — Conversion scoring
- 🔍 **Complaint Intelligence** — RAG Q&A
- 🧠 **Customer Intel** — Combined analysis *(Demo page)*
- 📊 **Monitoring** — Live metrics & drift
""")
    st.divider()
    st.caption("Built with FastAPI + ChromaDB + Mistral")

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    border: 1px solid rgba(99,102,241,0.2);
">
    <div style="font-size: 3.5rem; margin-bottom: 8px">🏦</div>
    <h1 style="
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    ">Meridian Financial Intelligence</h1>
    <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 12px; max-width: 600px; margin-left: auto; margin-right: auto;">
        Production-grade AI platform combining ML conversion prediction and 
        RAG-based complaint intelligence.
    </p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background:#1e293b;border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:20px;text-align:center;height:140px">
        <div style="font-size:2rem">🤖</div>
        <div style="font-weight:700;color:#818cf8;margin-top:8px">ML Predictions</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:4px">XGBoost · 87% ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#1e293b;border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:20px;text-align:center;height:140px">
        <div style="font-size:2rem">🔍</div>
        <div style="font-weight:700;color:#818cf8;margin-top:8px">Complaint RAG</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:4px">ChromaDB · Mistral · 41k chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:#1e293b;border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:20px;text-align:center;height:140px">
        <div style="font-size:2rem">🧠</div>
        <div style="font-weight:700;color:#818cf8;margin-top:8px">Customer Intel</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:4px">ML + RAG combined</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background:#1e293b;border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:20px;text-align:center;height:140px">
        <div style="font-size:2rem">📊</div>
        <div style="font-weight:700;color:#818cf8;margin-top:8px">Monitoring</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:4px">Evidently · Drift · Metrics</div>
    </div>
    """, unsafe_allow_html=True)

# Quick start
st.divider()
st.markdown("### 👈 Select a page from the sidebar to get started")
if not healthy:
    st.warning(f"⚠️ Backend is offline. Set **API_BASE_URL** to your FastAPI endpoint.  \nCurrent: `{CONFIG.api_base_url}`")
