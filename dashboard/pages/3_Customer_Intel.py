"""
dashboard/pages/3_Customer_Intel.py
--------------------------------------
Customer Intelligence page — PRIMARY DEMO SHOWCASE.

Features:
  - Full customer profile form
  - Complaint question + filters
  - Calls POST /customer-intel
  - Conversion probability gauge + band card
  - Complaint themes bar chart
  - Risk cards (ML + RAG combined view)
  - Cited complaint IDs
  - Segment insights
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.utils.api_client import customer_intel
from dashboard.utils.config import CONFIG
from dashboard.utils.helpers import (
    PRODUCT_OPTIONS, ISSUE_OPTIONS, format_probability, format_latency,
    conversion_band_color, conversion_band_emoji
)
from dashboard.components.cards import (
    conversion_band_card, kpi_card, alert, refusal_card,
    sufficiency_note_card, evidence_card
)
from dashboard.components.charts import (
    probability_gauge, complaint_themes_bar, probability_bar
)

st.set_page_config(page_title="Customer Intel · Meridian", page_icon="🧠", layout="wide")

_CSS = Path(__file__).parents[1] / "assets" / "styles.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#312e81,#1e1b4b);border-radius:12px;padding:24px 28px;margin-bottom:24px">
    <h1 style="margin:0;color:#e0e7ff;font-size:2rem">🧠 Customer Intelligence</h1>
    <p style="margin:8px 0 0;color:#a5b4fc">Combined ML conversion prediction + grounded complaint intelligence</p>
    <span style="font-size:0.75rem;color:#6366f1;font-weight:600;letter-spacing:0.1em">⭐ PRIMARY DEMO PAGE</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧠 Customer Intel")
    st.caption("Primary demo page")
    st.caption(f"Backend: `{CONFIG.api_base_url}`")
    st.divider()
    st.info("This page combines ML prediction + RAG complaint intelligence into a single customer view.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
input_tab, result_tab = st.tabs(["📝 Input", "📊 Results"])

with input_tab:
    st.markdown("### Customer Profile + Intelligence Query")

    with st.form("intel_form"):
        st.markdown("#### 👤 Customer Profile (ML Features)")
        p1, p2, p3 = st.columns(3)

        with p1:
            age = st.slider("Age", 17, 98, 42)
            duration = st.number_input("Call Duration (s)", 0, 5000, 300)
            campaign = st.number_input("Campaign Contacts", 1, 50, 1)
            previous = st.number_input("Previous Contacts", 0, 30, 0)
            pdays = st.number_input("Days Since Last Contact", 0, 999, 999)

        with p2:
            emp_var_rate = st.number_input("Emp. Variation Rate", -5.0, 5.0, -1.8, step=0.1)
            cons_price_idx = st.number_input("Consumer Price Index", 90.0, 100.0, 93.075, step=0.001)
            cons_conf_idx = st.number_input("Confidence Index", -55.0, -20.0, -47.1, step=0.1)
            euribor3m = st.number_input("Euribor 3M", 0.0, 6.0, 1.334, step=0.001)
            nr_employed = st.number_input("Nr. Employees (k)", 4900.0, 5300.0, 5099.1, step=0.1)

        with p3:
            job = st.selectbox("Job", ["management", "admin.", "technician", "blue-collar",
                                        "services", "retired", "self-employed", "entrepreneur",
                                        "housemaid", "student", "unemployed", "unknown"])
            marital = st.selectbox("Marital", ["married", "single", "divorced", "unknown"])
            education = st.selectbox("Education", ["university.degree", "high.school",
                                                    "professional.course", "basic.9y",
                                                    "basic.6y", "basic.4y", "illiterate", "unknown"])
            default = st.selectbox("Credit Default", ["no", "yes", "unknown"])
            housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"])
            loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
            contact = st.selectbox("Contact Type", ["cellular", "telephone"])
            month = st.selectbox("Month", ["may", "jun", "jul", "aug", "oct", "nov",
                                            "dec", "mar", "apr", "sep", "jan", "feb"])
            day_of_week = st.selectbox("Day", ["mon", "tue", "wed", "thu", "fri"])
            poutcome = st.selectbox("Prev. Outcome", ["nonexistent", "failure", "success"])

        st.divider()
        st.markdown("#### 🔍 Complaint Intelligence Query")

        q_col1, q_col2, q_col3 = st.columns([2, 1, 1])
        with q_col1:
            question = st.text_input(
                "Complaint question (optional)",
                placeholder="e.g. What billing complaints do customers like this report?",
            )
        with q_col2:
            product_filter = st.selectbox("Product filter", PRODUCT_OPTIONS)
        with q_col3:
            issue_filter = st.selectbox("Issue filter", ISSUE_OPTIONS)

        top_k = st.slider("Evidence chunks", 1, 10, 5)
        submit = st.form_submit_button("🧠 Analyse Customer", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
if submit:
    profile = dict(
        age=age, duration=duration, campaign=campaign, previous=previous,
        pdays=pdays, emp_var_rate=emp_var_rate, cons_price_idx=cons_price_idx,
        cons_conf_idx=cons_conf_idx, euribor3m=euribor3m, nr_employed=nr_employed,
        job=job, marital=marital, education=education, default=default,
        housing=housing, loan=loan, contact=contact, month=month,
        day_of_week=day_of_week, poutcome=poutcome,
    )

    with st.spinner("Running customer intelligence analysis…"):
        result = customer_intel(
            customer_profile=profile,
            question=question.strip() if question else None,
            product_filter=product_filter if product_filter else None,
            issue_filter=issue_filter if issue_filter else None,
            top_k=top_k,
        )

    with result_tab:
        if result.get("_error"):
            alert(result["_message"], kind="error")
        else:
            prob = result["conversion_probability"]
            pred = result["conversion_prediction"]
            band = result["conversion_band"]
            model_ver = result.get("model_version", "")
            ml_lat = result.get("ml_latency_ms", 0)
            rag_lat = result.get("rag_latency_ms", 0)
            total_lat = result.get("total_latency_ms", 0)
            complaint_answer = result.get("complaint_answer", "")
            complaint_refused = result.get("complaint_refused", False)
            complaint_question = result.get("complaint_question", "")
            themes = result.get("complaint_themes", [])
            cited_ids = result.get("cited_complaint_ids", [])
            sufficiency = result.get("evidence_sufficiency", "")

            # ── KPI strip ─────────────────────────────────────────────────────
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi_card("Conversion Prob.", format_probability(prob),
                         color=conversion_band_color(band))
            with k2:
                kpi_card("Conversion Band", f"{conversion_band_emoji(band)} {band}",
                         color=conversion_band_color(band))
            with k3:
                kpi_card("ML Latency", f"{ml_lat:.0f} ms", color="#6366f1")
            with k4:
                kpi_card("RAG Latency", f"{rag_lat:.0f} ms", color="#a78bfa")

            st.markdown("")

            # ── ML + RAG side-by-side ─────────────────────────────────────────
            ml_col, rag_col = st.columns([1, 1])

            with ml_col:
                st.markdown("### 🤖 ML Prediction")
                conversion_band_card(prob, band, pred, 0.5)
                st.plotly_chart(probability_gauge(prob, 0.5, "Conversion Probability"),
                                use_container_width=True)

            with rag_col:
                st.markdown("### 🔍 Complaint Intelligence")
                if complaint_refused:
                    refusal_card(complaint_answer)
                else:
                    st.markdown(f"""
                    <div style="
                        background:#1e293b;border-left:4px solid #6366f1;
                        border-radius:0 12px 12px 0;padding:16px 20px;
                        color:#e2e8f0;line-height:1.7;
                        font-size:0.9rem;
                    ">
                    {complaint_answer}
                    </div>
                    """, unsafe_allow_html=True)
                    if sufficiency:
                        sufficiency_note_card(sufficiency)

            st.divider()

            # ── Themes + cited IDs ────────────────────────────────────────────
            theme_col, ids_col = st.columns([2, 1])

            with theme_col:
                st.markdown("### 📊 Complaint Theme Distribution")
                st.plotly_chart(complaint_themes_bar(themes), use_container_width=True)

            with ids_col:
                st.markdown("### 🔗 Cited Complaint IDs")
                if cited_ids:
                    for i, cid in enumerate(cited_ids[:10]):
                        evidence_card(i, cid)
                    if len(cited_ids) > 10:
                        st.caption(f"…and {len(cited_ids) - 10} more")
                else:
                    st.caption("No evidence IDs cited.")

            # ── Full response ─────────────────────────────────────────────────
            with st.expander("📋 Full API Response"):
                st.json(result)
else:
    with result_tab:
        st.info("👈 Fill in the form on the **Input** tab and click **Analyse Customer** to see results here.")
