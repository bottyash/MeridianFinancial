"""
dashboard/pages/1_ML_Predictions.py
-------------------------------------
ML conversion prediction page.

Features:
  - Customer profile form (all 20 features)
  - Calls POST /predict
  - Probability gauge chart
  - Conversion band card
  - Horizontal probability bar
  - Batch scoring section (CSV upload)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

from dashboard.utils.api_client import predict, batch_score
from dashboard.utils.config import CONFIG
from dashboard.utils.helpers import PRODUCT_OPTIONS
from dashboard.components.cards import (
    conversion_band_card, kpi_card, alert, status_banner
)
from dashboard.components.charts import probability_gauge, probability_bar
from dashboard.components.tables import batch_results_table

st.set_page_config(page_title="ML Predictions · Meridian", page_icon="🤖", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = Path(__file__).parents[1] / "assets" / "styles.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 ML Predictions")
st.markdown("Predict campaign conversion probability for a bank marketing customer profile.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ML Predictions")
    st.caption(f"Backend: `{CONFIG.api_base_url}`")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### Customer Profile")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Numeric Features**")
        age = st.slider("Age", 17, 98, 42)
        duration = st.number_input("Last Contact Duration (s)", 0, 5000, 300)
        campaign = st.number_input("Contacts This Campaign", 1, 50, 1)
        previous = st.number_input("Previous Contacts", 0, 30, 0)
        pdays = st.number_input("Days Since Last Contact (999=never)", 0, 999, 999)

    with c2:
        st.markdown("**Economic Indicators**")
        emp_var_rate = st.number_input("Employment Variation Rate", -5.0, 5.0, -1.8, step=0.1)
        cons_price_idx = st.number_input("Consumer Price Index", 90.0, 100.0, 93.075, step=0.001)
        cons_conf_idx = st.number_input("Consumer Confidence Index", -55.0, -20.0, -47.1, step=0.1)
        euribor3m = st.number_input("Euribor 3M Rate", 0.0, 6.0, 1.334, step=0.001)
        nr_employed = st.number_input("Nr. Employees (thousands)", 4900.0, 5300.0, 5099.1, step=0.1)

    with c3:
        st.markdown("**Categorical Features**")
        job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid",
                                    "management", "retired", "self-employed", "services",
                                    "student", "technician", "unemployed", "unknown"])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y",
                                                "high.school", "illiterate", "professional.course",
                                                "university.degree", "unknown"])
        default = st.selectbox("Credit Default", ["no", "yes", "unknown"])
        housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"])
        loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
        contact = st.selectbox("Contact Type", ["cellular", "telephone"])
        month = st.selectbox("Last Contact Month",
                              ["jan", "feb", "mar", "apr", "may", "jun",
                               "jul", "aug", "sep", "oct", "nov", "dec"])
        day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
        poutcome = st.selectbox("Previous Campaign Outcome",
                                 ["nonexistent", "failure", "success"])

    submitted = st.form_submit_button("🔮 Predict Conversion", use_container_width=True, type="primary")

# ── Prediction result ─────────────────────────────────────────────────────────
if submitted:
    profile = dict(
        age=age, duration=duration, campaign=campaign, previous=previous,
        pdays=pdays, emp_var_rate=emp_var_rate, cons_price_idx=cons_price_idx,
        cons_conf_idx=cons_conf_idx, euribor3m=euribor3m, nr_employed=nr_employed,
        job=job, marital=marital, education=education, default=default,
        housing=housing, loan=loan, contact=contact, month=month,
        day_of_week=day_of_week, poutcome=poutcome,
    )

    with st.spinner("Running ML inference…"):
        result = predict(profile)

    if result.get("_error"):
        alert(result["_message"], kind="error")
    else:
        prob = result["probability"]
        pred = result["prediction"]
        threshold = result.get("threshold", 0.5)
        model_ver = result.get("model_version", "")
        latency = result.get("latency_ms", 0)

        # Derive band locally for display
        band = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"

        st.divider()
        st.markdown("### 📊 Prediction Results")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            conversion_band_card(prob, band, pred, threshold)
            st.markdown("")
            kpi_col1, kpi_col2 = st.columns(2)
            with kpi_col1:
                kpi_card("Model Version", model_ver, color="#6366f1")
            with kpi_col2:
                kpi_card("Latency", f"{latency:.1f} ms", color="#22c55e")

        with res_col2:
            st.plotly_chart(probability_gauge(prob, threshold), use_container_width=True)
            st.plotly_chart(probability_bar(prob), use_container_width=True)

        with st.expander("📋 View Full API Response"):
            st.json(result)

# ── Batch scoring ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📦 Batch Scoring")
st.caption("Upload a CSV file (max 500 rows) with the same 20 feature columns to score all records at once.")

uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
        st.dataframe(df.head(5), use_container_width=True)

        if len(df) > 500:
            alert("File has more than 500 rows. Only the first 500 will be scored.", kind="warning")
            df = df.head(500)

        if st.button("⚡ Run Batch Score", type="primary"):
            records = df.to_dict(orient="records")
            with st.spinner(f"Scoring {len(records)} records…"):
                batch_result = batch_score(records)

            if batch_result.get("_error"):
                alert(batch_result["_message"], kind="error")
            else:
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    kpi_card("Total", str(batch_result["total"]), color="#6366f1")
                with b_col2:
                    kpi_card("Succeeded", str(batch_result["succeeded"]), color="#22c55e")
                with b_col3:
                    kpi_card("Failed", str(batch_result["failed"]), color="#ef4444")

                batch_results_table(batch_result.get("results", []))
    except Exception as exc:
        alert(f"Could not parse CSV: {exc}", kind="error")
