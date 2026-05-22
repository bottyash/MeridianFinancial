"""
dashboard/pages/2_Complaint_Intelligence.py
---------------------------------------------
RAG-grounded complaint Q&A page.

Features:
  - Question input with optional product/issue filters
  - Calls POST /ask-complaints
  - Answer display with evidence cards
  - Retrieval score bar chart
  - Evidence metadata table
  - Refusal handling
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.utils.api_client import ask_complaints
from dashboard.utils.config import CONFIG
from dashboard.utils.helpers import PRODUCT_OPTIONS, ISSUE_OPTIONS
from dashboard.components.cards import (
    alert, refusal_card, sufficiency_note_card, evidence_card, kpi_card
)
from dashboard.components.charts import evidence_scores_bar
from dashboard.components.tables import evidence_table

st.set_page_config(page_title="Complaint Intelligence · Meridian", page_icon="🔍", layout="wide")

_CSS = Path(__file__).parents[1] / "assets" / "styles.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Complaint Intelligence")
st.markdown("Ask questions about consumer complaints. Answers are grounded in the CFPB complaint corpus via RAG.")
st.divider()

with st.sidebar:
    st.markdown("## 🔍 Complaint Intelligence")
    st.caption(f"Backend: `{CONFIG.api_base_url}`")
    st.divider()
    st.markdown("**About**")
    st.caption("Uses ChromaDB semantic search + Mistral LLM to answer complaint-related questions.")
    st.caption("Answers include evidence IDs and a sufficiency rating.")

# ── Query form ────────────────────────────────────────────────────────────────
st.markdown("### Ask a Question")

example_questions = [
    "What are the most common credit card billing disputes?",
    "How do consumers report mortgage payment issues?",
    "What debt collection harassment complaints exist?",
    "What student loan servicing problems are reported?",
    "How do consumers report credit report errors?",
]

use_example = st.selectbox("💡 Example questions (or type your own below)", ["— custom —"] + example_questions)

with st.form("complaint_form"):
    question = st.text_area(
        "Your question",
        value="" if use_example == "— custom —" else use_example,
        height=100,
        placeholder="e.g. What are the most common credit card billing disputes?",
    )

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        product_filter = st.selectbox("Product filter", PRODUCT_OPTIONS)
    with f_col2:
        top_k = st.slider("Evidence chunks (top-k)", 1, 20, 5)
    with f_col3:
        st.markdown("")
        st.markdown("")
        submitted = st.form_submit_button("🔍 Search Complaints", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    if not question or len(question.strip()) < 3:
        alert("Please enter a question (minimum 3 characters).", kind="warning")
    else:
        with st.spinner("Retrieving evidence and generating answer…"):
            result = ask_complaints(
                question=question.strip(),
                top_k=top_k,
                product_filter=product_filter if product_filter else None,
            )

        if result.get("_error"):
            alert(result["_message"], kind="error")
        else:
            refused = result.get("refused", False)
            answer = result.get("answer", "")
            evidence_ids = result.get("evidence_ids", [])
            sufficiency = result.get("evidence_sufficiency", "")
            prompt_version = result.get("prompt_version", "")
            retrieval_count = result.get("retrieval_count", 0)
            latency = result.get("latency_ms", 0)
            token_usage = result.get("token_usage", {})

            st.divider()

            # ── KPI strip ─────────────────────────────────────────────────────
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi_card("Status", "🚫 Refused" if refused else "✅ Answered",
                         color="#ef4444" if refused else "#22c55e")
            with k2:
                kpi_card("Evidence IDs", str(len(evidence_ids)), color="#6366f1")
            with k3:
                kpi_card("Latency", f"{latency:.0f} ms", color="#f59e0b")
            with k4:
                kpi_card("Tokens Used", str(token_usage.get("total_tokens", 0)), color="#a78bfa")

            st.markdown("")

            if refused:
                refusal_card(answer)
            else:
                # ── Answer ─────────────────────────────────────────────────────
                st.markdown("### 💬 Grounded Answer")
                st.markdown(f"""
                <div style="
                    background:#1e293b;
                    border-left:4px solid #6366f1;
                    border-radius:0 12px 12px 0;
                    padding:20px 24px;
                    color:#e2e8f0;
                    line-height:1.7;
                    margin-bottom:16px;
                ">
                {answer}
                </div>
                """, unsafe_allow_html=True)

                # ── Sufficiency note ───────────────────────────────────────────
                sufficiency_note_card(sufficiency)

                # ── Evidence ───────────────────────────────────────────────────
                st.markdown("### 📎 Evidence Chunks")
                ev_col1, ev_col2 = st.columns([1, 1])

                with ev_col1:
                    st.markdown("**Retrieved chunk IDs:**")
                    for i, eid in enumerate(evidence_ids):
                        evidence_card(i, eid)

                with ev_col2:
                    st.plotly_chart(
                        evidence_scores_bar(evidence_ids),
                        use_container_width=True,
                    )

                # ── Metadata table ─────────────────────────────────────────────
                with st.expander("📋 Evidence Metadata Table"):
                    evidence_table(evidence_ids)

                # ── Meta info ─────────────────────────────────────────────────
                with st.expander("🔧 Request Metadata"):
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.markdown(f"**Prompt version:** `{prompt_version}`")
                        st.markdown(f"**Retrieval count:** {retrieval_count}")
                    with meta_col2:
                        st.markdown(f"**Prompt tokens:** {token_usage.get('prompt_tokens', 0)}")
                        st.markdown(f"**Completion tokens:** {token_usage.get('completion_tokens', 0)}")

                with st.expander("📋 Full API Response"):
                    st.json(result)
