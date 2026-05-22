"""
dashboard/components/tables.py
--------------------------------
Reusable Streamlit table/dataframe rendering helpers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def evidence_table(evidence_ids: list[str], scores: list[float] | None = None) -> None:
    """Render a styled table of evidence chunk IDs and scores."""
    if not evidence_ids:
        st.info("No evidence chunks retrieved.")
        return

    rows = []
    for i, eid in enumerate(evidence_ids):
        row = {"#": i + 1, "Chunk ID": eid}
        if scores and i < len(scores):
            row["Similarity"] = f"{scores[i]:.3f}"
            tier = "🟢 HIGH" if scores[i] >= 0.7 else "🟡 MEDIUM" if scores[i] >= 0.4 else "🔴 LOW"
            row["Quality"] = tier
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def batch_results_table(results: list[dict]) -> None:
    """Render batch scoring results as a sortable dataframe."""
    if not results:
        st.info("No results.")
        return

    rows = []
    for r in results:
        rows.append({
            "Row": r.get("row_index", ""),
            "Probability": f"{r.get('probability', 0):.4f}",
            "Prediction": "✅ Subscribe" if r.get("prediction") == 1 else "❌ Won't",
            "Error": r.get("error") or "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def metrics_summary_table(metrics: dict) -> None:
    """Render key metrics as a two-column summary table."""
    rows = [
        ("Total Requests", metrics.get("total_requests", 0)),
        ("Error Count", metrics.get("error_count", 0)),
        ("Error Rate", f"{metrics.get('error_rate', 0):.2%}"),
        ("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.1f} ms"),
        ("Uptime", f"{metrics.get('uptime_seconds', 0):.0f} s"),
    ]
    pred = metrics.get("prediction_distribution", {})
    if pred:
        rows += [
            ("Total Predictions", pred.get("total_predictions", 0)),
            ("Positive Rate", f"{pred.get('positive_rate', 0):.2%}"),
        ]
    rag = metrics.get("rag_retrieval_stats", {})
    if rag:
        rows += [
            ("RAG Total Queries", rag.get("total_queries", 0)),
            ("RAG Refusal Rate", f"{rag.get('refusal_rate', 0):.2%}"),
            ("Avg Evidence / Query", f"{rag.get('avg_evidence_ids_per_query', 0):.1f}"),
        ]

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)
