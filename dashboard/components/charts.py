"""
dashboard/components/charts.py
--------------------------------
Reusable Plotly chart builders for the Meridian Financial dashboard.

All functions return a ``plotly.graph_objects.Figure`` ready for
``st.plotly_chart(..., use_container_width=True)``.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Any


# ── Color palette ─────────────────────────────────────────────────────────────
PRIMARY = "#6366f1"      # indigo
SUCCESS = "#22c55e"      # green
WARNING = "#f59e0b"      # amber
DANGER = "#ef4444"       # red
NEUTRAL = "#6b7280"      # gray
BG_DARK = "#0f172a"
BG_CARD = "#1e293b"
TEXT = "#e2e8f0"

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ---------------------------------------------------------------------------
# Gauge — conversion probability
# ---------------------------------------------------------------------------

def probability_gauge(probability: float, threshold: float = 0.5, title: str = "Conversion Probability") -> go.Figure:
    """Donut-style gauge chart for a 0–1 probability.

    Parameters
    ----------
    probability:
        Value 0.0–1.0.
    threshold:
        Decision threshold (drawn as a marker).
    title:
        Chart title.
    """
    color = SUCCESS if probability >= threshold else DANGER
    pct = probability * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": TEXT}},
        delta={"reference": threshold * 100, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT, "tickfont": {"color": TEXT}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": BG_CARD,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(239,68,68,0.15)"},
                {"range": [threshold * 100, 100], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": WARNING, "width": 3},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
        title={"text": title, "font": {"color": TEXT, "size": 14}},
    ))
    fig.update_layout(**_LAYOUT_BASE, height=280)
    return fig


# ---------------------------------------------------------------------------
# Horizontal bar — probability breakdown
# ---------------------------------------------------------------------------

def probability_bar(probability: float) -> go.Figure:
    """Single horizontal bar showing probability vs (1-probability)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[probability * 100],
        y=["Probability"],
        orientation="h",
        marker_color=SUCCESS if probability >= 0.5 else DANGER,
        name="Will Subscribe",
        text=[f"{probability * 100:.1f}%"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=[(1 - probability) * 100],
        y=["Probability"],
        orientation="h",
        marker_color="rgba(107,114,128,0.3)",
        name="Won't Subscribe",
        text=[f"{(1 - probability) * 100:.1f}%"],
        textposition="inside",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="stack",
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, font=dict(color=TEXT)),
        height=100,
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Retrieval score bars — evidence chunks
# ---------------------------------------------------------------------------

def evidence_scores_bar(evidence_ids: list[str], scores: list[float] | None = None) -> go.Figure:
    """Horizontal bar chart for evidence chunk similarity scores.

    Parameters
    ----------
    evidence_ids:
        List of chunk ID strings (used as y-axis labels, truncated).
    scores:
        Optional list of similarity floats. If None, shows placeholder bars.
    """
    n = len(evidence_ids)
    if not n:
        fig = go.Figure()
        fig.add_annotation(text="No evidence retrieved", showarrow=False, font=dict(color=NEUTRAL))
        fig.update_layout(**_LAYOUT_BASE, height=150)
        return fig

    labels = [f"Chunk {i+1}: {eid[:8]}…" for i, eid in enumerate(evidence_ids)]
    vals = scores if scores and len(scores) == n else [0.75] * n

    colors = [SUCCESS if v >= 0.7 else WARNING if v >= 0.4 else DANGER for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in vals],
        textposition="auto",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=max(150, n * 40 + 60),
        xaxis=dict(range=[0, 1], title="Similarity Score", color=TEXT),
        yaxis=dict(color=TEXT),
        title=dict(text="Evidence Chunk Similarity Scores", font=dict(color=TEXT, size=13)),
    )
    return fig


# ---------------------------------------------------------------------------
# KPI trend line
# ---------------------------------------------------------------------------

def kpi_line(values: list[float], label: str, color: str = PRIMARY) -> go.Figure:
    """Simple sparkline for a KPI trend."""
    fig = go.Figure(go.Scatter(
        y=values,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=120,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        title=dict(text=label, font=dict(color=TEXT, size=12)),
    )
    return fig


# ---------------------------------------------------------------------------
# Prediction distribution donut
# ---------------------------------------------------------------------------

def prediction_donut(positive: int, negative: int) -> go.Figure:
    """Donut chart showing positive vs negative prediction split."""
    total = positive + negative
    if total == 0:
        positive, negative = 1, 4  # placeholder

    fig = go.Figure(go.Pie(
        labels=["Will Subscribe", "Won't Subscribe"],
        values=[positive, negative],
        hole=0.65,
        marker=dict(colors=[SUCCESS, NEUTRAL]),
        textfont=dict(color=TEXT),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=240,
        showlegend=True,
        legend=dict(font=dict(color=TEXT), orientation="h", y=-0.1),
        title=dict(text="Prediction Distribution", font=dict(color=TEXT, size=13)),
    )
    return fig


# ---------------------------------------------------------------------------
# RAG metrics bar
# ---------------------------------------------------------------------------

def rag_metrics_bar(stats: dict) -> go.Figure:
    """Grouped bar chart for RAG retrieval statistics."""
    hit_rate = stats.get("hit_rate", stats.get("avg_evidence_ids_per_query", 0))
    refusal_rate = stats.get("refusal_rate", 0)

    # Normalise: hit_rate might be >1 if it's avg evidence IDs
    if isinstance(hit_rate, float) and hit_rate > 1:
        hit_rate = min(hit_rate / 10, 1.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Hit Rate",
        x=["Hit Rate"],
        y=[hit_rate],
        marker_color=SUCCESS,
        text=[f"{hit_rate:.1%}"],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Refusal Rate",
        x=["Refusal Rate"],
        y=[refusal_rate],
        marker_color=DANGER,
        text=[f"{refusal_rate:.1%}"],
        textposition="outside",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        yaxis=dict(range=[0, 1.1], tickformat=".0%", color=TEXT),
        xaxis=dict(color=TEXT),
        showlegend=False,
        title=dict(text="RAG Retrieval Stats", font=dict(color=TEXT, size=13)),
    )
    return fig


# ---------------------------------------------------------------------------
# Complaint theme bar
# ---------------------------------------------------------------------------

def complaint_themes_bar(themes: list[dict]) -> go.Figure:
    """Horizontal bar chart for complaint theme evidence counts."""
    if not themes:
        fig = go.Figure()
        fig.add_annotation(text="No themes available", showarrow=False, font=dict(color=NEUTRAL))
        fig.update_layout(**_LAYOUT_BASE, height=120)
        return fig

    labels = [t.get("theme", f"Theme {i+1}") for i, t in enumerate(themes)]
    counts = [len(t.get("evidence_ids", [])) for t in themes]

    fig = go.Figure(go.Bar(
        y=labels,
        x=counts,
        orientation="h",
        marker_color=PRIMARY,
        text=counts,
        textposition="auto",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=max(120, len(themes) * 45 + 60),
        xaxis=dict(title="Evidence Chunks", color=TEXT),
        yaxis=dict(color=TEXT),
        title=dict(text="Complaint Theme Distribution", font=dict(color=TEXT, size=13)),
    )
    return fig
