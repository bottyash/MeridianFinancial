"""
dashboard/components/cards.py
-------------------------------
Reusable Streamlit card components using st.markdown with custom CSS.

Cards render styled metric blocks, status indicators, and alert boxes
without requiring external CSS frameworks.
"""

from __future__ import annotations

import streamlit as st

from dashboard.utils.helpers import (
    conversion_band_color,
    conversion_band_emoji,
    format_probability,
    format_latency,
    format_int,
    sufficiency_color,
    api_status_badge,
)


# ---------------------------------------------------------------------------
# KPI metric card
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str = "", color: str = "#6366f1") -> None:
    """Render a styled KPI metric card."""
    delta_html = f"<div style='font-size:0.75rem;color:#9ca3af;margin-top:2px'>{delta}</div>" if delta else ""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-left: 4px solid {color};
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 8px;
    ">
        <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;font-weight:600">{label}</div>
        <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-top:4px">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Conversion band card
# ---------------------------------------------------------------------------

def conversion_band_card(probability: float, band: str, prediction: int, threshold: float) -> None:
    """Render a conversion band result card."""
    color = conversion_band_color(band)
    emoji = conversion_band_emoji(band)
    decision = "✅ Will Subscribe" if prediction == 1 else "❌ Won't Subscribe"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid {color}55;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    ">
        <div style="font-size:3rem">{emoji}</div>
        <div style="font-size:1rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px">Conversion Band</div>
        <div style="font-size:2rem;font-weight:800;color:{color};margin-top:4px">{band}</div>
        <div style="font-size:2.5rem;font-weight:700;color:#f1f5f9;margin-top:8px">{format_probability(probability)}</div>
        <div style="font-size:0.9rem;color:#94a3b8;margin-top:4px">probability (threshold: {threshold:.0%})</div>
        <div style="
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 8px 16px;
            margin-top: 16px;
            font-size:1rem;
            font-weight:600;
            color:#e2e8f0;
        ">{decision}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Evidence card
# ---------------------------------------------------------------------------

def evidence_card(index: int, chunk_id: str, score: float | None = None) -> None:
    """Render a single evidence chunk card."""
    score_html = ""
    if score is not None:
        from dashboard.utils.helpers import sufficiency_color
        color = "#22c55e" if score >= 0.7 else "#f59e0b" if score >= 0.4 else "#ef4444"
        score_html = f"""<span style="
            background:{color}22;color:{color};
            padding:2px 8px;border-radius:12px;
            font-size:0.75rem;font-weight:600;margin-left:8px
        ">sim={score:.2f}</span>"""

    st.markdown(f"""
    <div style="
        background:#1e293b;
        border:1px solid rgba(255,255,255,0.08);
        border-radius:10px;
        padding:12px 16px;
        margin-bottom:6px;
        display:flex;
        align-items:center;
    ">
        <span style="color:#6366f1;font-weight:700;margin-right:12px">#{index+1}</span>
        <code style="color:#94a3b8;font-size:0.8rem">{chunk_id}</code>
        {score_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------

def status_banner(healthy: bool, model_version: str = "", environment: str = "") -> None:
    """Render the backend status banner in the sidebar."""
    if healthy:
        color, icon, text = "#22c55e", "🟢", "Backend Online"
    else:
        color, icon, text = "#ef4444", "🔴", "Backend Offline"

    meta = ""
    if model_version:
        meta += f"<div style='font-size:0.7rem;color:#94a3b8'>Model: {model_version}</div>"
    if environment:
        meta += f"<div style='font-size:0.7rem;color:#94a3b8'>Env: {environment}</div>"

    st.markdown(f"""
    <div style="
        background:{color}15;
        border:1px solid {color}44;
        border-radius:10px;
        padding:10px 14px;
        margin-bottom:12px;
    ">
        <div style="font-weight:700;color:{color}">{icon} {text}</div>
        {meta}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Alert box
# ---------------------------------------------------------------------------

def alert(message: str, kind: str = "error") -> None:
    """Render a styled alert box.

    Parameters
    ----------
    message: str
    kind: "error" | "warning" | "info" | "success"
    """
    configs = {
        "error":   ("#ef4444", "❌"),
        "warning": ("#f59e0b", "⚠️"),
        "info":    ("#6366f1", "ℹ️"),
        "success": ("#22c55e", "✅"),
    }
    color, icon = configs.get(kind, configs["info"])
    st.markdown(f"""
    <div style="
        background:{color}15;
        border:1px solid {color}44;
        border-radius:10px;
        padding:12px 16px;
        margin:8px 0;
        color:#e2e8f0;
    ">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sufficiency note card
# ---------------------------------------------------------------------------

def sufficiency_note_card(note: str) -> None:
    """Render the evidence sufficiency note."""
    color = sufficiency_color(note)
    st.markdown(f"""
    <div style="
        background:{color}15;
        border-left:3px solid {color};
        border-radius:0 8px 8px 0;
        padding:10px 14px;
        margin:8px 0;
        font-size:0.85rem;
        color:#e2e8f0;
    ">
        📊 {note}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Refusal card
# ---------------------------------------------------------------------------

def refusal_card(message: str) -> None:
    """Render a styled refusal message card."""
    st.markdown(f"""
    <div style="
        background:rgba(239,68,68,0.08);
        border:1px solid rgba(239,68,68,0.3);
        border-radius:12px;
        padding:20px;
        text-align:center;
        margin:12px 0;
    ">
        <div style="font-size:2rem">🚫</div>
        <div style="font-weight:700;color:#ef4444;margin-top:6px">Insufficient Evidence</div>
        <div style="color:#94a3b8;margin-top:8px;font-size:0.9rem">{message}</div>
    </div>
    """, unsafe_allow_html=True)
