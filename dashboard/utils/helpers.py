"""
dashboard/utils/helpers.py
---------------------------
Shared formatting and UI helper utilities for the dashboard.
No ML logic — pure display helpers only.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Conversion band helpers
# ---------------------------------------------------------------------------

BAND_COLORS = {
    "HIGH": "#22c55e",    # green
    "MEDIUM": "#f59e0b",  # amber
    "LOW": "#ef4444",     # red
}

BAND_EMOJI = {
    "HIGH": "🟢",
    "MEDIUM": "🟡",
    "LOW": "🔴",
}


def conversion_band_color(band: str) -> str:
    """Return the hex color for a conversion band label."""
    return BAND_COLORS.get(band.upper(), "#6b7280")


def conversion_band_emoji(band: str) -> str:
    """Return the emoji for a conversion band label."""
    return BAND_EMOJI.get(band.upper(), "⚪")


def format_probability(prob: float) -> str:
    """Format a 0–1 probability as a percentage string."""
    return f"{prob * 100:.1f}%"


def format_latency(ms: float) -> str:
    """Format a latency value for display."""
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def format_int(value: int | float) -> str:
    """Format an integer with comma thousands separators."""
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Evidence sufficiency helpers
# ---------------------------------------------------------------------------

SUFFICIENCY_COLORS = {
    "HIGH": "#22c55e",
    "MEDIUM": "#f59e0b",
    "LOW": "#ef4444",
    "NO": "#9ca3af",
}


def sufficiency_color(note: str) -> str:
    """Extract quality tier from sufficiency note and return color."""
    upper = note.upper()
    for tier in ("HIGH", "MEDIUM", "LOW"):
        if tier in upper:
            return SUFFICIENCY_COLORS[tier]
    return SUFFICIENCY_COLORS["NO"]


# ---------------------------------------------------------------------------
# API status helpers
# ---------------------------------------------------------------------------

def api_status_badge(healthy: bool) -> str:
    """Return a colored status badge string."""
    return "🟢 Online" if healthy else "🔴 Offline"


# ---------------------------------------------------------------------------
# Safe dict access
# ---------------------------------------------------------------------------

def safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


# ---------------------------------------------------------------------------
# Product / Issue filter options (pulled from CFPB taxonomy)
# ---------------------------------------------------------------------------

PRODUCT_OPTIONS = [
    "",
    "Credit card",
    "Mortgage",
    "Student loan",
    "Consumer loan",
    "Debt collection",
    "Credit reporting",
    "Bank account or service",
    "Payday loan",
    "Money transfers",
    "Prepaid card",
    "Other financial service",
]

ISSUE_OPTIONS = [
    "",
    "Billing disputes",
    "Incorrect information on credit report",
    "Loan modification",
    "Debt is not mine",
    "Communication tactics",
    "Managing the loan or lease",
    "Problems with credit report",
    "Struggling to pay mortgage",
    "Improper use of credit report",
]
