"""
src/serving/schemas.py
------------------------
Pydantic v2 request / response schemas for the Meridian Financial ML API.

All field names match the UCI Bank Marketing feature set (post snake_case
normalisation applied in phase-1 ingestion).

Design
------
* Request schema mirrors the feature pipeline exactly — no silent coercions
* Response schemas are versioned and include prediction metadata
* Optional / nullable fields return explicit ``None`` so downstream consumers
  can distinguish "missing" from "zero"
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Prediction request
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input schema for ``POST /predict``.

    All 20 feature columns are required.  Values must match the ranges and
    categories seen during training — the model handles unknown categoricals
    gracefully (``handle_unknown='ignore'``), but out-of-range numerics are
    flagged by the validator.
    """

    # Numeric features
    age: int = Field(..., ge=17, le=120, description="Client age in years.")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds.")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign.")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign.")
    pdays: int = Field(
        ..., ge=0,
        description="Days since last contact (999 = never contacted).",
    )
    emp_var_rate: float = Field(..., description="Employment variation rate (quarterly).")
    cons_price_idx: float = Field(..., description="Consumer price index (monthly).")
    cons_conf_idx: float = Field(..., description="Consumer confidence index (monthly).")
    euribor3m: float = Field(..., description="Euribor 3-month rate (daily).")
    nr_employed: float = Field(..., description="Number of employees (quarterly, thousands).")

    # Categorical features
    job: str = Field(..., description="Client job type.")
    marital: str = Field(..., description="Marital status.")
    education: str = Field(..., description="Education level.")
    default: str = Field(..., description="Has credit in default? (yes/no/unknown)")
    housing: str = Field(..., description="Has housing loan? (yes/no/unknown)")
    loan: str = Field(..., description="Has personal loan? (yes/no/unknown)")
    contact: str = Field(..., description="Contact communication type.")
    month: str = Field(..., description="Last contact month (lowercase, e.g. 'may').")
    day_of_week: str = Field(..., description="Last contact day of week (e.g. 'mon').")
    poutcome: str = Field(..., description="Outcome of previous campaign.")

    model_config = {"json_schema_extra": {
        "example": {
            "age": 35,
            "duration": 180,
            "campaign": 2,
            "previous": 0,
            "pdays": 999,
            "emp_var_rate": -1.8,
            "cons_price_idx": 93.075,
            "cons_conf_idx": -47.1,
            "euribor3m": 1.334,
            "nr_employed": 5099.1,
            "job": "admin.",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "poutcome": "nonexistent",
        }
    }}

    @field_validator("month")
    @classmethod
    def normalise_month(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("day_of_week")
    @classmethod
    def normalise_day(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("job", "marital", "education", "default", "housing", "loan",
                     "contact", "poutcome")
    @classmethod
    def normalise_categorical(cls, v: str) -> str:
        return v.strip().lower()


# ---------------------------------------------------------------------------
# Prediction response
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """Response schema for ``POST /predict``."""

    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model's predicted probability of campaign conversion.",
    )
    prediction: int = Field(
        ..., ge=0, le=1,
        description="Binary prediction: 1 = will subscribe, 0 = will not.",
    )
    threshold: float = Field(
        ..., description="Decision threshold used to derive ``prediction``."
    )
    model_version: str = Field(
        ..., description="Identifier of the model that produced this prediction."
    )
    latency_ms: float = Field(
        ..., description="End-to-end request processing time in milliseconds."
    )


# ---------------------------------------------------------------------------
# Health check response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response schema for ``GET /health``."""

    status: str = Field(..., description="'ok' when the service is healthy.")
    model_version: str = Field(
        ..., description="Identifier of the currently loaded prediction model."
    )
    vector_index_version: Optional[str] = Field(
        default=None,
        description="Version of the ChromaDB vector index (placeholder for RAG phases).",
    )


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""

    error: str = Field(..., description="Machine-readable error type.")
    detail: str = Field(..., description="Human-readable error description.")
    status_code: int = Field(..., description="HTTP status code.")
