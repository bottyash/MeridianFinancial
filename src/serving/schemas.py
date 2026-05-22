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


# ---------------------------------------------------------------------------
# RAG / complaint answering schemas (Phase 6)
# ---------------------------------------------------------------------------

class AskComplaintsRequest(BaseModel):
    """Input schema for ``POST /ask-complaints``."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question about consumer complaints.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of complaint chunks to retrieve as evidence.",
    )
    product_filter: Optional[str] = Field(
        default=None,
        description="Optional ChromaDB metadata filter on product category.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What are the most common credit card billing issues?",
            "top_k": 5,
            "product_filter": None,
        }
    }}


class TokenUsage(BaseModel):
    """LLM token consumption breakdown."""

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class AskComplaintsResponse(BaseModel):
    """Response schema for ``POST /ask-complaints``."""

    question: str = Field(..., description="Original question as received.")
    answer: str = Field(..., description="Grounded answer or refusal message.")
    refused: bool = Field(
        ...,
        description="True when the engine refused due to insufficient evidence.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs used as evidence context.",
    )
    evidence_sufficiency: str = Field(
        default="",
        description="Human-readable note about evidence quality.",
    )
    prompt_version: str = Field(
        ...,
        description="Version tag of the prompt template used.",
    )
    retrieval_count: int = Field(
        default=0,
        description="Total chunks retrieved (before sufficiency filtering).",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="LLM token consumption.",
    )
    latency_ms: float = Field(
        ...,
        description="End-to-end request processing time in milliseconds.",
    )
    model: str = Field(
        default="",
        description="Mistral model identifier used for generation.",
    )


# ---------------------------------------------------------------------------
# Phase 7 — Integration endpoint schemas
# ---------------------------------------------------------------------------

# ── /batch-score ─────────────────────────────────────────────────────────────

class BatchScoreItem(BaseModel):
    """A single result row within a batch-score response."""

    row_index: int = Field(..., description="Zero-based index of this record in the input.")
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1)
    threshold: float
    error: Optional[str] = Field(
        default=None,
        description="Per-row error message (None when successful).",
    )


class BatchScoreRequest(BaseModel):
    """Input schema for ``POST /batch-score``."""

    records: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of prediction request records (max 500 per call).",
    )


class BatchScoreResponse(BaseModel):
    """Response schema for ``POST /batch-score``."""

    total: int = Field(..., description="Total number of input records.")
    succeeded: int = Field(..., description="Records scored without error.")
    failed: int = Field(..., description="Records that raised an error.")
    model_version: str = Field(..., description="Model used for all predictions.")
    latency_ms: float = Field(..., description="Total batch processing time in ms.")
    results: list[BatchScoreItem] = Field(..., description="Per-record results.")


# ── /customer-intel ───────────────────────────────────────────────────────────

class ComplaintTheme(BaseModel):
    """A single complaint theme extracted from retrieved evidence."""

    theme: str = Field(..., description="Short description of the complaint theme.")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that support this theme.",
    )


class CustomerIntelRequest(BaseModel):
    """Input schema for ``POST /customer-intel``."""

    # ML prediction features (full PredictRequest subset is embedded)
    customer_profile: PredictRequest = Field(
        ..., description="Customer feature profile for conversion scoring."
    )
    # RAG question
    question: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional complaint intelligence question. "
                    "Defaults to 'What are common product complaints?'.",
    )
    # Filters
    product_filter: Optional[str] = Field(
        default=None,
        description="Filter complaints by product category.",
    )
    issue_filter: Optional[str] = Field(
        default=None,
        description="Filter complaints by issue type.",
    )
    date_filter: Optional[str] = Field(
        default=None,
        description="Filter complaints by date (format: YYYY-MM-DD).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of complaint chunks to retrieve.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "customer_profile": {
                "age": 42, "duration": 220, "campaign": 1, "previous": 0,
                "pdays": 999, "emp_var_rate": -1.8, "cons_price_idx": 93.075,
                "cons_conf_idx": -47.1, "euribor3m": 1.334, "nr_employed": 5099.1,
                "job": "management", "marital": "married",
                "education": "university.degree", "default": "no",
                "housing": "yes", "loan": "no", "contact": "cellular",
                "month": "may", "day_of_week": "thu", "poutcome": "nonexistent",
            },
            "question": "What are the most common credit card complaints?",
            "product_filter": "Credit card",
            "top_k": 5,
        }
    }}


class CustomerIntelResponse(BaseModel):
    """Response schema for ``POST /customer-intel``."""

    # ML output
    conversion_probability: float = Field(..., ge=0.0, le=1.0)
    conversion_prediction: int = Field(..., ge=0, le=1)
    conversion_band: str = Field(
        ...,
        description="Human label: 'HIGH' (>= 0.7), 'MEDIUM' (>= 0.4), or 'LOW' (< 0.4).",
    )
    model_version: str

    # RAG output
    complaint_question: str = Field(..., description="Complaint intelligence question answered.")
    complaint_answer: str = Field(..., description="Grounded answer to the complaint question.")
    complaint_refused: bool
    complaint_themes: list[ComplaintTheme] = Field(
        default_factory=list,
        description="Extracted complaint themes with cited evidence IDs.",
    )
    cited_complaint_ids: list[str] = Field(
        default_factory=list,
        description="Unique complaint IDs cited across all evidence chunks.",
    )
    evidence_sufficiency: str = Field(default="")

    # Latency
    ml_latency_ms: float
    rag_latency_ms: float
    total_latency_ms: float


# ── /metrics ──────────────────────────────────────────────────────────────────

class PredictionDistribution(BaseModel):
    """Distribution of binary predictions seen since startup."""

    total_predictions: int = Field(default=0)
    positive_predictions: int = Field(default=0, description="prediction == 1")
    negative_predictions: int = Field(default=0, description="prediction == 0")
    positive_rate: float = Field(default=0.0)


class RAGRetrievalStats(BaseModel):
    """Aggregate statistics for RAG retrieval calls."""

    total_queries: int = Field(default=0)
    refused_queries: int = Field(default=0)
    refusal_rate: float = Field(default=0.0)
    avg_evidence_ids_per_query: float = Field(default=0.0)


class MetricsResponse(BaseModel):
    """Response schema for ``GET /metrics``."""

    uptime_seconds: float = Field(..., description="Seconds since application startup.")
    total_requests: int
    error_count: int
    error_rate: float
    avg_latency_ms: float = Field(..., description="Average request latency across all endpoints.")
    prediction_distribution: PredictionDistribution
    rag_retrieval_stats: RAGRetrievalStats
