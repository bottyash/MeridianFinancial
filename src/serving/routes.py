"""
src/serving/routes.py
----------------------
FastAPI route handlers for the Meridian Financial ML serving layer.

Routes implemented (Phase 4):
  GET  /health    — liveness + readiness check
  POST /predict   — single-record conversion probability prediction

Each handler:
  * validates the payload via Pydantic schemas
  * logs the request with structured metadata
  * tracks and returns end-to-end latency
  * returns a typed response model
  * raises ``HTTPException`` with a structured body on errors
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from src.serving.model_loader import ModelBundle, get_model_bundle
from src.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger("meridian.routes")

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_to_dataframe(payload: PredictRequest) -> pd.DataFrame:
    """Convert a ``PredictRequest`` to a single-row DataFrame.

    Column names must match those expected by the fitted ColumnTransformer
    (i.e., the ``ALL_FEATURE_COLUMNS`` list from ``features.py``).

    Parameters
    ----------
    payload:
        Validated Pydantic request model.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 20 feature columns.
    """
    return pd.DataFrame([payload.model_dump()])


def _predict_single(
    bundle: ModelBundle,
    payload: PredictRequest,
) -> tuple[float, int]:
    """Run preprocessing and inference for a single record.

    Parameters
    ----------
    bundle:
        Loaded :class:`ModelBundle`.
    payload:
        Validated prediction request.

    Returns
    -------
    tuple[float, int]
        ``(probability, binary_prediction)``
    """
    from src.data_pipeline.features import ALL_FEATURE_COLUMNS

    df = _request_to_dataframe(payload)
    X: np.ndarray = bundle.preprocessor.transform(df[ALL_FEATURE_COLUMNS])
    probability: float = float(bundle.model.predict_proba(X)[0, 1])
    prediction: int = int(probability >= bundle.threshold)
    return probability, prediction


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
async def health_check() -> HealthResponse:
    """Liveness and readiness probe.

    Returns the service status, the currently loaded model version, and a
    placeholder for the RAG vector index version (populated in phase 5+).

    Returns
    -------
    HealthResponse
    """
    try:
        bundle = get_model_bundle()
        model_version = bundle.model_version
    except Exception as exc:  # noqa: BLE001
        logger.error("Health check: model bundle unavailable — %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {exc}",
        ) from exc

    logger.info("Health check OK — model_version=%s", model_version)
    return HealthResponse(
        status="ok",
        model_version=model_version,
        vector_index_version=None,  # populated in RAG phase
    )


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict campaign conversion probability",
    tags=["Prediction"],
    responses={
        422: {"description": "Validation error — invalid payload"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    """Predict the probability that a client will subscribe to a term deposit.

    Parameters
    ----------
    payload:
        20-feature client profile (see schema for field descriptions).

    Returns
    -------
    PredictResponse
        Probability, binary decision, model version, and latency.
    """
    t0 = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", "N/A")

    logger.info(
        "POST /predict — request_id=%s  age=%s  job=%s  contact=%s",
        request_id, payload.age, payload.job, payload.contact,
    )

    try:
        bundle = get_model_bundle()
        probability, prediction = _predict_single(bundle, payload)
    except FileNotFoundError as exc:
        logger.error("Model artifact missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed for request_id=%s: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        ) from exc

    latency_ms = (time.perf_counter() - t0) * 1_000

    logger.info(
        "POST /predict DONE — request_id=%s  probability=%.4f  "
        "prediction=%d  latency_ms=%.2f",
        request_id, probability, prediction, latency_ms,
    )

    return PredictResponse(
        probability=round(probability, 6),
        prediction=prediction,
        threshold=bundle.threshold,
        model_version=bundle.model_version,
        latency_ms=round(latency_ms, 3),
    )
