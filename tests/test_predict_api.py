"""
tests/test_predict_api.py
--------------------------
Integration tests for Phase 4: FastAPI ML serving layer.

Uses FastAPI ``TestClient`` (httpx transport) — no real server required.

Covers:
  * GET /health — ok response, model version, vector index placeholder
  * GET /health — 503 when model bundle unavailable
  * POST /predict — valid payload → probability in [0,1], binary decision
  * POST /predict — probability/threshold/model_version/latency_ms present
  * POST /predict — invalid payload (missing field) → 422
  * POST /predict — invalid age → 422
  * POST /predict — extra unknown fields are ignored
  * POST /predict — response model_version matches loaded model name
  * POST /predict — decision matches threshold rule
  * Request timing header present (X-Process-Time-Ms)
  * OpenAPI docs endpoint reachable
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Minimal valid payload (matches PredictRequest schema)
# ---------------------------------------------------------------------------
VALID_PAYLOAD: dict = {
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_bundle(probability: float = 0.35, model_version: str = "improved_model"):
    """Return a MagicMock ModelBundle that produces a fixed probability."""
    from src.serving.model_loader import ModelBundle

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 63))

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - probability, probability]])

    bundle = ModelBundle(
        preprocessor=mock_preprocessor,
        model=mock_model,
        feature_schema={
            "numeric_features": [
                "age", "duration", "campaign", "previous", "pdays",
                "emp_var_rate", "cons_price_idx", "cons_conf_idx",
                "euribor3m", "nr_employed",
            ],
            "categorical_features": [
                "job", "marital", "education", "default", "housing",
                "loan", "contact", "month", "day_of_week", "poutcome",
            ],
            "all_feature_columns": [],
        },
        model_version=model_version,
        threshold=0.5,
    )
    return bundle


@pytest.fixture()
def client():
    """TestClient with a mocked model bundle so no artifacts need to exist."""
    from src.serving.app import app
    from src.serving.model_loader import reset_model_bundle_cache

    reset_model_bundle_cache()
    bundle = _make_mock_bundle()

    with patch("src.serving.model_loader.get_model_bundle", return_value=bundle), \
         patch("src.serving.routes.get_model_bundle", return_value=bundle):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
    reset_model_bundle_cache()


@pytest.fixture()
def client_high_prob():
    """TestClient wired to return high probability (above 0.5 threshold)."""
    from src.serving.app import app
    from src.serving.model_loader import reset_model_bundle_cache

    reset_model_bundle_cache()
    bundle = _make_mock_bundle(probability=0.8)

    with patch("src.serving.model_loader.get_model_bundle", return_value=bundle), \
         patch("src.serving.routes.get_model_bundle", return_value=bundle):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
    reset_model_bundle_cache()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_ok_status(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_string(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_returns_model_version(self, client):
        resp = client.get("/health")
        assert "model_version" in resp.json()
        assert resp.json()["model_version"] == "improved_model"

    def test_health_vector_index_version_is_none(self, client):
        resp = client.get("/health")
        assert resp.json()["vector_index_version"] is None

    def test_health_503_when_bundle_unavailable(self):
        from src.serving.app import app
        from src.serving.model_loader import reset_model_bundle_cache

        reset_model_bundle_cache()
        with patch(
            "src.serving.routes.get_model_bundle",
            side_effect=FileNotFoundError("artifacts missing"),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/health")
        assert resp.status_code == 503
        reset_model_bundle_cache()


# ---------------------------------------------------------------------------
# POST /predict — happy path
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_probability_in_range(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        prob = resp.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_binary_decision_is_0_or_1(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["prediction"] in (0, 1)

    def test_predict_threshold_present(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert "threshold" in resp.json()
        assert isinstance(resp.json()["threshold"], float)

    def test_predict_model_version_present(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert "model_version" in resp.json()
        assert resp.json()["model_version"] == "improved_model"

    def test_predict_latency_ms_present_and_positive(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert "latency_ms" in resp.json()
        assert resp.json()["latency_ms"] >= 0

    def test_predict_decision_follows_threshold_low_prob(self, client):
        """Mock returns 0.35 probability → below 0.5 threshold → prediction=0."""
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert data["probability"] < data["threshold"]
        assert data["prediction"] == 0

    def test_predict_decision_follows_threshold_high_prob(self, client_high_prob):
        """Mock returns 0.80 probability → above 0.5 threshold → prediction=1."""
        resp = client_high_prob.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert data["probability"] >= data["threshold"]
        assert data["prediction"] == 1

    def test_predict_response_has_all_required_keys(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        required = {"probability", "prediction", "threshold", "model_version", "latency_ms"}
        assert required.issubset(resp.json().keys())

    def test_timing_header_present(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert "x-process-time-ms" in resp.headers


# ---------------------------------------------------------------------------
# POST /predict — validation errors
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_missing_required_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_age_too_young_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "age": 5}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_age_too_old_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "age": 200}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_duration_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "duration": -1}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_campaign_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "campaign": 0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_month_normalised_to_lowercase(self, client):
        """Validator should normalise 'MAY' → 'may' without error."""
        payload = {**VALID_PAYLOAD, "month": "MAY"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_day_normalised_to_lowercase(self, client):
        payload = {**VALID_PAYLOAD, "day_of_week": "MON"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# OpenAPI docs
# ---------------------------------------------------------------------------

class TestOpenAPIDocs:
    def test_openapi_schema_reachable(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_contains_predict_path(self, client):
        schema = client.get("/openapi.json").json()
        assert "/predict" in schema["paths"]

    def test_openapi_contains_health_path(self, client):
        schema = client.get("/openapi.json").json()
        assert "/health" in schema["paths"]

    def test_docs_page_reachable(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
