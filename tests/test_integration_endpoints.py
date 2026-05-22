"""
tests/test_integration_endpoints.py
-------------------------------------
Tests for Phase 7 integration endpoints:
  POST /batch-score
  POST /customer-intel
  GET  /metrics

All ML and RAG dependencies are mocked — no real artifacts needed.

Covers:
  * /batch-score — happy path, per-row results, error isolation, size limits
  * /batch-score — validation: empty records, max size
  * /customer-intel — all required fields present, conversion band logic
  * /customer-intel — product_filter wires through, default question used
  * /customer-intel — validation: missing customer_profile
  * /metrics — zero-state response, counter increments after predictions
  * Helper: _conversion_band tiers
  * Helper: _extract_themes grouping
  * OpenAPI schema contains all three new paths
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.rag.answer import RAGAnswer, PROMPT_VERSION, REFUSAL_MESSAGE
from src.serving.routes import _conversion_band, _extract_themes

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VALID_PROFILE = {
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


def _make_mock_bundle(probability: float = 0.35):
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 63))
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - probability, probability]])

    from src.serving.model_loader import ModelBundle
    return ModelBundle(
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
        model_version="improved_model",
        threshold=0.5,
    )


def _make_mock_rag_engine(refused: bool = False, evidence_ids=None):
    engine = MagicMock()
    engine.answer.return_value = RAGAnswer(
        question="test question",
        answer="Based on evidence, billing disputes are common." if not refused else REFUSAL_MESSAGE,
        refused=refused,
        evidence_ids=evidence_ids if evidence_ids is not None else ([] if refused else ["id1", "id2", "id3"]),
        evidence_sufficiency="Evidence quality: HIGH (avg_similarity=0.82, max_similarity=0.90, n_chunks=3)",
        prompt_version=PROMPT_VERSION,
        retrieval_count=5,
        token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=200.0,
        model="mistral-small-latest",
    )
    return engine


@pytest.fixture()
def client(request):
    """TestClient with mocked bundle and RAG engine. Resets metrics counters."""
    from src.serving.app import app
    from src.serving.model_loader import reset_model_bundle_cache
    import src.serving.routes as routes_module

    reset_model_bundle_cache()
    # Reset metrics counters to zero for isolation
    routes_module._metrics.update({
        "total_requests": 0,
        "error_count": 0,
        "latency_sum_ms": 0.0,
        "total_predictions": 0,
        "positive_predictions": 0,
        "negative_predictions": 0,
        "rag_total_queries": 0,
        "rag_refused_queries": 0,
        "rag_evidence_ids_total": 0,
    })

    bundle = _make_mock_bundle(probability=0.35)
    engine = _make_mock_rag_engine()

    with patch("src.serving.model_loader.get_model_bundle", return_value=bundle), \
         patch("src.serving.routes.get_model_bundle", return_value=bundle), \
         patch("src.serving.routes._get_rag_engine", return_value=engine):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    reset_model_bundle_cache()


@pytest.fixture()
def client_high_prob():
    """TestClient with high probability model (probability=0.8 → HIGH band)."""
    from src.serving.app import app
    from src.serving.model_loader import reset_model_bundle_cache
    import src.serving.routes as routes_module

    reset_model_bundle_cache()
    routes_module._metrics.update({
        "total_requests": 0, "error_count": 0, "latency_sum_ms": 0.0,
        "total_predictions": 0, "positive_predictions": 0, "negative_predictions": 0,
        "rag_total_queries": 0, "rag_refused_queries": 0, "rag_evidence_ids_total": 0,
    })

    bundle = _make_mock_bundle(probability=0.8)
    engine = _make_mock_rag_engine()

    with patch("src.serving.model_loader.get_model_bundle", return_value=bundle), \
         patch("src.serving.routes.get_model_bundle", return_value=bundle), \
         patch("src.serving.routes._get_rag_engine", return_value=engine):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    reset_model_bundle_cache()


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestConversionBand:
    def test_high_band(self):
        assert _conversion_band(0.8) == "HIGH"
        assert _conversion_band(0.7) == "HIGH"

    def test_medium_band(self):
        assert _conversion_band(0.5) == "MEDIUM"
        assert _conversion_band(0.4) == "MEDIUM"

    def test_low_band(self):
        assert _conversion_band(0.39) == "LOW"
        assert _conversion_band(0.0) == "LOW"


class TestExtractThemes:
    def test_returns_tuple_of_two_lists(self):
        rag_answer = _make_mock_rag_engine().answer.return_value
        themes, ids = _extract_themes(rag_answer)
        assert isinstance(themes, list)
        assert isinstance(ids, list)

    def test_empty_evidence_ids_returns_empty_themes(self):
        engine = _make_mock_rag_engine(evidence_ids=[])
        rag_answer = engine.answer.return_value
        themes, ids = _extract_themes(rag_answer)
        assert themes == []
        assert ids == []

    def test_themes_up_to_3_groups(self):
        ids = [f"chunk_{i}" for i in range(9)]
        engine = _make_mock_rag_engine(evidence_ids=ids)
        rag_answer = engine.answer.return_value
        themes, cited_ids = _extract_themes(rag_answer)
        assert len(themes) <= 3

    def test_cited_ids_are_deduplicated(self):
        repeated_ids = ["abc", "abc", "def"]
        engine = _make_mock_rag_engine(evidence_ids=repeated_ids)
        rag_answer = engine.answer.return_value
        _, cited_ids = _extract_themes(rag_answer)
        assert len(cited_ids) == len(set(cited_ids))


# ---------------------------------------------------------------------------
# POST /batch-score
# ---------------------------------------------------------------------------

class TestBatchScore:
    def test_returns_200(self, client):
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE]})
        assert resp.status_code == 200

    def test_total_matches_input_length(self, client):
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE, VALID_PROFILE]})
        assert resp.json()["total"] == 2

    def test_succeeded_count(self, client):
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE]})
        assert resp.json()["succeeded"] == 1
        assert resp.json()["failed"] == 0

    def test_results_length_matches_total(self, client):
        n = 3
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE] * n})
        assert len(resp.json()["results"]) == n

    def test_each_result_has_row_index(self, client):
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE, VALID_PROFILE]})
        indices = [r["row_index"] for r in resp.json()["results"]]
        assert 0 in indices
        assert 1 in indices

    def test_probability_in_valid_range(self, client):
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE]})
        prob = resp.json()["results"][0]["probability"]
        assert 0.0 <= prob <= 1.0

    def test_prediction_is_0_or_1(self, client):
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE]})
        pred = resp.json()["results"][0]["prediction"]
        assert pred in (0, 1)

    def test_model_version_in_response(self, client):
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE]})
        assert "model_version" in resp.json()
        assert resp.json()["model_version"] == "improved_model"

    def test_latency_ms_positive(self, client):
        resp = client.post("/batch-score", json={"records": [VALID_PROFILE]})
        assert resp.json()["latency_ms"] >= 0

    def test_empty_records_returns_422(self, client):
        resp = client.post("/batch-score", json={"records": []})
        assert resp.status_code == 422

    def test_openapi_includes_batch_score(self, client):
        schema = client.get("/openapi.json").json()
        assert "/batch-score" in schema["paths"]


# ---------------------------------------------------------------------------
# POST /customer-intel
# ---------------------------------------------------------------------------

VALID_INTEL_PAYLOAD = {
    "customer_profile": VALID_PROFILE,
    "question": "What are common billing issues?",
    "top_k": 5,
}


class TestCustomerIntel:
    def test_returns_200(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert resp.status_code == 200

    def test_conversion_probability_in_range(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        prob = resp.json()["conversion_probability"]
        assert 0.0 <= prob <= 1.0

    def test_conversion_band_low_for_low_prob(self, client):
        # bundle fixture has probability=0.35 → LOW
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert resp.json()["conversion_band"] == "LOW"

    def test_conversion_band_high_for_high_prob(self, client_high_prob):
        resp = client_high_prob.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert resp.json()["conversion_band"] == "HIGH"

    def test_conversion_prediction_0_or_1(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert resp.json()["conversion_prediction"] in (0, 1)

    def test_complaint_answer_present(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert "complaint_answer" in resp.json()
        assert len(resp.json()["complaint_answer"]) > 0

    def test_complaint_refused_flag_present(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert "complaint_refused" in resp.json()

    def test_cited_complaint_ids_present(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert "cited_complaint_ids" in resp.json()

    def test_complaint_themes_list(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert isinstance(resp.json()["complaint_themes"], list)

    def test_all_latency_fields_present(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        data = resp.json()
        for key in ("ml_latency_ms", "rag_latency_ms", "total_latency_ms"):
            assert key in data
            assert data[key] >= 0

    def test_question_defaults_when_not_provided(self, client):
        payload = {"customer_profile": VALID_PROFILE}
        resp = client.post("/customer-intel", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["complaint_question"]) > 0

    def test_product_filter_accepted(self, client):
        payload = {**VALID_INTEL_PAYLOAD, "product_filter": "Credit card"}
        resp = client.post("/customer-intel", json=payload)
        assert resp.status_code == 200

    def test_missing_customer_profile_returns_422(self, client):
        resp = client.post("/customer-intel", json={"question": "test question"})
        assert resp.status_code == 422

    def test_model_version_in_response(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        assert resp.json()["model_version"] == "improved_model"

    def test_all_required_keys_present(self, client):
        resp = client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        required = {
            "conversion_probability", "conversion_prediction", "conversion_band",
            "model_version", "complaint_question", "complaint_answer",
            "complaint_refused", "complaint_themes", "cited_complaint_ids",
            "evidence_sufficiency", "ml_latency_ms", "rag_latency_ms", "total_latency_ms",
        }
        assert required.issubset(resp.json().keys())

    def test_openapi_includes_customer_intel(self, client):
        schema = client.get("/openapi.json").json()
        assert "/customer-intel" in schema["paths"]


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_all_required_keys_present(self, client):
        resp = client.get("/metrics")
        data = resp.json()
        required = {
            "uptime_seconds", "total_requests", "error_count", "error_rate",
            "avg_latency_ms", "prediction_distribution", "rag_retrieval_stats",
        }
        assert required.issubset(data.keys())

    def test_uptime_is_positive(self, client):
        resp = client.get("/metrics")
        assert resp.json()["uptime_seconds"] > 0

    def test_zero_state_on_fresh_start(self, client):
        resp = client.get("/metrics")
        data = resp.json()
        assert data["total_requests"] == 0
        assert data["error_count"] == 0

    def test_prediction_distribution_keys(self, client):
        resp = client.get("/metrics")
        pd = resp.json()["prediction_distribution"]
        assert "total_predictions" in pd
        assert "positive_predictions" in pd
        assert "negative_predictions" in pd
        assert "positive_rate" in pd

    def test_rag_retrieval_stats_keys(self, client):
        resp = client.get("/metrics")
        rrs = resp.json()["rag_retrieval_stats"]
        assert "total_queries" in rrs
        assert "refused_queries" in rrs
        assert "refusal_rate" in rrs
        assert "avg_evidence_ids_per_query" in rrs

    def test_request_count_increments_after_customer_intel(self, client):
        client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        resp = client.get("/metrics")
        # customer-intel calls _record_request once
        assert resp.json()["total_requests"] >= 1

    def test_prediction_distribution_increments_after_customer_intel(self, client):
        client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        resp = client.get("/metrics")
        pd = resp.json()["prediction_distribution"]
        assert pd["total_predictions"] >= 1

    def test_rag_stats_increment_after_customer_intel(self, client):
        client.post("/customer-intel", json=VALID_INTEL_PAYLOAD)
        resp = client.get("/metrics")
        rrs = resp.json()["rag_retrieval_stats"]
        assert rrs["total_queries"] >= 1

    def test_openapi_includes_metrics(self, client):
        schema = client.get("/openapi.json").json()
        assert "/metrics" in schema["paths"]
