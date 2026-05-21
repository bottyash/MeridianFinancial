"""
tests/test_rag_api.py
----------------------
Unit and API integration tests for Phase 6: Grounded RAG answering service.

Strategy
--------
* All LLM and retrieval calls are mocked — no Mistral API key required.
* The FastAPI TestClient tests the full HTTP layer with mocked engine.
* Pure-function tests (build_evidence_block, assess_evidence_sufficiency,
  evaluate_single_case) run without any external dependencies.

Covers:
  * RAGAnswer dataclass — to_dict, fields
  * build_evidence_block — formatting
  * assess_evidence_sufficiency — quality tiers
  * RAGAnswerEngine.answer — happy path with mocked LLM and retriever
  * RAGAnswerEngine.answer — empty question refusal
  * RAGAnswerEngine.answer — retrieval returns no sufficient chunks → refusal
  * RAGAnswerEngine.answer — LLM failure → refusal with error
  * POST /ask-complaints — valid payload → 200
  * POST /ask-complaints — required fields present in response
  * POST /ask-complaints — refused flag wired correctly
  * POST /ask-complaints — invalid payload (short question) → 422
  * POST /ask-complaints — empty question → 422
  * POST /ask-complaints — OpenAPI schema includes /ask-complaints
  * evaluate_single_case — pass / fail logic
  * run_evaluation — produces report JSON with summary
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.rag.answer import (
    PROMPT_VERSION,
    REFUSAL_MESSAGE,
    RAGAnswer,
    RAGAnswerEngine,
    assess_evidence_sufficiency,
    build_evidence_block,
)
from src.rag.rag_eval import evaluate_single_case, run_evaluation
from src.rag.retrieve import RetrievalResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_result(
    chunk_id: str = "abc123",
    text: str = "The bank refused to refund my charge.",
    distance: float = 0.2,
    similarity: float = 0.9,
    complaint_id: str = "42",
    product: str = "Credit card",
    company: str = "ACME Bank",
    date: str = "2023-01-01",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        distance=distance,
        similarity=similarity,
        metadata={
            "complaint_id": complaint_id,
            "product": product,
            "issue": "Billing dispute",
            "company": company,
            "date": date,
            "chunk_index": 0,
        },
    )


def _make_mock_engine(
    answer_text: str = "Based on the evidence, billing disputes are common.",
    refused: bool = False,
    evidence_ids: list[str] | None = None,
) -> MagicMock:
    """Return a MagicMock that behaves like a RAGAnswerEngine."""
    engine = MagicMock()
    rag_answer = RAGAnswer(
        question="test question",
        answer=answer_text if not refused else REFUSAL_MESSAGE,
        refused=refused,
        evidence_ids=evidence_ids or (["abc123", "def456"] if not refused else []),
        evidence_sufficiency="Evidence quality: HIGH (avg_similarity=0.80, max_similarity=0.90, n_chunks=2)",
        prompt_version=PROMPT_VERSION,
        retrieval_count=5,
        token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=250.0,
        model="mistral-small-latest",
    )
    engine.answer.return_value = rag_answer
    return engine


# ---------------------------------------------------------------------------
# RAGAnswer dataclass
# ---------------------------------------------------------------------------

class TestRAGAnswer:
    def test_to_dict_contains_required_keys(self):
        ra = RAGAnswer(
            question="q",
            answer="a",
            refused=False,
            evidence_ids=["x"],
            evidence_sufficiency="HIGH",
            prompt_version="v1.0",
        )
        d = ra.to_dict()
        required = {
            "question", "answer", "refused", "evidence_ids",
            "evidence_sufficiency", "prompt_version",
            "retrieval_count", "token_usage", "latency_ms", "model",
        }
        assert required.issubset(d.keys())

    def test_refused_true_when_no_evidence(self):
        ra = RAGAnswer(question="q", answer=REFUSAL_MESSAGE, refused=True)
        assert ra.refused is True
        assert ra.evidence_ids == []


# ---------------------------------------------------------------------------
# build_evidence_block
# ---------------------------------------------------------------------------

class TestBuildEvidenceBlock:
    def test_non_empty_output(self):
        chunks = [_make_result()]
        block = build_evidence_block(chunks)
        assert len(block) > 0

    def test_contains_complaint_id(self):
        chunks = [_make_result(complaint_id="99")]
        block = build_evidence_block(chunks)
        assert "99" in block

    def test_contains_chunk_text(self):
        chunks = [_make_result(text="bank refused to refund")]
        block = build_evidence_block(chunks)
        assert "bank refused to refund" in block

    def test_multiple_chunks_numbered(self):
        chunks = [_make_result(chunk_id=f"c{i}") for i in range(3)]
        block = build_evidence_block(chunks)
        assert "1." in block
        assert "2." in block
        assert "3." in block

    def test_empty_list_returns_empty_string(self):
        assert build_evidence_block([]) == ""


# ---------------------------------------------------------------------------
# assess_evidence_sufficiency
# ---------------------------------------------------------------------------

class TestAssessEvidenceSufficiency:
    def test_no_chunks_returns_no_evidence(self):
        note = assess_evidence_sufficiency([], threshold=0.25)
        assert "No relevant evidence" in note

    def test_high_similarity_tagged_high(self):
        chunks = [_make_result(similarity=0.85), _make_result(similarity=0.80)]
        note = assess_evidence_sufficiency(chunks, threshold=0.25)
        assert "HIGH" in note

    def test_medium_similarity_tagged_medium(self):
        chunks = [_make_result(similarity=0.5)]
        note = assess_evidence_sufficiency(chunks, threshold=0.25)
        assert "MEDIUM" in note

    def test_low_similarity_tagged_low(self):
        chunks = [_make_result(similarity=0.15)]
        note = assess_evidence_sufficiency(chunks, threshold=0.25)
        assert "LOW" in note

    def test_note_contains_n_chunks(self):
        chunks = [_make_result() for _ in range(3)]
        note = assess_evidence_sufficiency(chunks, threshold=0.25)
        assert "n_chunks=3" in note


# ---------------------------------------------------------------------------
# RAGAnswerEngine (mocked LLM and retriever)
# ---------------------------------------------------------------------------

def _make_engine_with_mock_llm(
    retrieval_results: list[RetrievalResult],
    llm_answer: str = "Answer based on evidence.",
    llm_tokens: dict | None = None,
    llm_raises: Exception | None = None,
) -> RAGAnswerEngine:
    """Build a RAGAnswerEngine with mocked retriever and LLM."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = retrieval_results

    def mock_llm(question, evidence_block, n_chunks, model, api_key, **kwargs):
        if llm_raises:
            raise llm_raises
        return llm_answer, llm_tokens or {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80}

    engine = RAGAnswerEngine(
        retriever=mock_retriever,
        refusal_similarity_threshold=0.25,
        top_k=5,
        llm_caller=mock_llm,
    )
    return engine


class TestRAGAnswerEngine:
    def test_happy_path_returns_answer(self):
        chunks = [_make_result(similarity=0.85)]
        engine = _make_engine_with_mock_llm(chunks)
        result = engine.answer("What billing issues are common?")
        assert isinstance(result, RAGAnswer)
        assert not result.refused
        assert "Answer based on evidence." in result.answer

    def test_happy_path_evidence_ids_present(self):
        chunks = [_make_result(chunk_id="abc", similarity=0.85)]
        engine = _make_engine_with_mock_llm(chunks)
        result = engine.answer("billing issues")
        assert "abc" in result.evidence_ids

    def test_happy_path_prompt_version_set(self):
        chunks = [_make_result(similarity=0.85)]
        engine = _make_engine_with_mock_llm(chunks)
        result = engine.answer("billing issues")
        assert result.prompt_version == PROMPT_VERSION

    def test_happy_path_token_usage_present(self):
        chunks = [_make_result(similarity=0.85)]
        engine = _make_engine_with_mock_llm(chunks)
        result = engine.answer("billing issues")
        assert "total_tokens" in result.token_usage
        assert result.token_usage["total_tokens"] == 80

    def test_empty_question_returns_refusal(self):
        engine = _make_engine_with_mock_llm([])
        result = engine.answer("")
        assert result.refused is True

    def test_whitespace_question_returns_refusal(self):
        engine = _make_engine_with_mock_llm([])
        result = engine.answer("   ")
        assert result.refused is True

    def test_no_sufficient_chunks_returns_refusal(self):
        # All chunks below similarity threshold
        chunks = [_make_result(similarity=0.10)]
        engine = _make_engine_with_mock_llm(chunks)
        result = engine.answer("some question")
        assert result.refused is True
        assert result.evidence_ids == []

    def test_llm_failure_returns_refusal_not_exception(self):
        chunks = [_make_result(similarity=0.85)]
        engine = _make_engine_with_mock_llm(
            chunks, llm_raises=RuntimeError("API timeout")
        )
        result = engine.answer("billing dispute")
        assert result.refused is True
        assert "LLM error" in result.answer

    def test_latency_ms_is_positive(self):
        chunks = [_make_result(similarity=0.85)]
        engine = _make_engine_with_mock_llm(chunks)
        result = engine.answer("billing issues")
        assert result.latency_ms >= 0

    def test_retrieval_count_recorded(self):
        chunks = [_make_result(similarity=0.85)] * 3
        engine = _make_engine_with_mock_llm(chunks)
        result = engine.answer("billing issues")
        assert result.retrieval_count == 3


# ---------------------------------------------------------------------------
# POST /ask-complaints API tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def rag_client():
    """TestClient with mocked RAGAnswerEngine and model bundle."""
    from src.serving.app import app
    from src.serving.model_loader import reset_model_bundle_cache

    reset_model_bundle_cache()

    mock_bundle = MagicMock()
    mock_bundle.model_version = "improved_model"
    mock_bundle.threshold = 0.5

    mock_engine = _make_mock_engine()

    with patch("src.serving.model_loader.get_model_bundle", return_value=mock_bundle), \
         patch("src.serving.routes.get_model_bundle", return_value=mock_bundle), \
         patch("src.serving.routes._get_rag_engine", return_value=mock_engine):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    reset_model_bundle_cache()


@pytest.fixture()
def rag_client_refused():
    """TestClient where the engine returns a refusal."""
    from src.serving.app import app
    from src.serving.model_loader import reset_model_bundle_cache

    reset_model_bundle_cache()
    mock_bundle = MagicMock()
    mock_bundle.model_version = "improved_model"
    mock_bundle.threshold = 0.5
    mock_engine = _make_mock_engine(refused=True)

    with patch("src.serving.model_loader.get_model_bundle", return_value=mock_bundle), \
         patch("src.serving.routes.get_model_bundle", return_value=mock_bundle), \
         patch("src.serving.routes._get_rag_engine", return_value=mock_engine):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    reset_model_bundle_cache()


VALID_RAG_PAYLOAD = {"question": "What are common credit card billing issues?", "top_k": 5}


class TestAskComplaintsEndpoint:
    def test_returns_200(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_answer(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert "answer" in resp.json()
        assert len(resp.json()["answer"]) > 0

    def test_response_has_refused_flag(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert "refused" in resp.json()

    def test_response_has_evidence_ids(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert "evidence_ids" in resp.json()

    def test_response_has_prompt_version(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert "prompt_version" in resp.json()
        assert resp.json()["prompt_version"] == PROMPT_VERSION

    def test_response_has_evidence_sufficiency(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert "evidence_sufficiency" in resp.json()

    def test_response_has_latency_ms(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert resp.json()["latency_ms"] >= 0

    def test_response_has_token_usage(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert "token_usage" in resp.json()

    def test_refused_flag_true_when_engine_refuses(self, rag_client_refused):
        resp = rag_client_refused.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["refused"] is True

    def test_refused_evidence_ids_empty(self, rag_client_refused):
        resp = rag_client_refused.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        assert resp.json()["evidence_ids"] == []

    def test_all_required_keys_present(self, rag_client):
        resp = rag_client.post("/ask-complaints", json=VALID_RAG_PAYLOAD)
        required = {
            "question", "answer", "refused", "evidence_ids",
            "evidence_sufficiency", "prompt_version",
            "retrieval_count", "token_usage", "latency_ms", "model",
        }
        assert required.issubset(resp.json().keys())


class TestAskComplaintsValidation:
    def test_short_question_returns_422(self, rag_client):
        resp = rag_client.post("/ask-complaints", json={"question": "hi"})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self, rag_client):
        resp = rag_client.post("/ask-complaints", json={})
        assert resp.status_code == 422

    def test_top_k_zero_returns_422(self, rag_client):
        resp = rag_client.post("/ask-complaints", json={"question": "valid question here", "top_k": 0})
        assert resp.status_code == 422

    def test_top_k_too_large_returns_422(self, rag_client):
        resp = rag_client.post("/ask-complaints", json={"question": "valid question here", "top_k": 100})
        assert resp.status_code == 422

    def test_product_filter_optional(self, rag_client):
        resp = rag_client.post("/ask-complaints", json={
            "question": "valid question here",
            "product_filter": "Credit card",
        })
        assert resp.status_code == 200

    def test_openapi_includes_ask_complaints(self, rag_client):
        schema = rag_client.get("/openapi.json").json()
        assert "/ask-complaints" in schema["paths"]


# ---------------------------------------------------------------------------
# evaluate_single_case / run_evaluation
# ---------------------------------------------------------------------------

class TestEvaluateSingleCase:
    def test_pass_when_answer_and_evidence_present(self):
        engine = _make_mock_engine(refused=False, evidence_ids=["e1"])
        case = {"id": "t1", "question": "test question?", "expect_refused": False}
        result = evaluate_single_case(case, engine)
        assert result["passed"] is True

    def test_pass_when_refusal_expected_and_received(self):
        engine = _make_mock_engine(refused=True)
        case = {"id": "t2", "question": "nonsense", "expect_refused": True}
        result = evaluate_single_case(case, engine)
        assert result["passed"] is True

    def test_fail_when_refusal_unexpected(self):
        engine = _make_mock_engine(refused=True)
        case = {"id": "t3", "question": "billing issues?", "expect_refused": False}
        result = evaluate_single_case(case, engine)
        assert result["passed"] is False

    def test_result_contains_criteria_dict(self):
        engine = _make_mock_engine(refused=False)
        case = {"id": "t4", "question": "valid question?", "expect_refused": False}
        result = evaluate_single_case(case, engine)
        assert "criteria" in result
        assert "answer_present" in result["criteria"]

    def test_result_contains_evidence_ids(self):
        engine = _make_mock_engine(evidence_ids=["abc"])
        case = {"id": "t5", "question": "valid question?", "expect_refused": False}
        result = evaluate_single_case(case, engine)
        assert result["evidence_ids"] == ["abc"]


class TestRunEvaluation:
    def test_produces_json_report(self, tmp_path):
        engine = _make_mock_engine()
        test_cases = [
            {"id": "tc1", "question": "billing issues?", "expect_refused": False},
            {"id": "tc2", "question": "nonsense", "expect_refused": True},
        ]
        # For tc2: expect refused but engine returns not-refused → will fail
        report = run_evaluation(engine, test_cases=test_cases, report_path=tmp_path / "r.json")
        assert (tmp_path / "r.json").exists()

    def test_report_has_summary(self, tmp_path):
        engine = _make_mock_engine()
        test_cases = [{"id": "tc1", "question": "valid question?", "expect_refused": False}]
        report = run_evaluation(engine, test_cases=test_cases, report_path=tmp_path / "r.json")
        assert "summary" in report
        assert "total_cases" in report["summary"]
        assert "passed" in report["summary"]

    def test_report_has_results_list(self, tmp_path):
        engine = _make_mock_engine()
        test_cases = [{"id": "tc1", "question": "valid question?", "expect_refused": False}]
        report = run_evaluation(engine, test_cases=test_cases, report_path=tmp_path / "r.json")
        assert isinstance(report["results"], list)
        assert len(report["results"]) == 1

    def test_report_total_cases_correct(self, tmp_path):
        engine = _make_mock_engine()
        n = 5
        test_cases = [
            {"id": f"tc{i}", "question": "billing issues?", "expect_refused": False}
            for i in range(n)
        ]
        report = run_evaluation(engine, test_cases=test_cases, report_path=tmp_path / "r.json")
        assert report["summary"]["total_cases"] == n

    def test_json_report_is_valid(self, tmp_path):
        engine = _make_mock_engine()
        test_cases = [{"id": "tc1", "question": "valid question?", "expect_refused": False}]
        run_evaluation(engine, test_cases=test_cases, report_path=tmp_path / "r.json")
        with open(tmp_path / "r.json") as f:
            loaded = json.load(f)
        assert "timestamp" in loaded
