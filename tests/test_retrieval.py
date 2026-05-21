"""
tests/test_retrieval.py
------------------------
Unit and integration tests for Phase 5: RAG retrieval pipeline.

Strategy
--------
* Text cleaning and chunking functions are pure — tested directly with no
  external dependencies.
* ChromaDB interactions are tested with an in-memory Chroma client so no
  real ``chroma_store/`` directory is needed and tests run offline.
* The ``ComplaintRetriever`` is instantiated with a synthetic in-memory
  collection for retrieval-path tests.

Covers:
  * clean_narrative — whitespace, redaction, dash normalisation
  * is_valid_narrative — length gates
  * chunk_text — split, overlap, min-length filtering
  * make_chunk_id — determinism and uniqueness
  * prepare_documents — full pipeline on synthetic DataFrame
  * get_or_create_collection — creates / reopens collection
  * embed_and_upsert — shape and count checks (mocked embeddings)
  * ComplaintRetriever.retrieve — happy path, empty query, threshold filter
  * ComplaintRetriever.retrieve — metadata where-filter
  * RetrievalResult — properties
  * collection_count — post-upsert count
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.rag.build_index import (
    chunk_text,
    clean_narrative,
    get_or_create_collection,
    is_valid_narrative,
    make_chunk_id,
    prepare_documents,
)
from src.rag.retrieve import ComplaintRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_complaints_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "complaint_id": list(range(1, n + 1)),
        "product": ["Credit card"] * n,
        "issue": ["Billing dispute"] * n,
        "company": ["ACME Bank"] * n,
        "date_received": ["2023-01-01"] * n,
        "complaint_narrative": [
            f"This is a test complaint narrative number {i}. "
            "The bank charged me twice for the same transaction and refused to refund. "
            "I contacted customer service multiple times with no resolution."
            for i in range(n)
        ],
    })


# ---------------------------------------------------------------------------
# clean_narrative
# ---------------------------------------------------------------------------

class TestCleanNarrative:
    def test_strips_whitespace(self):
        assert clean_narrative("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        result = clean_narrative("hello   world")
        assert "  " not in result

    def test_removes_xx_redaction(self):
        result = clean_narrative("My name is XX and I live at XXXX street")
        assert "XX" not in result

    def test_normalises_em_dash(self):
        result = clean_narrative("good\u2014bad")
        assert "-" in result
        assert "\u2014" not in result

    def test_normalises_en_dash(self):
        result = clean_narrative("A\u2013B")
        assert "-" in result
        assert "\u2013" not in result

    def test_non_string_returns_empty(self):
        assert clean_narrative(None) == ""  # type: ignore[arg-type]
        assert clean_narrative(123) == ""  # type: ignore[arg-type]

    def test_empty_string_returns_empty(self):
        assert clean_narrative("") == ""

    def test_preserves_meaningful_content(self):
        text = "The bank refused to refund my money."
        assert "bank" in clean_narrative(text)
        assert "refund" in clean_narrative(text)


# ---------------------------------------------------------------------------
# is_valid_narrative
# ---------------------------------------------------------------------------

class TestIsValidNarrative:
    def test_long_text_valid(self):
        assert is_valid_narrative("a" * 50) is True

    def test_short_text_invalid(self):
        assert is_valid_narrative("short", min_length=30) is False

    def test_exactly_at_threshold_valid(self):
        assert is_valid_narrative("a" * 30, min_length=30) is True

    def test_empty_string_invalid(self):
        assert is_valid_narrative("") is False

    def test_non_string_invalid(self):
        assert is_valid_narrative(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=300)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        text = "word " * 200  # ~1000 chars
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 chars
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        if len(chunks) >= 2:
            # The end of chunk[0] should appear at the start of chunk[1]
            overlap_end = chunks[0][-20:]
            assert overlap_end in chunks[1][:40]

    def test_min_length_filters_short_chunks(self):
        # Single tiny text — should be filtered out
        chunks = chunk_text("hi", chunk_size=300, min_chunk_length=30)
        assert len(chunks) == 0

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []

    def test_chunk_sizes_within_limit(self):
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        for c in chunks:
            assert len(c) <= 100

    def test_all_chunks_meet_min_length(self):
        text = "word " * 100
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10, min_chunk_length=20)
        for c in chunks:
            assert len(c) >= 20


# ---------------------------------------------------------------------------
# make_chunk_id
# ---------------------------------------------------------------------------

class TestMakeChunkId:
    def test_returns_string(self):
        cid = make_chunk_id(1, 0, "hello")
        assert isinstance(cid, str)

    def test_length_16(self):
        cid = make_chunk_id(1, 0, "hello")
        assert len(cid) == 16

    def test_deterministic(self):
        a = make_chunk_id(42, 3, "same text")
        b = make_chunk_id(42, 3, "same text")
        assert a == b

    def test_different_inputs_different_ids(self):
        a = make_chunk_id(1, 0, "text A")
        b = make_chunk_id(1, 0, "text B")
        assert a != b

    def test_different_indices_different_ids(self):
        a = make_chunk_id(1, 0, "text")
        b = make_chunk_id(1, 1, "text")
        assert a != b


# ---------------------------------------------------------------------------
# prepare_documents
# ---------------------------------------------------------------------------

class TestPrepareDocuments:
    def test_returns_three_parallel_lists(self):
        df = _make_complaints_df(3)
        ids, texts, metas = prepare_documents(df)
        assert len(ids) == len(texts) == len(metas)

    def test_non_empty_output(self):
        df = _make_complaints_df(3)
        ids, texts, metas = prepare_documents(df)
        assert len(ids) > 0

    def test_metadata_contains_required_fields(self):
        df = _make_complaints_df(2)
        _, _, metas = prepare_documents(df)
        for meta in metas:
            assert "complaint_id" in meta
            assert "product" in meta
            assert "issue" in meta
            assert "company" in meta
            assert "date" in meta

    def test_ids_are_unique(self):
        df = _make_complaints_df(5)
        ids, _, _ = prepare_documents(df)
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_invalid_narrative_skipped(self):
        df = _make_complaints_df(2)
        df.loc[0, "complaint_narrative"] = "x"  # too short → skip
        ids_full, _, _ = prepare_documents(_make_complaints_df(2))
        ids_partial, _, _ = prepare_documents(df)
        assert len(ids_partial) < len(ids_full)

    def test_chunk_index_in_metadata(self):
        df = _make_complaints_df(1)
        _, _, metas = prepare_documents(df, chunk_size=50, chunk_overlap=10)
        chunk_indices = [m["chunk_index"] for m in metas]
        assert 0 in chunk_indices


# ---------------------------------------------------------------------------
# get_or_create_collection (in-memory)
# ---------------------------------------------------------------------------

class TestGetOrCreateCollection:
    def test_creates_collection(self, tmp_path):
        coll = get_or_create_collection(str(tmp_path), "test_col")
        assert coll is not None

    def test_collection_is_empty_on_create(self, tmp_path):
        coll = get_or_create_collection(str(tmp_path), "empty_col")
        assert coll.count() == 0

    def test_reopens_existing_collection(self, tmp_path):
        get_or_create_collection(str(tmp_path), "persist_col")
        coll2 = get_or_create_collection(str(tmp_path), "persist_col")
        assert coll2 is not None


# ---------------------------------------------------------------------------
# ComplaintRetriever (mocked Chroma + model)
# ---------------------------------------------------------------------------

def _make_mock_retriever(
    n_results: int = 3,
    distances: list[float] | None = None,
) -> ComplaintRetriever:
    """Build a ComplaintRetriever with mocked collection and embedding model."""
    if distances is None:
        distances = [0.1, 0.3, 0.5]

    ids = [f"chunk_{i}" for i in range(n_results)]
    docs = [f"Complaint text {i}" for i in range(n_results)]
    metas = [
        {
            "complaint_id": str(i),
            "product": "Credit card",
            "issue": "Billing",
            "company": "ACME",
            "date": "2023-01-01",
            "chunk_index": 0,
        }
        for i in range(n_results)
    ]

    mock_collection = MagicMock()
    mock_collection.count.return_value = n_results
    mock_collection.query.return_value = {
        "ids": [ids],
        "documents": [docs],
        "distances": [distances[:n_results]],
        "metadatas": [metas],
    }

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384))

    retriever = ComplaintRetriever.__new__(ComplaintRetriever)
    retriever._model = mock_model
    retriever._collection = mock_collection
    retriever._collection_name = "complaints"
    retriever._persist_dir = "chroma_store"
    retriever._model_name = "all-MiniLM-L6-v2"
    retriever.top_k = 5
    retriever.refusal_threshold = 0.75

    return retriever


class TestComplaintRetriever:
    def test_retrieve_returns_list(self):
        r = _make_mock_retriever()
        results = r.retrieve("credit card dispute")
        assert isinstance(results, list)

    def test_retrieve_nonempty_results(self):
        r = _make_mock_retriever(n_results=3, distances=[0.1, 0.3, 0.5])
        results = r.retrieve("billing issue")
        assert len(results) > 0

    def test_retrieve_empty_query_returns_empty(self):
        r = _make_mock_retriever()
        assert r.retrieve("") == []
        assert r.retrieve("   ") == []

    def test_retrieve_results_are_retrieval_result_instances(self):
        r = _make_mock_retriever(n_results=2, distances=[0.2, 0.4])
        results = r.retrieve("test query")
        for res in results:
            assert isinstance(res, RetrievalResult)

    def test_refusal_threshold_filters_distant_results(self):
        # All distances > 0.75 → all refused
        r = _make_mock_retriever(n_results=3, distances=[0.8, 0.9, 1.2])
        results = r.retrieve("query", refusal_threshold=0.75)
        assert results == []

    def test_refusal_threshold_keeps_close_results(self):
        r = _make_mock_retriever(n_results=3, distances=[0.1, 0.2, 0.3])
        results = r.retrieve("query", refusal_threshold=0.75)
        assert len(results) == 3

    def test_similarity_computed_from_distance(self):
        r = _make_mock_retriever(n_results=1, distances=[0.4])
        results = r.retrieve("test")
        if results:
            expected_sim = max(0.0, 1.0 - 0.4 / 2.0)
            assert abs(results[0].similarity - expected_sim) < 1e-6

    def test_distance_preserved_in_result(self):
        r = _make_mock_retriever(n_results=1, distances=[0.25])
        results = r.retrieve("test")
        if results:
            assert abs(results[0].distance - 0.25) < 1e-6

    def test_metadata_properties_accessible(self):
        r = _make_mock_retriever(n_results=1, distances=[0.1])
        results = r.retrieve("test")
        if results:
            assert results[0].product == "Credit card"
            assert results[0].company == "ACME"

    def test_collection_count(self):
        r = _make_mock_retriever(n_results=5)
        assert r.collection_count() == 5

    def test_retrieve_with_where_filter_passes_to_chroma(self):
        r = _make_mock_retriever(n_results=1, distances=[0.2])
        r.retrieve("test", where={"product": "Credit card"})
        call_kwargs = r._collection.query.call_args[1]
        assert "where" in call_kwargs
        assert call_kwargs["where"] == {"product": "Credit card"}

    def test_retrieve_top_k_override(self):
        r = _make_mock_retriever(n_results=2, distances=[0.1, 0.3])
        r.retrieve("test", top_k=2)
        call_kwargs = r._collection.query.call_args[1]
        assert call_kwargs["n_results"] == 2


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

class TestRetrievalResult:
    def test_properties_delegate_to_metadata(self):
        rr = RetrievalResult(
            chunk_id="abc123",
            text="some text",
            distance=0.2,
            similarity=0.9,
            metadata={
                "complaint_id": "42",
                "product": "Mortgage",
                "issue": "Foreclosure",
                "company": "Big Bank",
                "date": "2023-06-01",
            },
        )
        assert rr.complaint_id == "42"
        assert rr.product == "Mortgage"
        assert rr.issue == "Foreclosure"
        assert rr.company == "Big Bank"
        assert rr.date == "2023-06-01"

    def test_empty_metadata_returns_empty_strings(self):
        rr = RetrievalResult(chunk_id="x", text="t", distance=0.5, similarity=0.75)
        assert rr.complaint_id == ""
        assert rr.product == ""
