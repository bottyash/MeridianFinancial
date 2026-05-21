"""
src/rag/retrieve.py
--------------------
Complaint retrieval engine for Meridian Financial RAG pipeline.

Features
--------
* Top-k semantic similarity search over the ChromaDB complaint index
* Configurable refusal threshold — results below minimum similarity are filtered
* Metadata filtering support (product, company, etc.)
* Reusable ``ComplaintRetriever`` class with lazy-loaded embedding model
* Structured ``RetrievalResult`` dataclass for typed outputs

Usage
-----
  from src.rag.retrieve import ComplaintRetriever

  retriever = ComplaintRetriever()
  results = retriever.retrieve("credit card dispute unfair charges", top_k=5)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("meridian.rag.retrieve")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "rag_config.yaml"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieved chunk with similarity score and metadata.

    Attributes
    ----------
    chunk_id:
        Content-addressed ID from build_index.
    text:
        The actual chunk text returned from ChromaDB.
    distance:
        ChromaDB cosine distance (0 = identical, 2 = opposite).
    similarity:
        ``1 - distance / 2`` normalised to [0, 1] for cosine distance.
    metadata:
        Dict of complaint_id, product, issue, company, date, chunk_index.
    """

    chunk_id: str
    text: str
    distance: float
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def complaint_id(self) -> str:
        return self.metadata.get("complaint_id", "")

    @property
    def product(self) -> str:
        return self.metadata.get("product", "")

    @property
    def issue(self) -> str:
        return self.metadata.get("issue", "")

    @property
    def company(self) -> str:
        return self.metadata.get("company", "")

    @property
    def date(self) -> str:
        return self.metadata.get("date", "")


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_config(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class ComplaintRetriever:
    """Semantic retriever over the ChromaDB complaint index.

    The embedding model and Chroma client are lazily initialised on first use
    to keep import time fast.

    Parameters
    ----------
    config_path:
        Path to ``rag_config.yaml``.
    persist_dir:
        Override for ChromaDB persistence directory.
    collection_name:
        Override for collection name.
    model_name:
        Override for embedding model name.
    top_k:
        Default number of results to return.
    refusal_threshold:
        Cosine distance above which results are discarded.
        (ChromaDB cosine distance: 0=identical, 1=orthogonal, 2=opposite)
        With a threshold of 0.75, only chunks with distance < 0.75 are returned.
    """

    def __init__(
        self,
        config_path: Path = _DEFAULT_CONFIG,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        model_name: str | None = None,
        top_k: int | None = None,
        refusal_threshold: float | None = None,
    ) -> None:
        cfg = _load_config(config_path)

        self._persist_dir = persist_dir or str(
            _REPO_ROOT / cfg["chroma"]["persist_directory"]
        )
        self._collection_name = collection_name or cfg["chroma"]["collection_name"]
        self._model_name = model_name or cfg["embedding"]["model_name"]
        self.top_k: int = top_k if top_k is not None else cfg["retrieval"]["top_k"]
        self.refusal_threshold: float = (
            refusal_threshold
            if refusal_threshold is not None
            else cfg["retrieval"]["refusal_threshold"]
        )

        self._model: Optional[SentenceTransformer] = None
        self._collection: Optional[chromadb.Collection] = None

        logger.info(
            "ComplaintRetriever init — collection=%s  top_k=%d  refusal_threshold=%.2f",
            self._collection_name, self.top_k, self.refusal_threshold,
        )

    # ── Lazy properties ────────────────────────────────────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = client.get_collection(self._collection_name)
            logger.info(
                "Opened ChromaDB collection '%s' (%d items)",
                self._collection_name, self._collection.count(),
            )
        return self._collection

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        refusal_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Run a semantic similarity search against the complaint index.

        Parameters
        ----------
        query:
            Natural-language question or search phrase.
        top_k:
            Override the instance ``top_k`` for this call.
        where:
            ChromaDB ``where`` filter dict for metadata filtering.
            Example: ``{"product": "Credit card"}``
        refusal_threshold:
            Override the instance ``refusal_threshold`` for this call.
            Results with ``distance >= threshold`` are removed.

        Returns
        -------
        list[RetrievalResult]
            Top-k chunks ordered by ascending distance (most similar first).
            Empty list if query is empty or all results are below threshold.
        """
        if not query or not query.strip():
            logger.warning("retrieve() called with empty query — returning []")
            return []

        k = top_k if top_k is not None else self.top_k
        threshold = refusal_threshold if refusal_threshold is not None else self.refusal_threshold

        # Embed query
        query_embedding = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
        )[0].tolist()

        # Query ChromaDB
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            raw = self.collection.query(**query_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("ChromaDB query failed: %s", exc)
            return []

        # Unpack results
        results: list[RetrievalResult] = []
        ids_list = raw.get("ids", [[]])[0]
        docs_list = raw.get("documents", [[]])[0]
        dist_list = raw.get("distances", [[]])[0]
        meta_list = raw.get("metadatas", [[]])[0]

        for chunk_id, text, distance, metadata in zip(
            ids_list, docs_list, dist_list, meta_list
        ):
            # Refusal: filter chunks above distance threshold
            if distance >= threshold:
                logger.debug(
                    "Chunk %s refused (distance=%.4f >= threshold=%.2f)",
                    chunk_id, distance, threshold,
                )
                continue

            # ChromaDB cosine distance → similarity: sim = 1 - distance/2
            similarity = max(0.0, 1.0 - distance / 2.0)

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    distance=distance,
                    similarity=similarity,
                    metadata=metadata or {},
                )
            )

        logger.info(
            "retrieve('%s...') — returned %d / %d results (threshold=%.2f)",
            query[:40], len(results), k, threshold,
        )
        return results

    def retrieve_by_complaint_id(self, complaint_id: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve all chunks for a specific complaint by its ID.

        Parameters
        ----------
        complaint_id:
            The complaint identifier to filter on.
        top_k:
            Maximum number of chunks to return.

        Returns
        -------
        list[RetrievalResult]
        """
        # Use a trivially true distance threshold to skip refusal
        return self.retrieve(
            query="complaint narrative",
            top_k=top_k,
            where={"complaint_id": str(complaint_id)},
            refusal_threshold=2.0,  # accept everything
        )

    def collection_count(self) -> int:
        """Return the total number of chunks in the index."""
        return self.collection.count()
