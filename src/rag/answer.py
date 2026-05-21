"""
src/rag/answer.py
------------------
Grounded RAG answering engine for Meridian Financial.

Architecture
------------
1. Retrieve top-k complaint chunks from ChromaDB (phase 5)
2. Detect refusal conditions (empty retrieval / low similarity)
3. Build a versioned, evidence-grounded prompt
4. Call the Mistral API with structured context
5. Return a typed ``RAGAnswer`` dataclass including:
   * answer text
   * evidence chunk IDs
   * evidence sufficiency note
   * prompt version
   * token usage
   * latency

Refusal rules
-------------
* No chunks retrieved → refuse
* All retrieved chunks have similarity < refusal_similarity_threshold → refuse
* Refusal message is returned (not an exception) so the API can respond 200
  with a clear "insufficient evidence" explanation.

Usage
-----
  from src.rag.answer import RAGAnswerEngine

  engine = RAGAnswerEngine()
  result = engine.answer("What are the most common credit card complaints?")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.rag.retrieve import ComplaintRetriever, RetrievalResult

logger = logging.getLogger("meridian.rag.answer")

# ---------------------------------------------------------------------------
# Prompt versioning
# ---------------------------------------------------------------------------

PROMPT_VERSION: str = "v1.0"

SYSTEM_PROMPT: str = """You are a financial consumer complaint analyst for Meridian Financial.
Your role is to answer questions about consumer complaints using ONLY the evidence provided.

Rules:
- Base your answer exclusively on the provided complaint excerpts.
- If the evidence is insufficient, say so clearly.
- Always cite the complaint IDs you used.
- Be concise, factual, and professional.
- Do not fabricate information not present in the evidence."""

ANSWER_PROMPT_TEMPLATE: str = """Based on the following consumer complaint excerpts, please answer the question.

Question: {question}

Evidence ({n_chunks} complaint excerpts):
{evidence_block}

Instructions:
- Answer based only on the evidence above.
- Cite the complaint IDs you relied on.
- If the evidence does not sufficiently answer the question, state that clearly.

Answer:"""

REFUSAL_MESSAGE: str = (
    "I cannot provide a reliable answer to this question. "
    "The retrieved complaint evidence does not contain sufficient relevant information "
    "to address your query with confidence. Please try rephrasing your question or "
    "providing more context."
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGAnswer:
    """Structured output from the RAG answering engine.

    Attributes
    ----------
    question:
        Original user question.
    answer:
        LLM-generated answer (or refusal message).
    refused:
        True when the engine refused to answer due to insufficient evidence.
    evidence_ids:
        List of chunk IDs used as context (empty when refused).
    evidence_sufficiency:
        Human-readable note about evidence quality.
    prompt_version:
        Version tag of the prompt template used.
    retrieval_count:
        Number of chunks retrieved (before refusal filtering).
    token_usage:
        Dict with ``prompt_tokens``, ``completion_tokens``, ``total_tokens``
        (empty when refused or when using a mock LLM).
    latency_ms:
        Total end-to-end latency in milliseconds.
    model:
        Mistral model identifier used.
    """

    question: str
    answer: str
    refused: bool
    evidence_ids: list[str] = field(default_factory=list)
    evidence_sufficiency: str = ""
    prompt_version: str = PROMPT_VERSION
    retrieval_count: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "evidence_ids": self.evidence_ids,
            "evidence_sufficiency": self.evidence_sufficiency,
            "prompt_version": self.prompt_version,
            "retrieval_count": self.retrieval_count,
            "token_usage": self.token_usage,
            "latency_ms": self.latency_ms,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def build_evidence_block(chunks: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a numbered evidence block for the prompt.

    Parameters
    ----------
    chunks:
        Retrieved complaint chunks.

    Returns
    -------
    str
        Formatted evidence string.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = (
            f"[Complaint ID: {chunk.complaint_id} | "
            f"Product: {chunk.product} | "
            f"Company: {chunk.company} | "
            f"Date: {chunk.date}]"
        )
        lines.append(f"{i}. {meta}\n   {chunk.text}")
    return "\n\n".join(lines)


def assess_evidence_sufficiency(chunks: list[RetrievalResult], threshold: float) -> str:
    """Return a human-readable evidence sufficiency note.

    Parameters
    ----------
    chunks:
        Retrieved chunks (already filtered by refusal threshold).
    threshold:
        Minimum similarity score used for filtering.

    Returns
    -------
    str
    """
    if not chunks:
        return "No relevant evidence found."
    avg_sim = sum(c.similarity for c in chunks) / len(chunks)
    max_sim = max(c.similarity for c in chunks)
    if avg_sim >= 0.7:
        quality = "HIGH"
    elif avg_sim >= 0.4:
        quality = "MEDIUM"
    else:
        quality = "LOW"
    return (
        f"Evidence quality: {quality} "
        f"(avg_similarity={avg_sim:.3f}, max_similarity={max_sim:.3f}, "
        f"n_chunks={len(chunks)})"
    )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _call_mistral(
    question: str,
    evidence_block: str,
    n_chunks: int,
    model: str,
    api_key: str,
    temperature: float = 0.1,
) -> tuple[str, dict[str, int]]:
    """Call the Mistral chat API and return (answer_text, token_usage).

    Parameters
    ----------
    question, evidence_block, n_chunks:
        Inputs for the prompt template.
    model:
        Mistral model identifier.
    api_key:
        Mistral API key.
    temperature:
        LLM temperature (low for factual RAG).

    Returns
    -------
    tuple[str, dict[str, int]]
        ``(answer_text, {"prompt_tokens": …, "completion_tokens": …, "total_tokens": …})``
    """
    from mistralai import Mistral

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        question=question,
        n_chunks=n_chunks,
        evidence_block=evidence_block,
    )

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=512,
    )

    answer_text: str = response.choices[0].message.content or ""
    usage = response.usage
    token_usage = {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
    return answer_text.strip(), token_usage


# ---------------------------------------------------------------------------
# RAG Answer Engine
# ---------------------------------------------------------------------------

class RAGAnswerEngine:
    """Orchestrates retrieval + grounded answer generation.

    Parameters
    ----------
    retriever:
        Pre-configured :class:`ComplaintRetriever` instance.
        If ``None``, a default retriever is built from ``rag_config.yaml``.
    mistral_model:
        Mistral model identifier (default from settings or env).
    refusal_similarity_threshold:
        Chunks below this similarity are excluded; refuse if none remain.
    top_k:
        Number of chunks to retrieve per query.
    llm_caller:
        Callable for LLM calls — injected for testing (defaults to
        :func:`_call_mistral`).
    """

    def __init__(
        self,
        retriever: Optional[ComplaintRetriever] = None,
        mistral_model: Optional[str] = None,
        refusal_similarity_threshold: float = 0.25,
        top_k: int = 5,
        llm_caller=None,
    ) -> None:
        self.retriever = retriever or ComplaintRetriever(top_k=top_k)
        self.top_k = top_k
        self.refusal_similarity_threshold = refusal_similarity_threshold

        # Model and API key
        try:
            from src.common.config import settings
            self.mistral_model = mistral_model or settings.mistral_model
            self._api_key = settings.mistral_api_key.get_secret_value()
        except Exception:
            self.mistral_model = mistral_model or os.getenv(
                "MISTRAL_MODEL", "mistral-small-latest"
            )
            self._api_key = os.getenv("MISTRAL_API_KEY", "")

        self._llm_caller = llm_caller or _call_mistral

        logger.info(
            "RAGAnswerEngine ready — model=%s  top_k=%d  refusal_threshold=%.2f",
            self.mistral_model, self.top_k, self.refusal_similarity_threshold,
        )

    def answer(
        self,
        question: str,
        where: dict[str, Any] | None = None,
    ) -> RAGAnswer:
        """Answer *question* using retrieved complaint evidence.

        Parameters
        ----------
        question:
            Natural-language user question.
        where:
            Optional ChromaDB metadata filter (e.g. ``{"product": "Mortgage"}``).

        Returns
        -------
        RAGAnswer
        """
        t0 = time.perf_counter()

        if not question or not question.strip():
            return RAGAnswer(
                question=question,
                answer="Please provide a non-empty question.",
                refused=True,
                evidence_sufficiency="No question provided.",
                latency_ms=0.0,
                model=self.mistral_model,
            )

        # 1. Retrieve chunks
        chunks = self.retriever.retrieve(
            question,
            top_k=self.top_k,
            where=where,
        )
        retrieval_count = len(chunks)

        # 2. Filter by similarity threshold
        sufficient_chunks = [
            c for c in chunks if c.similarity >= self.refusal_similarity_threshold
        ]

        logger.info(
            "answer('%s...') — retrieved=%d  sufficient=%d",
            question[:40], retrieval_count, len(sufficient_chunks),
        )

        # 3. Refusal check
        if not sufficient_chunks:
            latency_ms = (time.perf_counter() - t0) * 1_000
            logger.warning(
                "REFUSAL: insufficient evidence for question '%s...' "
                "(retrieved=%d, sufficient=0)",
                question[:40], retrieval_count,
            )
            return RAGAnswer(
                question=question,
                answer=REFUSAL_MESSAGE,
                refused=True,
                evidence_ids=[],
                evidence_sufficiency="No relevant evidence above similarity threshold.",
                prompt_version=PROMPT_VERSION,
                retrieval_count=retrieval_count,
                token_usage={},
                latency_ms=round(latency_ms, 3),
                model=self.mistral_model,
            )

        # 4. Build evidence block
        evidence_block = build_evidence_block(sufficient_chunks)
        evidence_ids = [c.chunk_id for c in sufficient_chunks]
        sufficiency_note = assess_evidence_sufficiency(
            sufficient_chunks, self.refusal_similarity_threshold
        )

        # 5. Call LLM
        try:
            answer_text, token_usage = self._llm_caller(
                question=question,
                evidence_block=evidence_block,
                n_chunks=len(sufficient_chunks),
                model=self.mistral_model,
                api_key=self._api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed: %s", exc)
            latency_ms = (time.perf_counter() - t0) * 1_000
            return RAGAnswer(
                question=question,
                answer=f"LLM error: {exc}",
                refused=True,
                evidence_ids=evidence_ids,
                evidence_sufficiency=sufficiency_note,
                prompt_version=PROMPT_VERSION,
                retrieval_count=retrieval_count,
                token_usage={},
                latency_ms=round(latency_ms, 3),
                model=self.mistral_model,
            )

        latency_ms = (time.perf_counter() - t0) * 1_000

        logger.info(
            "answer DONE — latency_ms=%.1f  tokens=%s  evidence_ids=%d",
            latency_ms,
            token_usage.get("total_tokens", "N/A"),
            len(evidence_ids),
        )

        return RAGAnswer(
            question=question,
            answer=answer_text,
            refused=False,
            evidence_ids=evidence_ids,
            evidence_sufficiency=sufficiency_note,
            prompt_version=PROMPT_VERSION,
            retrieval_count=retrieval_count,
            token_usage=token_usage,
            latency_ms=round(latency_ms, 3),
            model=self.mistral_model,
        )
