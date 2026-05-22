"""
src/monitoring/rag_monitor.py
------------------------------
RAG pipeline monitoring for Meridian Financial.

Tracks the following per-query statistics:
  * retrieval hit-rate     — fraction of queries with ≥ 1 result
  * refusal rate           — fraction of queries that were refused
  * avg top-k similarity   — mean cosine similarity of returned chunks
  * latency                — per-query wall-clock latency in ms
  * token usage            — prompt + completion + total tokens per query
  * empty retrieval count  — queries returning 0 chunks

This module can run in two modes:
1. **Simulation mode** (``python src/monitoring/rag_monitor.py``) — generates
   synthetic query events to produce realistic monitoring output without
   requiring a live ChromaDB or Mistral API connection.
2. **Library mode** — ``RAGMonitor`` can be integrated into the live serving
   layer by passing real query results from ``ComplaintRetriever``.

Outputs
-------
``monitoring/metrics.json``  — flushed via ``MetricsLogger``

Usage
-----
  python src/monitoring/rag_monitor.py
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so `src.*` imports resolve when this
# script is run directly: python src/monitoring/rag_monitor.py
_REPO_ROOT_STR = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("meridian.monitoring.rag_monitor")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = _REPO_ROOT / "monitoring" / "metrics.json"


# ---------------------------------------------------------------------------
# RAGQueryEvent dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class RAGQueryEvent:
    """Captures all observable attributes of one RAG retrieval+answer call.

    Attributes
    ----------
    query:
        User question text.
    n_results:
        Number of chunks returned by retrieval (before refusal filter).
    refused:
        True if the engine refused to answer.
    similarities:
        Cosine similarity scores for all returned chunks.
    latency_ms:
        End-to-end query latency in milliseconds.
    token_usage:
        Dict with prompt_tokens, completion_tokens, total_tokens.
    """
    query: str
    n_results: int
    refused: bool
    similarities: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)

    @property
    def avg_similarity(self) -> float:
        return sum(self.similarities) / len(self.similarities) if self.similarities else 0.0

    @property
    def empty_retrieval(self) -> bool:
        return self.n_results == 0


# ---------------------------------------------------------------------------
# RAGMonitor
# ---------------------------------------------------------------------------

class RAGMonitor:
    """Collects, aggregates, and logs RAG query events.

    Parameters
    ----------
    metrics_logger:
        A ``MetricsLogger`` instance. If ``None``, a default logger is created.
    """

    def __init__(self, metrics_logger=None) -> None:
        if metrics_logger is None:
            from src.monitoring.metrics_logger import MetricsLogger
            self._ml = MetricsLogger()
        else:
            self._ml = metrics_logger

        self._events: list[RAGQueryEvent] = []
        logger.info("RAGMonitor initialised")

    def record(self, event: RAGQueryEvent) -> None:
        """Record a RAG query event and log it to the MetricsLogger.

        Parameters
        ----------
        event:
            A completed :class:`RAGQueryEvent`.
        """
        self._events.append(event)
        self._ml.log_rag_query(
            query=event.query,
            n_results=event.n_results,
            refused=event.refused,
            avg_similarity=event.avg_similarity,
            latency_ms=event.latency_ms,
            token_usage=event.token_usage,
        )

    def summarise(self) -> dict[str, Any]:
        """Compute and return aggregate statistics across all recorded events.

        Returns
        -------
        dict
            Hit-rate, refusal-rate, avg similarity, latency, token usage,
            empty retrieval count.
        """
        events = self._events
        n = len(events)

        if n == 0:
            return {"total_queries": 0}

        refused_count = sum(1 for e in events if e.refused)
        empty_count = sum(1 for e in events if e.empty_retrieval)
        hit_events = [e for e in events if not e.refused]
        sims = [e.avg_similarity for e in events if e.similarities]
        latencies = [e.latency_ms for e in events]
        tokens = [e.token_usage.get("total_tokens", 0) for e in events]

        summary = {
            "total_queries": n,
            "hit_rate": round((n - refused_count) / n, 4),
            "refusal_rate": round(refused_count / n, 4),
            "refused_count": refused_count,
            "empty_retrieval_count": empty_count,
            "avg_top_k_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0,
            "avg_latency_ms": round(sum(latencies) / n, 3) if latencies else 0.0,
            "avg_token_usage": round(sum(tokens) / n, 1) if tokens else 0.0,
            "total_tokens_used": sum(tokens),
        }

        logger.info(
            "RAG summary — total=%d  hit_rate=%.2f  refusal_rate=%.2f  "
            "avg_sim=%.3f  avg_latency=%.1fms",
            n, summary["hit_rate"], summary["refusal_rate"],
            summary["avg_top_k_similarity"], summary["avg_latency_ms"],
        )
        return summary

    def flush(self) -> Path:
        """Flush accumulated events and return the metrics file path."""
        return self._ml.flush()


# ---------------------------------------------------------------------------
# Synthetic query simulation
# ---------------------------------------------------------------------------

_SAMPLE_QUERIES = [
    "What are common credit card billing disputes?",
    "How do banks handle mortgage payment issues?",
    "Are there complaints about debt collection harassment?",
    "What problems occur with student loan servicers?",
    "How do consumers report identity theft incidents?",
    "What issues arise with checking account overdrafts?",
    "Are there complaints about credit score reporting errors?",
    "What are auto loan payment complaints?",
    "How do consumers resolve insurance claim disputes?",
    "What payday loan complaints exist?",
    "xyzzy nonsense query that will not match anything",  # refusal case
    "qwerty irrelevant completely random topic",           # refusal case
]


def simulate_rag_queries(
    monitor: RAGMonitor,
    n_queries: int = 50,
    refusal_rate: float = 0.15,
    random_state: int = 42,
) -> None:
    """Generate synthetic RAG query events and record them to *monitor*.

    Parameters
    ----------
    monitor:
        :class:`RAGMonitor` to record events into.
    n_queries:
        Total number of synthetic queries to generate.
    refusal_rate:
        Fraction of queries that will be simulated as refused.
    random_state:
        Seed for reproducibility.
    """
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    logger.info("Simulating %d RAG queries (refusal_rate=%.0f%%)", n_queries, refusal_rate * 100)

    for i in range(n_queries):
        query = rng.choice(_SAMPLE_QUERIES)
        refused = rng.random() < refusal_rate

        if refused:
            event = RAGQueryEvent(
                query=query,
                n_results=rng.randint(0, 2),
                refused=True,
                similarities=[float(np_rng.uniform(0.05, 0.20))],
                latency_ms=float(np_rng.uniform(30, 80)),
                token_usage={},
            )
        else:
            n_results = rng.randint(3, 5)
            similarities = [float(np_rng.uniform(0.55, 0.95)) for _ in range(n_results)]
            event = RAGQueryEvent(
                query=query,
                n_results=n_results,
                refused=False,
                similarities=similarities,
                latency_ms=float(np_rng.uniform(150, 600)),
                token_usage={
                    "prompt_tokens": rng.randint(200, 500),
                    "completion_tokens": rng.randint(40, 120),
                    "total_tokens": rng.randint(250, 620),
                },
            )

        monitor.record(event)

    logger.info("Simulation complete — %d events recorded", n_queries)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    n_queries: int = 50,
    metrics_path=DEFAULT_METRICS_PATH,
) -> dict:
    """Run the RAG monitoring simulation and flush metrics.

    Parameters
    ----------
    n_queries:
        Number of synthetic queries to simulate.
    metrics_path:
        Output path for the metrics JSON file.

    Returns
    -------
    dict
        Aggregate summary statistics.
    """
    from src.monitoring.metrics_logger import MetricsLogger

    ml = MetricsLogger(output_path=metrics_path)
    monitor = RAGMonitor(metrics_logger=ml)

    simulate_rag_queries(monitor, n_queries=n_queries)

    summary = monitor.summarise()
    monitor.flush()

    return summary


if __name__ == "__main__":
    summary = run()
    print("\nRAG monitoring simulation complete:")
    for key, val in summary.items():
        print(f"  {key:35s} = {val}")
    print(f"\n  Metrics written => monitoring/metrics.json")
