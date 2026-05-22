"""
src/monitoring/metrics_logger.py
----------------------------------
Thread-safe, JSON-persisted metrics logger for the Meridian Financial
monitoring stack.

Provides a ``MetricsLogger`` class that accumulates numerical event records
in-memory and periodically (or on-demand) flushes a structured JSON snapshot
to disk.  The same logger is shared by both ``ml_drift.py`` and
``rag_monitor.py`` so all monitoring output lands in one file.

Output
------
``monitoring/metrics.json``  — flat list of event records, plus a summary
                               section computed at flush time.

Usage
-----
  from src.monitoring.metrics_logger import MetricsLogger

  logger = MetricsLogger()
  logger.log_event("rag_query", {"latency_ms": 240, "refused": False, "n_results": 5})
  logger.flush()   # writes monitoring/metrics.json
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger("meridian.monitoring.metrics_logger")

# ---------------------------------------------------------------------------
# Default output path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH: Path = _REPO_ROOT / "monitoring" / "metrics.json"


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Thread-safe, JSON-persisted event logger.

    Parameters
    ----------
    output_path:
        Path for the flushed JSON metrics file.
    """

    def __init__(self, output_path: Path = DEFAULT_METRICS_PATH) -> None:
        self._output_path = output_path
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    # ── Core API ──────────────────────────────────────────────────────────────

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Append a timestamped event record.

        Parameters
        ----------
        event_type:
            Category label (e.g. ``"rag_query"``, ``"ml_prediction"``,
            ``"drift_detected"``).
        data:
            Arbitrary key-value metrics for this event.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        with self._lock:
            self._events.append(record)

        _log.debug("Logged event '%s': %s", event_type, data)

    def log_ml_prediction(
        self,
        probability: float,
        prediction: int,
        latency_ms: float,
    ) -> None:
        """Shorthand for logging a single ML prediction event."""
        self.log_event("ml_prediction", {
            "probability": probability,
            "prediction": prediction,
            "latency_ms": latency_ms,
        })

    def log_rag_query(
        self,
        query: str,
        n_results: int,
        refused: bool,
        avg_similarity: float,
        latency_ms: float,
        token_usage: dict[str, int],
    ) -> None:
        """Shorthand for logging a single RAG query event."""
        self.log_event("rag_query", {
            "query_preview": query[:60],
            "n_results": n_results,
            "refused": refused,
            "avg_similarity": avg_similarity,
            "latency_ms": latency_ms,
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        })

    # ── Summary helpers ───────────────────────────────────────────────────────

    def _compute_summary(self) -> dict[str, Any]:
        """Compute aggregate summary statistics across all logged events."""
        with self._lock:
            events = list(self._events)

        if not events:
            return {"total_events": 0}

        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in events:
            by_type[e["event_type"]].append(e)

        summary: dict[str, Any] = {
            "total_events": len(events),
            "event_types": {k: len(v) for k, v in by_type.items()},
        }

        # ML prediction summary
        ml_events = by_type.get("ml_prediction", [])
        if ml_events:
            probs = [e["probability"] for e in ml_events]
            preds = [e["prediction"] for e in ml_events]
            latencies = [e["latency_ms"] for e in ml_events]
            summary["ml_prediction_summary"] = {
                "count": len(ml_events),
                "avg_probability": round(sum(probs) / len(probs), 4),
                "positive_rate": round(sum(preds) / len(preds), 4),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
            }

        # RAG query summary
        rag_events = by_type.get("rag_query", [])
        if rag_events:
            refusals = [e for e in rag_events if e.get("refused", False)]
            empty = [e for e in rag_events if e.get("n_results", -1) == 0]
            sims = [e["avg_similarity"] for e in rag_events if e.get("avg_similarity", 0) > 0]
            latencies = [e["latency_ms"] for e in rag_events]
            tokens = [e.get("total_tokens", 0) for e in rag_events]
            summary["rag_query_summary"] = {
                "count": len(rag_events),
                "hit_rate": round(1 - len(refusals) / len(rag_events), 4),
                "refusal_rate": round(len(refusals) / len(rag_events), 4),
                "empty_retrieval_count": len(empty),
                "avg_top_k_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
                "avg_token_usage": round(sum(tokens) / len(tokens), 1),
            }

        return summary

    # ── Persistence ───────────────────────────────────────────────────────────

    def flush(self, output_path: Path | None = None) -> Path:
        """Write all accumulated events and a summary snapshot to JSON.

        Parameters
        ----------
        output_path:
            Override the default output path.

        Returns
        -------
        Path
            The path the file was written to.
        """
        path = output_path or self._output_path
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self._compute_summary(),
            "events": list(self._events),
        }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        _log.info("Metrics flushed to %s  (%d events)", path, len(self._events))
        return path

    def reset(self) -> None:
        """Clear all accumulated events."""
        with self._lock:
            self._events.clear()

    def event_count(self) -> int:
        """Return the number of accumulated events."""
        with self._lock:
            return len(self._events)
