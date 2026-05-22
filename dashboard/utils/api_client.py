"""
dashboard/utils/api_client.py
------------------------------
Reusable, deployment-aware HTTP client for the Meridian Financial FastAPI backend.

All dashboard pages import from here — no page should make raw `requests` calls.
Handles:
  * timeouts
  * structured error returns (never raises — returns typed error dicts)
  * connection failures (backend offline)
  * JSON parsing errors
  * request logging
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from dashboard.utils.config import CONFIG

logger = logging.getLogger("meridian.dashboard.api_client")


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

# All API methods return a dict. On error, the dict contains:
#   {"_error": True, "_message": str, "_status_code": int | None}
# Callers check `response.get("_error")` to detect failures.

ApiResponse = dict[str, Any]

_ERROR_OFFLINE: ApiResponse = {
    "_error": True,
    "_message": "Backend is offline or unreachable. Check API_BASE_URL and network connectivity.",
    "_status_code": None,
}


def _error(message: str, status_code: int | None = None) -> ApiResponse:
    return {"_error": True, "_message": message, "_status_code": status_code}


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, **kwargs) -> ApiResponse:
    t0 = time.perf_counter()
    try:
        resp = requests.get(url, timeout=CONFIG.request_timeout, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("GET %s → %d  (%.1f ms)", url, resp.status_code, latency_ms)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning("GET %s — connection refused", url)
        return _ERROR_OFFLINE
    except requests.exceptions.Timeout:
        return _error(f"Request timed out after {CONFIG.request_timeout}s")
    except requests.exceptions.HTTPError as exc:
        return _error(str(exc), status_code=exc.response.status_code if exc.response else None)
    except Exception as exc:  # noqa: BLE001
        return _error(f"Unexpected error: {exc}")


def _post(url: str, payload: dict, **kwargs) -> ApiResponse:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=CONFIG.request_timeout,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("POST %s → %d  (%.1f ms)", url, resp.status_code, latency_ms)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning("POST %s — connection refused", url)
        return _ERROR_OFFLINE
    except requests.exceptions.Timeout:
        return _error(f"Request timed out after {CONFIG.request_timeout}s")
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return _error(detail, status_code=exc.response.status_code if exc.response else None)
    except Exception as exc:  # noqa: BLE001
        return _error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Public API methods
# ---------------------------------------------------------------------------

def health_check() -> ApiResponse:
    """GET /health — liveness + readiness probe."""
    return _get(CONFIG.api_health_url)


def predict(profile: dict) -> ApiResponse:
    """POST /predict — single-record ML conversion prediction."""
    return _post(CONFIG.api_predict_url, profile)


def batch_score(records: list[dict]) -> ApiResponse:
    """POST /batch-score — batch ML conversion scoring."""
    return _post(CONFIG.api_batch_score_url, {"records": records})


def ask_complaints(
    question: str,
    top_k: int = 5,
    product_filter: str | None = None,
) -> ApiResponse:
    """POST /ask-complaints — RAG-grounded complaint Q&A."""
    payload: dict[str, Any] = {"question": question, "top_k": top_k}
    if product_filter:
        payload["product_filter"] = product_filter
    return _post(CONFIG.api_ask_complaints_url, payload)


def customer_intel(
    customer_profile: dict,
    question: str | None = None,
    product_filter: str | None = None,
    issue_filter: str | None = None,
    top_k: int = 5,
) -> ApiResponse:
    """POST /customer-intel — combined ML + RAG intelligence."""
    payload: dict[str, Any] = {
        "customer_profile": customer_profile,
        "top_k": top_k,
    }
    if question:
        payload["question"] = question
    if product_filter:
        payload["product_filter"] = product_filter
    if issue_filter:
        payload["issue_filter"] = issue_filter
    return _post(CONFIG.api_customer_intel_url, payload)


def get_metrics() -> ApiResponse:
    """GET /metrics — aggregate service metrics."""
    return _get(CONFIG.api_metrics_url)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def is_backend_online() -> bool:
    """Return True if the backend /health endpoint responds with status=ok."""
    resp = health_check()
    return not resp.get("_error") and resp.get("status") == "ok"
