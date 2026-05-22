"""
src/serving/app.py
-------------------
FastAPI application factory for the Meridian Financial Customer Intelligence Platform.

Wires together:
  * application metadata (title, version, OpenAPI docs)
  * startup event — pre-warms the model bundle cache
  * middleware — request timing logged as a structured JSON header
  * route registration — ML prediction routes (phase 4)
  * global exception handler — consistent error envelope

Usage
-----
  # Development
  uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

  # Production
  uvicorn src.serving.app:app --workers 2 --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.serving.model_loader import get_model_bundle, reset_model_bundle_cache
from src.serving.routes import router
from src.serving.schemas import ErrorResponse

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("meridian.app")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm the model bundle on startup so the first request is fast."""
    logger.info("Starting Meridian Financial API — pre-warming model bundle ...")
    try:
        bundle = get_model_bundle()
        logger.info(
            "Model bundle ready — version=%s  threshold=%.2f",
            bundle.model_version, bundle.threshold,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "STARTUP WARNING: model bundle failed to load (%s). "
            "Predictions will fail until artifacts are available.",
            exc,
        )
    yield
    # Shutdown: nothing to clean up for this phase
    logger.info("Meridian Financial API shutting down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance.

    Returns
    -------
    FastAPI
    """
    app = FastAPI(
        title="Meridian Financial — Customer Intelligence API",
        description=(
            "Production ML + RAG serving layer for campaign conversion prediction "
            "and complaint intelligence."
        ),
        version="0.4.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS middleware ───────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # tighten in production via settings
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def log_request_timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1_000
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.3f}"
        logger.info(
            "%s %s — status=%d  latency_ms=%.2f",
            request.method, request.url.path,
            response.status_code, latency_ms,
        )
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_server_error",
                detail=str(exc),
                status_code=500,
            ).model_dump(),
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router)

    return app


# ---------------------------------------------------------------------------
# Module-level singleton (used by uvicorn)
# ---------------------------------------------------------------------------
app: FastAPI = create_app()
