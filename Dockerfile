# syntax=docker/dockerfile:1
# ─── Meridian Financial — Production Dockerfile ───────────────────────────────
# Multi-stage build:
#   builder  — installs Python deps into an isolated virtual environment
#   runtime  — lean final image (venv + source only, no build tools)
#
# Build:   docker build -t meridian-financial:latest .
# Run:     docker run --env-file .env -p 8000:8000 meridian-financial:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed only at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Create an isolated venv inside the image and install all deps
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Meridian Financial API"
LABEL org.opencontainers.image.description="Customer Intelligence Platform — FastAPI serving layer"
LABEL org.opencontainers.image.source="https://github.com/bottyash/MeridianFinancial"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv

# Add venv to PATH and configure Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Number of uvicorn workers (override via env)
    APP_WORKERS=2 \
    APP_PORT=8000 \
    APP_ENV=production \
    # Logging
    LOG_LEVEL=info

# Copy source code and configuration
COPY src/ ./src/
COPY config/ ./config/

# Copy smoke-test script (used by CI and healthcheck)
COPY scripts/smoke_test.py ./scripts/smoke_test.py

# Create runtime directories (data / artifact volumes mounted externally)
RUN mkdir -p \
        data/samples \
        artifacts/features \
        artifacts/models \
        artifacts/reports \
        chroma_store \
        mlruns \
        monitoring \
        reports

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false meridian \
    && chown -R meridian:meridian /app
USER meridian

EXPOSE 8000

# Healthcheck — poll /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:${APP_PORT:-8000}/health')" \
    || exit 1

# Parameterised CMD — override workers/port via environment variables
CMD uvicorn src.serving.app:app \
    --host 0.0.0.0 \
    --port ${APP_PORT:-8000} \
    --workers ${APP_WORKERS:-2} \
    --log-level ${LOG_LEVEL:-info}
