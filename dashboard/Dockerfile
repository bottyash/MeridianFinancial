# ─── Meridian Financial Dashboard — Dockerfile ───────────────────────────────
# Lightweight Streamlit dashboard container.
#
# Compatible with:
#   - local docker-compose
#   - HuggingFace Spaces (Streamlit SDK)
#   - AWS ECS / AppRunner
#
# Build:   docker build -t meridian-dashboard:latest -f dashboard/Dockerfile .
# Run:     docker run -e API_BASE_URL=http://your-api:8000 -p 8501:8501 meridian-dashboard:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Meridian Financial Dashboard"
LABEL org.opencontainers.image.description="Streamlit operations dashboard"
LABEL org.opencontainers.image.source="https://github.com/bottyash/MeridianFinancial"

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────────
COPY dashboard/requirements.txt ./dashboard/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r dashboard/requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY dashboard/ ./dashboard/

# ── Streamlit configuration ───────────────────────────────────────────────────
RUN mkdir -p /app/.streamlit
RUN cat > /app/.streamlit/config.toml << 'EOF'
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
primaryColor = "#6366f1"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#f1f5f9"
font = "sans serif"

[browser]
gatherUsageStats = false
EOF

# ── Environment variables ─────────────────────────────────────────────────────
ENV API_BASE_URL=http://localhost:8000 \
    ENVIRONMENT=production \
    REQUEST_TIMEOUT=30 \
    DASHBOARD_TITLE="Meridian Financial Intelligence" \
    PAGE_ICON=🏦 \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd --no-create-home --shell /bin/false streamlit_user \
    && chown -R streamlit_user:streamlit_user /app
USER streamlit_user

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

# ── Startup ───────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
