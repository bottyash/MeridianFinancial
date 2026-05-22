# Deployment Guide — Meridian Financial

## Prerequisites

- Docker 24+ and Docker Compose v2
- Python 3.11 (for local development)
- `MISTRAL_API_KEY` from [console.mistral.ai](https://console.mistral.ai)
- 4 GB RAM, 10 GB disk (for model artifacts + ChromaDB + MLflow)

---

## Option A — Docker Compose (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/bottyash/MeridianFinancial.git
cd MeridianFinancial
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env:
#   MISTRAL_API_KEY=your-key-here
#   APP_PORT=8000       (optional, default 8000)
#   APP_WORKERS=2       (optional, default 2)
#   LOG_LEVEL=info      (optional)
```

### 3. Prepare artifacts (first run only)

```bash
# Create a local venv for data prep
python -m venv .venv && .venv/Scripts/activate   # Windows
# source .venv/bin/activate                      # Linux/macOS

pip install -r requirements.txt

# Run data pipeline
python src/data_pipeline/ingest.py
python src/data_pipeline/features.py

# Train model
python src/training/train.py

# Build RAG vector index (~5–10 minutes)
python src/rag/build_index.py
```

### 4. Start services

```bash
docker compose up --build
```

Services started:
| Service | URL | Description |
|---------|-----|-------------|
| FastAPI API | `http://localhost:8000` | ML + RAG serving |
| MLflow UI | `http://localhost:5000` | Experiment tracking |
| OpenAPI docs | `http://localhost:8000/docs` | Interactive API documentation |

### 5. Verify deployment

```bash
# Health check
curl http://localhost:8000/health

# Run smoke tests
python scripts/smoke_test.py --base-url http://localhost:8000
```

Expected output:
```
Smoke tests against http://localhost:8000
============================================================
  [PASS] GET /health  —  status == ok
  [PASS] GET /metrics  —  keys present: [...]
  [PASS] GET /openapi.json  —  all 6 endpoints registered
============================================================
Result: 3 passed, 0 failed
```

### 6. Stop services

```bash
docker compose down

# Remove volumes (clears ChromaDB + MLflow data):
docker compose down -v
```

---

## Option B — Local Development (without Docker)

```bash
# Activate venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Start API
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Option C — GitHub Actions CI/CD

CI runs automatically on push to `main` / `develop`:
1. Lint → Test → Eval gate → Docker build → Smoke test
2. On CI pass: Deploy workflow triggers automatically
3. Deploy workflow pushes image to GHCR and runs post-deploy smoke tests

Artifacts uploaded per run:
- JUnit XML test report
- Container logs
- Deployment summary JSON

---

## Environment Variable Reference

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `MISTRAL_API_KEY` | — | Yes (live RAG) | Mistral API key |
| `MISTRAL_MODEL` | `mistral-small-latest` | No | Mistral model name |
| `APP_PORT` | `8000` | No | API listen port |
| `APP_WORKERS` | `2` | No | Uvicorn workers |
| `APP_ENV` | `production` | No | Environment tag |
| `LOG_LEVEL` | `info` | No | Uvicorn log level |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | No | MLflow server URL |
| `COMPLAINT_SAMPLE_SEED` | `42` | No | Sampling seed for reproducibility |

---

## Volume Mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./data/samples` | `/app/data/samples` | Read-only sample CSVs |
| `./artifacts` | `/app/artifacts` | Model + preprocessor pkl files |
| `./chroma_store` | `/app/chroma_store` | ChromaDB vector index |
| `./monitoring` | `/app/monitoring` | Monitoring output (HTML, JSON) |
| `./mlruns` | `/app/mlruns` | MLflow experiment data |

---

## Troubleshooting

**API not healthy after `docker compose up`:**
```bash
docker compose logs api
# Common cause: model artifacts not present
# Fix: run src/training/train.py locally before building the image
```

**ChromaDB collection not found:**
```bash
# Run the RAG index build outside the container (host venv):
python src/rag/build_index.py
# The chroma_store/ directory will be mounted into the container
```

**MISTRAL_API_KEY not set:**
```bash
# All tests mock the API — tests will pass without a real key
# Only live /ask-complaints and /customer-intel calls require it
```
