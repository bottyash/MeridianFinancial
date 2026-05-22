# Deployment Guide — Meridian Financial

## Active Deployment: HuggingFace Spaces

The current production deployment uses **HuggingFace Spaces** for both the API backend and the Streamlit dashboard. This provides:

- ✅ Free-tier, permanent public hosting
- ✅ No infrastructure management
- ✅ Publicly accessible demo URLs
- ✅ CPU-only compatible (no GPU required)
- ✅ Lightweight startup

> **Future Deployment (Planned):** AWS EC2 for production-scale backend infrastructure. See [Future AWS Architecture](#future-aws-architecture-planned) below.

---

## Active Deployment Architecture

```
GitHub Repo
    │
    ▼
GitHub Actions (CI)
    │  Validation + Tests + Promotion Gate
    ▼
Build Docker Images
    │
    ▼
Deploy to HuggingFace Spaces
    ├── bottyash/meridian-api        → FastAPI backend
    └── bottyash/meridian-dashboard  → Streamlit dashboard
    │
    ▼
Public Demo URLs
    ├── https://huggingface.co/spaces/bottyash/meridian-api
    └── https://huggingface.co/spaces/bottyash/meridian-dashboard
```

---

## Prerequisites

- Docker 24+ and Docker Compose v2 (for local development)
- Python 3.11
- `MISTRAL_API_KEY` from [console.mistral.ai](https://console.mistral.ai)
- `HF_TOKEN` from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (for deployment)

---

## Option A — Local Development (Recommended for development)

### 1. Clone and set up environment

```bash
git clone https://github.com/bottyash/MeridianFinancial.git
cd MeridianFinancial

python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env:
#   MISTRAL_API_KEY=your-key-here
#   API_BASE_URL=http://localhost:8000
```

### 3. Prepare artifacts (first run only)

```bash
python src/data_pipeline/ingest.py
python src/data_pipeline/features.py
python src/training/train.py
python src/rag/build_index.py        # ~5–10 minutes
```

### 4. Start services

```bash
# Terminal 1 — FastAPI backend
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Streamlit dashboard
streamlit run dashboard/app.py
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

---

## Option B — Docker Compose (Local full-stack)

```bash
cp .env.example .env    # set MISTRAL_API_KEY

docker compose up --build
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| MLflow | http://localhost:5000 |

---

## Option C — HuggingFace Spaces (Active Production Deployment)

### Automated deployment (GitHub Actions)

Every push to `main` that passes CI automatically triggers the deploy workflow:

1. Validates configs and Dockerfiles
2. Builds Docker images with GHA layer cache
3. Pushes dashboard source to HF Spaces via `huggingface_hub`
4. Runs post-deploy smoke test

**Required GitHub Secrets:**

| Secret | Value | Required |
|--------|-------|----------|
| `HF_TOKEN` | HuggingFace write token | Yes |
| `HF_DASH_SPACE` | `username/meridian-dashboard` | Yes |
| `HF_API_SPACE` | `username/meridian-api` | Yes |
| `MISTRAL_API_KEY` | Mistral API key | For live RAG |

### Manual deployment

```bash
# Deploy backend
export HF_TOKEN=hf_your_token_here
export HF_API_SPACE=bottyash/meridian-api
bash scripts/deploy_backend.sh

# Deploy dashboard
export HF_TOKEN=hf_your_token_here
export HF_DASH_SPACE=bottyash/meridian-dashboard
export API_BASE_URL=https://bottyash-meridian-api.hf.space
bash scripts/deploy_dashboard.sh
```

### HuggingFace Spaces configuration

**Dashboard Space (`app.py` entry point):**
- SDK: `streamlit`
- Set `API_BASE_URL` as a Space secret pointing to your API Space

**API Space (`Dockerfile` runtime):**
- SDK: `docker`
- Set `MISTRAL_API_KEY` as a Space secret

---

## Environment Variable Reference

### Active (used in production)

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | — | Mistral API key (live RAG) |
| `API_BASE_URL` | `http://localhost:8000` | FastAPI backend URL |
| `ENVIRONMENT` | `development` | Runtime environment tag |
| `HF_TOKEN` | — | HuggingFace deploy token |
| `APP_PORT` | `8000` | API listen port |
| `APP_WORKERS` | `2` | Uvicorn worker count |
| `LOG_LEVEL` | `info` | Uvicorn log level |

---

## HuggingFace Spaces Limitations (Free Tier)

- **Sleep after inactivity** — Space sleeps after ~48h without requests; wakes on next visit
- **No persistent storage** — ChromaDB index must be bundled or loaded from HF Hub on startup
- **CPU only** — No GPU acceleration; embedding inference runs on CPU (~120 ms/query)
- **2 vCPU / 16 GB RAM** — Sufficient for demo workloads; not suitable for concurrent production traffic
- **Cold start latency** — First request after sleep may take 30–60s to load models

**Workarounds:**
- Pre-embed all chunks and commit `chroma_store/` to the Space repo (eliminates cold-start indexing)
- Use HF Hub to store model artifacts; load on startup

---

## Startup Sequence

```
Container starts
    │
    ▼
uvicorn src.serving.app:app
    │
    ▼
FastAPI lifespan → lazy-load ModelBundle + RAGAnswerEngine
    │  (first /predict call triggers ModelBundle load)
    │  (first /ask-complaints call triggers RAGAnswerEngine + ChromaDB load)
    ▼
Ready to serve
```

---

## Smoke Test

```bash
# Against any running instance
python scripts/smoke_test.py --base-url http://localhost:8000

# Expected:
#   [PASS] GET /health
#   [PASS] GET /metrics
#   [PASS] GET /openapi.json (all 6 endpoints)
```

---

## Future AWS Architecture (Planned)

> **Status: Not currently active. Planned for future production-scale deployment.**

When traffic demands exceed HuggingFace Spaces free-tier capacity, the planned production architecture is:

```
AWS EC2 (t3.medium or c5.xlarge)
    ├── Docker Compose or ECS
    ├── FastAPI backend (multi-worker)
    ├── Nginx reverse proxy + TLS
    ├── Persistent EBS for ChromaDB
    └── CloudWatch monitoring

AWS Application Load Balancer
    └── → EC2 target group

Route 53
    └── api.meridianfinancial.example.com
```

**Future deployment script (not yet active):**
```bash
# Planned: scripts/deploy_aws.sh
# ssh ubuntu@EC2_HOST "cd MeridianFinancial && docker compose up -d"
```

**Future secrets (not currently used):**
- `DEPLOY_HOST` — EC2 public IP
- `DEPLOY_USER` — SSH user
- `DEPLOY_KEY` — SSH private key
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — AWS credentials

See `docs/hardening_plan.md` for full AWS production hardening roadmap.
