# Meridian Financial — Customer Intelligence Platform


[![CI](https://github.com/bottyash/MeridianFinancial/actions/workflows/ci.yml/badge.svg)](https://github.com/bottyash/MeridianFinancial/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade AI platform combining:
- **ML campaign conversion prediction** (XGBoost, 87% ROC-AUC, MLflow-tracked)
- **RAG complaint intelligence** (ChromaDB + `all-MiniLM-L6-v2` + Mistral)
- **Shared FastAPI serving layer** (6 endpoints, Pydantic v2 validation)
- **Evidently drift monitoring** + **GitHub Actions CI/CD**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Meridian Financial Platform                    │
│                                                                   │
│  ┌──────────┐   ┌───────────┐   ┌─────────────┐                 │
│  │ Data     │   │ Training  │   │  RAG Index  │                 │
│  │ Pipeline │──▶│ Pipeline  │   │  Build      │                 │
│  │ (Phase 1)│   │ (Phase 3) │   │  (Phase 5)  │                 │
│  └──────────┘   └─────┬─────┘   └──────┬──────┘                 │
│                        │                │                         │
│                        ▼                ▼                         │
│               ┌────────────────────────────────┐                 │
│               │     FastAPI Serving Layer       │                 │
│               │   GET  /health                  │                 │
│               │   POST /predict                 │                 │
│               │   POST /batch-score             │                 │
│               │   POST /ask-complaints          │                 │
│               │   POST /customer-intel          │                 │
│               │   GET  /metrics                 │                 │
│               └────────────┬───────────────────┘                 │
│                            │                                      │
│              ┌─────────────┼───────────────┐                     │
│              ▼             ▼               ▼                     │
│        ┌──────────┐ ┌──────────┐  ┌──────────────┐             │
│        │ MLflow   │ │ChromaDB  │  │  Evidently   │             │
│        │ Tracking │ │ Vector   │  │  Monitoring  │             │
│        │ Server   │ │  Store   │  │  + Drift     │             │
│        └──────────┘ └──────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

**Repository layout:**
```
MeridianFinancial/
├── src/
│   ├── common/          # Shared config, logger, helpers
│   ├── data_pipeline/   # Ingestion, validation, feature engineering
│   ├── training/        # XGBoost training + MLflow + promotion gate
│   ├── rag/             # ChromaDB indexing + RAG answering engine
│   ├── serving/         # FastAPI app, schemas, routes, model loader
│   └── monitoring/      # Evidently drift + RAG monitor + metrics logger
├── tests/               # 288 pytest tests (all phases)
├── data/
│   ├── raw/             # Source datasets (not committed)
│   └── samples/         # Deterministic 10k-row samples (committed)
├── artifacts/
│   ├── features/        # Preprocessor pkl + feature schema JSON
│   ├── models/          # Trained model pkl
│   └── reports/         # rag_eval.json
├── chroma_store/        # Persisted ChromaDB vector index (41,653 chunks)
├── mlruns/              # MLflow experiment tracking
├── monitoring/          # drift_summary.json + metrics.json + report.html
├── config/              # rag_config.yaml, logging.yaml, prompts.yaml
├── scripts/             # smoke_test.py, gen_rag_eval.py
├── docs/                # Architecture, API, decision log, hardening plan
└── .github/workflows/   # ci.yml (5 jobs) + deploy.yml (4 jobs)
```

---

## Quick Start

> **Live Demo:** Hosted permanently on [HuggingFace Spaces](https://huggingface.co/spaces/bottyash/meridian-dashboard)  
> **API Backend:** [bottyash/meridian-api](https://huggingface.co/spaces/bottyash/meridian-api)  
> **Future Production Deployment:** AWS EC2 (planned — see [docs/deployment.md](docs/deployment.md#future-aws-architecture-planned))

### 1. Clone and create environment

```bash
git clone https://github.com/bottyash/MeridianFinancial.git
cd MeridianFinancial

# Create virtual environment
python -m venv .venv

# Activate
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env — set MISTRAL_API_KEY (required for live RAG)
# All tests mock the API key, so tests run without it
```

### 3. Run data pipeline + feature engineering

```bash
python src/data_pipeline/ingest.py
python src/data_pipeline/features.py
```

### 4. Train the model

```bash
python src/training/train.py
```

### 5. Build the RAG vector index

```bash
python src/rag/build_index.py
# Embeds 10,000 complaints → 41,653 chunks → chroma_store/
```

### 6. Start the API

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Run all tests

```bash
pytest tests/ -v
# Expected: 288 passed
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + readiness probe |
| `POST` | `/predict` | Single-record ML conversion prediction |
| `POST` | `/batch-score` | Batch scoring (up to 500 records) |
| `POST` | `/ask-complaints` | RAG-grounded complaint Q&A |
| `POST` | `/customer-intel` | Combined ML prediction + complaint intelligence |
| `GET` | `/metrics` | Aggregate service metrics |

### curl Examples

**Health check:**
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_version": "improved_model", "vector_index_version": null}
```

**Predict conversion:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42, "duration": 300, "campaign": 1, "previous": 0,
    "pdays": 999, "emp_var_rate": -1.8, "cons_price_idx": 93.075,
    "cons_conf_idx": -47.1, "euribor3m": 1.334, "nr_employed": 5099.1,
    "job": "management", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "thu", "poutcome": "nonexistent"
  }'
```
```json
{"probability": 0.312, "prediction": 0, "threshold": 0.5, "model_version": "improved_model", "latency_ms": 12.3}
```

**Ask complaints:**
```bash
curl -X POST http://localhost:8000/ask-complaints \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the most common credit card billing disputes?", "top_k": 5}'
```
```json
{
  "question": "What are the most common credit card billing disputes?",
  "answer": "Based on the evidence, consumers frequently report...",
  "refused": false,
  "evidence_ids": ["a1b2c3d4", "e5f6g7h8"],
  "evidence_sufficiency": "Evidence quality: HIGH (avg_similarity=0.82, n_chunks=5)",
  "prompt_version": "v1.0",
  "retrieval_count": 5,
  "latency_ms": 820.5
}
```

**Customer intelligence (combined ML + RAG):**
```bash
curl -X POST http://localhost:8000/customer-intel \
  -H "Content-Type: application/json" \
  -d '{
    "customer_profile": {"age": 42, ...},
    "question": "What billing complaints do customers like this report?",
    "product_filter": "Credit card"
  }'
```
```json
{
  "conversion_probability": 0.72,
  "conversion_band": "HIGH",
  "conversion_prediction": 1,
  "complaint_answer": "...",
  "complaint_themes": [{"theme": "Complaint theme 1", "evidence_ids": ["a1b2"]}],
  "cited_complaint_ids": ["a1b2", "c3d4"],
  "total_latency_ms": 1240.1
}
```

**Batch scoring:**
```bash
curl -X POST http://localhost:8000/batch-score \
  -H "Content-Type: application/json" \
  -d '{"records": [{"age": 42, ...}, {"age": 35, ...}]}'
```
```json
{"total": 2, "succeeded": 2, "failed": 0, "latency_ms": 24.1, "results": [...]}
```

**Metrics:**
```bash
curl http://localhost:8000/metrics
```
```json
{
  "uptime_seconds": 3602.1,
  "total_requests": 142,
  "error_count": 1,
  "avg_latency_ms": 48.2,
  "prediction_distribution": {"total_predictions": 98, "positive_rate": 0.21},
  "rag_retrieval_stats": {"total_queries": 44, "refusal_rate": 0.09}
}
```

---

## Docker / Deployment

### Docker Compose (recommended)

```bash
# Build and start all services
docker compose up --build

# Run in background
docker compose up -d

# Follow logs
docker compose logs -f api

# Stop
docker compose down
```

Services started:
- **`meridian_api`** — FastAPI on `localhost:8000`
- **`meridian_mlflow`** — MLflow UI on `localhost:5000`

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | — | Mistral API key (required for live RAG) |
| `APP_PORT` | `8000` | API listen port |
| `APP_WORKERS` | `2` | Uvicorn worker count |
| `APP_ENV` | `production` | Environment tag |
| `LOG_LEVEL` | `info` | Uvicorn log level |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |

---

## Monitoring

### ML Drift Detection

```bash
python src/monitoring/ml_drift.py
# Outputs:
#   monitoring/report.html       (Evidently HTML drift report)
#   monitoring/drift_summary.json (column-level drift stats)
#   monitoring/metrics.json      (event log)
```

Simulates 2σ mean shift + 30% category swap across 2,000 reference rows.  
All 4 monitored columns detected as drifted.

### RAG Monitoring

```bash
python src/monitoring/rag_monitor.py
# Simulates 50 RAG queries
# Outputs: monitoring/metrics.json
#   hit_rate=0.84, refusal_rate=0.16, avg_similarity=0.649
```

**Tracked metrics:**
- Retrieval hit-rate (queries with ≥1 result)
- Refusal rate
- Avg top-k cosine similarity
- Per-query latency
- Token usage (prompt + completion)
- Empty retrieval count

---

## CI/CD

### CI Pipeline (`.github/workflows/ci.yml`)

| Job | Trigger | Description |
|-----|---------|-------------|
| `lint` | push/PR | ruff style check |
| `test` | push/PR | Full pytest suite, JUnit XML artifact |
| `eval-gate` | after test | promotion_gate check (skips if no artifacts) |
| `docker-build` | after test | Buildx image build + GHA layer cache |
| `smoke-test` | after docker | `/health` + `/metrics` inside Compose |


### Deploy Pipeline (`.github/workflows/deploy.yml`)

**Active target: HuggingFace Spaces**

Triggered when CI passes on `main` (or manual dispatch):
1. Validate YAML configs and Dockerfiles
2. Build API + dashboard Docker images (no push — build validation only)
3. Push dashboard source to HF Spaces via `huggingface_hub`
4. Set `API_BASE_URL` as HF Space secret
5. Post-deploy smoke test (local Compose)

**Required GitHub Secrets for HF Spaces deployment:**

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace write token |
| `HF_DASH_SPACE` | Dashboard Space ID (e.g. `bottyash/meridian-dashboard`) |
| `HF_API_SPACE` | API Space ID (e.g. `bottyash/meridian-api`) |
| `MISTRAL_API_KEY` | Mistral API key (for live RAG) |

**Manual deployment:**
```bash
export HF_TOKEN=hf_xxx
bash scripts/deploy_dashboard.sh
bash scripts/deploy_backend.sh
```

---

## Known Limitations

1. **No real-time model retraining** — drift detection is batch-only; retraining requires manual trigger
2. **In-process metrics store** — `/metrics` resets on restart; production needs Prometheus/StatsD
3. **Mistral API dependency** — live `/ask-complaints` requires a valid `MISTRAL_API_KEY`; all tests mock it
4. **ChromaDB single-node** — not distributed; index grows linearly; sharding needed for >1M documents
5. **Complaint-ID theme extraction** — `_extract_themes` uses chunk IDs not actual complaint IDs; a second LLM call would produce richer theme labels
6. **No auth layer** — endpoints are unauthenticated; production needs API-key or OAuth2 middleware
7. **CPU-only embeddings** — `all-MiniLM-L6-v2` runs on CPU; GPU mount in Compose recommended for latency-sensitive workloads

---

## Future Production Architecture (Planned)

> **Status: Not currently active — planned for future production-scale deployment.**

When traffic demands exceed HuggingFace Spaces free-tier capacity:

- **AWS EC2** — dedicated backend runtime (t3.medium/c5.xlarge)
- **Docker Compose / ECS** — multi-worker FastAPI serving
- **Nginx + TLS** — reverse proxy with HTTPS
- **EBS persistent volumes** — ChromaDB + MLflow persistence
- **CloudWatch** — production monitoring
- **Route 53** — custom domain
- **ALB** — application load balancer for horizontal scaling

See `docs/hardening_plan.md` for the full production hardening roadmap.

---

## Reproducibility

All randomness is seeded:
- `COMPLAINT_SAMPLE_SEED=42` in `src/data_pipeline/ingest.py`
- `random_state=42` throughout training and monitoring scripts
- `numpy.random.default_rng(42)` in drift simulation

Re-running any phase script produces byte-identical outputs given the same raw data.

---

## License

MIT — see [LICENSE](LICENSE).
