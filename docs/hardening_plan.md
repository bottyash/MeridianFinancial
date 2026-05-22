# Production Hardening Plan — Meridian Financial

## Overview

This document describes the known gaps between the current implementation and a production-ready deployment, and the concrete steps required to close them.

Priority tiers:
- 🔴 **Critical** — must fix before public traffic
- 🟡 **High** — fix before sustained production use
- 🟢 **Medium** — improvements for scale / observability

---

## Security

### 🔴 Add API authentication

**Current state:** All 6 endpoints are unauthenticated.  
**Risk:** Anyone with network access can call `/predict` or `/ask-complaints`.

**Plan:**
```python
# Option A — API key header (fast to implement)
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@router.post("/predict")
async def predict(api_key: str = Security(api_key_header), ...):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403)
    ...
```

```python
# Option B — OAuth2 Bearer token via FastAPI OAuth2PasswordBearer
# Suitable if integrating with existing identity provider
```

**Effort:** 2–4 hours.

---

### 🔴 Secrets rotation

**Current state:** `MISTRAL_API_KEY` is a long-lived static secret set via `.env`.  
**Plan:**
- Integrate with HashiCorp Vault or AWS Secrets Manager
- Rotate key on 30-day schedule
- Use `python-decouple` or `pydantic-settings` with `SecretStr` type to prevent logging

---

### 🟡 Rate limiting

**Current state:** No per-client rate limits.  
**Plan:**
- Add `slowapi` (FastAPI-native rate limiter) for `/ask-complaints` and `/customer-intel`
- Limits: 60 req/min per IP for standard, 300/min for authenticated clients

---

### 🟡 Input sanitization

**Current state:** Question strings are passed directly to the Mistral prompt.  
**Risk:** Prompt injection via crafted question strings.  
**Plan:**
- Strip/escape `<|`, `>>>`, `[INST]` tokens from user input before prompt construction
- Add max question length server-side validation (already at 1000 chars) + profanity/injection filter

---

## Observability

### 🔴 Replace in-process metrics with Prometheus

**Current state:** `/metrics` reads from an in-process Python dict that resets on restart.  
**Plan:**
```python
# requirements.txt:  prometheus-client>=0.20.0
from prometheus_client import Counter, Histogram, make_asgi_app

REQUEST_COUNT = Counter("meridian_requests_total", "Total requests", ["endpoint"])
REQUEST_LATENCY = Histogram("meridian_request_latency_ms", "Latency", ["endpoint"])

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/prometheus-metrics", metrics_app)
```
- Replace `_record_request()` / `_record_prediction()` calls with `Counter.inc()` / `Histogram.observe()`
- Add Grafana dashboard scraping from `http://api:8000/prometheus-metrics`

**Effort:** 4–6 hours.

---

### 🟡 Structured logging with correlation IDs

**Current state:** Logs use `X-Request-ID` header but do not propagate it through the call stack.  
**Plan:**
- Use `structlog` + `contextvars` to propagate `request_id` automatically into every log call
- Output JSON lines for log aggregation in ELK/Datadog

---

### 🟡 Distributed tracing

**Plan:**
- Add `opentelemetry-instrumentation-fastapi` + `opentelemetry-instrumentation-requests`
- Export traces to Jaeger or Datadog APM
- Trace spans for: request intake → ML inference → RAG retrieval → LLM call → response

---

## Reliability

### 🔴 ChromaDB persistence and backup

**Current state:** `chroma_store/` is a local directory; data loss on container replacement.  
**Plan:**
- Mount as a persistent volume in production Docker/K8s
- Daily backup of `chroma_store/` to S3/GCS
- Add index version tag to `HealthResponse.vector_index_version`

---

### 🟡 Model artifact versioning

**Current state:** `improved_model.pkl` is a single file; no rollback path.  
**Plan:**
- Store model artifacts in MLflow model registry with `staging` / `production` stages
- Promotion gate sets `MlflowClient.transition_model_version_stage()`
- `model_loader.py` loads by stage (`production`) not by filename

---

### 🟡 Circuit breaker for Mistral API

**Current state:** Mistral API failures raise an unhandled exception → 500.  
**Plan:**
- Wrap Mistral calls in `circuitbreaker` decorator (or `tenacity` retry + fallback)
- On circuit open: return `refused=True` with `evidence_sufficiency="LLM unavailable"`
- Retry: 2 attempts with 2s backoff before refusing

---

### 🟢 ChromaDB sharding

**Current state:** Single ChromaDB node; linear query time growth.  
**Plan:**
- For > 500k chunks: use ChromaDB distributed mode or migrate to Weaviate cluster
- Shard by complaint product category for O(1) metadata filter performance

---

## Performance

### 🟡 GPU embedding inference

**Current state:** `all-MiniLM-L6-v2` runs on CPU; ~120 ms per query.  
**Plan:**
- Add `CUDA_VISIBLE_DEVICES` env var support
- In Docker Compose: `deploy.resources.reservations.devices: [{driver: nvidia, capabilities: [gpu]}]`
- Expected latency: ~15 ms on A10G

---

### 🟡 Async RAG retrieval

**Current state:** `ComplaintRetriever.query()` is synchronous → blocks the event loop.  
**Plan:**
- Wrap ChromaDB calls in `asyncio.get_event_loop().run_in_executor(None, ...)`
- Or: use `asyncio.to_thread()` (Python 3.9+)

---

### 🟢 Batch embedding for index builds

**Current state:** `build_index.py` encodes chunks one batch at a time (batch=32).  
**Plan:**
- Increase batch size to 256 on GPU
- Parallelize across complaints with `concurrent.futures.ThreadPoolExecutor`

---

## CI/CD

### 🟡 Add load test to CI

**Plan:**
```yaml
- name: Load test
  run: |
    pip install locust
    locust -f tests/locustfile.py --headless -u 20 -r 5 --run-time 60s \
           --host http://localhost:8000
```
- Gate: p99 latency < 2000 ms under 20 concurrent users

---

### 🟡 Semantic versioning and changelogs

**Current state:** Docker images tagged only with git SHA.  
**Plan:**
- Use `semantic-release` to auto-bump version on commit message conventions
- Publish `CHANGELOG.md` on each release
- Tag images with `major.minor.patch` + `latest`

---

## Data

### 🔴 PII scrubbing in production complaint data

**Current state:** Complaint samples are sourced from CFPB public data (PII already redacted by CFPB).  
**Risk:** If expanded to internal complaint data, PII must be redacted.  
**Plan:**
- Add `presidio-analyzer` + `presidio-anonymizer` as a pre-indexing step
- Entities to redact: `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `CREDIT_CARD`

---

### 🟡 Real-time index updates

**Current state:** Index is rebuilt from scratch by running `build_index.py`.  
**Plan:**
- Add a `POST /index-complaint` internal endpoint
- Triggered by a Kafka consumer / webhook on new complaint submission
- Content-addressed IDs ensure idempotent upserts (already implemented)
