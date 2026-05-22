# API Reference — Meridian Financial

Base URL (local): `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`  
OpenAPI schema: `http://localhost:8000/openapi.json`

---

## `GET /health`

Liveness and readiness probe.

**Response 200:**
```json
{
  "status": "ok",
  "model_version": "improved_model",
  "vector_index_version": null
}
```

**curl:**
```bash
curl http://localhost:8000/health
```

---

## `POST /predict`

Single-record ML campaign conversion prediction.

**Request body (20 required fields):**
```json
{
  "age": 42,
  "duration": 300,
  "campaign": 1,
  "previous": 0,
  "pdays": 999,
  "emp_var_rate": -1.8,
  "cons_price_idx": 93.075,
  "cons_conf_idx": -47.1,
  "euribor3m": 1.334,
  "nr_employed": 5099.1,
  "job": "management",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "thu",
  "poutcome": "nonexistent"
}
```

**Response 200:**
```json
{
  "probability": 0.312456,
  "prediction": 0,
  "threshold": 0.5,
  "model_version": "improved_model",
  "latency_ms": 12.3
}
```

**Error responses:**
- `422` — validation error (field missing or out of range)
- `503` — model artifact not loaded

---

## `POST /batch-score`

Score up to 500 records in a single request. Per-row errors are isolated.

**Request body:**
```json
{
  "records": [
    { ...profile_1... },
    { ...profile_2... }
  ]
}
```

**Response 200:**
```json
{
  "total": 2,
  "succeeded": 2,
  "failed": 0,
  "model_version": "improved_model",
  "latency_ms": 24.1,
  "results": [
    {
      "row_index": 0,
      "probability": 0.312456,
      "prediction": 0,
      "threshold": 0.5,
      "error": null
    },
    {
      "row_index": 1,
      "probability": 0.718200,
      "prediction": 1,
      "threshold": 0.5,
      "error": null
    }
  ]
}
```

**Limits:** `records` must have 1–500 items.

---

## `POST /ask-complaints`

Answer questions about consumer complaints using RAG (ChromaDB retrieval + Mistral).

**Request body:**
```json
{
  "question": "What are the most common credit card billing disputes?",
  "top_k": 5,
  "product_filter": "Credit card"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | 3–1000 chars |
| `top_k` | int | No (default 5) | 1–20 evidence chunks |
| `product_filter` | string | No | Filter retrieval by product category |

**Response 200 (answered):**
```json
{
  "question": "What are the most common credit card billing disputes?",
  "answer": "Based on the evidence, consumers frequently report...",
  "refused": false,
  "evidence_ids": ["a1b2c3d4", "e5f6g7h8", "i9j0k1l2"],
  "evidence_sufficiency": "Evidence quality: HIGH (avg_similarity=0.82, max_similarity=0.91, n_chunks=5)",
  "prompt_version": "v1.0",
  "retrieval_count": 5,
  "token_usage": {
    "prompt_tokens": 312,
    "completion_tokens": 89,
    "total_tokens": 401
  },
  "latency_ms": 820.5,
  "model": "mistral-small-latest"
}
```

**Response 200 (refused):**
```json
{
  "question": "xyzzy nonsense query",
  "answer": "I don't have sufficient evidence in the complaint database to answer this question accurately.",
  "refused": true,
  "evidence_ids": [],
  "evidence_sufficiency": "No relevant evidence found.",
  "prompt_version": "v1.0",
  "retrieval_count": 2,
  "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "latency_ms": 35.2,
  "model": "mistral-small-latest"
}
```

**Note:** Refusals return **200 OK** with `refused: true`, not a 4xx error.

---

## `POST /customer-intel`

Combined ML conversion prediction + complaint intelligence for a customer.

**Request body:**
```json
{
  "customer_profile": { ...20 feature fields... },
  "question": "What billing complaints do customers like this report?",
  "product_filter": "Credit card",
  "issue_filter": null,
  "date_filter": null,
  "top_k": 5
}
```

**Response 200:**
```json
{
  "conversion_probability": 0.72,
  "conversion_prediction": 1,
  "conversion_band": "HIGH",
  "model_version": "improved_model",
  "complaint_question": "What billing complaints do customers like this report?",
  "complaint_answer": "Based on the evidence...",
  "complaint_refused": false,
  "complaint_themes": [
    {"theme": "Complaint theme 1", "evidence_ids": ["a1b2", "c3d4"]},
    {"theme": "Complaint theme 2", "evidence_ids": ["e5f6"]}
  ],
  "cited_complaint_ids": ["a1b2", "c3d4", "e5f6"],
  "evidence_sufficiency": "Evidence quality: HIGH ...",
  "ml_latency_ms": 12.3,
  "rag_latency_ms": 820.5,
  "total_latency_ms": 834.1
}
```

**Conversion bands:**

| Band | Probability |
|------|-------------|
| `HIGH` | ≥ 0.70 |
| `MEDIUM` | 0.40 – 0.69 |
| `LOW` | < 0.40 |

---

## `GET /metrics`

Aggregate service metrics since last startup.

**Response 200:**
```json
{
  "uptime_seconds": 3602.1,
  "total_requests": 142,
  "error_count": 1,
  "error_rate": 0.007042,
  "avg_latency_ms": 48.2,
  "prediction_distribution": {
    "total_predictions": 98,
    "positive_predictions": 21,
    "negative_predictions": 77,
    "positive_rate": 0.214286
  },
  "rag_retrieval_stats": {
    "total_queries": 44,
    "refused_queries": 4,
    "refusal_rate": 0.090909,
    "avg_evidence_ids_per_query": 3.8
  }
}
```

**Note:** Metrics reset on process restart. Not suitable as a durable metrics store — swap with Prometheus for production.
