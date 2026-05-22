# Architecture — Meridian Financial Customer Intelligence Platform

## System Overview

The Meridian Financial platform is a modular, production-minded AI system that merges two ML subsystems — campaign conversion prediction and RAG-based complaint intelligence — into a unified FastAPI serving layer.

```
                         ┌──────────────────────────────────────────────────────┐
                         │                  Client / Upstream                    │
                         └────────────────────────┬─────────────────────────────┘
                                                  │ HTTP/JSON
                         ┌────────────────────────▼─────────────────────────────┐
                         │             FastAPI Serving Layer                     │
                         │                                                       │
                         │  GET  /health          POST /predict                  │
                         │  POST /batch-score     POST /ask-complaints           │
                         │  POST /customer-intel  GET  /metrics                  │
                         │                                                       │
                         │  Pydantic v2 validation │ Structured logging          │
                         │  Per-request latency    │ In-process metrics          │
                         └──────────┬──────────────────────────┬────────────────┘
                                    │                          │
               ┌────────────────────▼────┐        ┌───────────▼──────────────────┐
               │    ML Prediction Path    │        │       RAG Answering Path      │
               │                         │        │                               │
               │  ModelBundle (lazy)      │        │  RAGAnswerEngine (lazy)      │
               │  ├─ ColumnTransformer   │        │  ├─ ComplaintRetriever       │
               │  ├─ XGBoostClassifier   │        │  │   ├─ SentenceTransformer  │
               │  └─ threshold (0.5)     │        │  │   └─ ChromaDB collection  │
               │                         │        │  └─ Mistral chat completion  │
               └────────────┬────────────┘        └───────────┬──────────────────┘
                            │                                  │
               ┌────────────▼────────────┐        ┌───────────▼──────────────────┐
               │   artifacts/models/      │        │   chroma_store/              │
               │   improved_model.pkl     │        │   (41,653 embedded chunks)   │
               └─────────────────────────┘        └──────────────────────────────┘
```

---

## Component Breakdown

### 1. Data Pipeline (`src/data_pipeline/`)

| Module | Responsibility |
|--------|---------------|
| `ingest.py` | Load raw CSVs → validate schema → produce deterministic 10k-row complaint sample (seed=42) + UCI bank sample |
| `features.py` | Fit `ColumnTransformer` on training split → save preprocessor + feature schema → write Parquet |
| `validate.py` | Great Expectations-style column/type/range checks; raises `DataValidationError` on failure |

**Design choices:**
- Sample reproducibility via `random_state=42` — identical outputs on re-run
- Schema stored as JSON so serving layer can load it without importing training code
- Parquet preferred over CSV for type fidelity

---

### 2. Training Pipeline (`src/training/`)

| Module | Responsibility |
|--------|---------------|
| `train.py` | Grid-search XGBoost, log all runs to MLflow, call promotion gate |
| `promotion_gate.py` | Block promotion if PR-AUC improvement < 3 pp OR F1 degradation > 2 pp |

**Model selection:**  
XGBoost with `scale_pos_weight` tuning for class imbalance (~88.7% negative). Final model: `improved_model` tag in MLflow.

**Promotion gate logic:**
```
promote = (pr_auc_delta >= 0.03) AND (f1_delta >= -0.02)
```

---

### 3. RAG Index (`src/rag/`)

| Module | Responsibility |
|--------|---------------|
| `build_index.py` | Clean → chunk (512 tokens, 64-token overlap) → SHA-256 content ID → ChromaDB upsert |
| `retrieve.py` | `ComplaintRetriever` — lazy embedding model load, cosine similarity search, threshold filter (0.25) |
| `answer.py` | `RAGAnswerEngine` — retrieval → evidence sufficiency assessment → grounded Mistral prompt → `RAGAnswer` |
| `rag_eval.py` | 12-case evaluation harness → `artifacts/reports/rag_eval.json` |

**Chunk sizing rationale:**  
512-token chunks balance context length (enough for Mistral to reason from) vs. retrieval precision.  
64-token overlap preserves cross-chunk continuity for multi-sentence complaints.

**Refusal threshold:**  
Similarity < 0.25 → refuse; the engine returns `refused=True` with `REFUSAL_MESSAGE` rather than hallucinating.

---

### 4. Serving Layer (`src/serving/`)

| Module | Responsibility |
|--------|---------------|
| `app.py` | FastAPI app factory, lifespan, CORS, middleware |
| `routes.py` | Route handlers — metrics counters, lazy singletons, per-row error isolation |
| `schemas.py` | Pydantic v2 request/response models (16 schemas) |
| `model_loader.py` | Thread-safe `ModelBundle` singleton via `get_model_bundle()` |

**Lazy loading:**  
Both `_get_model_bundle()` and `_get_rag_engine()` initialize on first request (not at import time), enabling fast startup and test isolation via `patch()`.

**In-process metrics:**  
`_metrics` dict accumulates request counts, latency sum, prediction distribution, and RAG stats — reset on restart. Designed for swap-in with Prometheus in production.

---

### 5. Monitoring (`src/monitoring/`)

| Module | Responsibility |
|--------|---------------|
| `metrics_logger.py` | Thread-safe event accumulator → JSON flush |
| `ml_drift.py` | Evidently `DataDriftPreset` on reference vs. synthetic drifted data → HTML + JSON |
| `rag_monitor.py` | `RAGMonitor` + `RAGQueryEvent` dataclass → hit-rate, refusal rate, similarity, latency, token stats |

**Drift simulation:**  
2σ mean shift on numeric columns + 30% random category swap — guarantees all columns are flagged as drifted, demonstrating the full Evidently pipeline.

---

## Data Flow

```
Raw CSV
  │
  ▼ ingest.py (validate → sample → save)
data/samples/complaints_sample.csv
  │
  ├──▶ features.py → artifacts/features/preprocessor.pkl
  │                             feature_schema.json
  │                             bank_features.parquet
  │
  ├──▶ train.py → artifacts/models/improved_model.pkl
  │               mlruns/ (experiment tracking)
  │
  └──▶ build_index.py → chroma_store/ (ChromaDB)
                          41,653 chunks
                          all-MiniLM-L6-v2 embeddings
```

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML model | XGBoost | Strong tabular performance, interpretable feature importance, fast inference |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, CPU-friendly, proven semantic retrieval quality |
| Vector store | ChromaDB | Zero-config local deployment, supports metadata filtering for product/issue filters |
| LLM | Mistral small | Cost-efficient, JSON-mode compatible, 32k context for multi-chunk evidence |
| Serving | FastAPI | Async, OpenAPI auto-gen, Pydantic v2 native |
| Experiment tracking | MLflow | Open-source, artifact logging, promotion gate integration |
| Monitoring | Evidently | HTML reports, column-level drift stats, no external infrastructure |
