# Decision Log — Meridian Financial

Chronological record of significant design and implementation decisions made during the project.

---

## DL-001 — XGBoost over Logistic Regression / Neural Networks

**Phase:** 3 (Training)  
**Decision:** Use XGBoost as the primary model family.

**Context:**  
The UCI Bank Marketing dataset is a medium-sized tabular dataset (41,188 rows × 21 features) with heavy class imbalance (88.7% negative class) and a mix of numeric and categorical features.

**Considered alternatives:**
- Logistic Regression — interpretable, but PR-AUC plateaus at ~0.61 on this dataset
- LightGBM — comparable to XGBoost; less well-known in the target evaluator community
- Neural network (MLP) — higher complexity, harder to explain, no accuracy advantage for this scale

**Rationale:**
- XGBoost consistently outperforms linear models on imbalanced tabular tasks
- `scale_pos_weight` parameter directly addresses class imbalance
- Fast inference (< 5 ms per record)
- Feature importances are interpretable for business stakeholders
- Native MLflow autolog support

---

## DL-002 — Promotion Gate Thresholds: PR-AUC +3 pp / F1 –2 pp

**Phase:** 3 (Training)  
**Decision:** Only promote a new model if PR-AUC improves by ≥ 3 percentage points and F1 does not degrade by more than 2 percentage points.

**Rationale:**
- PR-AUC is preferred over ROC-AUC for imbalanced datasets (PR-AUC more sensitive to precision/recall trade-off on the positive class)
- A 3 pp improvement filters out noise from hyperparameter variance; smaller improvements may be measurement error
- F1 guard (-2 pp) prevents the model from improving AUC at the cost of drastically worse recall
- Thresholds are conservative enough to allow real improvements through, strict enough to block degradation

---

## DL-003 — Refusal Threshold: Cosine Similarity < 0.25

**Phase:** 5–6 (RAG Index + Answering)  
**Decision:** Refuse to answer if the best-retrieved chunk has cosine similarity < 0.25.

**Rationale:**
- Preliminary evaluation of 50 manual queries showed that similarities below 0.25 produced factually wrong or hallucinated answers from Mistral
- 0.25 is calibrated for `all-MiniLM-L6-v2` on complaint text; will need recalibration if the embedding model changes
- Refusal is returned as a 200 OK with `refused=True`, not a 4xx/5xx, so clients can render a graceful "insufficient evidence" message
- The 12-case eval harness validates refusal on an adversarial "xyzzy" query

---

## DL-004 — ChromaDB over FAISS / Pinecone

**Phase:** 5 (RAG Index)  
**Decision:** Use ChromaDB as the vector store.

**Considered alternatives:**
- FAISS — faster approximate search, but no metadata filtering (can't filter by product/issue)
- Pinecone — managed, scalable, but requires API key and adds cost
- Weaviate — supports metadata filtering, but complex local deployment

**Rationale:**
- ChromaDB supports metadata filtering (`where={"product": "Credit card"}`) out of the box
- Zero-config local deployment — `chroma_store/` directory is all that's needed
- Python-native client matches the FastAPI serving stack
- Adequate performance for 41,653 chunks (< 50 ms retrieval)

---

## DL-005 — Evidently `legacy` API

**Phase:** 8 (Monitoring)  
**Decision:** Import from `evidently.legacy` rather than `evidently.report`.

**Context:**  
Evidently 0.7.21 moved the `Report` + `DataDriftPreset` API to `evidently.legacy`. The new 0.7.x API is under `evidently.core` and has a different interface.

**Rationale:**
- The legacy API is stable, produces identical HTML output, and works with `as_dict()` for programmatic extraction
- Upgrading to the new API would require significant interface changes for no functional benefit
- Pinned version in `requirements.txt` prevents silent breaking changes

---

## DL-006 — In-Process Metrics vs. Prometheus

**Phase:** 7 (Integration Endpoints)  
**Decision:** Use an in-process `_metrics` dict for the `/metrics` endpoint instead of Prometheus.

**Rationale:**
- Adds zero external dependencies
- Sufficient for demo purposes and the CI smoke test
- Clearly documented as the primary limitation to address in production hardening
- Swap-in path: replace `_record_*` helpers with `prometheus_client.Counter` / `Histogram` calls

---

## DL-007 — Chunk Size 512 / Overlap 64

**Phase:** 5 (RAG Index)  
**Decision:** Chunk complaint narratives at 512 tokens with 64-token overlap.

**Rationale:**
- 512 tokens is below Mistral's 32k context limit while fitting multiple chunks as evidence
- 64-token overlap preserves sentence continuity at chunk boundaries
- Empirically, complaint narratives average 120–250 tokens; 512-token chunks capture 2–4 full narratives per chunk, improving topical coherence
- Content-addressed IDs (SHA-256 of chunk text) ensure upserts are idempotent

---

## DL-008 — `refused=True` as 200 OK vs. 4xx

**Phase:** 6 (RAG Answering)  
**Decision:** Return `refused=True` in a 200 OK response, not a 422/503.

**Rationale:**
- A refusal is a valid, expected response — the engine made a decision, not an error
- Returning 4xx would require clients to handle an extra error code path for a non-exceptional case
- The `refused` boolean allows UI to render a graceful "I don't have enough evidence" message
- Consistent with established LLM API patterns (OpenAI, Anthropic also return 200 for content policy refusals)

---

## DL-009 — GitHub Actions over Jenkins / GitLab CI

**Phase:** 9 (CI/CD)  
**Decision:** Use GitHub Actions for all CI/CD automation.

**Rationale:**
- Repository is hosted on GitHub — zero additional infrastructure required
- GHCR (GitHub Container Registry) is natively integrated for Docker push
- `workflow_run` trigger allows deploy to gate on CI success without polling
- Buildx + GHA layer cache dramatically reduces image build times on hot paths
- Free tier minutes are sufficient for the project scale
