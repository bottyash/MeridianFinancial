# Reflection — Meridian Financial Platform
---

## 1. Why XGBoost?

I chose XGBoost over logistic regression, random forest, and neural networks for three reasons.

First, **the data is tabular and medium-scale** (41k rows, 21 features). On this kind of dataset, gradient boosting consistently outperforms linear models on imbalanced binary classification tasks — empirically shown by our PR-AUC jumping from ~0.61 (logistic regression) to ~0.74 (XGBoost with `scale_pos_weight` tuning).

Second, **class imbalance handling is built-in**. The `scale_pos_weight` parameter directly penalizes misclassifying the positive class, which matters here (only 11.3% of clients subscribe). We tuned this by cross-validation rather than guessing.

Third, **inference speed**. XGBoost scores a single record in < 5 ms, making it viable for real-time `/predict` calls without batching. A neural network would require more infrastructure (GPU, ONNX export) for equivalent latency.

The alternative I seriously considered was LightGBM — performance is nearly identical, but XGBoost has better MLflow autolog integration and more community examples for this exact use case.

---

## 2. What broke during deployment?

**Three things broke, in order of pain:**

**1. Evidently 0.7.x API change** — `from evidently.report import Report` raised `ModuleNotFoundError`. Evidently had migrated the legacy API to `evidently.legacy.report` in 0.7.x. The fix was straightforward once diagnosed, but it cost an hour because the error message was cryptic.

**2. ChromaDB telemetry `asyncio.iscoroutinefunction` deprecation warning** — Python 3.14 deprecated `asyncio.iscoroutinefunction` in favour of `inspect.iscoroutinefunction`. ChromaDB 0.5.x still uses the deprecated form. This generates a warning in every test run and will break in Python 3.16. Worked around by pinning and documenting; the real fix requires a ChromaDB upstream patch.

**3. PowerShell shell quoting** — inline `python -c` commands with f-strings and JSON fail in PowerShell because `&&` is not a valid statement separator and quote escaping differs from bash. Solved by extracting all inline scripts into `.py` files under `scripts/` — a better pattern anyway.

---

## 3. Why the PR-AUC +3pp / F1 –2pp gate margin?

The margins were chosen to be **statistically meaningful but not excessively conservative**.

A 3-percentage-point improvement in PR-AUC corresponds to roughly 1.5× the cross-validation standard deviation observed during grid search (~2 pp). This filters out noise from random seed variation, different train/test splits, or minor hyperparameter tweaks that don't represent real signal.

The F1 guard of –2pp prevents a model from "gaming" PR-AUC at the expense of recall — for example, by raising the prediction threshold to boost precision while missing more subscribers. In a real marketing campaign, missed subscribers (false negatives) directly translate to lost revenue.

In production I would tighten these to +2pp / –1pp once we have more historical model data to estimate variance more precisely.

---

## 4. One failed RAG answer

**Question:** "What issues arise with credit card interest rate disputes?"

**Expected:** Specific complaints about APR changes, deferred interest, retroactive rate hikes.

**What the engine returned:** A blend of credit card billing disputes AND savings account interest rate complaints, because `all-MiniLM-L6-v2` embeds both into similar semantic space near the word "interest rate." The answer was factually plausible but not grounded solely in credit card complaints.

**Why it failed:** The retrieval step has no product-type awareness by default. Without a `product_filter="Credit card"` parameter, the top-5 chunks included 2 savings/CD-related complaints that semantically overlapped on "interest rate."

**What I'd fix:** The `/customer-intel` endpoint already threads `product_filter` through to the `where={"product": ...}` ChromaDB filter. The standalone `/ask-complaints` endpoint should prompt the user to add a product filter, or the RAG engine should automatically extract the product category from the question using a lightweight classifier.

---

## 5. Biggest remaining production risk

**The Mistral API is a single point of failure with no fallback.**

If Mistral's API is unavailable (rate limit, outage, key expiry), every `/ask-complaints` and `/customer-intel` call returns a 500. There is currently no circuit breaker, no retry, no fallback to a local model (e.g., Ollama with Mistral 7B), and no cached response mechanism.

In a real deployment this would be the first thing I'd address:
1. Add `tenacity` retry with exponential backoff (2 attempts, 2s base)
2. On circuit open: return `refused=True` with `evidence_sufficiency="LLM temporarily unavailable — please retry"`
3. For high-priority customers: route to a locally-hosted Mistral 7B via Ollama as fallback

The ML prediction path (`/predict`, `/batch-score`) is fully self-contained and does not share this risk.

---

## 6. What would a senior MLOps engineer criticize?

**Six things, in increasing severity:**

1. **The in-process metrics dict** — `_metrics` resets on restart and doesn't aggregate across multiple Uvicorn workers. In a 2-worker deployment the metrics are already incorrect. Should be Prometheus counters from day one.

2. **No model artifact versioning strategy** — `improved_model.pkl` is a single file. If a bad model is deployed, rollback requires manually replacing the file and restarting. The MLflow Model Registry's `staging` → `production` promotion flow should be wired to `model_loader.py`.

3. **ChromaDB is not backed up** — `chroma_store/` is a local directory mounted as a Docker volume. A host machine failure = vector index loss. Should have daily S3 snapshots.

4. **The promotion gate skips gracefully on missing artifacts** — this is correct for CI on forks, but in production the gate must be enforced, not skipped. The CI eval-gate job should fail loudly if called on the main deployment branch without artifacts.

5. **No canary deployment** — new models go 100% to production immediately. A senior MLOps engineer would require a canary: 5% traffic to new model, monitor PR-AUC and error rate for 24 hours, then full cutover.

6. **Complaint data is static** — the ChromaDB index is built once from a historical dump. In production, new CFPB complaints arrive daily. Without a real-time index update pipeline, the RAG answers become stale. This is the most important missing operational workflow.
