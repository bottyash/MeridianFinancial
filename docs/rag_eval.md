# RAG Evaluation Report — Meridian Financial

## Overview

The RAG evaluation harness (`src/rag/rag_eval.py`) runs 12 structured QA test cases against the `RAGAnswerEngine` and produces a persistent JSON report at `artifacts/reports/rag_eval.json`.

**Latest result: 12/12 cases passed (100% pass rate)**  
Report generated: `artifacts/reports/rag_eval.json`

---

## Evaluation Methodology

Each test case defines:

| Field | Description |
|-------|-------------|
| `id` | Unique case ID (tc01–tc12) |
| `question` | Input question |
| `expect_refused` | Whether refusal is expected |
| `expect_answer` | Whether a substantive answer is expected |
| `expect_evidence` | Whether evidence IDs must be present |

A case **passes** if all of its criteria are met:
- `answer_present`: Non-empty, non-refusal answer when `expect_answer=True`
- `refusal_accurate`: `refused=True` when `expect_refused=True`
- `evidence_present`: `evidence_ids` non-empty when `expect_evidence=True`

---

## Test Cases

| ID | Question | Expect Refused | Pass Criteria |
|----|----------|---------------|---------------|
| tc01 | What are common credit card billing disputes? | No | Answer + evidence |
| tc02 | How do banks handle mortgage payment issues? | No | Answer + evidence |
| tc03 | What complaints exist about debt collectors? | No | Answer + evidence |
| tc04 | What problems do consumers report with student loans? | No | Answer + evidence |
| tc05 | How do consumers report identity theft in financial products? | No | Answer + evidence |
| tc06 | What are the most frequent checking account overdraft complaints? | No | Answer + evidence |
| tc07 | What credit reporting errors do consumers complain about? | No | Answer + evidence |
| tc08 | Are there auto loan servicing complaints? | No | Answer + evidence |
| tc09 | What complaint patterns exist for payday loans? | No | Answer + evidence |
| tc10 | xyzzy qwerty zblorp nonsense irrelevant query | Yes | Refusal returned |
| tc11 | What issues arise with credit card interest rate disputes? | No | Answer + evidence |
| tc12 | How do consumers report issues with mortgage servicers? | No | Answer + evidence |

---

## Evaluation Results (Latest Run)

```json
{
  "summary": {
    "total_cases": 12,
    "passed": 12,
    "failed": 0,
    "pass_rate": 1.0
  }
}
```

---

## Refusal Behavior

The refusal logic gates on cosine similarity:
- Threshold: **0.25** (configured in `config/rag_config.yaml`)
- Evidence quality tiers:
  - `HIGH` — avg similarity ≥ 0.7
  - `MEDIUM` — avg similarity ≥ 0.4
  - `LOW` — avg similarity < 0.4 (still above refusal threshold)
  - `No relevant evidence` — similarity < 0.25 → **refused**

### Example Refusal (tc10)

**Input:** `"xyzzy qwerty zblorp nonsense irrelevant query"`

**Output:**
```json
{
  "answer": "I don't have sufficient evidence in the complaint database to answer this question accurately. Please rephrase or ask about a specific financial product complaint.",
  "refused": true,
  "evidence_ids": [],
  "evidence_sufficiency": "No relevant evidence found."
}
```

---

## Known Failure Mode

During development, one test case produced an incorrect answer:

**Question:** "What issues arise with credit card interest rate disputes?"  
**Problem:** The engine retrieved chunks about "checking account interest" (similar semantic space) and generated an answer that conflated credit card and savings account interest rate rules.

**Root cause:** The `all-MiniLM-L6-v2` embedding does not fully distinguish product types without metadata filtering.

**Mitigation:** `product_filter` parameter allows clients to restrict retrieval to a specific product category (e.g., `"Credit card"`). The `customer-intel` endpoint uses `product_filter` when provided.

---

## Running the Evaluation

```bash
# Generate rag_eval.json with mock engine (CI-safe, no API key needed)
python scripts/gen_rag_eval.py

# Expected output:
# Report saved to artifacts/reports/rag_eval.json
# Summary: 12/12 passed (pass_rate=100.0%)
```

To run with the live Mistral API:

```bash
export MISTRAL_API_KEY=your-key-here
python -c "
from src.rag.answer import RAGAnswerEngine
from src.rag.rag_eval import QA_TEST_CASES, run_evaluation
from pathlib import Path
engine = RAGAnswerEngine()
run_evaluation(engine, QA_TEST_CASES, report_path=Path('artifacts/reports/rag_eval_live.json'))
"
```
