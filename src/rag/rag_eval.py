"""
src/rag/rag_eval.py
--------------------
Evaluation harness for the Meridian Financial RAG answering pipeline.

Runs a battery of ≥10 QA test cases against the RAG engine, logs pass/fail
for each case, and persists structured results to ``artifacts/reports/rag_eval.json``.

Evaluation criteria
--------------------
* ``answer_present``       — answer is non-empty and not a refusal
* ``expected_refused``     — refusal cases match expectation
* ``evidence_ids_present`` — at least one evidence ID returned
* ``prompt_version_ok``    — prompt version matches expected tag

Usage
-----
  python src/rag/rag_eval.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("meridian.rag.rag_eval")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH: Path = _REPO_ROOT / "artifacts" / "reports" / "rag_eval.json"


# ---------------------------------------------------------------------------
# Test cases (≥10 required by spec)
# ---------------------------------------------------------------------------

QA_TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "tc01",
        "question": "What are the most common issues with credit cards?",
        "expect_refused": False,
        "description": "General credit card complaint query",
    },
    {
        "id": "tc02",
        "question": "How do banks handle billing disputes?",
        "expect_refused": False,
        "description": "Billing dispute resolution query",
    },
    {
        "id": "tc03",
        "question": "What problems do customers face with mortgage loans?",
        "expect_refused": False,
        "description": "Mortgage complaint query",
    },
    {
        "id": "tc04",
        "question": "Are there complaints about debt collection practices?",
        "expect_refused": False,
        "description": "Debt collection issues query",
    },
    {
        "id": "tc05",
        "question": "What are common issues with student loan servicers?",
        "expect_refused": False,
        "description": "Student loan servicing query",
    },
    {
        "id": "tc06",
        "question": "How frequently do consumers complain about identity theft?",
        "expect_refused": False,
        "description": "Identity theft complaint frequency",
    },
    {
        "id": "tc07",
        "question": "What issues arise with checking and savings accounts?",
        "expect_refused": False,
        "description": "Checking account complaints query",
    },
    {
        "id": "tc08",
        "question": "Do complaints mention issues with credit reporting?",
        "expect_refused": False,
        "description": "Credit reporting issues query",
    },
    {
        "id": "tc09",
        "question": "What are common auto loan complaints?",
        "expect_refused": False,
        "description": "Auto loan complaint query",
    },
    {
        "id": "tc10",
        "question": "xyzzyqwerty123456 nonsense query with no match",
        "expect_refused": True,
        "description": "Nonsense query — should be refused",
    },
    {
        "id": "tc11",
        "question": "What happens when a bank ignores a customer's dispute?",
        "expect_refused": False,
        "description": "Bank dispute resolution failure query",
    },
    {
        "id": "tc12",
        "question": "Are there payday loan complaints in the data?",
        "expect_refused": False,
        "description": "Payday loan query",
    },
]


# ---------------------------------------------------------------------------
# Single-case evaluation
# ---------------------------------------------------------------------------

def evaluate_single_case(
    case: dict[str, Any],
    engine: Any,  # RAGAnswerEngine
    expected_prompt_version: str = "v1.0",
) -> dict[str, Any]:
    """Run one test case through the engine and return a structured result.

    Parameters
    ----------
    case:
        Test case dict with keys: id, question, expect_refused, description.
    engine:
        Instantiated ``RAGAnswerEngine``.
    expected_prompt_version:
        Prompt version string to validate against.

    Returns
    -------
    dict
        Evaluation result with pass/fail flags for each criterion.
    """
    question = case["question"]
    expect_refused = case.get("expect_refused", False)

    logger.info("Evaluating [%s]: %s", case["id"], question[:60])

    rag_answer = engine.answer(question)

    # --- Criteria ---
    answer_present = bool(rag_answer.answer) and not rag_answer.refused
    refusal_correct = rag_answer.refused == expect_refused
    evidence_ids_present = len(rag_answer.evidence_ids) > 0
    prompt_version_ok = rag_answer.prompt_version == expected_prompt_version

    # Overall pass: refusal cases pass if refusal_correct; non-refusal cases
    # must also have answer and evidence present.
    if expect_refused:
        passed = refusal_correct
    else:
        passed = answer_present and evidence_ids_present and prompt_version_ok

    result = {
        "id": case["id"],
        "description": case.get("description", ""),
        "question": question,
        "expect_refused": expect_refused,
        "actual_refused": rag_answer.refused,
        "passed": passed,
        "criteria": {
            "answer_present": answer_present,
            "refusal_correct": refusal_correct,
            "evidence_ids_present": evidence_ids_present,
            "prompt_version_ok": prompt_version_ok,
        },
        "evidence_ids": rag_answer.evidence_ids,
        "evidence_sufficiency": rag_answer.evidence_sufficiency,
        "prompt_version": rag_answer.prompt_version,
        "retrieval_count": rag_answer.retrieval_count,
        "token_usage": rag_answer.token_usage,
        "latency_ms": rag_answer.latency_ms,
        "answer_preview": rag_answer.answer[:200] if rag_answer.answer else "",
    }

    status = "PASS" if passed else "FAIL"
    logger.info("[%s] %s — %s", case["id"], status, question[:50])
    return result


# ---------------------------------------------------------------------------
# Full harness
# ---------------------------------------------------------------------------

def run_evaluation(
    engine: Any,
    test_cases: list[dict[str, Any]] = QA_TEST_CASES,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Run all test cases and produce a structured evaluation report.

    Parameters
    ----------
    engine:
        Instantiated ``RAGAnswerEngine``.
    test_cases:
        List of QA test case dicts.
    report_path:
        Output path for the JSON report.

    Returns
    -------
    dict
        Summary report with per-case results and aggregate statistics.
    """
    logger.info("Starting RAG evaluation — %d test cases", len(test_cases))

    results = [evaluate_single_case(case, engine) for case in test_cases]

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0.0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 4),
        },
        "results": results,
    }

    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info(
        "RAG evaluation complete — %d/%d passed (%.1f%%)  report=%s",
        passed, total, pass_rate * 100, report_path,
    )
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    from src.rag.answer import RAGAnswerEngine
    engine = RAGAnswerEngine()
    report = run_evaluation(engine)
    print(
        f"\nEvaluation complete: "
        f"{report['summary']['passed']}/{report['summary']['total_cases']} passed"
    )
