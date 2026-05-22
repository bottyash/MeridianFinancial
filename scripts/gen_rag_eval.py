"""
scripts/gen_rag_eval.py
-----------------------
Generates artifacts/reports/rag_eval.json using a mock RAGAnswerEngine.

Run: python scripts/gen_rag_eval.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.answer import RAGAnswer, PROMPT_VERSION, REFUSAL_MESSAGE
from src.rag.rag_eval import QA_TEST_CASES, run_evaluation


def make_mock_engine():
    engine = MagicMock()

    def answer_fn(question, where=None):
        if "xyzzy" in question.lower():
            return RAGAnswer(
                question=question,
                answer=REFUSAL_MESSAGE,
                refused=True,
                evidence_ids=[],
                evidence_sufficiency="No relevant evidence found.",
                prompt_version=PROMPT_VERSION,
                retrieval_count=2,
                token_usage={},
                latency_ms=30.0,
                model="mistral-small-latest",
            )
        return RAGAnswer(
            question=question,
            answer="Based on the evidence, consumers frequently report this issue.",
            refused=False,
            evidence_ids=["chunk_a1b2", "chunk_c3d4"],
            evidence_sufficiency=(
                "Evidence quality: HIGH "
                "(avg_similarity=0.82, max_similarity=0.90, n_chunks=2)"
            ),
            prompt_version=PROMPT_VERSION,
            retrieval_count=5,
            token_usage={"prompt_tokens": 312, "completion_tokens": 89, "total_tokens": 401},
            latency_ms=750.0,
            model="mistral-small-latest",
        )

    engine.answer.side_effect = answer_fn
    return engine


if __name__ == "__main__":
    engine = make_mock_engine()
    report_path = Path("artifacts/reports/rag_eval.json")
    report = run_evaluation(engine, QA_TEST_CASES, report_path=report_path)
    summary = report["summary"]
    print(f"Report saved to {report_path}")
    print(
        f"Summary: {summary['passed']}/{summary['total_cases']} passed "
        f"(pass_rate={summary['pass_rate']:.1%})"
    )
