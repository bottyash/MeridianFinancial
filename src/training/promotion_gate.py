"""
src/training/promotion_gate.py
--------------------------------
Model promotion gate for Meridian Financial.

Enforces the mandatory promotion rules from master_prompt.md:

  PROMOTE only if:
    * PR-AUC (improved) - PR-AUC (baseline) >= 0.03  (3 percentage points)
    * F1 (baseline) - F1 (improved) <= 0.02          (degradation ≤ 2pp)

  Blocked promotions are logged and a rejection reason is saved to
  ``artifacts/models/gate_log.json``.

Usage
-----
  from src.training.promotion_gate import evaluate_promotion_gate
  result = evaluate_promotion_gate(baseline_metrics, improved_metrics)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("meridian.promotion_gate")

# ---------------------------------------------------------------------------
# Gate thresholds (from master_prompt.md)
# ---------------------------------------------------------------------------
PR_AUC_IMPROVEMENT_MIN: float = float(
    os.getenv("GATE_PR_AUC_MIN_IMPROVEMENT", "0.03")
)
F1_DEGRADATION_MAX: float = float(
    os.getenv("GATE_F1_MAX_DEGRADATION", "0.02")
)

# Default output path
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_LOG: Path = Path(
    os.getenv("GATE_LOG_PATH", str(_REPO_ROOT / "artifacts" / "models" / "gate_log.json"))
)


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def check_promotion_gate(
    baseline_metrics: dict[str, float],
    improved_metrics: dict[str, float],
    pr_auc_min_improvement: float = PR_AUC_IMPROVEMENT_MIN,
    f1_max_degradation: float = F1_DEGRADATION_MAX,
) -> dict[str, Any]:
    """Apply the promotion gate rules and return a structured result.

    Parameters
    ----------
    baseline_metrics:
        Evaluation metrics for the baseline (Logistic Regression) model.
    improved_metrics:
        Evaluation metrics for the improved (XGBoost) model.
    pr_auc_min_improvement:
        Minimum required increase in PR-AUC for promotion (default 0.03).
    f1_max_degradation:
        Maximum allowed F1 decrease for promotion (default 0.02).

    Returns
    -------
    dict
        Keys:
          * ``promoted`` (bool)           — whether the gate passed
          * ``pr_auc_delta`` (float)       — actual PR-AUC improvement
          * ``f1_delta`` (float)           — actual F1 change (negative = degradation)
          * ``pr_auc_check`` (bool)        — did PR-AUC criterion pass?
          * ``f1_check`` (bool)            — did F1 criterion pass?
          * ``rejection_reasons`` (list)   — human-readable rejection messages
          * ``gate_config`` (dict)         — thresholds used
    """
    pr_auc_delta = improved_metrics["pr_auc"] - baseline_metrics["pr_auc"]
    f1_delta = improved_metrics["f1"] - baseline_metrics["f1"]
    f1_degradation = -f1_delta  # positive means degradation

    pr_auc_check = pr_auc_delta >= pr_auc_min_improvement
    f1_check = f1_degradation <= f1_max_degradation + 1e-9  # inclusive boundary

    rejection_reasons: list[str] = []
    if not pr_auc_check:
        rejection_reasons.append(
            f"PR-AUC improvement {pr_auc_delta:+.4f} is below the required "
            f"+{pr_auc_min_improvement:.2f} threshold."
        )
    if not f1_check:
        rejection_reasons.append(
            f"F1 degradation {f1_degradation:.4f} exceeds the allowed "
            f"{f1_max_degradation:.2f} limit."
        )

    promoted = pr_auc_check and f1_check

    result: dict[str, Any] = {
        "promoted": promoted,
        "pr_auc_delta": round(pr_auc_delta, 6),
        "f1_delta": round(f1_delta, 6),
        "pr_auc_check": pr_auc_check,
        "f1_check": f1_check,
        "rejection_reasons": rejection_reasons,
        "gate_config": {
            "pr_auc_min_improvement": pr_auc_min_improvement,
            "f1_max_degradation": f1_max_degradation,
        },
        "baseline_metrics": {
            "pr_auc": baseline_metrics["pr_auc"],
            "f1": baseline_metrics["f1"],
            "roc_auc": baseline_metrics.get("roc_auc"),
        },
        "improved_metrics": {
            "pr_auc": improved_metrics["pr_auc"],
            "f1": improved_metrics["f1"],
            "roc_auc": improved_metrics.get("roc_auc"),
        },
    }

    if promoted:
        logger.info(
            "PROMOTION GATE PASSED -- delta PR-AUC: %+.4f (>= %.2f)  delta F1: %+.4f (degradation <= %.2f)",
            pr_auc_delta, pr_auc_min_improvement,
            f1_delta, f1_max_degradation,
        )
    else:
        logger.warning(
            "PROMOTION GATE BLOCKED -- %s",
            "; ".join(rejection_reasons),
        )

    return result


def save_gate_log(
    gate_result: dict[str, Any],
    output_path: Path = DEFAULT_GATE_LOG,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    """Append *gate_result* (with timestamp) to the gate log JSON file.

    The log accumulates all gate evaluations as a JSON list, so historical
    decisions are preserved across training runs.

    Parameters
    ----------
    gate_result:
        Output of :func:`check_promotion_gate`.
    output_path:
        Destination JSON file (created / appended to).
    run_metadata:
        Optional extra fields to include (e.g., MLflow run IDs).

    Returns
    -------
    Path
        Absolute path to the gate log.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing log (or start fresh)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as fh:
            log: list[dict[str, Any]] = json.load(fh)
    else:
        log = []

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **gate_result,
    }
    if run_metadata:
        entry["run_metadata"] = run_metadata

    log.append(entry)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    logger.info(
        "Gate log saved to %s (%d total entries)",
        output_path, len(log),
    )
    return output_path


def evaluate_promotion_gate(
    baseline_metrics: dict[str, float],
    improved_metrics: dict[str, float],
    gate_log_path: Path = DEFAULT_GATE_LOG,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check the promotion gate, log the result, and return the gate dict.

    This is the primary public API for the promotion gate.

    Parameters
    ----------
    baseline_metrics:
        Metrics dict for the baseline model.
    improved_metrics:
        Metrics dict for the improved model.
    gate_log_path:
        Path to ``gate_log.json`` for persistence.
    run_metadata:
        Optional extra context (e.g., MLflow run IDs) to embed in the log.

    Returns
    -------
    dict
        Gate result from :func:`check_promotion_gate`.
    """
    result = check_promotion_gate(baseline_metrics, improved_metrics)
    save_gate_log(result, gate_log_path, run_metadata)
    return result


def print_gate_report(gate_result: dict[str, Any]) -> None:
    """Print a human-readable gate decision report to stdout."""
    border = "=" * 60
    status = "[PASS] PROMOTED" if gate_result["promoted"] else "[BLOCK] BLOCKED"
    print(border)
    print(f"  Promotion Gate Decision -- {status}")
    print(border)
    print(f"  PR-AUC improvement : {gate_result['pr_auc_delta']:+.4f}  "
          f"(required >= {gate_result['gate_config']['pr_auc_min_improvement']:.2f})"
          f"  {'PASS' if gate_result['pr_auc_check'] else 'FAIL'}")
    print(f"  F1 change          : {gate_result['f1_delta']:+.4f}  "
          f"(degradation <= {gate_result['gate_config']['f1_max_degradation']:.2f})"
          f"  {'PASS' if gate_result['f1_check'] else 'FAIL'}")
    if gate_result["rejection_reasons"]:
        print("  Rejection reasons:")
        for reason in gate_result["rejection_reasons"]:
            print(f"    - {reason}")
    print(border)
