"""
src/training/evaluate.py
--------------------------
Model evaluation utilities for Meridian Financial.

Computes all mandatory metrics:
  * ROC-AUC
  * PR-AUC (area under precision-recall curve)
  * F1 score (binary, threshold=0.5)
  * Precision
  * Recall
  * Calibration score (Brier score)
  * Confusion matrix elements (TP, FP, TN, FN)

All functions are pure and reusable — they operate on arrays, not on paths or
MLflow state, so they can be called from training, serving, or test code.

Usage
-----
  python src/training/evaluate.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("meridian.evaluate")


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full set of evaluation metrics for *model* on the test set.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible estimator with ``predict_proba``.
    X_test:
        Test feature matrix (numpy array).
    y_test:
        True binary labels (numpy array).
    threshold:
        Decision threshold for converting probabilities to class predictions.

    Returns
    -------
    dict[str, float]
        All metrics as a flat, JSON-serialisable dict.
    """
    # Probability of the positive class
    y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]
    y_pred: np.ndarray = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_test, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "brier_score": round(brier, 6),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    logger.info(
        "Metrics — ROC-AUC: %.4f  PR-AUC: %.4f  F1: %.4f  "
        "Precision: %.4f  Recall: %.4f  Brier: %.4f",
        roc_auc, pr_auc, f1, precision, recall, brier,
    )
    return metrics


def compare_metrics(
    baseline_metrics: dict[str, float],
    improved_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute deltas between improved and baseline metrics.

    Parameters
    ----------
    baseline_metrics:
        Metrics dict from the baseline model.
    improved_metrics:
        Metrics dict from the improved model.

    Returns
    -------
    dict[str, float]
        Delta for each scalar metric (improved − baseline).
    """
    scalar_keys = {"roc_auc", "pr_auc", "f1", "precision", "recall", "brier_score"}
    deltas = {
        f"delta_{k}": round(improved_metrics[k] - baseline_metrics[k], 6)
        for k in scalar_keys
        if k in baseline_metrics and k in improved_metrics
    }
    logger.info(
        "Comparison — ΔPR-AUC: %+.4f  ΔF1: %+.4f  ΔROC-AUC: %+.4f",
        deltas.get("delta_pr_auc", 0),
        deltas.get("delta_f1", 0),
        deltas.get("delta_roc_auc", 0),
    )
    return deltas


def print_evaluation_report(
    model_name: str,
    metrics: dict[str, float],
) -> None:
    """Print a human-readable evaluation report to stdout."""
    border = "=" * 60
    print(border)
    print(f"  Evaluation Report — {model_name}")
    print(border)
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC     : {metrics['pr_auc']:.4f}")
    print(f"  F1 Score   : {metrics['f1']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TP={metrics['true_positives']}  FP={metrics['false_positives']}")
    print(f"    FN={metrics['false_negatives']}  TN={metrics['true_negatives']}")
    print(border)


# ---------------------------------------------------------------------------
# Entry point (runs evaluation against saved artifacts)
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    """Load saved model artifacts and evaluate on the bank marketing sample."""
    import os

    from src.data_pipeline.features import (
        ALL_FEATURE_COLUMNS,
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
        build_preprocessing_pipeline,
        encode_target,
        fit_pipeline,
        load_bank_sample,
        transform_features,
    )
    from sklearn.model_selection import train_test_split
    from src.training.train import MODELS_DIR, RANDOM_SEED, TEST_SIZE, load_model

    _repo_root = Path(__file__).resolve().parents[2]
    bank_path = Path(
        os.getenv("BANK_SAMPLE_PATH", str(_repo_root / "data" / "samples" / "bank_sample.csv"))
    )

    df = load_bank_sample(bank_path)
    y = encode_target(df[TARGET_COLUMN])

    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    pipeline = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    fit_pipeline(train_df, pipeline, ALL_FEATURE_COLUMNS)
    X_test = transform_features(test_df, pipeline, ALL_FEATURE_COLUMNS).values

    for name in ["baseline_model", "improved_model"]:
        model = load_model(name, MODELS_DIR)
        metrics = compute_metrics(model, X_test, y_test.values)
        print_evaluation_report(name, metrics)


if __name__ == "__main__":
    run_evaluation()
