"""
src/training/train.py
----------------------
Training pipeline for Meridian Financial campaign conversion models.

Trains two models against the same feature-engineered dataset:
  1. Logistic Regression  — interpretable baseline
  2. XGBoost              — improved challenger

Each run is tracked in MLflow (params, metrics, artifacts).
Fitted models are persisted as joblib pickles under ``artifacts/models/``.

Usage
-----
  python src/training/train.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_pipeline.features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_MAP,
    build_preprocessing_pipeline,
    encode_target,
    fit_pipeline,
    load_bank_sample,
    transform_features,
    validate_feature_columns,
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
logger = logging.getLogger("meridian.train")

# ---------------------------------------------------------------------------
# Path / experiment constants (overridable via env)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR: Path = Path(
    os.getenv("MODELS_DIR", str(_REPO_ROOT / "artifacts" / "models"))
)
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "meridian-campaign-conversion"
)

# Deterministic train/test split
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    bank_sample_path: Path | None = None,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Load, preprocess, and split the bank marketing sample.

    Parameters
    ----------
    bank_sample_path:
        Path to ``bank_sample.csv``.  Defaults to ``settings.bank_sample_path``.
    test_size:
        Fraction of data reserved for test (default 0.2).
    seed:
        Random seed for reproducible splits.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test, fitted_pipeline)``
    """
    if bank_sample_path is None:
        from src.common.config import settings
        bank_sample_path = settings.bank_sample_path

    df = load_bank_sample(bank_sample_path)
    validate_feature_columns(df, ALL_FEATURE_COLUMNS + [TARGET_COLUMN])

    y = encode_target(df[TARGET_COLUMN])

    # Build and fit preprocessing pipeline on training split only
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=seed, stratify=y
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    logger.info(
        "Split: %d train / %d test  (positive rate train=%.3f test=%.3f)",
        len(train_df), len(test_df),
        y_train.mean(), y_test.mean(),
    )

    pipeline = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    pipeline = fit_pipeline(train_df, pipeline, ALL_FEATURE_COLUMNS)

    X_train = transform_features(train_df, pipeline, ALL_FEATURE_COLUMNS).values
    X_test = transform_features(test_df, pipeline, ALL_FEATURE_COLUMNS).values

    return X_train, X_test, y_train.values, y_test.values, pipeline


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_baseline_model(seed: int = RANDOM_SEED) -> LogisticRegression:
    """Return an unfitted Logistic Regression (interpretable baseline).

    Parameters
    ----------
    seed:
        Random state for reproducibility.

    Returns
    -------
    LogisticRegression
    """
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",   # handles class imbalance
        random_state=seed,
        n_jobs=-1,
    )


def build_improved_model(seed: int = RANDOM_SEED) -> XGBClassifier:
    """Return an unfitted XGBoost classifier (improved challenger).

    Parameters
    ----------
    seed:
        Random state for reproducibility.

    Returns
    -------
    XGBClassifier
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=7,        # ~36k negative / ~4.6k positive ≈ 7.9
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Any, float]:
    """Fit *model* on training data and return the fitted model + latency.

    Parameters
    ----------
    model:
        Unfitted sklearn-compatible estimator.
    X_train:
        Training feature matrix.
    y_train:
        Training labels.

    Returns
    -------
    tuple[model, float]
        (fitted model, training latency in seconds)
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    latency = time.perf_counter() - t0
    logger.info(
        "Trained %s in %.2fs", type(model).__name__, latency
    )
    return model, latency


def save_model(model: Any, name: str, output_dir: Path = MODELS_DIR) -> Path:
    """Persist *model* to ``{output_dir}/{name}.pkl`` via joblib.

    Parameters
    ----------
    model:
        Fitted estimator.
    name:
        Artifact name (without ``.pkl`` extension).
    output_dir:
        Destination directory.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.pkl"
    joblib.dump(model, out_path)
    logger.info("Model saved to %s", out_path)
    return out_path


def load_model(name: str, models_dir: Path = MODELS_DIR) -> Any:
    """Load a joblib-persisted model from ``{models_dir}/{name}.pkl``.

    Parameters
    ----------
    name:
        Artifact name (without ``.pkl`` extension).
    models_dir:
        Directory containing the pickle file.

    Returns
    -------
    Fitted estimator.
    """
    pkl_path = models_dir / f"{name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {pkl_path}")
    model = joblib.load(pkl_path)
    logger.info("Model loaded from %s", pkl_path)
    return model


# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------

def _setup_mlflow() -> None:
    """Configure MLflow tracking URI and ensure the experiment exists."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(
        "MLflow: tracking_uri=%s  experiment=%s",
        MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    )


def run_training_with_tracking(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[Any, dict[str, float], str]:
    """Train *model*, evaluate it, and log everything to MLflow.

    Parameters
    ----------
    model:
        Unfitted estimator.
    model_name:
        Human-readable name used for MLflow run tagging and artifact naming.
    X_train, X_test, y_train, y_test:
        Pre-split numpy arrays.

    Returns
    -------
    tuple
        ``(fitted_model, metrics_dict, mlflow_run_id)``
    """
    from src.training.evaluate import compute_metrics

    _setup_mlflow()

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("model_name", model_name)

        # Log hyper-parameters
        params = model.get_params()
        # Flatten any nested dicts (XGBoost can produce them)
        flat_params = {
            k: str(v) for k, v in params.items() if v is not None
        }
        mlflow.log_params(flat_params)

        # Train
        fitted_model, latency = train_model(model, X_train, y_train)
        mlflow.log_metric("training_latency_s", round(latency, 4))

        # Evaluate on test set
        metrics = compute_metrics(fitted_model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Persist artifact
        artifact_path = save_model(fitted_model, model_name)
        mlflow.log_artifact(str(artifact_path), artifact_path="models")

        run_id = run.info.run_id
        logger.info(
            "[%s] run_id=%s  PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f",
            model_name, run_id,
            metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
        )

    return fitted_model, metrics, run_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_training() -> dict[str, Any]:
    """End-to-end training pipeline.

    1. Prepare data (load → preprocess → split)
    2. Train Logistic Regression baseline
    3. Train XGBoost improved model
    4. Log both runs to MLflow

    Returns
    -------
    dict
        Keys: ``baseline_metrics``, ``improved_metrics``, ``baseline_run_id``,
        ``improved_run_id``.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()

    baseline_model = build_baseline_model()
    _, baseline_metrics, baseline_run_id = run_training_with_tracking(
        baseline_model, "baseline_model",
        X_train, X_test, y_train, y_test,
    )

    improved_model = build_improved_model()
    _, improved_metrics, improved_run_id = run_training_with_tracking(
        improved_model, "improved_model",
        X_train, X_test, y_train, y_test,
    )

    logger.info("Training complete.")
    logger.info("Baseline  — PR-AUC: %.4f  F1: %.4f", baseline_metrics["pr_auc"], baseline_metrics["f1"])
    logger.info("Improved  — PR-AUC: %.4f  F1: %.4f", improved_metrics["pr_auc"], improved_metrics["f1"])

    return {
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "baseline_run_id": baseline_run_id,
        "improved_run_id": improved_run_id,
    }


if __name__ == "__main__":
    run_training()
