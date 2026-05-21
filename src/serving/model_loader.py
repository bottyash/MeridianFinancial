"""
src/serving/model_loader.py
-----------------------------
Thread-safe, singleton model loader for the Meridian Financial serving layer.

Responsibilities
----------------
* Load the fitted sklearn Pipeline (preprocessor) from ``artifacts/features/``
* Load the trained XGBoost model (``improved_model.pkl``) from ``artifacts/models/``
* Load the feature schema JSON so the serving layer can validate inputs
* Provide a single ``ModelBundle`` dataclass to avoid repeated disk I/O
* Expose a ``get_model_bundle()`` function cached with ``lru_cache``

The loader is intentionally decoupled from FastAPI lifecycle hooks so it can
be used in tests, CLI scripts, and batch-scoring jobs without any HTTP
server context.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.data_pipeline.features import load_feature_schema

logger = logging.getLogger("meridian.model_loader")

# ---------------------------------------------------------------------------
# Path defaults (overridable via environment variables)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURES_ARTIFACTS_DIR: Path = Path(
    os.getenv("FEATURES_ARTIFACTS_DIR", str(_REPO_ROOT / "artifacts" / "features"))
)
MODELS_DIR: Path = Path(
    os.getenv("MODELS_DIR", str(_REPO_ROOT / "artifacts" / "models"))
)
MODEL_NAME: str = os.getenv("SERVING_MODEL_NAME", "improved_model")
DECISION_THRESHOLD: float = float(os.getenv("DECISION_THRESHOLD", "0.5"))


# ---------------------------------------------------------------------------
# Model bundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Container for all serving-time artefacts.

    Attributes
    ----------
    preprocessor:
        Fitted sklearn Pipeline (ColumnTransformer) from phase-2.
    model:
        Fitted XGBoost (or LR) classifier from phase-3.
    feature_schema:
        Dict loaded from ``feature_schema.json``.
    model_version:
        Human-readable identifier (filename stem of the model pickle).
    threshold:
        Decision threshold for binary classification.
    """

    preprocessor: Any
    model: Any
    feature_schema: dict[str, Any]
    model_version: str
    threshold: float


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model_bundle(
    features_artifacts_dir: Path = FEATURES_ARTIFACTS_DIR,
    models_dir: Path = MODELS_DIR,
    model_name: str = MODEL_NAME,
    threshold: float = DECISION_THRESHOLD,
) -> ModelBundle:
    """Load and return a fully-populated :class:`ModelBundle`.

    Parameters
    ----------
    features_artifacts_dir:
        Directory containing ``preprocessor.pkl`` and ``feature_schema.json``.
    models_dir:
        Directory containing ``{model_name}.pkl``.
    model_name:
        Stem of the model pickle file (without ``.pkl``).
    threshold:
        Binary decision threshold.

    Returns
    -------
    ModelBundle

    Raises
    ------
    FileNotFoundError
        If any required artifact is missing.
    """
    preprocessor_path = features_artifacts_dir / "preprocessor.pkl"
    model_path = models_dir / f"{model_name}.pkl"

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {preprocessor_path}. "
            "Run src/data_pipeline/features.py first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run src/training/train.py first."
        )

    logger.info("Loading preprocessor from %s", preprocessor_path)
    preprocessor = joblib.load(preprocessor_path)

    logger.info("Loading model '%s' from %s", model_name, model_path)
    model = joblib.load(model_path)

    schema = load_feature_schema(features_artifacts_dir)

    bundle = ModelBundle(
        preprocessor=preprocessor,
        model=model,
        feature_schema=schema,
        model_version=model_name,
        threshold=threshold,
    )

    logger.info(
        "ModelBundle ready — model_version=%s  threshold=%.2f  "
        "numeric_features=%d  categorical_features=%d",
        model_name, threshold,
        len(schema.get("numeric_features", [])),
        len(schema.get("categorical_features", [])),
    )
    return bundle


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """Return the singleton :class:`ModelBundle` (cached after first call).

    Uses ``lru_cache`` so the artifacts are loaded from disk exactly once per
    process lifetime, keeping prediction latency low.

    Returns
    -------
    ModelBundle
    """
    return load_model_bundle()


def reset_model_bundle_cache() -> None:
    """Evict the cached ``ModelBundle`` (useful in tests and hot-reload scenarios)."""
    get_model_bundle.cache_clear()
    logger.info("ModelBundle cache cleared.")
