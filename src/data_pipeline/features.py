"""
src/data_pipeline/features.py
------------------------------
Reusable, inference-safe feature engineering pipeline for Meridian Financial.

Design principles
-----------------
* Train-serving parity: the sklearn Pipeline/ColumnTransformer is fit once on
  training data and persisted; the same object is loaded at serving time.
* Deterministic: fixed random state throughout; no stochastic steps.
* Modular: each concern (schema, pipeline construction, fit/transform, I/O)
  lives in its own function.
* Parquet output: transformed features are stored as Parquet for downstream
  efficient loading.

Artifacts produced (under ``artifacts/features/``):
  * ``preprocessor.pkl``         — fitted sklearn pipeline (joblib)
  * ``feature_schema.json``      — column names, types, and pipeline metadata
  * ``transformed_train.parquet``— fully processed feature matrix (X)

Usage
-----
  python src/data_pipeline/features.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("meridian.features")

# ---------------------------------------------------------------------------
# Path constants (overridable via environment variables)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

BANK_SAMPLE_PATH: Path = Path(
    os.getenv("BANK_SAMPLE_PATH", str(_REPO_ROOT / "data" / "samples" / "bank_sample.csv"))
)
ARTIFACTS_DIR: Path = Path(
    os.getenv("FEATURES_ARTIFACTS_DIR", str(_REPO_ROOT / "artifacts" / "features"))
)

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

# Mandatory features per phase-2 spec
NUMERIC_FEATURES: list[str] = [
    "age",
    "duration",
    "campaign",
    "previous",
    "pdays",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]

CATEGORICAL_FEATURES: list[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

TARGET_COLUMN: str = "y"
ALL_FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Mapping for binary target variable
TARGET_MAP: dict[str, int] = {"yes": 1, "no": 0}


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def build_feature_schema(
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str,
) -> dict[str, Any]:
    """Return a JSON-serialisable schema describing the feature pipeline.

    The schema is the contract between training and serving; it must be
    saved alongside the preprocessor so that serving can validate incoming
    requests against the expected column set.

    Parameters
    ----------
    numeric_features:
        Names of numeric input columns.
    categorical_features:
        Names of categorical input columns.
    target_column:
        Name of the binary label column.

    Returns
    -------
    dict
        JSON-serialisable schema dict.
    """
    return {
        "version": "1.0",
        "target_column": target_column,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_feature_columns": numeric_features + categorical_features,
        "numeric_steps": ["median_imputer", "standard_scaler"],
        "categorical_steps": ["constant_imputer", "onehot_encoder"],
        "onehot_handle_unknown": "ignore",
        "scaler": "StandardScaler",
        "imputer_numeric_strategy": "median",
        "imputer_categorical_fill": "missing",
    }


def save_feature_schema(schema: dict[str, Any], output_dir: Path) -> Path:
    """Persist *schema* to ``feature_schema.json`` in *output_dir*.

    Parameters
    ----------
    schema:
        Schema dict as returned by :func:`build_feature_schema`.
    output_dir:
        Directory in which to write the JSON file.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "feature_schema.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)
    logger.info("Feature schema saved to %s", out_path)
    return out_path


def load_feature_schema(artifacts_dir: Path) -> dict[str, Any]:
    """Load a previously saved ``feature_schema.json``.

    Parameters
    ----------
    artifacts_dir:
        Directory containing ``feature_schema.json``.

    Returns
    -------
    dict
    """
    schema_path = artifacts_dir / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    logger.info("Feature schema loaded from %s", schema_path)
    return schema


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Construct a sklearn Pipeline wrapping a ColumnTransformer.

    The pipeline is intentionally lean and deterministic:

    Numeric branch
    --------------
    1. ``SimpleImputer(strategy="median")``  — handles missing values robustly
    2. ``StandardScaler()``                  — zero-mean, unit-variance scaling

    Categorical branch
    ------------------
    1. ``SimpleImputer(strategy="constant", fill_value="missing")``
    2. ``OneHotEncoder(handle_unknown="ignore", sparse_output=False)``
       — ``handle_unknown="ignore"`` prevents train-serving skew when new
         categories appear at inference time.

    Parameters
    ----------
    numeric_features:
        Column names for the numeric branch.
    categorical_features:
        Column names for the categorical branch.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "constant_imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "onehot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # drop any extra columns — prevents silent skew
        verbose_feature_names_out=True,
    )

    pipeline = Pipeline(
        steps=[("preprocessor", column_transformer)],
    )

    logger.info(
        "Preprocessing pipeline built: %d numeric + %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )
    return pipeline


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_bank_sample(path: Path = BANK_SAMPLE_PATH) -> pd.DataFrame:
    """Load the bank marketing sample CSV.

    Parameters
    ----------
    path:
        Path to ``bank_sample.csv``.

    Returns
    -------
    pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Bank sample not found at {path}. Run ingest.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded bank sample: %d rows × %d columns from %s", *df.shape, path)
    return df


def encode_target(series: pd.Series, target_map: dict[str, int] = TARGET_MAP) -> pd.Series:
    """Map target column string values to integers (``yes``→1, ``no``→0).

    Parameters
    ----------
    series:
        Raw target column (string-valued).
    target_map:
        Mapping from string label to integer.

    Returns
    -------
    pd.Series
        Integer-encoded target.

    Raises
    ------
    ValueError
        If any values in *series* are not in *target_map*.
    """
    unexpected = set(series.dropna().unique()) - set(target_map.keys())
    if unexpected:
        raise ValueError(
            f"Unexpected target values (not in {set(target_map.keys())}): {unexpected}"
        )
    return series.map(target_map).astype(int)


def validate_feature_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ``ValueError`` if any required feature columns are missing.

    Parameters
    ----------
    df:
        Input dataframe.
    required:
        Column names that must exist.

    Raises
    ------
    ValueError
        Lists all missing columns in one error.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


# ---------------------------------------------------------------------------
# Fit / transform
# ---------------------------------------------------------------------------

def fit_pipeline(
    df: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
) -> Pipeline:
    """Fit the preprocessing *pipeline* on *df[feature_columns]*.

    Parameters
    ----------
    df:
        Training dataframe.
    pipeline:
        Unfitted sklearn pipeline.
    feature_columns:
        Columns to use as input features.

    Returns
    -------
    Pipeline
        Fitted pipeline (same object, mutated in place).
    """
    validate_feature_columns(df, feature_columns)
    X = df[feature_columns]
    pipeline.fit(X)
    logger.info("Pipeline fitted on %d rows", len(X))
    return pipeline


def transform_features(
    df: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Apply a *fitted* pipeline and return a named-column DataFrame.

    Using ``get_feature_names_out()`` ensures that column names are always
    deterministic and available for downstream inspection and serving-skew
    detection.

    Parameters
    ----------
    df:
        Input dataframe (train or inference).
    pipeline:
        Fitted sklearn pipeline.
    feature_columns:
        Columns used as input to the pipeline.

    Returns
    -------
    pd.DataFrame
        Transformed feature matrix with named columns.
    """
    validate_feature_columns(df, feature_columns)
    X = df[feature_columns]
    X_transformed: np.ndarray = pipeline.transform(X)
    feature_names_out = pipeline.get_feature_names_out()
    result = pd.DataFrame(X_transformed, columns=feature_names_out, index=df.index)
    logger.info(
        "Features transformed: %d rows × %d columns",
        *result.shape,
    )
    return result


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------

def save_preprocessor(pipeline: Pipeline, output_dir: Path) -> Path:
    """Persist the fitted *pipeline* to ``preprocessor.pkl`` via joblib.

    Parameters
    ----------
    pipeline:
        Fitted sklearn pipeline.
    output_dir:
        Destination directory.

    Returns
    -------
    Path
        Absolute path to the saved pickle file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "preprocessor.pkl"
    joblib.dump(pipeline, out_path)
    logger.info("Preprocessor saved to %s", out_path)
    return out_path


def load_preprocessor(artifacts_dir: Path) -> Pipeline:
    """Load the fitted pipeline from ``preprocessor.pkl``.

    Parameters
    ----------
    artifacts_dir:
        Directory containing ``preprocessor.pkl``.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline ready for ``transform()``.
    """
    pkl_path = artifacts_dir / "preprocessor.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {pkl_path}")
    pipeline: Pipeline = joblib.load(pkl_path)
    logger.info("Preprocessor loaded from %s", pkl_path)
    return pipeline


def save_transformed_parquet(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write transformed feature DataFrame to ``transformed_train.parquet``.

    Parameters
    ----------
    df:
        Transformed feature DataFrame (output of :func:`transform_features`).
    output_dir:
        Destination directory.

    Returns
    -------
    Path
        Absolute path to the Parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "transformed_train.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(
        "Transformed features saved to %s (%d rows × %d columns)",
        out_path, *df.shape,
    )
    return out_path


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_feature_engineering(
    bank_sample_path: Path = BANK_SAMPLE_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Path]:
    """End-to-end feature engineering pipeline.

    Steps
    -----
    1. Load bank sample CSV
    2. Validate required columns
    3. Encode target column
    4. Build and fit preprocessing pipeline
    5. Transform training features
    6. Persist: preprocessor.pkl, feature_schema.json, transformed_train.parquet

    Parameters
    ----------
    bank_sample_path:
        Path to ``bank_sample.csv`` (output of phase-1 ingestion).
    artifacts_dir:
        Directory for output artifacts.

    Returns
    -------
    dict
        Paths to generated artifacts keyed by artifact name.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_bank_sample(bank_sample_path)

    # 2. Validate that all required columns are present
    validate_feature_columns(df, ALL_FEATURE_COLUMNS + [TARGET_COLUMN])

    # 3. Encode target (not passed through the pipeline — kept separate)
    y = encode_target(df[TARGET_COLUMN])
    logger.info(
        "Target encoded: %d positive (yes), %d negative (no)",
        int(y.sum()), int((y == 0).sum()),
    )

    # 4. Build schema
    schema = build_feature_schema(NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN)

    # 5. Build + fit pipeline
    pipeline = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    pipeline = fit_pipeline(df, pipeline, ALL_FEATURE_COLUMNS)

    # 6. Transform
    X_transformed = transform_features(df, pipeline, ALL_FEATURE_COLUMNS)

    # 7. Persist all artifacts
    schema_path = save_feature_schema(schema, artifacts_dir)
    preprocessor_path = save_preprocessor(pipeline, artifacts_dir)
    parquet_path = save_transformed_parquet(X_transformed, artifacts_dir)

    logger.info("Feature engineering complete. Artifacts: %s", artifacts_dir)

    return {
        "feature_schema": schema_path,
        "preprocessor": preprocessor_path,
        "transformed_train": parquet_path,
    }


if __name__ == "__main__":
    run_feature_engineering()
