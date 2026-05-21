"""
tests/test_features.py
-----------------------
Unit tests for src/data_pipeline/features.py

Covers:
  * feature schema construction and persistence
  * pipeline construction (structure checks)
  * fit / transform round-trip
  * target encoding
  * column validation
  * artifact save / load (preprocessor + parquet)
  * inference-safety: unknown categories handled without error
  * no train-serving skew: same pipeline object produces identical output
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_MAP,
    build_feature_schema,
    build_preprocessing_pipeline,
    encode_target,
    fit_pipeline,
    load_feature_schema,
    load_preprocessor,
    save_feature_schema,
    save_preprocessor,
    save_transformed_parquet,
    transform_features,
    validate_feature_columns,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_bank_df(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Return a minimal synthetic bank marketing DataFrame."""
    rng = np.random.default_rng(seed)
    jobs = ["admin.", "blue-collar", "technician", "services", "management"]
    marital = ["married", "single", "divorced"]
    education = ["university.degree", "high.school", "basic.9y", "professional.course"]
    contact = ["cellular", "telephone"]
    month = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    day_of_week = ["mon", "tue", "wed", "thu", "fri"]
    poutcome = ["nonexistent", "failure", "success"]
    yn = ["yes", "no"]

    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "job": rng.choice(jobs, n),
            "marital": rng.choice(marital, n),
            "education": rng.choice(education, n),
            "default": rng.choice(yn, n),
            "housing": rng.choice(yn, n),
            "loan": rng.choice(yn, n),
            "contact": rng.choice(contact, n),
            "month": rng.choice(month, n),
            "day_of_week": rng.choice(day_of_week, n),
            "duration": rng.integers(0, 600, n),
            "campaign": rng.integers(1, 10, n),
            "pdays": rng.integers(0, 999, n),
            "previous": rng.integers(0, 5, n),
            "poutcome": rng.choice(poutcome, n),
            "emp_var_rate": rng.uniform(-3.5, 1.5, n),
            "cons_price_idx": rng.uniform(92.0, 95.0, n),
            "cons_conf_idx": rng.uniform(-50.0, -26.0, n),
            "euribor3m": rng.uniform(0.5, 5.0, n),
            "nr_employed": rng.uniform(4900.0, 5250.0, n),
            "y": rng.choice(yn, n),
        }
    )


# ---------------------------------------------------------------------------
# build_feature_schema
# ---------------------------------------------------------------------------

class TestBuildFeatureSchema:
    def test_schema_keys_present(self):
        schema = build_feature_schema(NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN)
        required_keys = {
            "version", "target_column", "numeric_features",
            "categorical_features", "all_feature_columns",
        }
        assert required_keys.issubset(schema.keys())

    def test_schema_target(self):
        schema = build_feature_schema(NUMERIC_FEATURES, CATEGORICAL_FEATURES, "y")
        assert schema["target_column"] == "y"

    def test_schema_numeric_features(self):
        schema = build_feature_schema(["age", "duration"], [], "y")
        assert schema["numeric_features"] == ["age", "duration"]
        assert schema["categorical_features"] == []

    def test_schema_all_feature_columns_concatenated(self):
        schema = build_feature_schema(["age"], ["job"], "y")
        assert schema["all_feature_columns"] == ["age", "job"]

    def test_schema_is_json_serialisable(self):
        schema = build_feature_schema(NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN)
        serialised = json.dumps(schema)
        assert isinstance(serialised, str)


# ---------------------------------------------------------------------------
# save_feature_schema / load_feature_schema
# ---------------------------------------------------------------------------

class TestFeatureSchemaPersistence:
    def test_roundtrip(self, tmp_path):
        schema = build_feature_schema(NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN)
        save_feature_schema(schema, tmp_path)
        loaded = load_feature_schema(tmp_path)
        assert loaded == schema

    def test_file_created(self, tmp_path):
        schema = build_feature_schema([], [], "y")
        path = save_feature_schema(schema, tmp_path)
        assert path.exists()
        assert path.name == "feature_schema.json"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_feature_schema(tmp_path)


# ---------------------------------------------------------------------------
# build_preprocessing_pipeline
# ---------------------------------------------------------------------------

class TestBuildPreprocessingPipeline:
    def test_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        assert isinstance(p, Pipeline)

    def test_pipeline_has_preprocessor_step(self):
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        assert "preprocessor" in p.named_steps

    def test_column_transformer_has_both_branches(self):
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        ct = p.named_steps["preprocessor"]
        names = [name for name, _, _ in ct.transformers]
        assert "numeric" in names
        assert "categorical" in names

    def test_numeric_branch_has_scaler(self):
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        ct = p.named_steps["preprocessor"]
        transformer_map = {name: t for name, t, _ in ct.transformers}
        numeric_pipe = transformer_map["numeric"]
        step_names = [name for name, _ in numeric_pipe.steps]
        assert "standard_scaler" in step_names

    def test_categorical_branch_has_ohe(self):
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        ct = p.named_steps["preprocessor"]
        transformer_map = {name: t for name, t, _ in ct.transformers}
        cat_pipe = transformer_map["categorical"]
        step_names = [name for name, _ in cat_pipe.steps]
        assert "onehot_encoder" in step_names


# ---------------------------------------------------------------------------
# validate_feature_columns
# ---------------------------------------------------------------------------

class TestValidateFeatureColumns:
    def test_pass_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        validate_feature_columns(df, ["a", "b"])  # should not raise

    def test_fail_missing_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required feature columns"):
            validate_feature_columns(df, ["a", "b"])

    def test_empty_required_list(self):
        df = pd.DataFrame({"a": [1]})
        validate_feature_columns(df, [])  # should not raise


# ---------------------------------------------------------------------------
# encode_target
# ---------------------------------------------------------------------------

class TestEncodeTarget:
    def test_yes_maps_to_1(self):
        s = pd.Series(["yes", "no", "yes"])
        encoded = encode_target(s)
        assert list(encoded) == [1, 0, 1]

    def test_no_maps_to_0(self):
        s = pd.Series(["no"])
        encoded = encode_target(s)
        assert int(encoded.iloc[0]) == 0

    def test_unexpected_value_raises(self):
        s = pd.Series(["yes", "maybe"])
        with pytest.raises(ValueError, match="Unexpected target values"):
            encode_target(s)

    def test_returns_integer_dtype(self):
        s = pd.Series(["yes", "no"])
        encoded = encode_target(s)
        assert pd.api.types.is_integer_dtype(encoded)

    def test_custom_map(self):
        s = pd.Series(["pos", "neg"])
        encoded = encode_target(s, target_map={"pos": 1, "neg": 0})
        assert list(encoded) == [1, 0]


# ---------------------------------------------------------------------------
# fit_pipeline / transform_features
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_fit_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        df = _make_bank_df()
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fitted = fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        assert isinstance(fitted, Pipeline)

    def test_transform_returns_dataframe(self):
        df = _make_bank_df()
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        X = transform_features(df, p, ALL_FEATURE_COLUMNS)
        assert isinstance(X, pd.DataFrame)

    def test_transform_row_count(self):
        df = _make_bank_df(n=30)
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        X = transform_features(df, p, ALL_FEATURE_COLUMNS)
        assert len(X) == 30

    def test_transform_no_nulls(self):
        df = _make_bank_df()
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        X = transform_features(df, p, ALL_FEATURE_COLUMNS)
        assert not X.isnull().any().any()

    def test_transform_column_names_deterministic(self):
        """Same data + same pipeline → same column names."""
        df = _make_bank_df(seed=0)
        p1 = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p1, ALL_FEATURE_COLUMNS)
        cols1 = transform_features(df, p1, ALL_FEATURE_COLUMNS).columns.tolist()

        p2 = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p2, ALL_FEATURE_COLUMNS)
        cols2 = transform_features(df, p2, ALL_FEATURE_COLUMNS).columns.tolist()

        assert cols1 == cols2

    def test_transform_values_deterministic(self):
        """Same data + same pipeline → identical transformed values."""
        df = _make_bank_df(seed=7)
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        X1 = transform_features(df, p, ALL_FEATURE_COLUMNS)
        X2 = transform_features(df, p, ALL_FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(X1, X2)

    def test_fit_missing_column_raises(self):
        df = _make_bank_df().drop(columns=["age"])
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        with pytest.raises(ValueError, match="Missing required feature columns"):
            fit_pipeline(df, p, ALL_FEATURE_COLUMNS)

    def test_numeric_columns_scaled(self):
        """After StandardScaler, numeric features should have near-zero mean."""
        df = _make_bank_df(n=200, seed=1)
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        X = transform_features(df, p, ALL_FEATURE_COLUMNS)
        # Numeric output columns are prefixed with 'numeric__'
        numeric_out_cols = [c for c in X.columns if c.startswith("numeric__")]
        assert len(numeric_out_cols) > 0
        for col in numeric_out_cols:
            mean = X[col].mean()
            assert abs(mean) < 0.5, f"Column {col} mean {mean:.4f} not near zero"


# ---------------------------------------------------------------------------
# Inference-safety: unknown categories
# ---------------------------------------------------------------------------

class TestInferenceSafety:
    def test_unknown_category_does_not_raise(self):
        """OneHotEncoder with handle_unknown='ignore' must not raise on new
        category values at inference time — preventing train-serving skew."""
        train_df = _make_bank_df(n=50, seed=10)
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(train_df, p, ALL_FEATURE_COLUMNS)

        # Inject an unseen job category at inference time
        infer_df = _make_bank_df(n=5, seed=99)
        infer_df["job"] = "UNSEEN_JOB_CATEGORY"

        # Should NOT raise
        X = transform_features(infer_df, p, ALL_FEATURE_COLUMNS)
        assert len(X) == 5

    def test_output_shape_consistent_with_unknown_category(self):
        """Column count must be identical whether or not unknown categories
        appear — critical for serving-layer correctness."""
        train_df = _make_bank_df(n=50, seed=10)
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(train_df, p, ALL_FEATURE_COLUMNS)

        known_X = transform_features(train_df, p, ALL_FEATURE_COLUMNS)

        infer_df = _make_bank_df(n=5, seed=99)
        infer_df["job"] = "UNSEEN_JOB_CATEGORY"
        unknown_X = transform_features(infer_df, p, ALL_FEATURE_COLUMNS)

        assert known_X.shape[1] == unknown_X.shape[1]


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

class TestArtifactPersistence:
    def test_save_load_preprocessor_roundtrip(self, tmp_path):
        df = _make_bank_df(n=30)
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        X_before = transform_features(df, p, ALL_FEATURE_COLUMNS)

        save_preprocessor(p, tmp_path)
        p_loaded = load_preprocessor(tmp_path)
        X_after = transform_features(df, p_loaded, ALL_FEATURE_COLUMNS)

        pd.testing.assert_frame_equal(X_before, X_after)

    def test_save_preprocessor_creates_file(self, tmp_path):
        df = _make_bank_df()
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        path = save_preprocessor(p, tmp_path)
        assert path.exists()
        assert path.name == "preprocessor.pkl"

    def test_load_preprocessor_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_preprocessor(tmp_path)

    def test_save_transformed_parquet_creates_file(self, tmp_path):
        df = _make_bank_df(n=10)
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        X = transform_features(df, p, ALL_FEATURE_COLUMNS)
        path = save_transformed_parquet(X, tmp_path)
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_parquet_roundtrip_preserves_data(self, tmp_path):
        df = _make_bank_df(n=20)
        p = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        fit_pipeline(df, p, ALL_FEATURE_COLUMNS)
        X = transform_features(df, p, ALL_FEATURE_COLUMNS)
        save_transformed_parquet(X, tmp_path)

        X_loaded = pd.read_parquet(tmp_path / "transformed_train.parquet")
        pd.testing.assert_frame_equal(X, X_loaded)
