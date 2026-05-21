"""
tests/test_training.py
-----------------------
Unit tests for Phase 3: training, evaluation, and promotion gate.

Covers:
  * compute_metrics — all mandatory metric keys, valid ranges
  * compare_metrics — delta computation
  * build_baseline_model / build_improved_model — type and param checks
  * train_model — fits and returns latency
  * save_model / load_model — round-trip persistence
  * check_promotion_gate — pass, block (PR-AUC), block (F1), block (both)
  * save_gate_log / load — appends correctly, JSON structure
  * evaluate_promotion_gate — integration (gate + log in one call)
  * print_gate_report — smoke test (no exceptions)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.training.evaluate import (
    compare_metrics,
    compute_metrics,
    print_evaluation_report,
)
from src.training.promotion_gate import (
    check_promotion_gate,
    evaluate_promotion_gate,
    print_gate_report,
    save_gate_log,
)
from src.training.train import (
    build_baseline_model,
    build_improved_model,
    load_model,
    save_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_binary_data(
    n: int = 200,
    n_features: int = 10,
    positive_rate: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, X_test, y_train, y_test) with controlled class balance."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Make target somewhat separable so PR-AUC is meaningful
    w = rng.standard_normal(n_features)
    logit = X @ w
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > (1 - positive_rate)).astype(int)

    split = int(n * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture(scope="module")
def trained_lr():
    """Return a fitted LogisticRegression on synthetic data."""
    X_train, X_test, y_train, y_test = _make_binary_data()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_all_required_keys(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics = compute_metrics(model, X_test, y_test)
        required = {
            "roc_auc", "pr_auc", "f1", "precision",
            "recall", "brier_score",
            "true_positives", "true_negatives",
            "false_positives", "false_negatives",
        }
        assert required.issubset(metrics.keys())

    def test_roc_auc_in_valid_range(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics = compute_metrics(model, X_test, y_test)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_pr_auc_in_valid_range(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics = compute_metrics(model, X_test, y_test)
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_f1_in_valid_range(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics = compute_metrics(model, X_test, y_test)
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_brier_score_in_valid_range(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics = compute_metrics(model, X_test, y_test)
        assert 0.0 <= metrics["brier_score"] <= 1.0

    def test_confusion_matrix_sums_to_n(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics = compute_metrics(model, X_test, y_test)
        total = (
            metrics["true_positives"] + metrics["true_negatives"]
            + metrics["false_positives"] + metrics["false_negatives"]
        )
        assert total == len(y_test)

    def test_metrics_are_floats_or_ints(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics = compute_metrics(model, X_test, y_test)
        for k, v in metrics.items():
            assert isinstance(v, (int, float)), f"{k} should be numeric, got {type(v)}"

    def test_metrics_are_json_serialisable(self, trained_lr):
        model, X_test, y_test = trained_lr
        metrics = compute_metrics(model, X_test, y_test)
        # Should not raise
        json.dumps(metrics)


# ---------------------------------------------------------------------------
# compare_metrics
# ---------------------------------------------------------------------------

class TestCompareMetrics:
    def test_delta_keys_present(self):
        base = {"roc_auc": 0.7, "pr_auc": 0.4, "f1": 0.5, "precision": 0.6, "recall": 0.45, "brier_score": 0.15}
        improved = {"roc_auc": 0.75, "pr_auc": 0.45, "f1": 0.52, "precision": 0.61, "recall": 0.46, "brier_score": 0.13}
        deltas = compare_metrics(base, improved)
        assert "delta_pr_auc" in deltas
        assert "delta_f1" in deltas
        assert "delta_roc_auc" in deltas

    def test_delta_values_correct(self):
        base = {"roc_auc": 0.70, "pr_auc": 0.40, "f1": 0.50, "precision": 0.60, "recall": 0.45, "brier_score": 0.15}
        improved = {"roc_auc": 0.75, "pr_auc": 0.45, "f1": 0.52, "precision": 0.63, "recall": 0.47, "brier_score": 0.13}
        deltas = compare_metrics(base, improved)
        assert abs(deltas["delta_pr_auc"] - 0.05) < 1e-5
        assert abs(deltas["delta_f1"] - 0.02) < 1e-5

    def test_negative_delta_when_improved_is_worse(self):
        base = {"roc_auc": 0.80, "pr_auc": 0.50, "f1": 0.60, "precision": 0.65, "recall": 0.55, "brier_score": 0.10}
        improved = {"roc_auc": 0.75, "pr_auc": 0.45, "f1": 0.55, "precision": 0.60, "recall": 0.50, "brier_score": 0.12}
        deltas = compare_metrics(base, improved)
        assert deltas["delta_pr_auc"] < 0


# ---------------------------------------------------------------------------
# build_baseline_model / build_improved_model
# ---------------------------------------------------------------------------

class TestModelBuilders:
    def test_build_baseline_returns_lr(self):
        from sklearn.linear_model import LogisticRegression
        model = build_baseline_model()
        assert isinstance(model, LogisticRegression)

    def test_build_improved_returns_xgb(self):
        from xgboost import XGBClassifier
        model = build_improved_model()
        assert isinstance(model, XGBClassifier)

    def test_baseline_deterministic(self):
        m1 = build_baseline_model(seed=42)
        m2 = build_baseline_model(seed=42)
        assert m1.get_params() == m2.get_params()

    def test_improved_deterministic(self):
        m1 = build_improved_model(seed=42)
        m2 = build_improved_model(seed=42)
        assert m1.get_params() == m2.get_params()

    def test_baseline_has_class_weight_balanced(self):
        model = build_baseline_model()
        assert model.class_weight == "balanced"


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_returns_fitted_model_and_latency(self):
        X_train, X_test, y_train, _ = _make_binary_data(n=100)
        model = build_baseline_model()
        fitted, latency = train_model(model, X_train, y_train)
        assert latency > 0
        # Fitted model can predict
        preds = fitted.predict(X_test)
        assert len(preds) == len(X_test)

    def test_latency_is_positive_float(self):
        X_train, _, y_train, _ = _make_binary_data(n=100)
        model = build_baseline_model()
        _, latency = train_model(model, X_train, y_train)
        assert isinstance(latency, float)
        assert latency > 0


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_save_creates_pkl_file(self, tmp_path):
        X_train, _, y_train, _ = _make_binary_data(n=100)
        model = build_baseline_model()
        model.fit(X_train, y_train)
        path = save_model(model, "test_model", tmp_path)
        assert path.exists()
        assert path.suffix == ".pkl"
        assert path.name == "test_model.pkl"

    def test_load_returns_fitted_model(self, tmp_path):
        X_train, X_test, y_train, _ = _make_binary_data(n=100)
        model = build_baseline_model()
        model.fit(X_train, y_train)
        save_model(model, "roundtrip_model", tmp_path)
        loaded = load_model("roundtrip_model", tmp_path)
        preds = loaded.predict(X_test)
        assert len(preds) == len(X_test)

    def test_load_produces_identical_predictions(self, tmp_path):
        X_train, X_test, y_train, _ = _make_binary_data(n=100)
        model = build_baseline_model()
        model.fit(X_train, y_train)
        preds_before = model.predict_proba(X_test)
        save_model(model, "idempotent_model", tmp_path)
        loaded = load_model("idempotent_model", tmp_path)
        preds_after = loaded.predict_proba(X_test)
        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model", tmp_path)


# ---------------------------------------------------------------------------
# check_promotion_gate
# ---------------------------------------------------------------------------

class TestCheckPromotionGate:
    def _base(self) -> dict[str, float]:
        return {"pr_auc": 0.40, "f1": 0.55, "roc_auc": 0.75, "brier_score": 0.15}

    def test_gate_passes_when_pr_auc_and_f1_ok(self):
        base = self._base()
        improved = {**base, "pr_auc": base["pr_auc"] + 0.05, "f1": base["f1"] + 0.01}
        result = check_promotion_gate(base, improved)
        assert result["promoted"] is True
        assert result["rejection_reasons"] == []

    def test_gate_blocked_pr_auc_insufficient(self):
        base = self._base()
        # Only +0.01 improvement — below the +0.03 threshold
        improved = {**base, "pr_auc": base["pr_auc"] + 0.01}
        result = check_promotion_gate(base, improved)
        assert result["promoted"] is False
        assert result["pr_auc_check"] is False
        assert len(result["rejection_reasons"]) >= 1

    def test_gate_blocked_f1_degradation_exceeded(self):
        base = self._base()
        # PR-AUC ok, but F1 degrades by 0.03 > 0.02 allowed
        improved = {**base, "pr_auc": base["pr_auc"] + 0.05, "f1": base["f1"] - 0.03}
        result = check_promotion_gate(base, improved)
        assert result["promoted"] is False
        assert result["f1_check"] is False

    def test_gate_blocked_both_criteria_fail(self):
        base = self._base()
        improved = {**base, "pr_auc": base["pr_auc"] + 0.01, "f1": base["f1"] - 0.05}
        result = check_promotion_gate(base, improved)
        assert result["promoted"] is False
        assert len(result["rejection_reasons"]) == 2

    def test_gate_result_contains_required_keys(self):
        base = self._base()
        improved = {**base, "pr_auc": base["pr_auc"] + 0.05}
        result = check_promotion_gate(base, improved)
        for key in ["promoted", "pr_auc_delta", "f1_delta",
                    "pr_auc_check", "f1_check", "rejection_reasons", "gate_config"]:
            assert key in result

    def test_pr_auc_delta_computed_correctly(self):
        base = self._base()
        improved = {**base, "pr_auc": 0.46}  # delta = 0.06
        result = check_promotion_gate(base, improved)
        assert abs(result["pr_auc_delta"] - 0.06) < 1e-5

    def test_custom_thresholds_respected(self):
        base = self._base()
        improved = {**base, "pr_auc": base["pr_auc"] + 0.02}
        # With a lenient threshold of 0.01, this should pass
        result = check_promotion_gate(base, improved, pr_auc_min_improvement=0.01)
        assert result["promoted"] is True

    def test_exactly_at_pr_auc_threshold_passes(self):
        base = self._base()
        improved = {**base, "pr_auc": base["pr_auc"] + 0.03}  # exactly at threshold
        result = check_promotion_gate(base, improved)
        assert result["pr_auc_check"] is True

    def test_exactly_at_f1_degradation_limit_passes(self):
        base = self._base()
        improved = {**base, "pr_auc": base["pr_auc"] + 0.05, "f1": base["f1"] - 0.02}
        result = check_promotion_gate(base, improved)
        assert result["f1_check"] is True


# ---------------------------------------------------------------------------
# save_gate_log
# ---------------------------------------------------------------------------

class TestSaveGateLog:
    def _make_gate_result(self, promoted: bool = True) -> dict:
        return {
            "promoted": promoted,
            "pr_auc_delta": 0.05,
            "f1_delta": 0.01,
            "pr_auc_check": True,
            "f1_check": True,
            "rejection_reasons": [],
            "gate_config": {"pr_auc_min_improvement": 0.03, "f1_max_degradation": 0.02},
            "baseline_metrics": {"pr_auc": 0.40, "f1": 0.55, "roc_auc": 0.75},
            "improved_metrics": {"pr_auc": 0.45, "f1": 0.56, "roc_auc": 0.78},
        }

    def test_creates_json_file(self, tmp_path):
        result = self._make_gate_result()
        path = save_gate_log(result, tmp_path / "gate_log.json")
        assert path.exists()

    def test_log_is_valid_json_list(self, tmp_path):
        result = self._make_gate_result()
        path = tmp_path / "gate_log.json"
        save_gate_log(result, path)
        with open(path) as f:
            log = json.load(f)
        assert isinstance(log, list)
        assert len(log) == 1

    def test_log_entry_has_timestamp(self, tmp_path):
        result = self._make_gate_result()
        path = tmp_path / "gate_log.json"
        save_gate_log(result, path)
        with open(path) as f:
            log = json.load(f)
        assert "timestamp" in log[0]

    def test_log_appends_across_calls(self, tmp_path):
        path = tmp_path / "gate_log.json"
        save_gate_log(self._make_gate_result(True), path)
        save_gate_log(self._make_gate_result(False), path)
        with open(path) as f:
            log = json.load(f)
        assert len(log) == 2
        assert log[0]["promoted"] is True
        assert log[1]["promoted"] is False

    def test_run_metadata_included(self, tmp_path):
        result = self._make_gate_result()
        path = tmp_path / "gate_log.json"
        save_gate_log(result, path, run_metadata={"baseline_run_id": "abc123"})
        with open(path) as f:
            log = json.load(f)
        assert log[0]["run_metadata"]["baseline_run_id"] == "abc123"


# ---------------------------------------------------------------------------
# evaluate_promotion_gate (integration)
# ---------------------------------------------------------------------------

class TestEvaluatePromotionGate:
    def test_integration_pass_and_log(self, tmp_path):
        base = {"pr_auc": 0.40, "f1": 0.55, "roc_auc": 0.75, "brier_score": 0.15}
        improved = {**base, "pr_auc": 0.45, "f1": 0.56}
        log_path = tmp_path / "gate_log.json"
        result = evaluate_promotion_gate(base, improved, gate_log_path=log_path)
        assert result["promoted"] is True
        assert log_path.exists()

    def test_integration_block_and_log(self, tmp_path):
        base = {"pr_auc": 0.40, "f1": 0.55, "roc_auc": 0.75, "brier_score": 0.15}
        improved = {**base, "pr_auc": 0.41}  # insufficient improvement
        log_path = tmp_path / "gate_log.json"
        result = evaluate_promotion_gate(base, improved, gate_log_path=log_path)
        assert result["promoted"] is False
        with open(log_path) as f:
            log = json.load(f)
        assert log[0]["promoted"] is False


# ---------------------------------------------------------------------------
# print helpers (smoke tests)
# ---------------------------------------------------------------------------

class TestPrintHelpers:
    def test_print_evaluation_report_no_exception(self, capsys):
        metrics = {
            "roc_auc": 0.80, "pr_auc": 0.50, "f1": 0.60,
            "precision": 0.65, "recall": 0.55, "brier_score": 0.10,
            "true_positives": 30, "true_negatives": 120,
            "false_positives": 10, "false_negatives": 20,
        }
        print_evaluation_report("test_model", metrics)
        captured = capsys.readouterr()
        assert "test_model" in captured.out

    def test_print_gate_report_promoted(self, capsys):
        result = {
            "promoted": True,
            "pr_auc_delta": 0.05, "f1_delta": 0.01,
            "pr_auc_check": True, "f1_check": True,
            "rejection_reasons": [],
            "gate_config": {"pr_auc_min_improvement": 0.03, "f1_max_degradation": 0.02},
        }
        print_gate_report(result)
        captured = capsys.readouterr()
        assert "PROMOTED" in captured.out

    def test_print_gate_report_blocked(self, capsys):
        result = {
            "promoted": False,
            "pr_auc_delta": 0.01, "f1_delta": -0.03,
            "pr_auc_check": False, "f1_check": False,
            "rejection_reasons": ["PR-AUC too low", "F1 degraded too much"],
            "gate_config": {"pr_auc_min_improvement": 0.03, "f1_max_degradation": 0.02},
        }
        print_gate_report(result)
        captured = capsys.readouterr()
        assert "BLOCKED" in captured.out
        assert "PR-AUC too low" in captured.out
