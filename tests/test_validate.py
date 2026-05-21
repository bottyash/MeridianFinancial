"""
tests/test_validate.py
-----------------------
Unit tests for src/data_pipeline/validate.py

Covers every public validator function with:
  * happy-path (all-valid data → PASS)
  * failure cases (deliberate violations → FAIL with correct failure count)
  * edge cases (missing column, empty DataFrame)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_pipeline.validate import (
    ValidationResult,
    ValidationSummary,
    validate_age_range,
    validate_complaint_dates,
    validate_complaints_dataset,
    validate_bank_dataset,
    validate_duration_values,
    validate_duplicates,
    validate_nonempty_complaints,
    validate_required_columns,
    validate_target_column,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bank_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid bank marketing DataFrame."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 65],
            "job": ["admin.", "blue-collar", "technician", "services", "management"],
            "marital": ["married"] * n,
            "education": ["university.degree"] * n,
            "default": ["no"] * n,
            "housing": ["yes"] * n,
            "loan": ["no"] * n,
            "contact": ["cellular"] * n,
            "month": ["may"] * n,
            "day_of_week": ["mon"] * n,
            "duration": [100, 200, 150, 300, 50],
            "campaign": [1] * n,
            "pdays": [999] * n,
            "previous": [0] * n,
            "poutcome": ["nonexistent"] * n,
            "emp_var_rate": [-1.8] * n,
            "cons_price_idx": [93.994] * n,
            "cons_conf_idx": [-36.4] * n,
            "euribor3m": [4.857] * n,
            "nr_employed": [5191.0] * n,
            "y": ["no", "yes", "no", "no", "yes"],
        }
    )


def _make_complaints_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid complaints DataFrame."""
    return pd.DataFrame(
        {
            "complaint_id": [str(i) for i in range(1000, 1000 + n)],
            "product": ["Mortgage"] * n,
            "issue": ["Loan modification"] * n,
            "company": ["Acme Bank"] * n,
            "date_received": ["2023-01-15"] * n,
            "complaint_narrative": [f"Complaint text {i}" for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# validate_required_columns
# ---------------------------------------------------------------------------

class TestValidateRequiredColumns:
    def test_pass_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = validate_required_columns(df, ["a", "b", "c"])
        assert result.passed is True
        assert result.failures == 0
        assert result.details["missing_columns"] == []

    def test_fail_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        result = validate_required_columns(df, ["a", "b", "c"])
        assert result.passed is False
        assert result.failures == 2
        assert set(result.details["missing_columns"]) == {"b", "c"}

    def test_empty_required_list(self):
        df = pd.DataFrame({"a": [1]})
        result = validate_required_columns(df, [])
        assert result.passed is True

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = validate_required_columns(df, ["x"])
        assert result.passed is False


# ---------------------------------------------------------------------------
# validate_duplicates
# ---------------------------------------------------------------------------

class TestValidateDuplicates:
    def test_pass_unique_ids(self):
        df = pd.DataFrame({"id": ["A", "B", "C"]})
        result = validate_duplicates(df, "id")
        assert result.passed is True
        assert result.failures == 0

    def test_fail_duplicate_ids(self):
        df = pd.DataFrame({"id": ["A", "B", "A"]})
        result = validate_duplicates(df, "id")
        assert result.passed is False
        assert result.failures == 1

    def test_fail_column_missing(self):
        df = pd.DataFrame({"other": [1]})
        result = validate_duplicates(df, "id")
        assert result.passed is False
        assert "not found" in result.details["error"]

    def test_multiple_duplicates(self):
        df = pd.DataFrame({"id": ["X", "X", "X", "Y"]})
        result = validate_duplicates(df, "id")
        assert result.failures == 2  # 2 non-first occurrences


# ---------------------------------------------------------------------------
# validate_age_range
# ---------------------------------------------------------------------------

class TestValidateAgeRange:
    def test_pass_valid_ages(self):
        df = pd.DataFrame({"age": [18, 40, 100]})
        result = validate_age_range(df)
        assert result.passed is True

    def test_fail_too_young(self):
        df = pd.DataFrame({"age": [10, 25, 35]})
        result = validate_age_range(df)
        assert result.passed is False
        assert result.failures >= 1

    def test_fail_too_old(self):
        df = pd.DataFrame({"age": [25, 101]})
        result = validate_age_range(df)
        assert result.passed is False

    def test_fail_null_age(self):
        df = pd.DataFrame({"age": [25, None, 40]})
        result = validate_age_range(df)
        assert result.passed is False
        assert result.failures == 1

    def test_fail_column_missing(self):
        df = pd.DataFrame({"other": [1]})
        result = validate_age_range(df)
        assert result.passed is False

    def test_boundary_values(self):
        df = pd.DataFrame({"age": [18, 100]})
        result = validate_age_range(df)
        assert result.passed is True


# ---------------------------------------------------------------------------
# validate_target_column
# ---------------------------------------------------------------------------

class TestValidateTargetColumn:
    def test_pass_valid_values(self):
        df = pd.DataFrame({"y": ["yes", "no", "no", "yes"]})
        result = validate_target_column(df, "y", frozenset({"yes", "no"}))
        assert result.passed is True

    def test_fail_invalid_value(self):
        df = pd.DataFrame({"y": ["yes", "maybe"]})
        result = validate_target_column(df, "y", frozenset({"yes", "no"}))
        assert result.passed is False
        assert result.failures >= 1

    def test_fail_nulls(self):
        df = pd.DataFrame({"y": ["yes", None]})
        result = validate_target_column(df, "y")
        assert result.passed is False

    def test_fail_column_missing(self):
        df = pd.DataFrame({"other": [1]})
        result = validate_target_column(df, "y")
        assert result.passed is False

    def test_no_value_constraint(self):
        df = pd.DataFrame({"y": ["yes", "no", "maybe"]})
        result = validate_target_column(df, "y", valid_values=None)
        assert result.passed is True  # no constraint, no nulls → pass


# ---------------------------------------------------------------------------
# validate_nonempty_complaints
# ---------------------------------------------------------------------------

class TestValidateNonemptyComplaints:
    def test_pass_all_filled(self):
        df = pd.DataFrame({"complaint_narrative": ["Text A", "Text B"]})
        result = validate_nonempty_complaints(df)
        assert result.passed is True

    def test_fail_null_narrative(self):
        df = pd.DataFrame({"complaint_narrative": ["Text", None]})
        result = validate_nonempty_complaints(df)
        assert result.passed is False
        assert result.failures == 1

    def test_fail_whitespace_narrative(self):
        df = pd.DataFrame({"complaint_narrative": ["Text", "   "]})
        result = validate_nonempty_complaints(df)
        assert result.passed is False
        assert result.failures == 1

    def test_fail_column_missing(self):
        df = pd.DataFrame({"other": [1]})
        result = validate_nonempty_complaints(df)
        assert result.passed is False


# ---------------------------------------------------------------------------
# validate_duration_values
# ---------------------------------------------------------------------------

class TestValidateDurationValues:
    def test_pass_valid_durations(self):
        df = pd.DataFrame({"duration": [0, 100, 300, 600]})
        result = validate_duration_values(df)
        assert result.passed is True

    def test_fail_negative_duration(self):
        df = pd.DataFrame({"duration": [100, -1, 200]})
        result = validate_duration_values(df)
        assert result.passed is False
        assert result.failures == 1

    def test_fail_null_duration(self):
        df = pd.DataFrame({"duration": [100, None]})
        result = validate_duration_values(df)
        assert result.passed is False

    def test_fail_column_missing(self):
        df = pd.DataFrame({"other": [1]})
        result = validate_duration_values(df)
        assert result.passed is False

    def test_zero_duration_allowed(self):
        df = pd.DataFrame({"duration": [0]})
        result = validate_duration_values(df)
        assert result.passed is True


# ---------------------------------------------------------------------------
# validate_complaint_dates
# ---------------------------------------------------------------------------

class TestValidateComplaintDates:
    def test_pass_valid_dates(self):
        df = pd.DataFrame({"date_received": ["2023-01-15", "2022-12-31", "2021-06-01"]})
        result = validate_complaint_dates(df)  # default format=%Y-%m-%d
        assert result.passed is True

    def test_fail_unparseable_date(self):
        df = pd.DataFrame({"date_received": ["2023-01-15", "not-a-date"]})
        result = validate_complaint_dates(df)
        assert result.passed is False
        assert result.failures == 1

    def test_fail_null_date(self):
        df = pd.DataFrame({"date_received": ["2023-01-15", None]})
        result = validate_complaint_dates(df)
        assert result.passed is False

    def test_fail_column_missing(self):
        df = pd.DataFrame({"other": [1]})
        result = validate_complaint_dates(df)
        assert result.passed is False

    def test_all_unparseable(self):
        df = pd.DataFrame({"date_received": ["bad", "also-bad"]})
        result = validate_complaint_dates(df)
        assert result.failures == 2


# ---------------------------------------------------------------------------
# validate_bank_dataset (orchestration)
# ---------------------------------------------------------------------------

class TestValidateBankDataset:
    def test_valid_bank_df_passes(self):
        df = _make_bank_df()
        summary = validate_bank_dataset(df)
        assert isinstance(summary, ValidationSummary)
        assert summary.dataset == "bank_marketing"
        assert summary.passed is True
        assert summary.failure_count == 0

    def test_invalid_bank_df_fails(self):
        df = _make_bank_df()
        df.loc[0, "age"] = 200  # out of range
        df.loc[1, "y"] = None   # null target
        summary = validate_bank_dataset(df)
        assert summary.passed is False
        assert summary.failure_count >= 1

    def test_returns_correct_check_names(self):
        df = _make_bank_df()
        summary = validate_bank_dataset(df)
        check_names = {r.check for r in summary.results}
        assert "required_columns" in check_names
        assert "age_range" in check_names
        assert "target_column" in check_names
        assert "duration_values" in check_names


# ---------------------------------------------------------------------------
# validate_complaints_dataset (orchestration)
# ---------------------------------------------------------------------------

class TestValidateComplaintsDataset:
    def test_valid_complaints_df_passes(self):
        df = _make_complaints_df()
        summary = validate_complaints_dataset(df)
        assert isinstance(summary, ValidationSummary)
        assert summary.dataset == "cfpb_complaints"
        assert summary.passed is True

    def test_invalid_complaints_df_fails(self):
        df = _make_complaints_df()
        df.loc[0, "complaint_narrative"] = ""  # empty narrative
        df.loc[1, "complaint_id"] = df.loc[0, "complaint_id"]  # duplicate
        summary = validate_complaints_dataset(df)
        assert summary.passed is False

    def test_returns_correct_check_names(self):
        df = _make_complaints_df()
        summary = validate_complaints_dataset(df)
        check_names = {r.check for r in summary.results}
        assert "required_columns" in check_names
        assert "no_duplicate_ids" in check_names
        assert "nonempty_complaints" in check_names
        assert "complaint_dates" in check_names


# ---------------------------------------------------------------------------
# ValidationResult / ValidationSummary helpers
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_str_pass(self):
        r = ValidationResult(check="foo", passed=True, failures=0)
        assert "[PASS]" in str(r)

    def test_str_fail(self):
        r = ValidationResult(check="bar", passed=False, failures=3)
        assert "[FAIL]" in str(r)
        assert "failures=3" in str(r)


class TestValidationSummary:
    def test_passed_property(self):
        results = [
            ValidationResult("a", True),
            ValidationResult("b", True),
        ]
        s = ValidationSummary("ds", results)
        assert s.passed is True

    def test_failed_property(self):
        results = [
            ValidationResult("a", True),
            ValidationResult("b", False),
        ]
        s = ValidationSummary("ds", results)
        assert s.passed is False
        assert s.failure_count == 1
