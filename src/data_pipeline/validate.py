"""
src/data_pipeline/validate.py
------------------------------
Reusable, structured validation for the Meridian Financial data pipeline.

Each validator is a pure function:
  * accepts a DataFrame + optional parameters
  * returns a ``ValidationResult`` dataclass (passed, failures, details)

A top-level ``run_validation()`` orchestrates both datasets and prints
a human-readable summary.

Business rules enforced (per phase-1 spec):
  1. No duplicate IDs
  2. Valid age range  [18, 100]
  3. Required columns present
  4. Target column exists and has valid values
  5. Complaint narrative non-empty
  6. Valid campaign duration  (duration >= 0)
  7. Complaint dates parse correctly
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("meridian.validate")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Structured outcome of a single validation check.

    Attributes
    ----------
    check:
        Short name of the validation rule.
    passed:
        ``True`` if the check succeeded.
    failures:
        Number of offending rows / issues found.
    details:
        Free-form extra context (row counts, bad values, etc.).
    """

    check: str
    passed: bool
    failures: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.check} | failures={self.failures} | {self.details}"


@dataclass
class ValidationSummary:
    """Aggregate of all validation results for one dataset."""

    dataset: str
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """``True`` only if every check passed."""
        return all(r.passed for r in self.results)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def print_summary(self) -> None:
        """Print a structured summary to stdout."""
        border = "=" * 60
        print(border)
        print(f"  Validation summary — {self.dataset}")
        print(border)
        for r in self.results:
            print(f"  {r}")
        print(f"\n  Overall: {'PASSED' if self.passed else 'FAILED'} "
              f"({self.failure_count} checks failed out of {len(self.results)})")
        print(border)


# ---------------------------------------------------------------------------
# Reusable validator functions
# ---------------------------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    required: list[str],
) -> ValidationResult:
    """Check that all required columns are present in *df*.

    Parameters
    ----------
    df:
        Input dataframe.
    required:
        Column names that must exist.

    Returns
    -------
    ValidationResult
    """
    missing = [c for c in required if c not in df.columns]
    passed = len(missing) == 0
    result = ValidationResult(
        check="required_columns",
        passed=passed,
        failures=len(missing),
        details={"missing_columns": missing},
    )
    if not passed:
        logger.warning("required_columns FAIL — missing: %s", missing)
    else:
        logger.info("required_columns PASS")
    return result


def validate_duplicates(
    df: pd.DataFrame,
    id_column: str,
) -> ValidationResult:
    """Ensure no duplicate values exist in *id_column*.

    Parameters
    ----------
    df:
        Input dataframe.
    id_column:
        Column that should be unique.

    Returns
    -------
    ValidationResult
    """
    if id_column not in df.columns:
        return ValidationResult(
            check="no_duplicate_ids",
            passed=False,
            failures=1,
            details={"error": f"Column '{id_column}' not found"},
        )

    dup_count = int(df[id_column].duplicated().sum())
    passed = dup_count == 0
    result = ValidationResult(
        check="no_duplicate_ids",
        passed=passed,
        failures=dup_count,
        details={"id_column": id_column, "duplicate_count": dup_count},
    )
    if not passed:
        logger.warning("no_duplicate_ids FAIL — %d duplicates in '%s'", dup_count, id_column)
    else:
        logger.info("no_duplicate_ids PASS — column '%s'", id_column)
    return result


def validate_age_range(
    df: pd.DataFrame,
    age_column: str = "age",
    min_age: int = 18,
    max_age: int = 100,
) -> ValidationResult:
    """Validate that all values in *age_column* fall within [min_age, max_age].

    Parameters
    ----------
    df:
        Input dataframe.
    age_column:
        Name of the age column.
    min_age:
        Minimum valid age (inclusive).
    max_age:
        Maximum valid age (inclusive).

    Returns
    -------
    ValidationResult
    """
    if age_column not in df.columns:
        return ValidationResult(
            check="age_range",
            passed=False,
            failures=1,
            details={"error": f"Column '{age_column}' not found"},
        )

    ages = pd.to_numeric(df[age_column], errors="coerce")
    out_of_range = int(((ages < min_age) | (ages > max_age) | ages.isna()).sum())
    passed = out_of_range == 0
    result = ValidationResult(
        check="age_range",
        passed=passed,
        failures=out_of_range,
        details={"min_age": min_age, "max_age": max_age, "out_of_range_count": out_of_range},
    )
    if not passed:
        logger.warning("age_range FAIL — %d rows outside [%d, %d]", out_of_range, min_age, max_age)
    else:
        logger.info("age_range PASS — all values in [%d, %d]", min_age, max_age)
    return result


def validate_target_column(
    df: pd.DataFrame,
    target_column: str = "y",
    valid_values: frozenset[str] | None = None,
) -> ValidationResult:
    """Check that the target column exists and contains only expected values.

    Parameters
    ----------
    df:
        Input dataframe.
    target_column:
        Name of the target / label column.
    valid_values:
        Allowed distinct values (``None`` skips value check).

    Returns
    -------
    ValidationResult
    """
    if target_column not in df.columns:
        return ValidationResult(
            check="target_column",
            passed=False,
            failures=1,
            details={"error": f"Target column '{target_column}' not found"},
        )

    null_count = int(df[target_column].isna().sum())
    invalid_count = 0
    unexpected_values: list[str] = []

    if valid_values is not None:
        mask = ~df[target_column].isin(valid_values) & df[target_column].notna()
        invalid_count = int(mask.sum())
        unexpected_values = df.loc[mask, target_column].unique().tolist()

    total_failures = null_count + invalid_count
    passed = total_failures == 0
    result = ValidationResult(
        check="target_column",
        passed=passed,
        failures=total_failures,
        details={
            "null_count": null_count,
            "invalid_value_count": invalid_count,
            "unexpected_values": unexpected_values,
        },
    )
    if not passed:
        logger.warning(
            "target_column FAIL — nulls=%d invalid=%d unexpected=%s",
            null_count, invalid_count, unexpected_values,
        )
    else:
        logger.info("target_column PASS — column '%s'", target_column)
    return result


def validate_nonempty_complaints(
    df: pd.DataFrame,
    narrative_column: str = "complaint_narrative",
) -> ValidationResult:
    """Ensure every complaint row has a non-empty narrative.

    Parameters
    ----------
    df:
        Input dataframe.
    narrative_column:
        Column containing the complaint text.

    Returns
    -------
    ValidationResult
    """
    if narrative_column not in df.columns:
        return ValidationResult(
            check="nonempty_complaints",
            passed=False,
            failures=1,
            details={"error": f"Column '{narrative_column}' not found"},
        )

    empty_mask = df[narrative_column].isna() | (df[narrative_column].str.strip() == "")
    empty_count = int(empty_mask.sum())
    passed = empty_count == 0
    result = ValidationResult(
        check="nonempty_complaints",
        passed=passed,
        failures=empty_count,
        details={"empty_narrative_count": empty_count},
    )
    if not passed:
        logger.warning("nonempty_complaints FAIL — %d empty narratives", empty_count)
    else:
        logger.info("nonempty_complaints PASS")
    return result


def validate_duration_values(
    df: pd.DataFrame,
    duration_column: str = "duration",
) -> ValidationResult:
    """Check that campaign call duration is non-negative.

    Parameters
    ----------
    df:
        Input dataframe.
    duration_column:
        Column containing call duration in seconds.

    Returns
    -------
    ValidationResult
    """
    if duration_column not in df.columns:
        return ValidationResult(
            check="duration_values",
            passed=False,
            failures=1,
            details={"error": f"Column '{duration_column}' not found"},
        )

    durations = pd.to_numeric(df[duration_column], errors="coerce")
    invalid_count = int((durations < 0).sum() + durations.isna().sum())
    passed = invalid_count == 0
    result = ValidationResult(
        check="duration_values",
        passed=passed,
        failures=invalid_count,
        details={"invalid_duration_count": invalid_count},
    )
    if not passed:
        logger.warning("duration_values FAIL — %d invalid durations", invalid_count)
    else:
        logger.info("duration_values PASS")
    return result


def validate_complaint_dates(
    df: pd.DataFrame,
    date_column: str = "date_received",
    date_format: str = "%Y-%m-%d",
) -> ValidationResult:
    """Verify that complaint dates can be parsed with the expected format.

    Parameters
    ----------
    df:
        Input dataframe.
    date_column:
        Column containing date strings.
    date_format:
        ``strptime``-compatible format string.

    Returns
    -------
    ValidationResult
    """
    if date_column not in df.columns:
        return ValidationResult(
            check="complaint_dates",
            passed=False,
            failures=1,
            details={"error": f"Column '{date_column}' not found"},
        )

    parsed = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
    unparseable = int(parsed.isna().sum())
    passed = unparseable == 0
    result = ValidationResult(
        check="complaint_dates",
        passed=passed,
        failures=unparseable,
        details={"date_column": date_column, "unparseable_count": unparseable},
    )
    if not passed:
        logger.warning("complaint_dates FAIL — %d unparseable dates", unparseable)
    else:
        logger.info("complaint_dates PASS")
    return result


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------

# Required columns for each dataset
BANK_REQUIRED_COLUMNS: list[str] = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp_var_rate", "cons_price_idx",
    "cons_conf_idx", "euribor3m", "nr_employed", "y",
]

COMPLAINT_REQUIRED_COLUMNS: list[str] = [
    "complaint_id", "product", "issue", "company",
    "date_received", "complaint_narrative",
]

BANK_TARGET_VALUES: frozenset[str] = frozenset({"yes", "no"})


def validate_bank_dataset(df: pd.DataFrame) -> ValidationSummary:
    """Run all bank marketing validation checks.

    Parameters
    ----------
    df:
        Bank marketing dataframe (post-ingestion).

    Returns
    -------
    ValidationSummary
    """
    results = [
        validate_required_columns(df, BANK_REQUIRED_COLUMNS),
        validate_age_range(df),
        validate_target_column(df, "y", BANK_TARGET_VALUES),
        validate_duration_values(df),
    ]
    summary = ValidationSummary(dataset="bank_marketing", results=results)
    logger.info(
        "Bank validation complete — %d checks, %d failed",
        len(results), summary.failure_count,
    )
    return summary


def validate_complaints_dataset(df: pd.DataFrame) -> ValidationSummary:
    """Run all complaints validation checks.

    Parameters
    ----------
    df:
        Complaints dataframe (post-ingestion).

    Returns
    -------
    ValidationSummary
    """
    results = [
        validate_required_columns(df, COMPLAINT_REQUIRED_COLUMNS),
        validate_duplicates(df, "complaint_id"),
        validate_nonempty_complaints(df),
        validate_complaint_dates(df),
    ]
    summary = ValidationSummary(dataset="cfpb_complaints", results=results)
    logger.info(
        "Complaints validation complete — %d checks, %d failed",
        len(results), summary.failure_count,
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_validation() -> tuple[ValidationSummary, ValidationSummary]:
    """Load samples and run full validation for both datasets.

    Returns
    -------
    tuple[ValidationSummary, ValidationSummary]
        (bank_summary, complaints_summary)
    """
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parents[2]
    samples_dir = _Path(
        __import__("os").getenv("SAMPLES_DIR", str(_repo_root / "data" / "samples"))
    )

    bank_path = samples_dir / "bank_sample.csv"
    complaints_path = samples_dir / "complaints_sample.csv"

    if not bank_path.exists():
        raise FileNotFoundError(
            f"Bank sample not found at {bank_path}. Run ingest.py first."
        )
    if not complaints_path.exists():
        raise FileNotFoundError(
            f"Complaints sample not found at {complaints_path}. Run ingest.py first."
        )

    logger.info("Loading bank sample from %s", bank_path)
    bank_df = pd.read_csv(bank_path, low_memory=False)
    logger.info("Loaded %d bank rows", len(bank_df))

    logger.info("Loading complaints sample from %s", complaints_path)
    complaints_df = pd.read_csv(complaints_path, low_memory=False, dtype=str)
    logger.info("Loaded %d complaint rows", len(complaints_df))

    bank_summary = validate_bank_dataset(bank_df)
    complaints_summary = validate_complaints_dataset(complaints_df)

    bank_summary.print_summary()
    complaints_summary.print_summary()

    return bank_summary, complaints_summary


if __name__ == "__main__":
    bank_s, complaints_s = run_validation()
    overall_ok = bank_s.passed and complaints_s.passed
    sys.exit(0 if overall_ok else 1)
