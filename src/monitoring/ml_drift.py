"""
src/monitoring/ml_drift.py
---------------------------
ML feature drift detection for Meridian Financial.

Pipeline
--------
1. Load reference dataset (complaints_sample or bank marketing sample)
2. Simulate a drifted "current" dataset by perturbing numeric distributions
3. Run Evidently DataDriftPreset report (reference → current)
4. Save HTML report to ``monitoring/report.html``
5. Extract drift summary and write ``monitoring/drift_summary.json``
6. Log structured drift detection events via ``MetricsLogger``

Drift simulation
-----------------
The "current" dataset is synthesised by applying controlled perturbations to
numeric features (mean shift + noise injection), plus randomly replacing a
fraction of categorical values.  This deterministic simulation lets the
monitoring pipeline exercise the full Evidently code path without needing a
live production feed.

Usage
-----
  python src/monitoring/ml_drift.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
logger = logging.getLogger("meridian.monitoring.ml_drift")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONITORING_DIR = _REPO_ROOT / "monitoring"
HTML_REPORT_PATH = _MONITORING_DIR / "report.html"
DRIFT_SUMMARY_PATH = _MONITORING_DIR / "drift_summary.json"
REFERENCE_DATA_PATH = _REPO_ROOT / "data" / "samples" / "complaints_sample.csv"

# Numeric features for drift detection (subset stable across both datasets)
_NUMERIC_FEATURES = ["complaint_id"]
_CATEGORICAL_FEATURES = ["product", "issue", "company"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reference_data(path: Path = REFERENCE_DATA_PATH, n: int = 2000) -> pd.DataFrame:
    """Load and subsample the reference dataset.

    Parameters
    ----------
    path:
        CSV path.
    n:
        Number of rows to use as reference (for speed).

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["complaint_id", "product"]).head(n).copy()
    logger.info("Reference data loaded: %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Drift simulation
# ---------------------------------------------------------------------------

def simulate_drift(
    reference: pd.DataFrame,
    numeric_shift: float = 2.0,
    category_swap_rate: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Produce a synthetic drifted version of *reference*.

    Perturbations applied:
    * Numeric columns: add ``numeric_shift * std`` mean shift + Gaussian noise
    * Categorical columns: randomly replace ``category_swap_rate`` fraction of
      values with a different category from the same column

    Parameters
    ----------
    reference:
        The un-drifted reference DataFrame.
    numeric_shift:
        Number of standard deviations to shift numeric means by.
    category_swap_rate:
        Fraction of categorical values to randomly replace.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Drifted copy (same columns and shape as *reference*).
    """
    rng = np.random.default_rng(random_state)
    current = reference.copy()

    # Numeric drift
    for col in _NUMERIC_FEATURES:
        if col in current.columns and pd.api.types.is_numeric_dtype(current[col]):
            std = current[col].std()
            shift = numeric_shift * std
            noise = rng.normal(0, std * 0.5, size=len(current))
            current[col] = current[col] + shift + noise

    # Categorical drift
    for col in _CATEGORICAL_FEATURES:
        if col in current.columns:
            cats = current[col].dropna().unique().tolist()
            if len(cats) < 2:
                continue
            mask = rng.random(len(current)) < category_swap_rate
            # Replace masked rows with a randomly chosen different category
            replacements = rng.choice(cats, size=mask.sum())
            current.loc[mask, col] = replacements

    logger.info(
        "Drift simulated — numeric_shift=%.1f  category_swap_rate=%.0f%%  n=%d",
        numeric_shift, category_swap_rate * 100, len(current),
    )
    return current


# ---------------------------------------------------------------------------
# Evidently drift report
# ---------------------------------------------------------------------------

def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_mapping=None,
) -> dict:
    """Run Evidently DataDriftPreset on reference vs current data.

    Parameters
    ----------
    reference, current:
        Pandas DataFrames with aligned columns.
    column_mapping:
        Optional Evidently ColumnMapping.

    Returns
    -------
    dict
        Raw Evidently JSON result dictionary.
    """
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])

    # Restrict to columns present in both, prefer numeric + a few categoricals
    cols = [c for c in _NUMERIC_FEATURES + _CATEGORICAL_FEATURES
            if c in reference.columns and c in current.columns]

    report.run(
        reference_data=reference[cols],
        current_data=current[cols],
        column_mapping=column_mapping,
    )
    logger.info("Evidently drift report computed — columns=%s", cols)
    return report.as_dict()


# ---------------------------------------------------------------------------
# HTML report save
# ---------------------------------------------------------------------------

def save_html_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path = HTML_REPORT_PATH,
) -> Path:
    """Generate and save the Evidently HTML drift report.

    Parameters
    ----------
    reference, current:
        DataFrames for the Evidently comparison.
    output_path:
        Path to write the ``.html`` file.

    Returns
    -------
    Path
        Written file path.
    """
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset

    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    cols = [c for c in _NUMERIC_FEATURES + _CATEGORICAL_FEATURES
            if c in reference.columns and c in current.columns]

    report.run(reference_data=reference[cols], current_data=current[cols])
    report.save_html(str(output_path))
    logger.info("HTML drift report saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Drift summary
# ---------------------------------------------------------------------------

def extract_drift_summary(result_dict: dict) -> dict:
    """Parse the Evidently result dictionary into a concise drift summary.

    Parameters
    ----------
    result_dict:
        Raw output of ``Report.as_dict()``.

    Returns
    -------
    dict
        Structured summary with per-column drift status and aggregate stats.
    """
    # Navigate Evidently 0.7.x dict structure
    metrics = result_dict.get("metrics", [])

    dataset_drift = False
    drifted_columns = []
    not_drifted = []
    column_details: list[dict] = []

    for metric in metrics:
        metric_id = metric.get("metric", "")
        result = metric.get("result", {})

        # DatasetDriftMetric
        if "DatasetDriftMetric" in metric_id or "dataset_drift" in result:
            dataset_drift = result.get("dataset_drift", False)

        # DataDriftTable / ColumnDriftMetric
        if "drift_by_columns" in result:
            for col_name, col_result in result["drift_by_columns"].items():
                drifted = col_result.get("drift_detected", False)
                score = col_result.get("drift_score", None)
                stat_test = col_result.get("stattest_name", "N/A")
                column_details.append({
                    "column": col_name,
                    "drift_detected": drifted,
                    "drift_score": round(score, 4) if score is not None else None,
                    "stattest": stat_test,
                })
                if drifted:
                    drifted_columns.append(col_name)
                else:
                    not_drifted.append(col_name)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_drift_detected": dataset_drift,
        "n_drifted_columns": len(drifted_columns),
        "n_not_drifted_columns": len(not_drifted),
        "drifted_columns": drifted_columns,
        "column_details": column_details,
    }

    logger.info(
        "Drift summary — dataset_drift=%s  drifted_cols=%d/%d",
        dataset_drift, len(drifted_columns), len(drifted_columns) + len(not_drifted),
    )
    return summary


def save_drift_summary(summary: dict, output_path: Path = DRIFT_SUMMARY_PATH) -> Path:
    """Persist drift summary to JSON.

    Parameters
    ----------
    summary:
        Dict from :func:`extract_drift_summary`.
    output_path:
        Destination path.

    Returns
    -------
    Path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Drift summary saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    reference_path: Path = REFERENCE_DATA_PATH,
    html_path: Path = HTML_REPORT_PATH,
    summary_path: Path = DRIFT_SUMMARY_PATH,
    n_reference: int = 2000,
) -> dict:
    """End-to-end drift detection pipeline.

    1. Load reference data
    2. Simulate drifted current data
    3. Run Evidently DataDriftPreset report
    4. Save HTML report
    5. Extract and save drift summary JSON
    6. Log events via MetricsLogger

    Parameters
    ----------
    reference_path:
        Path to reference CSV.
    html_path:
        Output path for HTML report.
    summary_path:
        Output path for drift summary JSON.
    n_reference:
        Number of reference rows to use.

    Returns
    -------
    dict
        Drift summary dictionary.
    """
    from src.monitoring.metrics_logger import MetricsLogger

    ml_logger = MetricsLogger()

    # 1. Load
    reference = load_reference_data(reference_path, n=n_reference)

    # 2. Simulate drift
    current = simulate_drift(reference)

    # 3. Run Evidently report
    result_dict = run_drift_report(reference, current)

    # 4. Save HTML
    save_html_report(reference, current, output_path=html_path)

    # 5. Extract summary
    summary = extract_drift_summary(result_dict)
    save_drift_summary(summary, output_path=summary_path)

    # 6. Log to MetricsLogger
    ml_logger.log_event("drift_detection", {
        "dataset_drift_detected": summary["dataset_drift_detected"],
        "n_drifted_columns": summary["n_drifted_columns"],
        "drifted_columns": summary["drifted_columns"],
    })

    # Log simulated prediction distributions
    rng = np.random.default_rng(0)
    ref_probs = rng.beta(2, 8, size=500)   # baseline — skewed low
    cur_probs = rng.beta(4, 6, size=500)   # drifted — shifted higher
    for prob in ref_probs[:20]:
        ml_logger.log_ml_prediction(float(prob), int(prob >= 0.5), latency_ms=45.0)
    for prob in cur_probs[:20]:
        ml_logger.log_ml_prediction(float(prob), int(prob >= 0.5), latency_ms=48.0)

    ml_logger.flush()

    logger.info("ML drift pipeline complete.")
    return summary


if __name__ == "__main__":
    summary = run()
    print(f"\nDrift detection complete:")
    print(f"  dataset_drift_detected = {summary['dataset_drift_detected']}")
    print(f"  drifted_columns = {summary['n_drifted_columns']}")
    print(f"  HTML report     => monitoring/report.html")
    print(f"  Drift summary   => monitoring/drift_summary.json")
    print(f"  Metrics log     => monitoring/metrics.json")
