"""
src/data_pipeline/ingest.py
----------------------------
Reproducible ingestion pipeline for:
  1. UCI Bank Marketing dataset  (structured CSV, semicolon-separated)
  2. CFPB Complaint dataset sample (complaints.csv)

Design principles
-----------------
* Deterministic: fixed random seed, reproducible samples
* Configurable: all paths / sizes driven by parameters / env-vars
* Modular: each dataset has its own loader function
* No side-effects at import time; entry-point is guarded by __main__
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("meridian.ingest")

# ---------------------------------------------------------------------------
# Path constants (overridable via environment variables)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

BANK_RAW_DIR: Path = Path(
    os.getenv("BANK_RAW_DIR", str(_REPO_ROOT / "data" / "raw" / "bank_marketing"))
)
COMPLAINTS_RAW_FILE: Path = Path(
    os.getenv(
        "COMPLAINTS_RAW_FILE",
        str(_REPO_ROOT / "data" / "raw" / "complaints" / "complaints.csv"),
    )
)
SAMPLES_DIR: Path = Path(
    os.getenv("SAMPLES_DIR", str(_REPO_ROOT / "data" / "samples"))
)

# Sampling configuration
COMPLAINT_SAMPLE_SIZE: int = int(os.getenv("COMPLAINT_SAMPLE_SIZE", "10000"))
COMPLAINT_SAMPLE_SEED: int = int(os.getenv("COMPLAINT_SAMPLE_SEED", "42"))
BANK_SAMPLE_SIZE: Optional[int] = (
    int(os.getenv("BANK_SAMPLE_SIZE"))
    if os.getenv("BANK_SAMPLE_SIZE")
    else None  # None → keep all rows
)

# Columns to retain from complaints dataset (preserves required metadata)
COMPLAINT_COLUMNS = {
    "Complaint ID": "complaint_id",
    "Product": "product",
    "Issue": "issue",
    "Company": "company",
    "Date received": "date_received",
    "Consumer complaint narrative": "complaint_narrative",
}


# ---------------------------------------------------------------------------
# Bank Marketing ingestion
# ---------------------------------------------------------------------------

def load_bank_marketing(
    raw_dir: Path = BANK_RAW_DIR,
    sample_size: Optional[int] = BANK_SAMPLE_SIZE,
    seed: int = COMPLAINT_SAMPLE_SEED,
) -> pd.DataFrame:
    """Load the UCI Bank Marketing dataset from the raw directory.

    Prefers ``bank-additional-full.csv`` (largest, richest feature set).
    Falls back to other CSVs in the folder if that file is absent.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw bank marketing CSV files.
    sample_size:
        Number of rows to keep (``None`` keeps all rows).
    seed:
        Random seed for deterministic sampling.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with snake_case column names.
    """
    candidate_files = [
        "bank-additional-full.csv",
        "bank-additional.csv",
        "bank-full.csv",
        "bank.csv",
    ]

    source_file: Optional[Path] = None
    for name in candidate_files:
        candidate = raw_dir / name
        if candidate.exists():
            source_file = candidate
            break

    if source_file is None:
        raise FileNotFoundError(
            f"No recognised bank marketing CSV found in {raw_dir}. "
            f"Expected one of: {candidate_files}"
        )

    logger.info("Loading bank marketing data from %s", source_file)
    df = pd.read_csv(source_file, sep=";", low_memory=False)
    logger.info("Loaded %d rows × %d columns from bank marketing dataset", *df.shape)

    # Normalise column names → snake_case, strip whitespace
    df.columns = [
        col.strip().lower().replace(" ", "_").replace(".", "_")
        for col in df.columns
    ]

    # Deterministic sample
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        logger.info("Sampled %d rows (seed=%d)", len(df), seed)
    else:
        logger.info("Keeping all %d rows (no sampling applied)", len(df))

    return df


# ---------------------------------------------------------------------------
# CFPB Complaint ingestion
# ---------------------------------------------------------------------------

def load_complaints(
    raw_file: Path = COMPLAINTS_RAW_FILE,
    sample_size: int = COMPLAINT_SAMPLE_SIZE,
    seed: int = COMPLAINT_SAMPLE_SEED,
    chunk_size: int = 100_000,
    pool_multiplier: int = 3,
) -> pd.DataFrame:
    """Load and deterministically sample the CFPB complaints dataset.

    Uses vectorized chunked reading to avoid loading the full ~8 GB file.
    Collects valid rows (non-empty narrative) into a pool of up to
    ``sample_size * pool_multiplier`` rows, then draws the final sample
    with a single ``pd.DataFrame.sample()`` call.

    Parameters
    ----------
    raw_file:
        Path to ``complaints.csv``.
    sample_size:
        Target number of rows in the sample (5 000 – 25 000).
    seed:
        Random seed for deterministic, reproducible output.
    chunk_size:
        Rows to read per CSV chunk (tune based on available RAM).
    pool_multiplier:
        The pool is capped at ``sample_size * pool_multiplier`` to bound
        memory usage while still providing a representative sample.

    Returns
    -------
    pd.DataFrame
        Sampled dataframe with standardised column names.
    """
    if not (5_000 <= sample_size <= 25_000):
        raise ValueError(
            f"sample_size must be in [5000, 25000]; got {sample_size}"
        )

    if not raw_file.exists():
        raise FileNotFoundError(f"Complaints CSV not found: {raw_file}")

    max_pool = sample_size * pool_multiplier
    logger.info(
        "Streaming complaints from %s "
        "(chunk_size=%d, target_sample=%d, max_pool=%d, seed=%d)",
        raw_file, chunk_size, sample_size, max_pool, seed,
    )

    usecols = list(COMPLAINT_COLUMNS.keys())
    pool_frames: list[pd.DataFrame] = []
    total_rows_seen: int = 0
    total_valid_rows: int = 0

    with pd.read_csv(
        raw_file,
        usecols=usecols,
        dtype=str,
        chunksize=chunk_size,
        on_bad_lines="skip",
    ) as reader:
        for chunk in reader:
            # Rename columns to standardised names
            chunk = chunk.rename(columns=COMPLAINT_COLUMNS)

            # Vectorized filter: keep rows with non-empty narrative
            mask = (
                chunk["complaint_narrative"].notna()
                & (chunk["complaint_narrative"].str.strip() != "")
            )
            valid = chunk[mask]
            total_rows_seen += len(chunk)
            total_valid_rows += len(valid)

            if len(valid):
                pool_frames.append(valid)

            # Stop early once pool is saturated
            current_pool_size = sum(len(f) for f in pool_frames)
            if current_pool_size >= max_pool:
                logger.info(
                    "Pool saturation reached (%d rows); stopping early after "
                    "%d raw rows streamed",
                    current_pool_size, total_rows_seen,
                )
                break

    if not pool_frames:
        raise RuntimeError(
            "No valid complaint rows found (with non-empty narrative) in the dataset."
        )

    pool_df = pd.concat(pool_frames, ignore_index=True)
    logger.info(
        "Pool built: %d valid rows from %d raw rows streamed",
        len(pool_df), total_rows_seen,
    )

    actual_sample = min(sample_size, len(pool_df))
    if actual_sample < sample_size:
        logger.warning(
            "Requested %d rows but only %d usable rows available; using all",
            sample_size, actual_sample,
        )

    df = pool_df.sample(n=actual_sample, random_state=seed).reset_index(drop=True)
    logger.info("Sampled %d complaint rows (seed=%d)", len(df), seed)
    return df


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_sample(df: pd.DataFrame, output_path: Path) -> None:
    """Persist a dataframe to CSV, creating parent directories as needed.

    Parameters
    ----------
    df:
        Dataframe to persist.
    output_path:
        Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ingestion() -> dict[str, Path]:
    """Execute the full ingestion pipeline and return output file paths.

    Returns
    -------
    dict
        ``{"bank": <path>, "complaints": <path>}``
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    bank_out = SAMPLES_DIR / "bank_sample.csv"
    complaints_out = SAMPLES_DIR / "complaints_sample.csv"

    # --- Bank Marketing ---
    bank_df = load_bank_marketing()
    save_sample(bank_df, bank_out)

    # --- Complaints ---
    complaints_df = load_complaints()
    save_sample(complaints_df, complaints_out)

    logger.info("Ingestion complete. Outputs: %s | %s", bank_out, complaints_out)
    return {"bank": bank_out, "complaints": complaints_out}


if __name__ == "__main__":
    run_ingestion()
