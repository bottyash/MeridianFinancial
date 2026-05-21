# data/README.md

# Meridian Financial — Data Directory

This directory holds all raw and processed data used by the Meridian Financial
Customer Intelligence Platform.

---

## Directory Layout

```
data/
├── raw/
│   ├── bank_marketing/          # UCI Bank Marketing dataset (local, not committed)
│   │   ├── bank-additional-full.csv   # Primary source (41,188 rows, 21 features)
│   │   ├── bank-additional.csv
│   │   ├── bank-full.csv
│   │   └── bank.csv
│   └── complaints/              # CFPB Consumer Complaint dataset (local, not committed)
│       └── complaints.csv       # Full dataset (~8.7 GB)
└── samples/                     # Deterministic samples committed to the repo
    ├── bank_sample.csv          # Cleaned bank marketing sample
    └── complaints_sample.csv    # 10 000-row complaint sample (with narratives)
```

---

## Datasets

### 1. UCI Bank Marketing Dataset

| Property        | Value                                          |
|-----------------|------------------------------------------------|
| Source          | UCI Machine Learning Repository               |
| File            | `bank-additional-full.csv`                    |
| Rows            | 41,188                                         |
| Target          | `y` — did client subscribe? (`yes` / `no`)    |
| Separator       | Semicolon (`;`)                               |
| Task            | Binary classification (campaign conversion)   |

Key features: `age`, `job`, `marital`, `education`, `duration`, `campaign`,
`pdays`, `previous`, `poutcome`, macro-economic indicators.

### 2. CFPB Consumer Complaint Dataset

| Property        | Value                                                    |
|-----------------|----------------------------------------------------------|
| Source          | Consumer Financial Protection Bureau (CFPB)             |
| File            | `complaints.csv`                                         |
| Size            | ~8.7 GB (full dataset)                                   |
| Sample size     | 10,000 rows (configurable via `COMPLAINT_SAMPLE_SIZE`)   |
| Task            | RAG-based complaint intelligence                         |

Retained columns (all others dropped):

| Raw Column                     | Standardised Name     |
|--------------------------------|-----------------------|
| `Complaint ID`                 | `complaint_id`        |
| `Product`                      | `product`             |
| `Issue`                        | `issue`               |
| `Company`                      | `company`             |
| `Date received`                | `date_received`       |
| `Consumer complaint narrative` | `complaint_narrative` |

---

## Ingestion Pipeline

Run the ingestion script to regenerate samples:

```bash
python src/data_pipeline/ingest.py
```

**Environment variables (all optional):**

| Variable               | Default                              | Description                         |
|------------------------|--------------------------------------|-------------------------------------|
| `BANK_RAW_DIR`         | `data/raw/bank_marketing/`           | Raw bank marketing directory        |
| `COMPLAINTS_RAW_FILE`  | `data/raw/complaints/complaints.csv` | Full complaints CSV path            |
| `SAMPLES_DIR`          | `data/samples/`                      | Output directory for samples        |
| `COMPLAINT_SAMPLE_SIZE`| `10000`                              | Rows to sample (5 000 – 25 000)     |
| `COMPLAINT_SAMPLE_SEED`| `42`                                 | Random seed for deterministic output|
| `BANK_SAMPLE_SIZE`     | _(empty — keep all rows)_            | Rows to sample from bank dataset    |

---

## Validation Pipeline

Run the validation script after ingestion:

```bash
python src/data_pipeline/validate.py
```

**Business rules enforced:**

| Rule                         | Dataset     | Check function                    |
|------------------------------|-------------|-----------------------------------|
| Required columns present     | Both        | `validate_required_columns()`     |
| No duplicate IDs             | Complaints  | `validate_duplicates()`           |
| Age in range [18, 100]       | Bank        | `validate_age_range()`            |
| Target column valid          | Bank        | `validate_target_column()`        |
| Complaint narrative non-empty| Complaints  | `validate_nonempty_complaints()`  |
| Duration ≥ 0                 | Bank        | `validate_duration_values()`      |
| Complaint dates parseable    | Complaints  | `validate_complaint_dates()`      |

---

## Reproducibility

Samples are fully reproducible given:
- The same raw source files
- `COMPLAINT_SAMPLE_SEED=42` (default)
- `COMPLAINT_SAMPLE_SIZE=10000` (default)

---

## Data Governance

- Raw datasets are **never committed** to version control (see `.gitignore`)
- Samples under `data/samples/` **are committed** for reproducibility
- Obvious PII is not actively present in sampled complaint narratives  
  (CFPB redacts personal information before publication)
- Full datasets are stored locally and must be sourced by each developer
