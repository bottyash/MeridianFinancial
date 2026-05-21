"""
src/rag/build_index.py
-----------------------
Complaint narrative indexing pipeline for Meridian Financial RAG system.

Pipeline
--------
1. Load complaint sample CSV
2. Clean and validate narratives
3. Chunk long narratives into overlapping windows
4. Generate embeddings with ``all-MiniLM-L6-v2``
5. Upsert chunks + metadata into a persisted ChromaDB collection

Artifacts
---------
* ``chroma_store/``  — persisted ChromaDB vector index (collection: complaints)

Design principles
-----------------
* Deterministic: same input → same chunks → same IDs (content-hash based)
* Idempotent: re-running upserts over existing data safely
* Configurable: all tuneable parameters live in ``config/rag_config.yaml``
* Modular: cleaning, chunking, and indexing are independent pure functions

Usage
-----
  python src/rag/build_index.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("meridian.rag.build_index")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "rag_config.yaml"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_rag_config(config_path: Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and return the RAG pipeline YAML configuration.

    Parameters
    ----------
    config_path:
        Path to ``rag_config.yaml``.

    Returns
    -------
    dict
    """
    if not config_path.exists():
        raise FileNotFoundError(f"RAG config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("RAG config loaded from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_REDACTION_RE = re.compile(r"\bXX+\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_narrative(text: str) -> str:
    """Normalise a CFPB complaint narrative.

    Transformations applied (in order):
    1. Strip leading / trailing whitespace
    2. Collapse repeated whitespace to a single space
    3. Remove ``XX``/``XXX`` redaction placeholders
    4. Normalise Unicode dashes to ASCII hyphens

    Parameters
    ----------
    text:
        Raw complaint narrative string.

    Returns
    -------
    str
        Cleaned narrative.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\u2014", "-").replace("\u2013", "-")  # em/en dash
    text = _REDACTION_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_valid_narrative(text: str, min_length: int = 30) -> bool:
    """Return True if *text* is non-empty after cleaning and meets *min_length*.

    Parameters
    ----------
    text:
        Cleaned narrative.
    min_length:
        Minimum character length to consider valid.

    Returns
    -------
    bool
    """
    return isinstance(text, str) and len(text.strip()) >= min_length


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    min_chunk_length: int = 30,
) -> list[str]:
    """Split *text* into overlapping character-level windows.

    Parameters
    ----------
    text:
        Input text (cleaned narrative).
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Number of characters overlapping between adjacent chunks.
    min_chunk_length:
        Discard chunks shorter than this threshold.

    Returns
    -------
    list[str]
        Non-empty chunks.
    """
    if not text or len(text) <= chunk_size:
        if len(text) >= min_chunk_length:
            return [text]
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_length:
            chunks.append(chunk)
        start += step

    return chunks


def make_chunk_id(complaint_id: int | str, chunk_index: int, text: str) -> str:
    """Generate a deterministic, content-addressed chunk ID.

    Uses a truncated SHA-256 of (complaint_id, chunk_index, text) so that
    re-indexing the same data produces the same IDs — enabling idempotent upserts.

    Parameters
    ----------
    complaint_id:
        Source complaint identifier.
    chunk_index:
        Zero-based index of this chunk within the complaint.
    text:
        Chunk text (used in the hash for content-addressing).

    Returns
    -------
    str
        16-character hex string.
    """
    raw = f"{complaint_id}:{chunk_index}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def prepare_documents(
    df: pd.DataFrame,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    min_chunk_length: int = 30,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Clean, chunk, and prepare all complaint documents for indexing.

    Parameters
    ----------
    df:
        Complaints DataFrame with columns: complaint_id, product, issue,
        company, date_received, complaint_narrative.
    chunk_size, chunk_overlap, min_chunk_length:
        Passed to :func:`chunk_text`.

    Returns
    -------
    tuple
        ``(ids, texts, metadatas)`` — parallel lists ready for Chroma upsert.
    """
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    skipped = 0
    for _, row in df.iterrows():
        narrative = clean_narrative(str(row.get("complaint_narrative", "")))
        if not is_valid_narrative(narrative, min_chunk_length):
            skipped += 1
            continue

        chunks = chunk_text(narrative, chunk_size, chunk_overlap, min_chunk_length)
        cid = row.get("complaint_id", "unknown")

        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(cid, i, chunk)
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "complaint_id": str(cid),
                "product": str(row.get("product", "")),
                "issue": str(row.get("issue", "")),
                "company": str(row.get("company", "")),
                "date": str(row.get("date_received", "")),
                "chunk_index": i,
            })

    logger.info(
        "Prepared %d chunks from %d complaints (%d skipped / invalid)",
        len(ids), len(df) - skipped, skipped,
    )
    return ids, texts, metadatas


def get_or_create_collection(
    persist_dir: str,
    collection_name: str,
    distance_metric: str = "cosine",
) -> chromadb.Collection:
    """Return (or create) a persisted ChromaDB collection.

    Parameters
    ----------
    persist_dir:
        Directory for ChromaDB persistence.
    collection_name:
        Name of the collection.
    distance_metric:
        Distance function: ``cosine``, ``l2``, or ``ip``.

    Returns
    -------
    chromadb.Collection
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_metric},
    )
    logger.info(
        "ChromaDB collection '%s' ready at %s  (count=%d)",
        collection_name, persist_dir, collection.count(),
    )
    return collection


def embed_and_upsert(
    collection: chromadb.Collection,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> int:
    """Generate embeddings and upsert into ChromaDB in batches.

    Parameters
    ----------
    collection:
        Target ChromaDB collection.
    ids, texts, metadatas:
        Parallel lists from :func:`prepare_documents`.
    model:
        Loaded SentenceTransformer model.
    batch_size:
        Number of chunks encoded and upserted per batch.

    Returns
    -------
    int
        Total number of chunks upserted.
    """
    total = len(ids)
    upserted = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        batch_texts = texts[start:end]
        batch_meta = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )
        upserted += len(batch_ids)
        logger.info(
            "Upserted %d / %d chunks", upserted, total
        )

    return upserted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_index(
    config_path: Path = _DEFAULT_CONFIG,
    complaints_path: Path | None = None,
) -> dict[str, Any]:
    """End-to-end index build pipeline.

    1. Load config
    2. Load complaints sample
    3. Prepare (clean + chunk) documents
    4. Load embedding model
    5. Embed and upsert into ChromaDB

    Parameters
    ----------
    config_path:
        Path to ``rag_config.yaml``.
    complaints_path:
        Override for complaints CSV path.

    Returns
    -------
    dict
        ``{"collection_name", "chunks_upserted", "collection_count"}``
    """
    cfg = load_rag_config(config_path)

    # Data path (CLI or config)
    if complaints_path is None:
        complaints_path = Path(
            os.getenv(
                "COMPLAINTS_SAMPLE_PATH",
                str(_REPO_ROOT / cfg["data"]["complaints_sample_path"]),
            )
        )

    if not complaints_path.exists():
        raise FileNotFoundError(
            f"Complaints sample not found: {complaints_path}. "
            "Run src/data_pipeline/ingest.py first."
        )

    logger.info("Loading complaints from %s", complaints_path)
    df = pd.read_csv(complaints_path, low_memory=False)
    logger.info("Loaded %d rows", len(df))

    # Chunking params
    chunk_cfg = cfg["chunking"]
    ids, texts, metadatas = prepare_documents(
        df,
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        min_chunk_length=chunk_cfg["min_chunk_length"],
    )

    if not ids:
        raise RuntimeError("No valid documents to index after cleaning.")

    # Embedding model
    emb_cfg = cfg["embedding"]
    logger.info("Loading embedding model: %s", emb_cfg["model_name"])
    model = SentenceTransformer(emb_cfg["model_name"])

    # ChromaDB collection
    chroma_cfg = cfg["chroma"]
    persist_dir = str(_REPO_ROOT / chroma_cfg["persist_directory"])
    collection = get_or_create_collection(
        persist_dir,
        chroma_cfg["collection_name"],
        chroma_cfg["distance_metric"],
    )

    # Embed and upsert
    upserted = embed_and_upsert(
        collection, ids, texts, metadatas, model,
        batch_size=emb_cfg["batch_size"],
    )

    final_count = collection.count()
    logger.info(
        "Index build complete — collection='%s'  chunks_upserted=%d  total_in_db=%d",
        chroma_cfg["collection_name"], upserted, final_count,
    )

    return {
        "collection_name": chroma_cfg["collection_name"],
        "chunks_upserted": upserted,
        "collection_count": final_count,
    }


if __name__ == "__main__":
    result = build_index()
    print(f"\nIndex build complete: {result}")
