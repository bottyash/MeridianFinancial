"""
src/common/config.py
---------------------
Centralised, environment-driven settings for Meridian Financial.

Uses ``pydantic-settings`` so every value is:
  * Type-validated at startup
  * Overridable via environment variable or ``.env`` file
  * Never hardcoded — secrets come from the environment only

Usage
-----
  from src.common.config import settings

  print(settings.app_env)
  print(settings.mistral_api_key.get_secret_value())
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / ``.env``.

    All fields have sensible defaults for local development.
    Production deployments **must** override ``MISTRAL_API_KEY`` and
    ``APP_ENV=production`` via real environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",            # tolerate extra env vars without errors
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment tier.",
    )
    app_host: str = Field(default="0.0.0.0", description="FastAPI bind host.")
    app_port: int = Field(default=8000, ge=1024, le=65535, description="FastAPI bind port.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root logger level.",
    )

    # ── Data paths ───────────────────────────────────────────────────────────
    bank_sample_path: Path = Field(
        default=Path("data/samples/bank_sample.csv"),
        description="Path to the processed bank marketing sample.",
    )
    complaints_sample_path: Path = Field(
        default=Path("data/samples/complaints_sample.csv"),
        description="Path to the processed CFPB complaints sample.",
    )
    features_artifacts_dir: Path = Field(
        default=Path("artifacts/features"),
        description="Directory for fitted preprocessor and feature schema.",
    )
    mlflow_tracking_uri: str = Field(
        default="mlruns",
        description="MLflow tracking URI (local dir or remote server).",
    )

    # ── Vector store ─────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(
        default=Path("chroma_store"),
        description="ChromaDB persistence directory.",
    )
    chroma_collection_name: str = Field(
        default="complaints",
        description="Name of the ChromaDB collection used for RAG.",
    )

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model name for RAG embeddings.",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    mistral_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Mistral AI API key.  Set via MISTRAL_API_KEY env var.",
    )
    mistral_model: str = Field(
        default="mistral-small-latest",
        description="Mistral model identifier.",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────
    complaint_sample_size: int = Field(
        default=10_000,
        ge=5_000,
        le=25_000,
        description="Target complaint sample size (5 000 – 25 000).",
    )
    complaint_sample_seed: int = Field(
        default=42,
        description="Random seed for deterministic complaint sampling.",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Settings("
            f"app_env={self.app_env!r}, "
            f"app_host={self.app_host!r}, "
            f"app_port={self.app_port}, "
            f"log_level={self.log_level!r}, "
            f"mistral_model={self.mistral_model!r}, "
            f"embedding_model={self.embedding_model!r}"
            f")"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton ``Settings`` instance (cached after first call).

    Using ``lru_cache`` ensures the ``.env`` file is parsed exactly once,
    keeping startup fast and avoiding repeated disk I/O.

    Returns
    -------
    Settings
    """
    return Settings()


# Module-level alias for convenient import
settings: Settings = get_settings()
