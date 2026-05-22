"""
dashboard/utils/config.py
--------------------------
Deployment-aware configuration for the Meridian Financial dashboard.

Reads from environment variables so the same code runs on:
  - local development (http://localhost:8000)
  - AWS EC2 backend   (https://your-ec2-host/api)
  - HuggingFace Spaces (public backend URL via HF Secret)

Environment variables
---------------------
API_BASE_URL       Base URL of the FastAPI backend (no trailing slash)
ENVIRONMENT        development | staging | production
REQUEST_TIMEOUT    HTTP request timeout in seconds (default 30)
DASHBOARD_TITLE    Browser tab / page title
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DashboardConfig:
    """Immutable runtime configuration loaded from environment."""

    api_base_url: str
    environment: str
    request_timeout: int
    dashboard_title: str
    page_icon: str

    # Derived helpers
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def api_health_url(self) -> str:
        return f"{self.api_base_url}/health"

    @property
    def api_predict_url(self) -> str:
        return f"{self.api_base_url}/predict"

    @property
    def api_batch_score_url(self) -> str:
        return f"{self.api_base_url}/batch-score"

    @property
    def api_ask_complaints_url(self) -> str:
        return f"{self.api_base_url}/ask-complaints"

    @property
    def api_customer_intel_url(self) -> str:
        return f"{self.api_base_url}/customer-intel"

    @property
    def api_metrics_url(self) -> str:
        return f"{self.api_base_url}/metrics"


def load_config() -> DashboardConfig:
    """Load configuration from environment variables with sensible defaults.

    Returns
    -------
    DashboardConfig
        Frozen configuration object.
    """
    return DashboardConfig(
        api_base_url=os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/"),
        environment=os.environ.get("ENVIRONMENT", "development"),
        request_timeout=int(os.environ.get("REQUEST_TIMEOUT", "30")),
        dashboard_title=os.environ.get("DASHBOARD_TITLE", "Meridian Financial Intelligence"),
        page_icon=os.environ.get("PAGE_ICON", "🏦"),
    )


# Module-level singleton — imported everywhere
CONFIG = load_config()
