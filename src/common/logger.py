"""
src/common/logger.py
---------------------
Reusable, structured logging factory for Meridian Financial.

Features
--------
* Consistent format across all modules
* Configurable level (controlled by ``settings.log_level``)
* No duplicate handlers even when ``get_logger`` is called multiple times
* JSON-compatible timestamps (ISO-8601)

Usage
-----
  from src.common.logger import get_logger

  logger = get_logger(__name__)
  logger.info("Ingestion started", extra={"rows": 41188})
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


# ── Default log format ────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _resolve_level(level: str) -> int:
    """Convert a string log level to its ``logging`` integer constant.

    Parameters
    ----------
    level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
        (case-insensitive).

    Returns
    -------
    int
        Corresponding ``logging`` level integer.

    Raises
    ------
    ValueError
        If *level* is not a recognised level name.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(
            f"Unknown log level {level!r}. "
            f"Choose from: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    return numeric


def configure_root_logger(level: Optional[str] = None) -> None:
    """Configure the root logger with the project-standard handler.

    Safe to call multiple times — idempotent.

    Parameters
    ----------
    level:
        Log level string.  Defaults to ``settings.log_level`` if ``None``.
    """
    # Avoid circular import: import settings lazily inside the function
    if level is None:
        try:
            from src.common.config import settings  # noqa: PLC0415
            level = settings.log_level
        except Exception:
            level = "INFO"

    root = logging.getLogger()

    # Only add a handler if none exist (prevents duplicate log lines)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)

    root.setLevel(_resolve_level(level))


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a named logger, ensuring the root logger is configured.

    Calling ``get_logger`` multiple times with the same *name* is safe and
    returns the same underlying ``logging.Logger`` instance (Python's logging
    module is a global registry).

    Parameters
    ----------
    name:
        Logger name — pass ``__name__`` from the calling module.
    level:
        Optional per-logger level override.  When ``None``, the logger
        inherits from the root logger.

    Returns
    -------
    logging.Logger

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Pipeline started")
    2024-01-01T12:00:00 | INFO     | my.module | Pipeline started
    """
    configure_root_logger()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(_resolve_level(level))
    return logger
