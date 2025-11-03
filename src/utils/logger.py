# src/utils/logger.py
"""
logger.py
----------
Central logging configuration for the Financial_Document_Analyzer project.
All modules should import `get_logger(__name__)`.
Logs are written both to console and to data/logs/pipeline.log.
"""

import logging
from pathlib import Path

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"


def setup_logger() -> None:
    """Configure global logging format, levels, and handlers."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),                  # Console
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),  # File
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for a specific module."""
    if not logging.getLogger().handlers:
        setup_logger()
    return logging.getLogger(name)