"""Centralized logger setup."""

from __future__ import annotations

import logging
import os

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger at the configured level."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger