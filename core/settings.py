"""
core/settings.py
================
Cached Settings singleton — import get_settings() anywhere.
"""
from __future__ import annotations
from functools import lru_cache
from .config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()