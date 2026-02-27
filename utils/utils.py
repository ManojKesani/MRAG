"""Misc helpers — file handling, ID generation, etc."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def generate_id() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())


def file_hash(path: str | Path, algo: str = "sha256") -> str:
    """Return hex digest of a file for deduplication."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if missing; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"