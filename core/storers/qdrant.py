"""
core/storers/qdrant.py
======================
Qdrant storer with dynamic collection schema.

Fix applied:
  Collection schema (vector dimension) is derived from the actual embeddings
  on the elements at store-time rather than being hardcoded.  This means
  swapping from all-MiniLM (384-d) to clip-ViT-B-32 (512-d) in config
  just works — the correct collection is created automatically.

  Also returns stored_count so node_store can log it.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..elements import BaseElement, ImageElement, TextElement
from .base import BaseStorer

logger = logging.getLogger(__name__)

VECTOR_NAME = "default"   # single named vector per point


class QdrantStorer(BaseStorer):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
    ):
        self.client = QdrantClient(host=host, port=port, api_key=api_key,https=False)
        logger.info(f"Connected to Qdrant at {host}:{port}")

    # ------------------------------------------------------------------
    def _ensure_collection(self, collection_name: str, vector_dim: int) -> None:
        """
        Create the collection if it doesn't exist.
        If it does exist, verify dimension compatibility and warn on mismatch.
        """
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    VECTOR_NAME: models.VectorParams(
                        size=vector_dim,
                        distance=models.Distance.COSINE,
                    )
                },
            )
            logger.info(
                f"Created collection '{collection_name}' with dim={vector_dim}"
            )
        else:
            # Validate existing dimension matches
            info = self.client.get_collection(collection_name)
            existing_dim = (
                info.config.params.vectors[VECTOR_NAME].size
                if isinstance(info.config.params.vectors, dict)
                else info.config.params.vectors.size
            )
            if existing_dim != vector_dim:
                logger.warning(
                    f"Collection '{collection_name}' exists with dim={existing_dim} "
                    f"but current embedder produces dim={vector_dim}. "
                    "Consider using a different collection_name to avoid mixed embeddings."
                )

    # ------------------------------------------------------------------
    def store(self, elements: List[BaseElement], collection_name: str) -> int:
        """
        Upload elements to Qdrant.  Returns the number of points stored.
        """
        if not elements:
            return 0

        # ── Derive vector dimension from first embedded element ───────
        embedded = [el for el in elements if getattr(el, "embedding", None)]
        if not embedded:
            logger.warning("No embedded elements to store — skipping.")
            return 0

        vector_dim: int = len(embedded[0].embedding)
        self._ensure_collection(collection_name, vector_dim)

        # ── Build point structs ───────────────────────────────────────
        points: List[models.PointStruct] = []
        for el in embedded:
            payload: Dict = {
                "type": type(el).__name__,
                "metadata": getattr(el, "metadata", {}),
            }
            if isinstance(el, TextElement):
                payload["content"] = el.content
            elif isinstance(el, ImageElement):
                payload["base64_data"] = getattr(el, "base64_data", None)

            points.append(
                models.PointStruct(
                    id=el.element_id,
                    vector={VECTOR_NAME: el.embedding},
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(
            f"✅ Stored {len(points)} points in '{collection_name}' (dim={vector_dim})"
        )
        return len(points)