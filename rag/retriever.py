"""
rag/retriever.py
================
Modernized Qdrant retriever using the Universal Query API (v1.10+).
Features: Store probing, multi-query fusion, and graceful fallbacks.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from .state import RetrievedChunk

logger = logging.getLogger(__name__)

# Must match your ingestion vector name (default is standard for Qdrant)
_PREFERRED_VECTOR_NAME = "default"

@dataclass
class StoreProbeResult:
    collection_exists: bool = False
    point_count: int = 0
    vector_name: Optional[str] = None
    initial_chunks: List[RetrievedChunk] = field(default_factory=list)
    diagnostics: str = ""

class MultiQueryRetriever:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        embedder_type: str = "text",
        embedder_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        score_threshold: float = 0.25,
    ):
        # Explicitly disable HTTPS for local Docker setups
        self.client = QdrantClient(host=host, port=port, api_key=api_key, https=False)
        self.top_k = top_k
        self.score_threshold = score_threshold

        # Load Embedder
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from core.embedders import get_embedder
        self.embedder = get_embedder(type=embedder_type, model_name=embedder_model)
        
        logger.info(f"Retriever Initialized | {embedder_model} | top_k={top_k}")

    def _detect_vector_name(self, collection_name: str) -> Optional[str]:
        """Detects if the collection uses named vectors or a single unnamed vector."""
        try:
            info = self.client.get_collection(collection_name)
            vectors_cfg = info.config.params.vectors
            if isinstance(vectors_cfg, dict):
                return _PREFERRED_VECTOR_NAME if _PREFERRED_VECTOR_NAME in vectors_cfg else next(iter(vectors_cfg))
            return None # Unnamed vector
        except Exception:
            return None

    def _search_safe(
        self,
        embedding: List[float],
        collection_name: str,
        vector_name: Optional[str],
        limit: int,
        threshold: float,
    ) -> List[Any]:
        """
        Executes search using the modern query_points API.
        Falls back to legacy search if query_points is unavailable.
        """
        try:
            # The modern way (Qdrant 1.10+)
            response = self.client.query_points(
                collection_name=collection_name,
                query=embedding,
                using=vector_name,
                limit=limit,
                score_threshold=threshold if threshold > 0 else None,
                with_payload=True
            )
            return response.points
        except AttributeError:
            # Fallback for unexpected environment issues
            logger.warning("query_points not found, falling back to legacy search")
            return self.client.search(
                collection_name=collection_name,
                query_vector=(vector_name, embedding) if vector_name else embedding,
                limit=limit,
                score_threshold=threshold if threshold > 0 else None,
                with_payload=True
            )

    def probe_collection(self, collection_name: str, raw_query: str) -> StoreProbeResult:
        """Verifies store health before agent begins heavy reasoning."""
        result = StoreProbeResult()
        try:
            if not self.client.collection_exists(collection_name):
                result.diagnostics = f"Collection '{collection_name}' missing."
                return result
            
            result.collection_exists = True
            info = self.client.get_collection(collection_name)
            result.point_count = info.points_count or 0
            result.vector_name = self._detect_vector_name(collection_name)

            if result.point_count > 0:
                [emb] = self.embedder.embed([raw_query])
                hits = self._search_safe(emb, collection_name, result.vector_name, self.top_k, self.score_threshold)
                result.initial_chunks = self._hits_to_chunks(hits)
                result.diagnostics = f"Ready. Found {len(result.initial_chunks)} relevant chunks."
            else:
                result.diagnostics = "Collection is empty."
        except Exception as e:
            result.diagnostics = f"Connection Error: {str(e)}"
        
        return result

    def _hits_to_chunks(self, hits: List[Any]) -> List[Dict[str, Any]]:
        """Convert Qdrant hits into a list of dictionaries."""
        return [
            {
                "id": str(hit.id),
                "content": hit.payload.get("content", ""),
                "score": float(hit.score),
                "metadata": hit.payload.get("metadata", {})
            } for hit in hits
        ]

    def retrieve(self, queries: List[str], collection_name: str) -> List[Dict[str, Any]]:
        """Performs multi-query batch retrieval and deduplication."""
        if not queries: return []
        
        v_name = self._detect_vector_name(collection_name)
        embeddings = self.embedder.embed(queries)
        
        all_chunks = {}
        for q_text, emb in zip(queries, embeddings):
            hits = self._search_safe(emb, collection_name, v_name, self.top_k, self.score_threshold)
            for chunk in self._hits_to_chunks(hits):
                # Use bracket notation [] because 'chunk' is a dict
                chunk_id = chunk["id"]
                if chunk_id not in all_chunks or chunk["score"] > all_chunks[chunk_id]["score"]:
                    all_chunks[chunk_id] = chunk

        # Sort by score and cap
        sorted_results = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:self.top_k * 2]