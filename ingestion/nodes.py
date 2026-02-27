"""
graph/nodes.py
==============
Pure execution nodes for the M-RAG ingestion pipeline.

Fixes applied vs original:
  1. next_step removed from every return dict  →  routing lives in graph.py only.
  2. status = "success" set explicitly in node_store  →  callers can rely on it.
  3. Hierarchical children are FLATTENED before embedding  →  every chunk gets
     an embedding and is stored individually (parents + children both stored).
  4. Image replacement uses element_id set (bulletproof, works with new reducer).
  5. model_name / vision_model pulled from config  →  fully runtime-configurable.
  6. Metrics always merged with existing dict (no silent loss).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from core.state import IngestionState
from core.loaders import get_loader_for_file
from core.processors.image_describer import ImageDescriber
from core.chunkers import get_chunker
from core.embedders import get_embedder
from core.storers.qdrant import QdrantStorer
from core.utils.logging import log_node
from core.elements import BaseElement, TextElement, ImageElement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _merge_metrics(state: IngestionState, new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new metric keys into the existing metrics dict."""
    base = dict(state.get("metrics") or {})
    base.update(new)
    return base


def _flatten_elements(elements: List[BaseElement]) -> List[BaseElement]:
    """
    Flatten hierarchical parent→children into a single list.
    Parents are kept (they carry the full-context text); children are appended
    so every chunk gets embedded and stored.

    Fix for: "Hierarchical Chunking Children Are Never Embedded or Stored"
    """
    flat: List[BaseElement] = []
    for el in elements:
        flat.append(el)
        if el.children:
            flat.extend(el.children)
            el.children = []   # prevent double-processing on re-runs
    return flat


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@log_node
def node_load(state: IngestionState) -> Dict[str, Any]:
    """Load raw elements from file (PDF, TXT, MD, PNG, JPG, …)."""
    file_path: str = state["file_path"]
    start = time.perf_counter()

    loader = get_loader_for_file(file_path)
    raw_elements = loader.load(file_path)

    duration = time.perf_counter() - start
    logger.info(f"Loaded {len(raw_elements)} raw elements in {duration:.2f}s")

    return {
        "elements": raw_elements,
        "metrics": _merge_metrics(state, {"load_time": duration, "raw_elements": len(raw_elements)}),
    }


@log_node
def node_describe_images(state: IngestionState) -> Dict[str, Any]:
    """
    Vision-LLM image description + replacement.

    Uses element_id set for safe removal even after reducer transforms.
    Skips gracefully when no images are present.
    """
    elements: List[BaseElement] = state.get("elements", [])
    config: Dict[str, Any] = state.get("config", {})

    image_elements = [el for el in elements if isinstance(el, ImageElement)]
    if not image_elements:
        logger.info("No images to describe → skipping")
        return {}   # empty dict → no state mutation

    describer = ImageDescriber(
        prompt=config.get("image_description_prompt"),
        vision_model=config.get("vision_model"),        # runtime-configurable
    )

    start = time.perf_counter()
    described_texts = describer.describe(image_elements)
    duration = time.perf_counter() - start

    # Replace image elements with their text descriptions (safe by ID)
    image_ids = {img.element_id for img in image_elements}
    non_image = [el for el in elements if el.element_id not in image_ids]
    new_elements = non_image + described_texts

    logger.info(
        f"Described {len(image_elements)} images → {len(described_texts)} "
        f"TextElements in {duration:.2f}s"
    )
    return {
        "elements": new_elements,
        "metrics": _merge_metrics(
            state, {"describe_time": duration, "images_described": len(image_elements)}
        ),
    }


@log_node
def node_chunk(state: IngestionState) -> Dict[str, Any]:
    """
    Chunk elements with the method specified in config.
    Only passes kwargs the selected chunker actually accepts.
    """
    elements: List[BaseElement] = state.get("elements", [])
    config: Dict[str, Any] = state.get("config", {})
    method: str = config.get("chunking_method", "recursive").lower()

    kwargs: Dict[str, Any] = {"chunk_overlap": config.get("chunk_overlap", 100)}

    if method == "hierarchical":
        kwargs.update(
            {
                "parent_chunk_size": config.get("parent_chunk_size", 2000),
                "child_chunk_size": config.get("child_chunk_size", 800),
            }
        )
    elif method == "recursive":
        kwargs["chunk_size"] = config.get("chunk_size", 1000)
    elif method == "semantic":
        kwargs["breakpoint_threshold"] = config.get("semantic_breakpoint", 0.8)

    logger.debug(f"Creating '{method}' chunker with kwargs: {list(kwargs.keys())}")
    chunker = get_chunker(method=method, **kwargs)

    start = time.perf_counter()
    chunks = chunker.chunk(elements)
    duration = time.perf_counter() - start

    # ── Fix: flatten hierarchical parent→child before proceeding ──────────
    flat_chunks = _flatten_elements(chunks)

    logger.info(
        f"Chunked → {len(chunks)} chunks (flattened to {len(flat_chunks)}) "
        f"using '{method}' in {duration:.2f}s"
    )
    return {
        "elements": flat_chunks,
        "metrics": _merge_metrics(
            state,
            {
                "chunk_time": duration,
                "chunks_pre_flatten": len(chunks),
                "chunks": len(flat_chunks),
                "chunking_method": method,
            },
        ),
    }


@log_node
def node_embed(state: IngestionState) -> Dict[str, Any]:
    """
    Embed all elements using the embedder specified in config.

    Fix: embedder model is pulled from config so it is fully runtime-configurable.
    The Qdrant storer already reads vector dimension from elements, so changing
    the model here automatically propagates a correct collection schema.
    """
    elements: List[BaseElement] = state.get("elements", [])
    config: Dict[str, Any] = state.get("config", {})

    embedder_type: str = config.get("embedding_type", "text")
    embedder_model: str = config.get("embedding_model", "all-MiniLM-L6-v2")

    embedder = get_embedder(type=embedder_type, model_name=embedder_model)

    # Build input list — text content or base64 fallback
    items_to_embed: List[Any] = []
    for el in elements:
        if hasattr(el, "content") and isinstance(el.content, str) and el.content.strip():
            items_to_embed.append(el.content)
        elif hasattr(el, "base64_data") and el.base64_data:
            items_to_embed.append(el.metadata.get("description", "image"))
        else:
            items_to_embed.append(str(el.content) if el.content else "")

    start = time.perf_counter()
    embeddings = embedder.embed(items_to_embed)
    duration = time.perf_counter() - start

    for el, emb in zip(elements, embeddings):
        el.embedding = emb
        el.metadata["embedding_model"] = embedder.model_name
        el.metadata["embedding_dim"] = embedder.dimension

    logger.info(
        f"Embedded {len(elements)} chunks with '{embedder_type}' "
        f"({embedder.model_name}, dim={embedder.dimension}) in {duration:.2f}s"
    )
    return {
        "elements": elements,
        "metrics": _merge_metrics(
            state,
            {
                "embed_time": duration,
                "embedder": embedder.model_name,
                "embedding_dim": embedder.dimension,
            },
        ),
    }


@log_node
def node_store(state: IngestionState) -> Dict[str, Any]:
    """
    Store everything to Qdrant.

    Fix: sets status = "success" so callers always get a deterministic signal.
    The QdrantStorer now derives vector config from the actual embedding dimension
    on the elements (configurable embedder fix).
    """
    from core.settings import get_settings   # lazy import to avoid circular

    elements: List[BaseElement] = state.get("elements", [])
    config: Dict[str, Any] = state.get("config", {})
    settings = get_settings()

    collection: str = config.get("collection_name", settings.default_collection)

    storer = QdrantStorer(
        host=config.get("qdrant_host", settings.qdrant_host),
        port=config.get("qdrant_port", settings.qdrant_port),
        api_key=config.get("qdrant_api_key", settings.qdrant_api_key),
    )

    start = time.perf_counter()
    stored_count = storer.store(elements, collection_name=collection)
    duration = time.perf_counter() - start

    logger.info(f"Stored {stored_count} vectors to '{collection}' in {duration:.2f}s")

    return {
        "status": "success",
        "metrics": _merge_metrics(
            state,
            {"store_time": duration, "stored": stored_count, "collection": collection},
        ),
    }