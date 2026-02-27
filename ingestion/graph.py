"""
graph/graph.py
==============
LangGraph ingestion pipeline — clean conditional routing.

Fixes applied vs original:
  1. next_step removed from state entirely.  Routing functions inspect
     config directly so the graph is the single source of routing truth.
  2. node_router removed (was vestigial once next_step is gone).
  3. Langfuse callback wired via run_config, not a module-level singleton,
     so it picks up per-request metadata cleanly.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.state import IngestionState
from .nodes import node_load, node_describe_images, node_chunk, node_embed, node_store

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions — inspect config, NOT state.next_step
# ---------------------------------------------------------------------------
def _route_after_load(state: IngestionState) -> Literal["describe_images", "chunk"]:
    """Skip image description when disabled in config or when no images exist."""
    config = state.get("config", {})
    if not config.get("describe_images", True):
        return "chunk"
    # Check if any ImageElement was actually loaded
    from core.elements import ImageElement
    elements = state.get("elements", [])
    if any(isinstance(el, ImageElement) for el in elements):
        return "describe_images"
    return "chunk"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_ingestion_graph() -> CompiledStateGraph:
    """Build and compile the M-RAG ingestion graph."""
    workflow = StateGraph(IngestionState)

    workflow.add_node("load", node_load)
    workflow.add_node("describe_images", node_describe_images)
    workflow.add_node("chunk", node_chunk)
    workflow.add_node("embed", node_embed)
    workflow.add_node("store", node_store)

    workflow.set_entry_point("load")

    # Conditional branch after load
    workflow.add_conditional_edges(
        "load",
        _route_after_load,
        {"describe_images": "describe_images", "chunk": "chunk"},
    )

    # Linear tail
    workflow.add_edge("describe_images", "chunk")
    workflow.add_edge("chunk", "embed")
    workflow.add_edge("embed", "store")
    workflow.add_edge("store", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
async def run_ingestion(
    file_path: str,
    config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async entry point for a single ingestion job.

    Parameters
    ----------
    file_path : str
        Absolute path to the file to ingest.
    config : dict, optional
        Serialised PipelineConfig (call `.model_dump()` before passing).
    run_name : str, optional
        Human-readable label for Langfuse tracing.
    """
    from core.config import PipelineConfig
    from core.settings import get_settings

    settings = get_settings()
    document_id = os.path.basename(file_path).replace(".", "_").replace(" ", "_")
    start = time.perf_counter()

    logger.info(f"🚀 Starting ingestion for: {document_id}")

    if config is None:
        config = PipelineConfig.from_settings(settings).model_dump()

    graph = build_ingestion_graph()

    initial_state: IngestionState = {
        "file_path": file_path,
        "document_id": document_id,
        "elements": [],
        "config": config,
        "metrics": {},
        "status": "running",
        "error": None,
    }

    # ── Langfuse tracing (optional — skipped if env vars absent) ──────────
    run_config: Dict[str, Any] = {}
    try:
        from langfuse.langchain import CallbackHandler
        lf_handler = CallbackHandler()
        run_config = {
            "callbacks": [lf_handler],
            "run_name": run_name or f"Ingest_{document_id}",
            "tags": ["rag", "ingestion"],
            "metadata": {"file_path": file_path, "document_id": document_id},
        }
    except ImportError:
        logger.debug("Langfuse not installed — tracing skipped")

    try:
        final_state = await graph.ainvoke(initial_state, config=run_config or None)
        duration = time.perf_counter() - start
        logger.info(
            f"✅ Ingestion complete | Doc: {document_id} | "
            f"Status: {final_state.get('status')} | Time: {duration:.2f}s"
        )
        return final_state

    except Exception as exc:
        duration = time.perf_counter() - start
        logger.error(f"❌ Pipeline crashed for {document_id}: {exc}", exc_info=True)
        return {
            **initial_state,
            "status": "failed",
            "error": str(exc),
            "metrics": {**(initial_state.get("metrics") or {}), "total_time": duration},
        }