"""
rag/graph.py
============
Agentic RAG LangGraph with store probe + recursive loop.

Updated flow
------------

  init → probe_store ──┬── (store empty/missing) ──────────────► synthesize → END
                       │
                       └── (store ok) ──► plan ──► transform ──► retrieve
                                           ▲                          │
                                           │                       rerank
                                           │                          │
                                           └────── evaluate ◄─────────┘
                                                      │
                                           conf ≥ threshold
                                           OR iter = max ──────────► synthesize → END

Why probe first?
  Without it the agent would apply 5 iterations of strategy transformations
  against an empty or misconfigured store and still return nothing.
  The probe gate surfaces the problem immediately with a clear error message.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    node_evaluate,
    node_init,
    node_plan,
    node_probe,
    node_rerank,
    node_retrieve,
    node_synthesize,
    node_transform,
)
from .state import RAGState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────
def _route_after_probe(state: RAGState) -> Literal["plan", "synthesize"]:
    """If the store is empty or unreachable, skip straight to synthesize
    which will return a clear error message."""
    if not state.get("store_ready", False):
        logger.warning(
            f"[ROUTE] Store not ready ({state.get('probe_diagnostics')}) → synthesize"
        )
        return "synthesize"
    logger.info("[ROUTE] Store ready → plan")
    return "plan"


def _should_continue(state: RAGState) -> Literal["loop", "synthesize"]:
    """After evaluate: loop or finalize."""
    iteration  = state.get("iteration") or 0
    max_iter   = state.get("max_iterations") or 5
    confidence = state.get("answer_confidence") or 0.0
    threshold  = (state.get("config") or {}).get("confidence_threshold", 0.75)

    if confidence >= threshold:
        logger.info(f"[ROUTE] confidence={confidence:.2f} ≥ {threshold} → synthesize")
        return "synthesize"

    if iteration >= max_iter:
        logger.info(f"[ROUTE] max_iterations={max_iter} hit → synthesize")
        state["status"] = "max_iterations"
        return "synthesize"

    logger.info(f"[ROUTE] iter={iteration}/{max_iter} conf={confidence:.2f} → loop")
    return "loop"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_rag_graph() -> CompiledStateGraph:
    wf = StateGraph(RAGState)

    wf.add_node("init",       node_init)
    wf.add_node("probe",      node_probe)
    wf.add_node("plan",       node_plan)
    wf.add_node("transform",  node_transform)
    wf.add_node("retrieve",   node_retrieve)
    wf.add_node("rerank",     node_rerank)
    wf.add_node("evaluate",   node_evaluate)
    wf.add_node("synthesize", node_synthesize)

    wf.set_entry_point("init")
    wf.add_edge("init", "probe")

    # Gate: only proceed to planning if the store is usable
    wf.add_conditional_edges(
        "probe",
        _route_after_probe,
        {"plan": "plan", "synthesize": "synthesize"},
    )

    # Main pipeline
    wf.add_edge("plan",      "transform")
    wf.add_edge("transform", "retrieve")
    wf.add_edge("retrieve",  "rerank")
    wf.add_edge("rerank",    "evaluate")

    # Recursive back-edge
    wf.add_conditional_edges(
        "evaluate",
        _should_continue,
        {"loop": "plan", "synthesize": "synthesize"},
    )

    wf.add_edge("synthesize", END)
    return wf.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public runner
# ─────────────────────────────────────────────────────────────────────────────
async def run_rag(
    query: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .config import RAGConfig

    if config is None:
        config = RAGConfig().model_dump()

    config.setdefault("qdrant_host",    os.getenv("QDRANT_HOST", "localhost"))
    config.setdefault("qdrant_port",    int(os.getenv("QDRANT_PORT", "6333")))
    config.setdefault("qdrant_api_key", os.getenv("QDRANT_API_KEY"))

    graph = build_rag_graph()

    initial: RAGState = {
        "original_query":    query,
        "collection_name":   config.get("collection_name", "mrag_default"),
        "config":            config,
        "iteration":         0,
        "max_iterations":    config.get("max_iterations", 5),
        "status":            "running",
        "store_ready":       False,
        "vector_name":       None,
        "probe_diagnostics": "",
        "notepad":           [],
        "notepad_summary":   "",
        "active_queries":    [],
        "all_queries_tried": [],
        "retrieved_chunks":  [],
        "all_chunks":        [],
        "selected_chunks":   [],
        "chosen_strategy":   "",
        "current_task":      "",
        "answer_draft":      "",
        "answer_confidence": 0.0,
        "final_answer":      "",
        "answer_sources":    [],
        "error":             None,
    }
    try:
        from langfuse.langchain import CallbackHandler
        lf_handler = CallbackHandler()
        run_config = {
            "callbacks": [lf_handler],
            "run_name": "Rag",
            "tags": ["rag", "ingestion"],
        }
    except ImportError:
        logger.debug("Langfuse not installed — tracing skipped")

    try:
        final = await graph.ainvoke(initial,config=run_config or None)
        # final["answer_sources"] = []
        # for chunk in final.get("selected_chunks", []):
        #     if chunk.metadata.get('type') == 'image':  # Assuming images have 'type' metadata
        #         final["answer_sources"].append(chunk.metadata.get('source', 'Unknown image path'))
        #     else:
        #         # Full text for text chunks
        #         final["answer_sources"].append(chunk.page_content)

        return final
    except Exception as exc:
        logger.error(f"RAG pipeline crashed: {exc}", exc_info=True)
        return {**initial, "status": "failed", "error": str(exc), "final_answer": ""}