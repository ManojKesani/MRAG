"""
rag/nodes.py
============
LangGraph nodes for the agentic RAG loop.

Execution order
---------------
  init → probe_store → (store empty → synthesize error)
                     → (store ok)  → plan → transform → retrieve → rerank → evaluate
                                     ↑___________________________________|  (loop)
                                                                         → synthesize

probe_store  (new)
  Verifies the collection exists, has points, detects the named-vector field,
  and runs the original query verbatim.  Results seed the first iteration so the
  agent always has *something* before it applies strategies.

node_retrieve  (updated)
  Receives `vector_name` from state (set by probe) — no more guessing.
  Passes probe's initial chunks as a fallback if advanced retrieval returns nothing.

Token budget per node
---------------------
  node_probe       0 LLM   (pure Qdrant + embed)
  node_plan        ~300 t
  node_transform   ~350 t
  node_retrieve    0 LLM
  node_rerank      ~250 t
  node_evaluate    ~600 t
  node_synthesize  ~700 t
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .notepad import build_minimal_prompt, compress_notepad, make_notepad_entry
from .reranker import format_context, rerank
from .retriever import MultiQueryRetriever
from .state import RAGState, RetrievedChunk
from .strategies import ALL_STRATEGIES, run_strategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _get_llm(model: str, temperature: float, max_tokens: int) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(model=model, api_key=api_key,
                    temperature=temperature, max_tokens=max_tokens)


def _llm(cfg: Dict[str, Any]) -> ChatGroq:
    return _get_llm(
        model=cfg.get("llm_model", "llama3-70b-8192"),
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens_per_call", 512),
    )


def _cfg(state: RAGState) -> Dict[str, Any]:
    return state.get("config") or {}


def _build_retriever(cfg: Dict[str, Any]) -> MultiQueryRetriever:
    """Construct a MultiQueryRetriever from the run config."""
    return MultiQueryRetriever(
        host=cfg.get("qdrant_host", "localhost"),
        port=int(cfg.get("qdrant_port", 6333)),
        api_key=cfg.get("qdrant_api_key"),
        embedder_type=cfg.get("embedding_type", "text"),
        embedder_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        top_k=cfg.get("top_k", 5),
        score_threshold=cfg.get("score_threshold", 0.25),
    )


# ─────────────────────────────────────────────────────────────────────────────
# node_init
# ─────────────────────────────────────────────────────────────────────────────
def node_init(state: RAGState) -> Dict[str, Any]:
    cfg = _cfg(state)
    logger.info(f"[INIT] Query: {state['original_query'][:80]}")
    return {
        "iteration": 0,
        "max_iterations": cfg.get("max_iterations", 5),
        "status": "running",
        "store_ready": False,
        "vector_name": None,
        "probe_diagnostics": "",
        "notepad": [],
        "notepad_summary": f"Query: {state['original_query']}\nNo previous iterations.",
        "all_queries_tried": [],
        "all_chunks": [],
        "answer_confidence": 0.0,
        "answer_draft": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# node_probe  ★ NEW
# ─────────────────────────────────────────────────────────────────────────────
def node_probe(state: RAGState) -> Dict[str, Any]:
    """
    Verify the vector store before running any strategy.

    1. Check collection exists and is non-empty.
    2. Detect the correct named-vector field — stored in state so every
       subsequent retrieve call uses the right name without re-detecting.
    3. Run the original query verbatim (no transformation) as a baseline.
       If this already yields good chunks the planner knows the store is
       responsive; if scores are low it can choose aggressive strategies.

    Routing: if store is empty / missing, _route_after_probe → "no_store"
             which jumps directly to synthesize with a clear error message.
    """
    cfg = _cfg(state)
    query = state["original_query"]
    collection = cfg.get("collection_name", "mrag_default")

    retriever = _build_retriever(cfg)
    probe = retriever.probe_collection(collection, query)

    logger.info(f"[PROBE] {probe.diagnostics}")

    if not probe.collection_exists or probe.point_count == 0:
        return {
            "store_ready": False,
            "probe_diagnostics": probe.diagnostics,
            "status": "no_store",
        }

    # Seed the first iteration with probe's raw results — they're free
    # (already retrieved) and give the planner a baseline to improve on.
    initial_chunks = probe.initial_chunks or []

    # Build a useful notepad summary so node_plan knows what the probe found
    probe_summary = (
        f"Query: {query}\n"
        f"[probe] direct_search | {len(initial_chunks)} chunks | "
        f"{probe.diagnostics}"
    )

    return {
        "store_ready": True,
        "vector_name": probe.vector_name,
        "probe_diagnostics": probe.diagnostics,
        # Seed state with raw results — the first rerank/evaluate can use these
        "retrieved_chunks": initial_chunks,
        "all_chunks": initial_chunks,       # append reducer
        "notepad_summary": probe_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# node_plan
# ─────────────────────────────────────────────────────────────────────────────
def node_plan(state: RAGState) -> Dict[str, Any]:
    cfg = _cfg(state)
    iteration = (state.get("iteration") or 0) + 1
    strategy_pref = cfg.get("strategy", "auto")
    enabled = cfg.get("enabled_strategies", ALL_STRATEGIES)

    if strategy_pref != "auto":
        chosen = strategy_pref
        task = f"Apply {chosen} to the query."
    else:
        llm = _llm(cfg)
        notepad_summary = state.get("notepad_summary", "")
        already_tried = [e["strategy"] for e in (state.get("notepad") or [])]
        remaining = [s for s in enabled if s not in already_tried] or enabled

        prompt = build_minimal_prompt(
            system_role="RAG planner. Pick ONE strategy from the list.",
            notepad_summary=notepad_summary,
            current_task=(
                f"Iteration {iteration}. Available: {', '.join(remaining)}.\n"
                "Reply with ONLY the strategy name."
            ),
        )
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        chosen = next((s for s in remaining if s in raw), remaining[0])
        task = f"Apply {chosen} to find information not yet retrieved."

    logger.info(f"[PLAN] iter={iteration} strategy={chosen}")
    return {"iteration": iteration, "chosen_strategy": chosen, "current_task": task}


# ─────────────────────────────────────────────────────────────────────────────
# node_transform
# ─────────────────────────────────────────────────────────────────────────────
def node_transform(state: RAGState) -> Dict[str, Any]:
    cfg = _cfg(state)
    llm = _llm(cfg)
    strategy = state.get("chosen_strategy", "query_rewriting")
    query = state["original_query"]
    notepad_summary = state.get("notepad_summary", "")

    queries = run_strategy(strategy, llm, query, notepad_summary)

    # Deduplicate against already-tried queries
    tried = set(state.get("all_queries_tried") or [])
    fresh = [q for q in queries if q not in tried]
    if not fresh:
        fresh = queries

    logger.info(f"[TRANSFORM/{strategy}] {len(fresh)} queries")
    return {"active_queries": fresh}


# ─────────────────────────────────────────────────────────────────────────────
# node_retrieve  (updated — uses pre-detected vector_name from probe)
# ─────────────────────────────────────────────────────────────────────────────

def node_retrieve(state: RAGState) -> Dict[str, Any]:
    cfg = _cfg(state)
    queries = state.get("active_queries") or [state["original_query"]]
    collection = cfg.get("collection_name", "mrag_default")
    
    # We no longer pass vector_name here because the new retriever 
    # detects it automatically via the Universal Query API.
    retriever = _build_retriever(cfg)
    chunks = retriever.retrieve(
        queries=queries,
        collection_name=collection,
    )

    # Fallback: if strategy retrieval returns nothing, reuse probe's raw chunks
    if not chunks:
        fallback = state.get("retrieved_chunks") or []
        if fallback:
            logger.warning(
                "[RETRIEVE] Strategy returned 0 chunks — reusing probe baseline "
                f"({len(fallback)} chunks). Consider lowering score_threshold."
            )
            chunks = fallback

    logger.info(f"[RETRIEVE] {len(chunks)} chunks from {len(queries)} queries")
    return {
        "retrieved_chunks": chunks,
        "all_queries_tried": queries,
        "all_chunks": chunks,
    }
# ─────────────────────────────────────────────────────────────────────────────
# node_rerank
# ─────────────────────────────────────────────────────────────────────────────
def node_rerank(state: RAGState) -> Dict[str, Any]:
    cfg = _cfg(state)
    llm = _llm(cfg)
    chunks = state.get("retrieved_chunks") or []
    query = state["original_query"]
    mode = cfg.get("rerank_mode", "llm")
    top_n = cfg.get("rerank_top_n", 4)

    selected = rerank(llm, query, chunks, mode=mode, top_n=top_n)
    logger.info(f"[RERANK] {len(selected)} chunks selected (mode={mode})")
    return {"selected_chunks": selected}


# ─────────────────────────────────────────────────────────────────────────────
# node_evaluate
# ─────────────────────────────────────────────────────────────────────────────
def node_evaluate(state: RAGState) -> Dict[str, Any]:
    cfg = _cfg(state)
    llm = _llm(cfg)
    iteration = state.get("iteration", 1)
    query = state["original_query"]
    selected = state.get("selected_chunks") or []
    draft = state.get("answer_draft") or ""

    context = format_context(selected, max_chars=1200)
    notepad_summary = state.get("notepad_summary", "")

    task = (
        "Given the context, produce a short answer draft.\n"
        "Then on a new line: CONFIDENCE: <0.0–1.0>\n"
        "Then: KEY_FINDING: <one sentence>\n"
        + (f"Previous draft: {draft[:200]}" if draft else "")
    )

    prompt = build_minimal_prompt(
        system_role="RAG evaluator. Be concise.",
        notepad_summary=notepad_summary,
        current_task=task,
        context_snippet=context,
    )

    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    conf_match = re.search(r"CONFIDENCE[:\s]+([\d.]+)", raw, re.IGNORECASE)
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    confidence = max(0.0, min(1.0, confidence))

    kf_match = re.search(r"KEY_FINDING[:\s]+(.+?)(?:\n|$)", raw, re.IGNORECASE)
    key_finding = kf_match.group(1).strip() if kf_match else raw[:100]

    answer_part = re.split(r"CONFIDENCE[:\s]", raw, flags=re.IGNORECASE)[0].strip()
    new_draft = answer_part if answer_part else draft

    logger.info(f"[EVALUATE] iter={iteration} confidence={confidence:.2f}")

    entry = make_notepad_entry(
        iteration=iteration,
        strategy=state.get("chosen_strategy", "?"),
        queries_used=state.get("active_queries") or [],
        chunks_found=len(selected),
        key_findings=key_finding,
    )
    new_entries = (state.get("notepad") or []) + [entry]
    new_summary = compress_notepad(new_entries, query)

    return {
        "answer_draft": new_draft,
        "answer_confidence": confidence,
        "notepad": [entry],
        "notepad_summary": new_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# node_synthesize
# ─────────────────────────────────────────────────────────────────────────────
def node_synthesize(state: RAGState) -> Dict[str, Any]:
    cfg = _cfg(state)
    status = state.get("status", "running")
    query = state["original_query"]

    # ── Hard-fail path: store was empty / missing ─────────────────────────
    if status == "no_store":
        diag = state.get("probe_diagnostics", "No data found in the vector store.")
        logger.warning(f"[SYNTHESIZE] Aborting — store not ready: {diag}")
        return {
            "final_answer": (
                f"❌ Cannot answer: {diag}\n\n"
                "Please ingest documents into the collection before querying."
            ),
            "answer_sources": [],
            "status": "failed",
        }

    llm = _llm(cfg)
    draft = state.get("answer_draft") or ""
    notepad_summary = state.get("notepad_summary", "")
    selected = state.get("selected_chunks") or []
    language = cfg.get("answer_language", "English")
    include_sources = cfg.get("include_sources", True)

    context = format_context(selected, max_chars=1400)

    task = (
        f"Write a complete, well-structured final answer in {language}.\n"
        f"Base it on the context and the draft below.\n"
        f"Draft: {draft[:300]}"
        + ("\nEnd with 'Sources: <list>'." if include_sources else "")
    )

    prompt = build_minimal_prompt(
        system_role="RAG answer synthesizer. Write clearly and cite evidence.",
        notepad_summary=notepad_summary,
        current_task=task,
        context_snippet=context,
    )

    final = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    sources: List[str] = []
    if include_sources:
        for c in selected:
            src = c["metadata"].get("source") or c["metadata"].get("page")
            if src and str(src) not in sources:
                sources.append(str(src))

    if status == "running":
        status = "success"

    logger.info(
        f"[SYNTHESIZE] Done. iters={state.get('iteration',0)} "
        f"conf={state.get('answer_confidence',0):.2f}"
    )
    return {"final_answer": final, "answer_sources": sources, "status": status}