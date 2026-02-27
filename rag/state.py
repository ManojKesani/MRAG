"""
rag/state.py
============
LangGraph state for the agentic RAG loop.

Token-efficiency design
-----------------------
The LLM never sees the full conversation history.
It only ever receives:
  • system_prompt        (~80 tokens, never changes)
  • notepad_summary      (~150 tokens, compressed rolling summary)
  • current_task         (~80 tokens, what this iteration must do)
  • retrieved_context    (~400 tokens, top-k re-ranked snippets)
  ─────────────────────────────────────────────────────────────
  Total per call        ≈ 700 tokens  (vs thousands in naive RAG)

Full retrieved chunks, intermediate reasoning, etc. live in state
but are NEVER sent raw to the LLM — only the compressed summary is.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


# ── Replace reducer (same as ingestion state) ─────────────────────────────
def _replace(left, right):
    return right if right is not None else left


def _append(left: List, right: Optional[List]) -> List:
    """Append-only reducer for logs / retrieved chunks across iterations."""
    if right is None:
        return left
    return left + right


# ---------------------------------------------------------------------------
class RetrievedChunk(TypedDict):
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class NotepadEntry(TypedDict):
    iteration: int
    strategy: str
    queries_used: List[str]
    chunks_found: int
    key_findings: str      # 1-3 sentence compressed finding — what this iteration learned


class RAGState(TypedDict, total=False):
    # ── Inputs (set once) ─────────────────────────────────────────────────
    original_query: str
    collection_name: str
    config: Dict[str, Any]        # RAGConfig serialised to dict

    # ── Agent loop controls ───────────────────────────────────────────────
    iteration: int                # current loop counter
    max_iterations: int           # recursion limit (default 5)
    status: str                   # "running" | "success" | "failed" | "max_iterations"

    # ── Query evolution ───────────────────────────────────────────────────
    active_queries: Annotated[List[str], _replace]   # queries used THIS iteration
    all_queries_tried: Annotated[List[str], _append] # every query ever tried

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_chunks: Annotated[List[RetrievedChunk], _replace]   # current iter chunks
    all_chunks: Annotated[List[RetrievedChunk], _append]          # deduplicated accumulation

    # ── Reranked / selected context ───────────────────────────────────────
    selected_chunks: Annotated[List[RetrievedChunk], _replace]    # after reranking

    # ── Token-efficient memory ────────────────────────────────────────────
    notepad: Annotated[List[NotepadEntry], _append]   # structured iteration log
    notepad_summary: str   # ≤150-token compressed summary sent to LLM each call

    # ── Current iteration plan ────────────────────────────────────────────
    chosen_strategy: str   # which strategy the planner chose this iteration
    current_task: str      # ≤80-token task description sent to LLM

    # ── Answer quality ────────────────────────────────────────────────────
    answer_draft: str      # working answer
    answer_confidence: float   # 0.0–1.0 self-assessed confidence
    final_answer: str          # set only when status = "success"
    answer_sources: List[str]  # source metadata for citations

    # ── Store probe (set by node_probe, read by all subsequent nodes) ────
    store_ready: bool               # True only when collection exists + has points
    vector_name: Optional[str]      # detected Qdrant vector name for this collection
    probe_diagnostics: str          # human-readable probe summary for notepad / UI

    # ── Error tracking ────────────────────────────────────────────────────
    error: Optional[str]