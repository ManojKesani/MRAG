"""
rag/reranker.py
===============
Re-ranking layer — sorts retrieved chunks by relevance to the query.

Two modes
---------
"llm"   — ask the LLM to score each chunk (token-efficient: one call, JSON list)
"score" — passthrough, keep Qdrant cosine scores as-is (zero extra tokens)

LLM re-ranking prompt is kept tiny:
  • ≤5 chunks passed at once, each truncated to 120 chars
  • LLM returns only a JSON list of IDs in ranked order
  • Total prompt ≈ 250 tokens
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from .state import RetrievedChunk

logger = logging.getLogger(__name__)

RerankMode = Literal["llm", "score"]

_RERANK_SYSTEM = (
    "You are a relevance ranker. Given a query and document snippets, "
    "return ONLY a JSON array of document IDs ordered best-to-worst. "
    "Example: [\"id3\", \"id1\", \"id2\"]"
)


def rerank(
    llm,
    query: str,
    chunks: List[RetrievedChunk],
    mode: RerankMode = "llm",
    top_n: int = 4,
) -> List[RetrievedChunk]:
    """
    Re-rank `chunks` by relevance to `query`.
    Returns at most `top_n` chunks.
    """
    if not chunks:
        return []

    if mode == "score" or len(chunks) <= 1:
        return chunks[:top_n]

    # ── LLM re-ranking ────────────────────────────────────────────────────
    # Build a compact snippet list (max 5 chunks, 120 chars each)
    candidates = chunks[:5]
    snippets = "\n".join(
        f'ID:{c["id"][:8]} | {c["content"][:120].replace(chr(10), " ")}'
        for c in candidates
    )
    prompt = f"Query: {query}\n\nSnippets:\n{snippets}\n\nReturn JSON array of IDs ranked best-first."

    try:
        raw = llm.invoke(
            [SystemMessage(content=_RERANK_SYSTEM), HumanMessage(content=prompt)]
        ).content.strip()

        # Parse JSON — be lenient
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if json_match:
            ranked_ids = json.loads(json_match.group())
        else:
            raise ValueError("No JSON array found")

        # Map partial IDs back to full IDs
        id_map = {c["id"][:8]: c for c in candidates}
        reranked = []
        for short_id in ranked_ids:
            if short_id in id_map:
                reranked.append(id_map[short_id])
        # Append any not mentioned by LLM at the end
        mentioned = {c["id"] for c in reranked}
        reranked += [c for c in candidates if c["id"] not in mentioned]

        # Append any beyond the 5-candidate window at original scores
        reranked += chunks[5:]

        logger.info(f"[rerank/llm] Reordered {len(reranked)} chunks")
        return reranked[:top_n]

    except Exception as exc:
        logger.warning(f"LLM reranking failed ({exc}) — falling back to score order")
        return chunks[:top_n]


def format_context(chunks: List[RetrievedChunk], max_chars: int = 1600) -> str:
    """
    Format selected chunks into a compact context string for prompt injection.
    Hard-truncated to max_chars to protect token budget.
    """
    parts = []
    total = 0
    for i, c in enumerate(chunks):
        src = c["metadata"].get("source", c["metadata"].get("page", "?"))
        snippet = f"[{i+1}|src:{src}] {c['content']}"
        if total + len(snippet) > max_chars:
            remaining = max_chars - total
            if remaining > 40:
                parts.append(snippet[:remaining] + "…")
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n---\n".join(parts)