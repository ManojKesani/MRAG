"""
rag/strategies.py
=================
Query transformation strategies.  Each strategy makes ONE small LLM call
and returns a list of query strings for retrieval.

Every call is designed to be ≤400 tokens total (prompt + response).

Strategies
----------
1. query_expansion      — add synonyms / related terms
2. query_rewriting      — rephrase for semantic search
3. query_decomposition  — split complex query into sub-questions
4. step_back_prompting  — abstract to a higher-level question
5. hyde                 — generate a hypothetical document excerpt
6. multi_query          — generate N diverse phrasings
7. sub_query            — identify atomic sub-queries (for multi-hop)
"""
from __future__ import annotations

import logging
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a search query specialist. "
    "Respond ONLY with the requested queries, one per line, no numbering, no preamble."
)


def _call(llm, messages) -> str:
    resp = llm.invoke(messages)
    return resp.content.strip()


def _parse_lines(raw: str, max_items: int = 5) -> List[str]:
    lines = [
        re.sub(r"^[\d\.\-\*\s]+", "", ln).strip()
        for ln in raw.splitlines()
        if ln.strip()
    ]
    return [l for l in lines if l][:max_items]


# ─────────────────────────────────────────────────────────────────────────────

def query_expansion(llm, query: str, notepad_summary: str) -> List[str]:
    """Add synonyms and related terms to the original query."""
    prompt = (
        f"Original query: {query}\n\n"
        "Write 3 expanded versions that add relevant synonyms or domain terms. "
        "One per line."
    )
    raw = _call(llm, [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    result = _parse_lines(raw)
    logger.info(f"[expansion] {len(result)} expanded queries")
    return result or [query]


def query_rewriting(llm, query: str, notepad_summary: str) -> List[str]:
    """Rewrite query for better semantic search retrieval."""
    prompt = (
        f"Original query: {query}\n\n"
        "Rewrite this as 2 different phrasings optimised for dense vector retrieval. "
        "Use noun phrases, avoid stop words. One per line."
    )
    raw = _call(llm, [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    result = _parse_lines(raw, max_items=3)
    logger.info(f"[rewriting] {len(result)} rewritten queries")
    return result or [query]


def query_decomposition(llm, query: str, notepad_summary: str) -> List[str]:
    """Break a complex query into independent sub-questions."""
    prompt = (
        f"Complex query: {query}\n\n"
        "Break this into 2-4 simpler, independent sub-questions that together answer it. "
        "One per line."
    )
    raw = _call(llm, [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    result = _parse_lines(raw, max_items=4)
    logger.info(f"[decomposition] {len(result)} sub-questions")
    return result or [query]


def step_back_prompting(llm, query: str, notepad_summary: str) -> List[str]:
    """Abstract the query to a higher-level concept."""
    prompt = (
        f"Specific query: {query}\n\n"
        "Write 1-2 more general / abstract questions that would provide "
        "background knowledge needed to answer the specific query. "
        "One per line."
    )
    raw = _call(llm, [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    result = _parse_lines(raw, max_items=2)
    # Always include original so we don't drift too far
    return result + [query] if result else [query]


def hyde(llm, query: str, notepad_summary: str) -> List[str]:
    """
    HyDE — generate a hypothetical document excerpt that would answer the query.
    The excerpt itself becomes the retrieval query (embed it, not the text).
    """
    prompt = (
        f"Query: {query}\n\n"
        "Write a short 2-sentence hypothetical document excerpt that would "
        "perfectly answer this query. Be factual and specific."
    )
    raw = _call(llm, [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    # Return both the hypothetical doc AND the original query
    hypothetical = raw.strip()[:300]
    logger.info(f"[hyde] hypothetical doc: {hypothetical[:80]}…")
    return [hypothetical, query]


def multi_query(llm, query: str, notepad_summary: str) -> List[str]:
    """Generate multiple diverse phrasings of the same question."""
    prompt = (
        f"Query: {query}\n\n"
        "Generate 4 diverse phrasings of this question that explore different angles. "
        "One per line."
    )
    raw = _call(llm, [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    result = _parse_lines(raw, max_items=4)
    logger.info(f"[multi_query] {len(result)} diverse queries")
    return result or [query]


def sub_query(llm, query: str, notepad_summary: str) -> List[str]:
    """
    Identify atomic sub-queries for multi-hop reasoning.
    Builds on what is already known (notepad_summary) to avoid redundant retrieval.
    """
    memory_hint = ""
    if "No previous" not in notepad_summary:
        memory_hint = f"\nAlready found: {notepad_summary[-200:]}"

    prompt = (
        f"Main query: {query}{memory_hint}\n\n"
        "List 2-3 specific atomic facts that still need to be retrieved to fully answer. "
        "One per line."
    )
    raw = _call(llm, [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    result = _parse_lines(raw, max_items=3)
    logger.info(f"[sub_query] {len(result)} sub-queries")
    return result or [query]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy registry
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_FN = {
    "query_expansion": query_expansion,
    "query_rewriting": query_rewriting,
    "query_decomposition": query_decomposition,
    "step_back_prompting": step_back_prompting,
    "hyde": hyde,
    "multi_query": multi_query,
    "sub_query": sub_query,
}

ALL_STRATEGIES = list(STRATEGY_FN.keys())


def run_strategy(
    strategy_name: str,
    llm,
    query: str,
    notepad_summary: str,
) -> List[str]:
    """Dispatch to the right strategy function."""
    fn = STRATEGY_FN.get(strategy_name)
    if fn is None:
        logger.warning(f"Unknown strategy '{strategy_name}' — falling back to original query")
        return [query]
    return fn(llm, query, notepad_summary)