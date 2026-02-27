"""
rag/notepad.py
==============
Token-efficient notepad and summary manager.

The notepad accumulates structured entries across iterations.
Before each LLM call the full notepad is compressed into a ≤150-token
summary string.  Only that summary (not the raw entries) is sent to the LLM.

This lets the agent run 5–10 iterations while keeping every individual
LLM request small.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .state import NotepadEntry

logger = logging.getLogger(__name__)

# Hard limits kept intentionally small
MAX_SUMMARY_SENTENCES = 4
MAX_FINDINGS_PER_ENTRY = 80   # chars


def compress_notepad(
    entries: List[NotepadEntry],
    original_query: str,
) -> str:
    """
    Convert a list of NotepadEntry dicts into a short plain-text summary
    that fits in ~150 tokens.

    Structure:
      Query: <original>
      [iter 1] Strategy=<s> | Queries=<n> | <key_findings>
      [iter 2] ...
      Overall: <what we know so far>
    """
    if not entries:
        return f"Query: {original_query}\nNo previous iterations."

    lines: List[str] = [f"Query: {original_query}"]
    for e in entries[-4:]:   # keep at most last 4 entries to cap tokens
        findings = e["key_findings"][:MAX_FINDINGS_PER_ENTRY].rstrip()
        lines.append(
            f"[iter {e['iteration']}] {e['strategy']} | "
            f"{e['chunks_found']} chunks | {findings}"
        )

    # Add a one-liner "what we know"
    if len(entries) > 1:
        all_findings = " ".join(e["key_findings"] for e in entries[-3:])
        # Truncate to ~60 words
        words = all_findings.split()[:60]
        lines.append("Cumulative: " + " ".join(words))

    summary = "\n".join(lines)
    logger.debug(f"Notepad summary ({len(summary.split())} words): {summary[:120]}…")
    return summary


def make_notepad_entry(
    iteration: int,
    strategy: str,
    queries_used: List[str],
    chunks_found: int,
    key_findings: str,
) -> NotepadEntry:
    return NotepadEntry(
        iteration=iteration,
        strategy=strategy,
        queries_used=queries_used,
        chunks_found=chunks_found,
        key_findings=key_findings[:MAX_FINDINGS_PER_ENTRY],
    )


def build_minimal_prompt(
    system_role: str,
    notepad_summary: str,
    current_task: str,
    context_snippet: Optional[str] = None,
) -> str:
    """
    Assemble the smallest useful prompt for a single LLM call.
    Total target: ≤700 tokens.

      [ROLE]          ~80 t
      [MEMORY]        ~150 t
      [TASK]          ~80 t
      [CONTEXT]       ~400 t (optional)
    """
    parts = [
        f"[ROLE] {system_role}",
        f"[MEMORY]\n{notepad_summary}",
        f"[TASK] {current_task}",
    ]
    if context_snippet:
        # Hard-truncate context to ~400 tokens ≈ 1600 chars
        snippet = context_snippet[:1600]
        parts.append(f"[CONTEXT]\n{snippet}")
    return "\n\n".join(parts)