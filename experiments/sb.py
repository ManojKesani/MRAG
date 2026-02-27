"""Sandbox / evaluation script — run quick RAG experiments."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.graph import run_rag

# ── Simple eval set ───────────────────────────────────────────────────────────
QUESTIONS = [
    "What is the main topic of the document?",
    "Summarize the key findings.",
    "What numbers or statistics are mentioned?",
]


async def evaluate(collection: str = "m_rag") -> dict:
    results = []
    for q in QUESTIONS:
        state = await run_rag(q, collection=collection)
        results.append({
            "question": q,
            "answer": state.get("answer", ""),
            "sources": state.get("sources", []),
        })
        print(f"\nQ: {q}\nA: {state.get('answer', '')}\n")

    metrics = {
        "total_questions": len(results),
        "answered": sum(1 for r in results if r["answer"]),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics:", metrics)
    return metrics


if __name__ == "__main__":
    collection = sys.argv[1] if len(sys.argv) > 1 else "m_rag"
    asyncio.run(evaluate(collection))