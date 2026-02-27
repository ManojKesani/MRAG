"""
rag/config.py
=============
Per-request RAG agent configuration.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


StrategyName = Literal[
    "query_expansion",
    "query_rewriting",
    "query_decomposition",
    "step_back_prompting",
    "hyde",
    "multi_query",
    "sub_query",
    "auto",     # let the planner choose each iteration
]

RerankMode = Literal["llm", "score"]


class RAGConfig(BaseModel):
    # ── Collection ────────────────────────────────────────────────────────
    collection_name: str = Field("mrag_default", description="Qdrant collection to search")

    # ── Strategy ──────────────────────────────────────────────────────────
    strategy: StrategyName = Field(
        "auto",
        description=(
            "'auto' lets the planner select and rotate strategies each iteration. "
            "Or fix to a single strategy."
        ),
    )
    enabled_strategies: List[str] = Field(
        default=[
            "query_expansion",
            "query_rewriting",
            "query_decomposition",
            "step_back_prompting",
            "hyde",
            "multi_query",
            "sub_query",
        ],
        description="Strategies available to the planner in auto mode.",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k: int = Field(5, ge=1, le=20, description="Chunks per query")
    score_threshold: float = Field(0.25, ge=0.0, le=1.0)

    # ── Re-ranking ────────────────────────────────────────────────────────
    rerank_mode: RerankMode = Field("llm", description="'llm' or 'score'")
    rerank_top_n: int = Field(4, ge=1, le=10, description="Chunks kept after reranking")

    # ── Agent loop ────────────────────────────────────────────────────────
    max_iterations: int = Field(5, ge=1, le=10, description="Recursion limit")
    confidence_threshold: float = Field(
        0.75, ge=0.0, le=1.0,
        description="Stop when answer confidence ≥ this value",
    )

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_model: str = Field("openai/gpt-oss-120b", description="Groq model for all agent calls")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens_per_call: int = Field(
        512, ge=64, le=2048,
        description="Max LLM output tokens per call (keeps requests small)",
    )

    # ── Embedding (must match ingestion) ─────────────────────────────────
    embedding_type: Literal["text", "multimodal", "image"] = "text"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Final answer ──────────────────────────────────────────────────────
    include_sources: bool = True
    answer_language: str = Field("English", description="Language for the final answer")