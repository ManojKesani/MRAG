"""
core/config.py
==============
Single source of truth for all configurable values.

Priority (highest → lowest):
  1. Per-request PipelineConfig payload (from frontend / API)
  2. Environment variables / .env file  (Settings)
  3. Hardcoded defaults below
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Infrastructure secrets — NEVER sent to the frontend
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Loaded once at startup from environment / .env file."""

    # ── Auth ──────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_host: str = Field("localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(None, alias="QDRANT_API_KEY")

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = Field(
        "postgresql+asyncpg://mrag:mrag_secret@localhost:5432/mrag",
        alias="DATABASE_URL",
    )

    # ── Default model names (overridable per-request) ─────────────────────
    default_llm_model: str = Field("llama3-70b-8192", alias="LLM_MODEL")
    default_vision_model: str = Field(
        "meta-llama/llama-4-scout-17b-16e-instruct", alias="VISION_MODEL"
    )
    default_embedding_model: str = Field(
        "all-MiniLM-L6-v2", alias="DEFAULT_EMBEDDING_MODEL"
    )
    default_embedding_type: Literal["text", "image", "multimodal"] = Field(
        "text", alias="DEFAULT_EMBEDDING_TYPE"
    )
    default_collection: str = Field("mrag_default", alias="DEFAULT_COLLECTION")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Per-request pipeline configuration — safe to expose to the frontend
# ---------------------------------------------------------------------------
class PipelineConfig(BaseModel):
    """
    Everything the caller can tune per ingestion job.
    Validated by FastAPI automatically when sent as JSON.
    """

    # ── Image description ─────────────────────────────────────────────────
    describe_images: bool = True
    image_description_prompt: str = (
        "Describe this image in rich detail for multimodal RAG retrieval. "
        "Include all visible text, objects, charts, layout, and context."
    )

    # ── Chunking ──────────────────────────────────────────────────────────
    chunking_method: Literal["recursive", "hierarchical", "semantic"] = "recursive"
    chunk_size: int = Field(1000, ge=100, le=8000, description="Tokens per chunk (recursive)")
    chunk_overlap: int = Field(100, ge=0, le=500)
    parent_chunk_size: int = Field(2000, ge=500, le=16000, description="Parent size (hierarchical)")
    child_chunk_size: int = Field(800, ge=100, le=4000, description="Child size (hierarchical)")
    semantic_breakpoint: float = Field(0.8, ge=0.0, le=1.0, description="Cosine sim threshold (semantic)")

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_type: Literal["text", "image", "multimodal"] = "text"
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="SentenceTransformer model name (e.g. all-MiniLM-L6-v2, clip-ViT-B-32)",
    )

    # ── Storage ───────────────────────────────────────────────────────────
    collection_name: str = Field("mrag_default", description="Qdrant collection")

    # ── LLM models ────────────────────────────────────────────────────────
    llm_model: str = "llama3-70b-8192"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    @classmethod
    def from_settings(cls, settings: Settings) -> "PipelineConfig":
        """Build a default config seeded from env/Settings."""
        return cls(
            embedding_type=settings.default_embedding_type,
            embedding_model=settings.default_embedding_model,
            collection_name=settings.default_collection,
            llm_model=settings.default_llm_model,
            vision_model=settings.default_vision_model,
        )