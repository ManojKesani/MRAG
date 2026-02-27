from .base import BaseChunker
from .recursive import RecursiveChunker
from .hierarchical import HierarchicalChunker
from .semantic import SemanticChunker

# A mapping for dynamic selection in your LangGraph nodes
CHUNKER_MAPPING = {
    "recursive": RecursiveChunker,
    "hierarchical": HierarchicalChunker,
    "semantic": SemanticChunker
}

def get_chunker(method: str = "recursive", **kwargs) -> BaseChunker:
    """Factory function to initialize a chunker with specific settings."""
    chunker_cls = CHUNKER_MAPPING.get(method.lower())
    if not chunker_cls:
        raise ValueError(f"Unknown chunking method: {method}")
    return chunker_cls(**kwargs)

__all__ = [
    "BaseChunker",
    "RecursiveChunker",
    "HierarchicalChunker",
    "SemanticChunker",
    "get_chunker"
]