from .base import BaseEmbedder
from .text import TextEmbedder
from .image import ImageEmbedder
from .multimodal import MultimodalEmbedder

# Registry for LangGraph nodes to select the embedding engine
EMBEDDER_MAPPING = {
    "text": TextEmbedder,
    "image": ImageEmbedder,
    "multimodal": MultimodalEmbedder
}

def get_embedder(type: str = "text", **kwargs) -> BaseEmbedder:
    """Factory to initialize the requested embedder."""
    embedder_cls = EMBEDDER_MAPPING.get(type.lower())
    if not embedder_cls:
        raise ValueError(f"Unknown embedder type: {type}. Choose from {list(EMBEDDER_MAPPING.keys())}")
    return embedder_cls(**kwargs)

__all__ = ["BaseEmbedder", "TextEmbedder", "ImageEmbedder", "MultimodalEmbedder", "get_embedder"]