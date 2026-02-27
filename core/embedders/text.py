import logging
from typing import List
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class TextEmbedder(BaseEmbedder):
    """Text-only embedder using SentenceTransformer (all-MiniLM, etc.)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        logger.info(f"Loading text embedder: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed list of texts."""
        if not texts:
            return []
        logger.debug(f"Embedding {len(texts)} texts with {self.model_name}...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name # Ensure this is set in __init__