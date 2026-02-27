import logging
from typing import List
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """CLIP image-only embedder."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self._model_name = model_name
        logger.info(f"Loading CLIP image embedder: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def embed(self, images: List[Image.Image]) -> List[List[float]]:
        """Batch embed PIL Images."""
        if not images:
            return []
        logger.debug(f"Embedding {len(images)} images with {self.model_name}...")
        embeddings = self.model.encode(
            images,
            batch_size=16,
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
        return self._model_name