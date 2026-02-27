import logging
from typing import List, Union
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """Unified CLIP embedder – text AND images share the exact same vector space."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self._model_name = model_name
        logger.info(f"Loading unified multimodal CLIP: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def embed(self, items: List[Union[str, Image.Image]]) -> List[List[float]]:
        """Accepts mixed list: str (text) or PIL.Image (image)."""
        if not items:
            return []

        logger.debug(f"Embedding {len(items)} multimodal items with {self.model_name}...")
        embeddings = self.model.encode(
            items,
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