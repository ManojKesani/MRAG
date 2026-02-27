import logging
from typing import List, Optional
import numpy as np

from ..elements import BaseElement, TextElement
from ..embedders.text import TextEmbedder  # Local import
from .base import BaseChunker

logger = logging.getLogger(__name__)

class SemanticChunker(BaseChunker):
    """
    Semantic similarity-based chunker using local TextEmbedder.
    Splits text into sentences, embeds them, and breaks when similarity drops.
    """

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        breakpoint_threshold: float = 0.8, # Adjust based on model (0.0 to 1.0)
    ):
        # Use local TextEmbedder logic
        self.embedder = embedder or TextEmbedder()
        self.breakpoint_threshold = breakpoint_threshold
        logger.info(f"SemanticChunker initialized with threshold {breakpoint_threshold}")

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter; can be swapped for nltk/spacy."""
        import re
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, elements: List[BaseElement]) -> List[BaseElement]:
        final_chunks: List[BaseElement] = []

        for el in elements:
            if not isinstance(el, TextElement) or not el.content.strip():
                final_chunks.append(el)
                continue

            sentences = self._split_sentences(el.content)
            if len(sentences) <= 1:
                final_chunks.append(el)
                continue

            # 1. Get embeddings for all sentences using your local logic
            embeddings = np.array(self.embedder.embed(sentences))
            
            # 2. Calculate cosine similarity between adjacent sentences
            # Formula: (A . B) / (||A|| * ||B||) - Note: TextEmbedder normalizes already
            clusters = []
            current_cluster = [sentences[0]]
            
            for i in range(len(sentences) - 1):
                similarity = np.dot(embeddings[i], embeddings[i+1])
                
                if similarity < self.breakpoint_threshold:
                    clusters.append(" ".join(current_cluster))
                    current_cluster = [sentences[i+1]]
                else:
                    current_cluster.append(sentences[i+1])
            
            clusters.append(" ".join(current_cluster))

            # 3. Create new TextElements
            for i, cluster_text in enumerate(clusters):
                final_chunks.append(
                    TextElement(
                        content=cluster_text,
                        metadata={
                            **el.metadata,
                            "chunk_index": i,
                            "parent_id": getattr(el, "element_id", None),
                            "chunking_method": "semantic_local",
                        },
                    )
                )

        logger.info(f"Semantic chunking complete. Generated {len(final_chunks)} chunks.")
        return final_chunks