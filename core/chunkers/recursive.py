import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..elements import BaseElement, TextElement
from .base import BaseChunker

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 128, chunk_overlap: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, elements: List[BaseElement]) -> List[BaseElement]:
        chunks: List[BaseElement] = []
        text_element_count = 0
        other_element_count = 0

        logger.debug(f"Starting chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")

        for el in elements:
            if isinstance(el, TextElement):
                text_element_count += 1
                # Optional: log very large text elements that might cause bottlenecks
                if len(el.content) > 5000:
                    logger.debug(f"Splitting large TextElement (ID: {el.element_id}, Chars: {len(el.content)})")
                
                texts = self.splitter.split_text(el.content)
                
                for i, txt in enumerate(texts):
                    chunks.append(TextElement(
                        content=txt,
                        metadata={
                            **el.metadata, 
                            "chunk_index": i, 
                            "parent_id": el.element_id
                        }
                    ))
            else:
                # Non-text elements (Images/Tables) are passed through
                other_element_count += 1
                chunks.append(el)

        logger.info(
            f"Chunking complete: {text_element_count} text elements split into {len(chunks) - other_element_count} chunks. "
            f"{other_element_count} non-text elements preserved."
        )

        return chunks