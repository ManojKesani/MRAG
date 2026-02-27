"""
core/chunkers/hierarchical.py
==============================
Hierarchical (Parent → Child) chunker.

Fix applied:
  Children are tagged so node_chunk can flatten them into a single list
  before embedding.  Every parent AND every child gets embedded and stored —
  parents give broad context for retrieval, children give precision.
"""
from __future__ import annotations

import logging
from typing import List

from ..elements import BaseElement, TextElement
from .base import BaseChunker
from .recursive import RecursiveChunker

logger = logging.getLogger(__name__)


class HierarchicalChunker(BaseChunker):
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.parent_chunker = RecursiveChunker(
            chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
        )
        self.child_chunker = RecursiveChunker(
            chunk_size=child_chunk_size, chunk_overlap=chunk_overlap // 2
        )
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    def chunk(self, elements: List[BaseElement]) -> List[BaseElement]:
        parents = self.parent_chunker.chunk(elements)
        final_chunks: List[BaseElement] = []

        logger.debug(
            f"Hierarchical chunking: parent_size={self.parent_chunk_size}, "
            f"child_size={self.child_chunk_size}"
        )

        for parent in parents:
            if not isinstance(parent, TextElement):
                final_chunks.append(parent)
                continue

            parent_id = parent.element_id
            children = self.child_chunker.chunk([parent])

            tagged_children: List[TextElement] = []
            for idx, child in enumerate(children):
                if isinstance(child, TextElement):
                    child.metadata.update(
                        {
                            "chunk_type": "child",
                            "parent_id": parent_id,
                            "chunking_method": "hierarchical",
                            "child_index": idx,
                        }
                    )
                    tagged_children.append(child)

            parent.metadata.update(
                {
                    "chunk_type": "parent",
                    "chunking_method": "hierarchical",
                    "child_count": len(tagged_children),
                }
            )
            # Store children on parent — node_chunk._flatten_elements() will
            # expand these into the flat embedding list.
            parent.children = tagged_children
            final_chunks.append(parent)

        logger.info(
            f"Hierarchical chunking complete: {len(parents)} parents, "
            f"{sum(len(p.children) for p in final_chunks if hasattr(p, 'children'))} children"
        )
        return final_chunks