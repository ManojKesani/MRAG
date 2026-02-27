from abc import ABC, abstractmethod
from typing import List
from ..elements import BaseElement

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, elements: List[BaseElement]) -> List[BaseElement]:
        """Return list of chunk elements (may have .children for hierarchy)."""
        ...