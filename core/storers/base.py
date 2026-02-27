from abc import ABC, abstractmethod
from typing import List
from ..elements import BaseElement

class BaseStorer(ABC):
    @abstractmethod
    def store(self, elements: List[BaseElement], collection_name: str):
        """Upload elements to the vector database."""
        ...