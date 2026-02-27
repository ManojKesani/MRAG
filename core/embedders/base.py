from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, items: List[Any]) -> List[List[float]]:
        """items = list of str (text) or list of PIL.Image (images)"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...