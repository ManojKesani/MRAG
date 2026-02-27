from abc import ABC, abstractmethod
from typing import List
from ..elements import BaseElement

class BaseLoader(ABC):
    @staticmethod
    @abstractmethod
    def load(file_path: str) -> List[BaseElement]:
        """Return flat list of elements. Pure. No side effects."""
        ...