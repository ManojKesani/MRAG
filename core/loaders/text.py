from typing import List
from ..elements import TextElement, BaseElement
from .base import BaseLoader

class TextLoader(BaseLoader):
    @staticmethod
    def load(file_path: str) -> List[BaseElement]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [TextElement(content=content, metadata={"source": file_path})]