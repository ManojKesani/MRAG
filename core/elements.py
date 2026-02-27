# import logging
# from pydantic import BaseModel, Field
# from typing import List, Dict, Any, Literal, Union, Optional
# from PIL import Image
# import uuid

# # Get logger for domain models
# logger = logging.getLogger(__name__)

# ElementType = Literal["text", "image", "table", "audio"]

# class BaseElement(BaseModel):
#     element_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     type: ElementType
#     content: Any
#     metadata: Dict[str, Any] = Field(default_factory=dict)
#     children: List["BaseElement"] = Field(default_factory=list)

#     embedding: Optional[List[float]] = None
#     embedding_model: Optional[str] = None

#     def get_text_for_retrieval(self) -> str:
#         """Unified text representation logic with logging for debugging retrieval paths."""
        
#         # 1. Check for processed multimodal descriptions
#         if isinstance(self, ImageElement) and self.description:
#             logger.debug(f"Retrieval: Using description for Image {self.element_id}")
#             return self.description
            
#         if isinstance(self, AudioElement) and self.transcription:
#             logger.debug(f"Retrieval: Using transcription for Audio {self.element_id}")
#             return self.transcription
            
#         # 2. Check for raw text content
#         if isinstance(self.content, str):
#             # We don't log here to avoid flooding, as this is the standard path
#             return self.content
            
#         # 3. Fallback path (potential quality issue)
#         logger.warning(
#             f"Retrieval fallback: Element {self.element_id} (type: {self.type}) "
#             f"has no specialized text. Using str() conversion."
#         )
#         return str(self.content)

#     class Config:
#         arbitrary_types_allowed = True 

# # --- Subclasses (Keep these as pure schemas) ---

# class TextElement(BaseElement):
#     type: ElementType = "text"
#     content: str

# class ImageElement(BaseElement):
#     type: ElementType = "image"
#     content: Image.Image
#     description: str = ""

# class TableElement(BaseElement):
#     type: ElementType = "table"
#     content: str # markdown

# class AudioElement(BaseElement):
#     type: ElementType = "audio"
#     content: str
#     transcription: str = ""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4
from PIL import Image

@dataclass
class BaseElement:
    element_id: str = field(default_factory=lambda: str(uuid4()))
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["BaseElement"] = field(default_factory=list) # For Hierarchical chunking

    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dict for LangGraph persistence or JSON export."""
        return {
            "element_id": self.element_id,
            "type": self.__class__.__name__,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children]
        }
    
    embedding: Optional[List[float]] = None
    
    def has_embedding(self) -> bool:
        return self.embedding is not None and len(self.embedding) > 0

@dataclass
class TextElement(BaseElement):
    content: str = ""
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["content"] = self.content
        return d

@dataclass
class ImageElement(BaseElement):
    content: Optional[Image.Image] = None  # Live object for local use
    base64_data: Optional[str] = None

@dataclass
class TableElement(BaseElement):
    content: str = "" # Usually Markdown string