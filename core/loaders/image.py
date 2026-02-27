# core/loaders/image.py
import io
import base64
from PIL import Image
from typing import List, Any
from ..elements import ImageElement, BaseElement
from .base import BaseLoader

class ImageLoader(BaseLoader):
    @staticmethod
    def to_base64(img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def process_raw_bytes(data: bytes, metadata: dict) -> ImageElement:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.load()
        
        # Add dimensions to metadata automatically
        metadata.update({"width": img.width, "height": img.height})
        
        return ImageElement(
            content=img,
            base64_data=ImageLoader.to_base64(img),
            metadata=metadata
        )

    @staticmethod
    def load(file_path: str, **kwargs) -> List[BaseElement]:
        with open(file_path, "rb") as f:
            data = f.read()
        return [ImageLoader.process_raw_bytes(data, {"source": file_path})]