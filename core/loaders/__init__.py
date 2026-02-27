from .text import TextLoader
from .image import ImageLoader
from .pdf import PDFLoader

LOADER_MAPPING = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".png": ImageLoader,
    ".jpg": ImageLoader,
    ".jpeg": ImageLoader,
}

def get_loader_for_file(file_path: str):
    import os
    ext = os.path.splitext(file_path.lower())[1]
    loader_cls = LOADER_MAPPING.get(ext)
    if not loader_cls:
        raise ValueError(f"No loader found for extension: {ext}")
    return loader_cls

__all__ = ["TextLoader", "ImageLoader", "PDFLoader", "get_loader_for_file"]