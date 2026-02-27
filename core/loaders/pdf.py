import fitz
import logging
from typing import List
from ..elements import BaseElement, TextElement, TableElement
from .base import BaseLoader
from .image import ImageLoader # Import the peer loader

logger = logging.getLogger(__name__)

class PDFLoader(BaseLoader):
    @staticmethod
    def load(file_path: str) -> List[BaseElement]:
        doc = fitz.open(file_path)
        elements: List[BaseElement] = []

        for page_num in range(len(doc)):
            page = doc[page_num] 
            curr = page_num + 1
            
            # 1. Text
            text = page.get_text("text").strip()
            if text:
                elements.append(TextElement(content=text, metadata={"page": curr}))

            # 2. Images - Using ImageLoader's logic
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                if base_img:
                    # Delegate processing to ImageLoader
                    img_el = ImageLoader.process_raw_bytes(
                        base_img["image"], 
                        {"page": curr, "xref": xref, "ext": base_img["ext"]}
                    )
                    elements.append(img_el)

            # # 3. Tables
            # try:
            #     for t in page.find_tables().tables:
            #         elements.append(TableElement(
            #             content=t.to_markdown(), 
            #             metadata={"page": curr}
            #         ))
            # except Exception:
            #     pass 

        doc.close()
        return elements