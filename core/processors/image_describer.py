"""
core/processors/image_describer.py
====================================
Vision-LLM image description node.

Fix: vision_model is accepted as a constructor argument so it can be
overridden per-request via PipelineConfig.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from PIL import Image

from ..elements import ImageElement, TextElement

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Describe this image in rich detail for multimodal RAG retrieval. "
    "Include all visible text, objects, charts, layout, and context."
)


class ImageDescriber:
    """Converts ImageElements → TextElements using a Vision LLM."""

    def __init__(
        self,
        prompt: Optional[str] = None,
        vision_model: Optional[str] = None,
    ):
        self.prompt = prompt or _DEFAULT_PROMPT
        # Override model globally so get_vision_llm() picks it up
        if vision_model:
            os.environ["VISION_MODEL"] = vision_model

    def describe(self, image_elements: List[ImageElement]) -> List[TextElement]:
        from ..llm_src import call_vision_llm   # local import keeps module lightweight

        described: List[TextElement] = []

        for img_el in image_elements:
            try:
                img_data: Optional[Image.Image] = img_el.content
                if img_data is None:
                    logger.warning(
                        f"ImageElement {img_el.element_id} has no PIL content — skipping."
                    )
                    continue

                description = call_vision_llm(img_data, self.prompt)

                text_el = TextElement(
                    element_id=img_el.element_id,   # preserve ID for safe dedup
                    content=description,
                    metadata={
                        **img_el.metadata,
                        "original_type": "image",
                        "description_method": "vision_llm",
                    },
                )
                described.append(text_el)
                logger.info(f"Described image {img_el.element_id} ({len(description)} chars)")

            except Exception as exc:
                logger.error(f"Failed to describe image {img_el.element_id}: {exc}")

        return described