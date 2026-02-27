"""LLM + vision model factory via LangChain-Groq."""

from __future__ import annotations

import os
import logging
from functools import lru_cache

from langchain_groq import ChatGroq

# Setup module-level logger
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
VISION_MODEL: str = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Return a cached ChatGroq LLM instance with config validation."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is missing from environment variables.")
        raise ValueError("GROQ_API_KEY not set.")

    logger.info(f"Initializing standard LLM | Model: {LLM_MODEL} | Temp: 0.0")
    
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=2048,
    )

@lru_cache(maxsize=1)
def get_vision_llm() -> ChatGroq:
    """Return a cached ChatGroq vision-capable instance with config validation."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is missing from environment variables.")
        raise ValueError("GROQ_API_KEY not set.")

    logger.info(f"Initializing Vision LLM | Model: {VISION_MODEL} | Temp: 0.0")
    
    return ChatGroq(
        model=VISION_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=512,
    )

import io
import base64
from PIL import Image
from langchain_core.messages import HumanMessage

def call_vision_llm(
    image: Image.Image,
    prompt: str = "Describe this image in detail for retrieval in a RAG system. Focus on visible objects, text, layout, and context."
) -> str:
    """Vision convenience wrapper used by ImageDescriber."""
    llm = get_vision_llm()
    
    # Convert PIL → base64 for Groq/Llama-3.2-vision
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
        }
    ])
    
    response = llm.invoke([message])
    return response.content.strip()