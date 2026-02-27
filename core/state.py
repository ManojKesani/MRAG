"""
core/state.py
=============
LangGraph state for the M-RAG ingestion pipeline.
"""
from __future__ import annotations
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from .elements import BaseElement

def _replace(left: List[BaseElement], right: Optional[List[BaseElement]]) -> List[BaseElement]:
    """
    Return `right` when provided, otherwise keep `left`.
    This gives REPLACE semantics — the node fully owns the elements list it returns.
    """
    if right is None:
        return left
    return right


class IngestionState(TypedDict, total=False):
    """
    State object passed between LangGraph nodes.

    `total=False` → every key is optional so nodes only need to return
    the keys they actually mutate.
    """
    # Required on entry
    file_path: str
    document_id: str

    # Core payload — uses REPLACE reducer so nodes can swap the whole list
    elements: Annotated[List[BaseElement], _replace]

    # Per-run configuration (PipelineConfig serialised to dict for JSON-safety)
    config: Dict[str, Any]

    # Accumulated timing / count metrics (merged manually in each node)
    metrics: Dict[str, Any]

    # Set by node_store; one of "success" | "failed"
    status: str
    error: Optional[str]