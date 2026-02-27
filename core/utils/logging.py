import time
import logging
import functools
from typing import Callable, Any

logger = logging.getLogger("ingestion_pipeline")

def log_node(func: Callable):
    @functools.wraps(func)
    def wrapper(state: Any, *args, **kwargs):
        node_name = func.__name__.replace("node_", "").upper()
        doc_id = state.get("document_id", "unknown")
        
        logger.info(f"[{node_name}] Starting | Doc: {doc_id}")
        start_time = time.perf_counter()
        
        try:
            result = func(state, *args, **kwargs)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Context-aware counts
            count = len(result.get("chunks", [])) or len(result.get("elements", []))
            logger.info(f"[{node_name}] Finished | Items: {count} | Time: {duration:.2f}s")
            
            return result
        except Exception as e:
            logger.error(f"[{node_name}] FAILED | Doc: {doc_id} | Error: {str(e)}")
            raise e
            
    return wrapper