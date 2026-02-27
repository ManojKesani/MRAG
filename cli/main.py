"""M-RAG CLI — ingest · chat · status"""

from __future__ import annotations
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ingestion.graph import run_ingestion
# from rag.graph import run_rag
from utils.logger import get_logger

from sqlalchemy import text
import httpx

import logging
from typing import Optional
from pathlib import Path
import asyncio
import typer
from rich.console import Console

from dotenv import load_dotenv
load_dotenv()


app = typer.Typer(name="m-rag", add_completion=False, rich_markup_mode="rich")
console = Console()
# Add this to your setup_logging function


def setup_logging(level: str):
    """Configures the global logging level."""
    
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Optional: Silence noisy 3rd party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    # 1. Silence the 'transformers' logger
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("core").setLevel(numeric_level)
    logging.getLogger("ingestion").setLevel(numeric_level)

    # 2. Disable the progress bars (the 'Loading weights' lines)
    # This environment variable tells Hugging Face to be quiet
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    # 3. Optional: Silence noisy network logs from Qdrant/HTTPX
    logging.getLogger("httpx").setLevel(logging.WARNING)

@app.callback()
def main(
    verbose: str = typer.Option(
        "INFO", 
        "--log", "-l", 
        help="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    ),
):
    """M-RAG CLI: Multimodal RAG Ingestion & Chat."""
    setup_logging(verbose)






# ── ingest ────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    path: Path = typer.Argument(..., help="PDF or image file / directory to ingest"),
    collection: str = typer.Option("m_rag", "--collection", "-c", help="Qdrant collection"),
) -> None:
    """Ingest a PDF or image into the vector store."""
    
    # We use a context manager to show a spinner while the log stream runs in the background
    with console.status("[bold str]Processing pipeline...") as status:
        try:
            config={
        "describe_images": True,
        "chunking_method": "recursive",   # or "semantic", "recursive"
        "embedding_type": "multimodal",      # best for mixed content
        "collection_name": collection,
    }
            result = asyncio.run(run_ingestion(str(path), config=config))
            console.print(f"\n[green]✓[/green] Ingestion Complete. Final Status: [bold]{result.get('status')}[/bold]")
        except Exception as e:
            console.print(f"\n[red]✗[/red] Ingestion Failed: {e}")
            raise typer.Exit(code=1)
    
# ── chat ──────────────────────────────────────────────────────────────────────

# @app.command()
# def chat(
#     query: str = typer.Argument(..., help="Your question"),
#     session_id: Optional[str] = typer.Option(None, "--session-id", "-s"),
#     collection: str = typer.Option("m_rag", "--collection", "-c"),
#     top_k: int = typer.Option(5, "--top-k", "-k", help="Chunks to retrieve"),
# ) -> None:
#     """Ask a question against the ingested knowledge base."""
#     tracer = get_tracer()
#     with tracer.trace("cli.chat", input={"query": query, "session_id": session_id}):
#         console.print(f"[cyan]Thinking…[/cyan]")
#         result = asyncio.run(
#             run_rag(query, session_id=session_id, collection=collection, top_k=top_k)
#         )
#         console.print("\n[bold green]Answer:[/bold green]")
#         console.print(result["answer"])
#         if result.get("sources"):
#             console.print("\n[dim]Sources:[/dim]")
#             for src in result["sources"]:
#                 console.print(f"  • {src}")


# ── status ────────────────────────────────────────────────────────────────────

@app.command()
def status(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show health of all system components."""
    import json
    from qdrant_client import QdrantClient
    from core.databases import check_db_connection
    import os
    from langfuse import Langfuse

    health: dict[str, str] = {}

    # Qdrant
    try:
        qc = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        qc.get_collections()
        health["qdrant"] = "ok"
    except Exception as e:
        health["qdrant"] = f"error: {e}"

    # Postgres
    try:
        asyncio.run(check_db_connection())
        health["postgres"] = "ok"
    except Exception as e:
        health["postgres"] = f"error: {e}"

    try:
        # This checks if the SDK can actually authenticate with your keys
        lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        )
        if lf.auth_check():
            health["langfuse"] = "ok"
        else:
            health["langfuse"] = "error: auth failed"
    except Exception:
        health["langfuse"] = "error: unreachable"

    # --- Render ---
    if as_json:
        console.print_json(json.dumps(health))
    else:
        table = Table(title="M-RAG System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        for component, stat in health.items():
            colour = "green" if stat == "ok" else "red"
            table.add_row(component, f"[{colour}]{stat}[/{colour}]")
        console.print(table)


if __name__ == "__main__":
    app()