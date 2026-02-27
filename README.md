# 🧠 M-RAG — Multimodal Agentic RAG Framework
<img width="1851" height="3180" alt="1" src="https://github.com/user-attachments/assets/d59c9eb7-a5f2-4bbc-a8d8-f1e495da5593" />
<img width="1851" height="4689" alt="2" src="https://github.com/user-attachments/assets/5cf13139-2c2c-4233-a4dd-6b275ec5b99e" />


> **Ingest PDFs and images. Query with an agentic loop. Get answers with citations.**
>
> Built on **LangGraph · LangChain · Groq · Qdrant · Streamlit**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Agentic RAG Pipeline](#agentic-rag-pipeline)
- [Query Strategies](#query-strategies)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Runbook: Local Setup](#runbook-local-setup)
- [Runbook: Running the Stack](#runbook-running-the-stack)
- [Runbook: CLI Usage](#runbook-cli-usage)
- [Runbook: API Usage](#runbook-api-usage)
- [Configuration Reference](#configuration-reference)
- [Environment Variables](#environment-variables)
- [Observability](#observability)
- [Troubleshooting](#troubleshooting)

---

## Overview

M-RAG is a production-ready **Multimodal Retrieval-Augmented Generation** framework that:

- **Ingests** PDFs (text + embedded images), standalone images, and plain text files
- **Describes** images using a Vision LLM so they become searchable alongside text
- **Chunks** content with three configurable strategies (recursive, hierarchical, semantic)
- **Embeds** using either text-only (`all-MiniLM`) or multimodal CLIP models
- **Stores** vectors in Qdrant with automatic collection schema management
- **Answers** questions via an agentic loop that selects query strategies, retrieves, re-ranks, and self-assesses confidence across up to N iterations

Everything is configurable per-request — no restarts needed to switch models, chunkers, or strategies.

---

## Architecture

### High-Level System View

```mermaid
graph TB
    UI[🖥️ Streamlit Frontend]
    API[⚡ FastAPI Backend]
    ING[📥 Ingestion Graph\nLangGraph]
    RAG[🔍 RAG Agent Graph\nLangGraph]
    QD[(🗄️ Qdrant\nVector Store)]
    PG[(🐘 PostgreSQL\nMetadata)]
    LF[📊 Langfuse\nObservability]
    GROQ[☁️ Groq LLM API\nLlama / GPT]

    UI -->|POST /ingest| API
    UI -->|POST /rag/query| API
    API --> ING
    API --> RAG
    ING --> QD
    ING --> PG
    RAG --> QD
    ING -.->|traces| LF
    RAG -.->|traces| LF
    ING --> GROQ
    RAG --> GROQ
```

---

## Project Structure

```
m-rag/
├── api/
│   ├── main.py              # FastAPI app, /ingest, /health, /collections
│   └── rag_router.py        # /rag/query (async), /rag/query/sync, /rag/strategies
├── core/
│   ├── config.py            # PipelineConfig (per-request) + Settings (env vars)
│   ├── settings.py          # Cached Settings singleton
│   ├── state.py             # LangGraph IngestionState TypedDict
│   ├── elements.py          # BaseElement, TextElement, ImageElement, TableElement
│   ├── databases.py         # Async SQLModel / PostgreSQL session management
│   ├── llm_src.py           # Groq LLM + Vision LLM factory (cached)
│   ├── loaders/             # PDFLoader, ImageLoader, TextLoader
│   ├── chunkers/            # RecursiveChunker, HierarchicalChunker, SemanticChunker
│   ├── embedders/           # TextEmbedder, ImageEmbedder, MultimodalEmbedder (CLIP)
│   ├── processors/
│   │   └── image_describer.py   # Vision LLM → TextElement conversion
│   └── storers/
│       └── qdrant.py        # Dynamic collection creation, upsert
├── ingestion/
│   ├── graph.py             # LangGraph ingestion pipeline builder + runner
│   └── nodes.py             # node_load, node_describe_images, node_chunk, node_embed, node_store
├── rag/
│   ├── graph.py             # LangGraph RAG agent builder + async runner
│   ├── nodes.py             # node_init, node_probe, node_plan, node_transform,
│   │                        #   node_retrieve, node_rerank, node_evaluate, node_synthesize
│   ├── state.py             # RAGState TypedDict (token-efficient memory design)
│   ├── config.py            # RAGConfig (per-request agent settings)
│   ├── strategies.py        # 7 query transformation strategies
│   ├── retriever.py         # MultiQueryRetriever with store probe
│   ├── reranker.py          # LLM + score-based re-ranker
│   └── notepad.py           # Token-efficient compressed memory manager
├── cli/
│   └── main.py              # Typer CLI: ingest, chat, status
├── frontend/
│   └── app.py               # Streamlit UI (Ingest + RAG tabs)
├── utils/
│   ├── logger.py            # Centralized logging setup
│   └── utils.py             # File hashing, ID generation helpers
├── tracing_langfuse/
│   └── docker-compose.yml   # Self-hosted Langfuse stack
├── docker-compose.yml       # Qdrant + PostgreSQL
├── pyproject.toml
└── .env                     # Secrets (not committed)
```

---

## Ingestion Pipeline

The ingestion pipeline is a **LangGraph directed graph** that processes a single file end-to-end.

```mermaid
flowchart LR
    A([📂 File Input]) --> B[node_load]
    B --> C{Has Images\nAND describe_images=true?}
    C -- Yes --> D[node_describe_images\nVision LLM → TextElement]
    C -- No --> E[node_chunk]
    D --> E
    E --> F{chunking_method}
    F -- recursive --> G[RecursiveChunker]
    F -- hierarchical --> H[HierarchicalChunker\nParent + Children flattened]
    F -- semantic --> I[SemanticChunker\nCosine similarity breakpoints]
    G & H & I --> J[node_embed\ntext / image / multimodal CLIP]
    J --> K[node_store\nQdrant upsert]
    K --> L([✅ status: success])
```

### Ingestion State Flow

```mermaid
stateDiagram-v2
    [*] --> Load : file_path
    Load --> DescribeImages : has ImageElements AND describe_images=true
    Load --> Chunk : no images OR describe_images=false
    DescribeImages --> Chunk : images replaced with TextElements
    Chunk --> Embed : flat list of elements
    Embed --> Store : elements with .embedding populated
    Store --> [*] : status=success, metrics dict
```

### Chunking Strategy Details

| Strategy | Best For | Key Parameters |
|---|---|---|
| `recursive` | General text, PDFs | `chunk_size`, `chunk_overlap` |
| `hierarchical` | Long docs needing broad + precise retrieval | `parent_chunk_size`, `child_chunk_size` |
| `semantic` | Coherent topic-based splits | `breakpoint_threshold` (cosine sim) |

> **Hierarchical note:** Both parent and child chunks are embedded and stored. Parents give broad context; children give precision.

---

## Agentic RAG Pipeline

The RAG agent is a **recursive LangGraph loop** with a probe gate, planner, and confidence-based exit.

```mermaid
flowchart TD
    START([🔍 Query Input]) --> INIT[node_init\nInitialize state]
    INIT --> PROBE[node_probe\nVerify Qdrant collection\nRun baseline query]

    PROBE --> GATE{Store ready?}
    GATE -- No --> SYNTH
    GATE -- Yes --> PLAN

    PLAN[node_plan\nLLM picks strategy\nor uses fixed config] --> TRANSFORM[node_transform\nApply query strategy\n→ N sub-queries]
    TRANSFORM --> RETRIEVE[node_retrieve\nMulti-query vector search\n+ deduplication]
    RETRIEVE --> RERANK[node_rerank\nLLM or score-based\nreranking → top-N]
    RERANK --> EVAL[node_evaluate\nDraft answer + confidence\n+ notepad entry]

    EVAL --> LOOP{confidence ≥ threshold\nOR iteration = max?}
    LOOP -- No → loop back --> PLAN
    LOOP -- Yes --> SYNTH[node_synthesize\nFinal polished answer\n+ citations]
    SYNTH --> END([📝 Final Answer])

    style GATE fill:#f0f0f0
    style LOOP fill:#f0f0f0
```

### Token Budget per Call

Each LLM call in the agent is capped to **~700 tokens** total:

```
[ROLE]    ~80 tokens   — static system persona
[MEMORY]  ~150 tokens  — compressed notepad summary
[TASK]    ~80 tokens   — current iteration objective
[CONTEXT] ~400 tokens  — top-K re-ranked chunk snippets
```

The full chunk list, raw retrieved documents, and reasoning history live in **LangGraph state** but are never sent raw to the LLM — only the compressed summary travels between calls.

---

## Query Strategies

The planner selects from 7 strategies each iteration. In `auto` mode it avoids strategies already tried.

```mermaid
mindmap
  root((Query\nStrategies))
    query_expansion
      Adds synonyms and domain terms
      Returns 3 expanded variants
    query_rewriting
      Rephrases for dense vector retrieval
      Noun-phrase focused
    query_decomposition
      Splits into 2-4 sub-questions
      Each answered independently
    step_back_prompting
      Abstracts to background-level question
      Provides foundational context
    hyde
      Generates a hypothetical answer excerpt
      Excerpt becomes the retrieval query
    multi_query
      4 diverse phrasings of same question
      Explores different retrieval angles
    sub_query
      Identifies still-missing atomic facts
      Builds on notepad memory
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq API (Llama 3 70B, GPT-OSS 120B) |
| **Vision LLM** | Groq API (Llama 4 Scout Vision) |
| **Orchestration** | LangGraph + LangChain |
| **Vector Store** | Qdrant (local Docker or cloud) |
| **Embeddings** | SentenceTransformers — `all-MiniLM-L6-v2` (text), `clip-ViT-B-32` (multimodal) |
| **PDF Parsing** | PyMuPDF (`fitz`) |
| **Database** | PostgreSQL + SQLModel + asyncpg |
| **API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Observability** | Langfuse (self-hosted or cloud) |
| **CLI** | Typer + Rich |
| **Package Manager** | uv |

---

## Prerequisites

- **Python 3.12+**
- **Docker + Docker Compose**
- **uv** — fast Python package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Groq API Key** — free tier available at [console.groq.com](https://console.groq.com)

---

## Runbook: Local Setup

### 1. Clone and enter the repo

```bash
git clone https://github.com/your-org/m-rag.git
cd m-rag
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
# Required
GROQ_API_KEY=gsk_...

# Qdrant (matches docker-compose.yml defaults)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# PostgreSQL (matches docker-compose.yml defaults)
DATABASE_URL=postgresql+asyncpg://mrag:mrag_secret@localhost:5433/mrag

# LLM model names (override if needed)
LLM_MODEL=openai/gpt-oss-120b
VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_EMBEDDING_TYPE=text
DEFAULT_COLLECTION=mrag_default

# Observability (optional — skip if not using Langfuse)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

### 3. Install Python dependencies

```bash
uv sync
```

> On first run this also downloads SentenceTransformer model weights (~90 MB for MiniLM, ~350 MB for CLIP). They are cached in `~/.cache/huggingface/`.

### 4. Start infrastructure services

```bash
docker compose up -d
```

This starts:
- **Qdrant** on `localhost:6333`
- **PostgreSQL** on `localhost:5433` (mapped from container 5432)

Verify both are healthy:

```bash
docker compose ps
# Both services should show "healthy" or "running"
```

### 5. Verify system health

```bash
uv run m-rag status
```

Expected output:

```
┏━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Component   ┃ Status ┃
┡━━━━━━━━━━━━━╇━━━━━━━━┩
│ qdrant      │ ok     │
│ postgres    │ ok     │
│ langfuse    │ ok     │  ← "error: unreachable" is OK if not using Langfuse
└─────────────┴────────┘
```

---

## Runbook: Running the Stack

### Start the API server

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Start the Streamlit frontend

In a separate terminal:

```bash
uv run streamlit run frontend/app.py
```

The UI opens at `http://localhost:8501`.

### Start Langfuse (optional observability)

```bash
cd tracing_langfuse
docker compose up -d
```

Langfuse UI at `http://localhost:3000`. Create a project and copy the public/secret keys into `.env`.

---

## Runbook: CLI Usage

### Ingest a single file

```bash
# Ingest a PDF into the default collection
uv run m-rag ingest path/to/document.pdf

# Ingest into a named collection
uv run m-rag ingest path/to/document.pdf --collection my_docs

# With verbose debug logging
uv run m-rag ingest path/to/document.pdf --log DEBUG
```

### Ingest a directory of files

```bash
# Ingest all supported files in a folder
for f in docs/*.pdf; do uv run m-rag ingest "$f" --collection research; done
```

### Check system status

```bash
uv run m-rag status

# JSON output (for scripting)
uv run m-rag status --json
```

### Run an evaluation script

```bash
# Run the built-in eval set against a collection
uv run python experiments/sb.py my_collection
```

---

## Runbook: API Usage

### Upload and ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@report.pdf" \
  -F 'config_json={"collection_name":"research","chunking_method":"recursive","chunk_size":1000}'
```

Response:
```json
{ "job_id": "abc-123", "status": "queued", "message": "Ingestion queued for 'report.pdf'" }
```

### Poll ingestion job status

```bash
curl http://localhost:8000/ingest/abc-123
```

Response when complete:
```json
{
  "job_id": "abc-123",
  "status": "success",
  "document_id": "report_pdf",
  "metrics": {
    "load_time": 0.42,
    "raw_elements": 24,
    "chunks": 87,
    "embed_time": 3.1,
    "store_time": 0.18,
    "stored": 87
  }
}
```

### Submit an async RAG query

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main conclusions?",
    "config": {
      "collection_name": "research",
      "max_iterations": 3,
      "confidence_threshold": 0.75,
      "strategy": "auto"
    }
  }'
```

### Synchronous RAG query (waits up to 120s)

```bash
curl -X POST http://localhost:8000/rag/query/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the key findings", "config": {"collection_name": "research"}}'
```

### List available Qdrant collections

```bash
curl http://localhost:8000/collections
# {"collections": ["mrag_default", "research"]}
```

### List available query strategies

```bash
curl http://localhost:8000/rag/strategies
# {"strategies": ["query_expansion", "query_rewriting", ...]}
```

---

## Configuration Reference

### PipelineConfig (Ingestion)

| Field | Type | Default | Description |
|---|---|---|---|
| `describe_images` | bool | `true` | Run Vision LLM on extracted images |
| `image_description_prompt` | str | (detailed default) | Prompt sent to vision model |
| `chunking_method` | enum | `recursive` | `recursive` / `hierarchical` / `semantic` |
| `chunk_size` | int | `1000` | Tokens per chunk (recursive only) |
| `chunk_overlap` | int | `100` | Overlap between consecutive chunks |
| `parent_chunk_size` | int | `2000` | Parent chunk size (hierarchical only) |
| `child_chunk_size` | int | `800` | Child chunk size (hierarchical only) |
| `semantic_breakpoint` | float | `0.8` | Cosine similarity split threshold (semantic only) |
| `embedding_type` | enum | `text` | `text` / `multimodal` / `image` |
| `embedding_model` | str | `all-MiniLM-L6-v2` | Any SentenceTransformer model name |
| `collection_name` | str | `mrag_default` | Target Qdrant collection |
| `llm_model` | str | `llama3-70b-8192` | Groq model for text tasks |
| `vision_model` | str | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model for image description |

### RAGConfig (Query Agent)

| Field | Type | Default | Description |
|---|---|---|---|
| `collection_name` | str | `mrag_default` | Qdrant collection to search |
| `strategy` | enum | `auto` | Fixed strategy or `auto` (planner selects) |
| `enabled_strategies` | list | all 7 | Strategies available to planner in auto mode |
| `top_k` | int | `5` | Chunks retrieved per query |
| `score_threshold` | float | `0.25` | Minimum cosine similarity to include a chunk |
| `rerank_mode` | enum | `llm` | `llm` (LLM-scored) or `score` (passthrough) |
| `rerank_top_n` | int | `4` | Chunks kept after reranking |
| `max_iterations` | int | `5` | Maximum agent loop iterations |
| `confidence_threshold` | float | `0.75` | Stop when agent self-reports confidence ≥ this |
| `llm_model` | str | `openai/gpt-oss-120b` | Groq model for all agent calls |
| `temperature` | float | `0.0` | LLM sampling temperature |
| `max_tokens_per_call` | int | `512` | Hard cap on output tokens per LLM call |
| `embedding_type` | enum | `text` | Must match what was used at ingestion time |
| `embedding_model` | str | `all-MiniLM-L6-v2` | Must match what was used at ingestion time |
| `include_sources` | bool | `true` | Append source citations to final answer |
| `answer_language` | str | `English` | Target language for the final answer |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq API key |
| `QDRANT_HOST` | | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | | `6333` | Qdrant port |
| `QDRANT_API_KEY` | | `null` | Qdrant API key (cloud only) |
| `DATABASE_URL` | | `postgresql+asyncpg://mrag:mrag_secret@localhost:5432/mrag` | Async PostgreSQL DSN |
| `LLM_MODEL` | | `openai/gpt-oss-120b` | Default Groq text model |
| `VISION_MODEL` | | `meta-llama/llama-4-scout-17b-16e-instruct` | Default Groq vision model |
| `DEFAULT_EMBEDDING_MODEL` | | `all-MiniLM-L6-v2` | Default SentenceTransformer model |
| `DEFAULT_EMBEDDING_TYPE` | | `text` | Default embedder type |
| `DEFAULT_COLLECTION` | | `mrag_default` | Default Qdrant collection name |
| `LOG_LEVEL` | | `INFO` | Python log level |
| `LANGFUSE_PUBLIC_KEY` | | — | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | | — | Langfuse project secret key |
| `LANGFUSE_HOST` | | `http://localhost:3000` | Langfuse server URL |

---

## Observability

M-RAG integrates with **Langfuse** for full LLM call tracing, latency measurement, and token counting.

Every ingestion run and RAG query automatically sends traces if `LANGFUSE_*` env vars are set. No code changes needed.

To self-host Langfuse:

```bash
cd tracing_langfuse
# Edit docker-compose.yml — replace all "CHANGEME" values with secure secrets
docker compose up -d
# Open http://localhost:3000, create an account, create a project,
# copy the API keys into your .env file
```

To use Langfuse Cloud instead, set:
```env
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

---

## Troubleshooting

### `GROQ_API_KEY not set` error

Ensure your `.env` file exists in the project root and contains `GROQ_API_KEY=gsk_...`. Run `uv run m-rag status` to confirm all components are healthy before ingesting.

### Qdrant collection dimension mismatch warning

This happens when you re-ingest using a different embedding model into an existing collection. Options:

1. Use a new `collection_name` for each embedding model (recommended)
2. Delete the old collection: `curl -X DELETE http://localhost:6333/collections/my_collection`

### `query_points not found` fallback in logs

Your Qdrant Docker image is older than v1.10. Pull the latest image:

```bash
docker compose pull qdrant
docker compose up -d qdrant
```

### Images not described / Vision LLM errors

- Confirm `describe_images: true` in your config
- Confirm `VISION_MODEL` is set to a vision-capable model on Groq (e.g. `meta-llama/llama-4-scout-17b-16e-instruct`)
- Check that the PDF actually contains embedded images (not scanned pages rendered as a single image)

### Low RAG confidence / poor answers

Try these config adjustments in the Streamlit UI or API:
- Lower `score_threshold` from `0.25` to `0.15` to retrieve more candidates
- Increase `top_k` from `5` to `8-10`
- Increase `max_iterations` from `5` to `7`
- Switch `embedding_type` to `multimodal` and re-ingest (if documents contain important diagrams)
- Enable more strategies in `enabled_strategies`

### PostgreSQL port conflict

The `docker-compose.yml` maps Postgres to port `5433` (not the default `5432`) to avoid conflicts with a locally installed Postgres. If you have no local Postgres, you can change `5433:5432` to `5432:5432` and update `DATABASE_URL` accordingly.

### Streamlit `API unreachable` error

Start the API first:
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
The Streamlit app reads `API_BASE_URL` (default `http://localhost:8000`). Set it if your API is on a different host:
```bash
API_BASE_URL=http://my-server:8000 uv run streamlit run frontend/app.py
```

---

## Development

### Run tests

```bash
uv run pytest
```

### Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

### Generate a full project dump (for sharing / debugging)

```bash
uv run python experiments/dump_project.py
# Creates mrag_project_dump.txt with all source files concatenated
```

---

## License

See `LICENSE.txt` for terms.
