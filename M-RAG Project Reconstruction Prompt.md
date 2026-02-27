# M-RAG Project Reconstruction Prompt

Use the following prompt to reconstruct the M-RAG (Multimodal Agentic RAG) project from scratch.

---

## PROMPT

Build a production-grade **Multimodal Agentic RAG (M-RAG) framework** in Python called `m-rag`. It ingests PDFs and images into a vector store and answers questions using a recursive LangGraph agent. Below are the complete specifications.

---

## 1. PROJECT STRUCTURE

```
m-rag/
├── api/
│   ├── main.py
│   └── rag_router.py
├── cli/
│   └── main.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── settings.py
│   ├── state.py
│   ├── elements.py
│   ├── databases.py
│   ├── llm_src.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pdf.py
│   │   ├── image.py
│   │   └── text.py
│   ├── chunkers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── recursive.py
│   │   ├── hierarchical.py
│   │   └── semantic.py
│   ├── embedders/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── text.py
│   │   ├── image.py
│   │   └── multimodal.py
│   ├── processors/
│   │   └── image_describer.py
│   ├── storers/
│   │   ├── base.py
│   │   └── qdrant.py
│   └── utils/
│       └── logging.py
├── ingestion/
│   ├── nodes.py
│   └── graph.py
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── state.py
│   ├── strategies.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── notepad.py
│   ├── nodes.py
│   └── graph.py
├── frontend/
│   └── app.py
├── utils/
│   ├── utils.py
│   └── logger.py
├── experiments/
│   └── sb.py
├── docker-compose.yml
└── pyproject.toml
```

---

## 2. DEPENDENCIES (`pyproject.toml`)

- Python >= 3.12
- Build backend: `hatchling`
- CLI entrypoint: `m-rag = "cli.main:app"`
- Runtime packages:
  - `langchain`, `langchain-groq`, `langchain-community`, `langgraph`
  - `qdrant-client`, `langchain-qdrant`
  - `sqlmodel`, `asyncpg`, `psycopg2-binary`
  - `sentence-transformers` (from git: UKPLab)
  - `pymupdf` (for PDF parsing via `fitz`), `pillow`
  - `langfuse` (observability)
  - `typer[standard]`, `rich` (CLI)
  - `httpx`, `python-dotenv`, `pydantic`, `pydantic-settings`, `tenacity`
  - `fastapi`, `uvicorn` (API)
  - `streamlit`, `requests` (frontend)
- Dev packages: `pytest`, `pytest-asyncio`, `ruff`, `mypy`
- PyTorch CPU index: `https://download.pytorch.org/whl/cpu`

---

## 3. INFRASTRUCTURE (`docker-compose.yml`)

Two services:
- **Postgres 16-alpine**: container `rag-postgres`, DB/user/pass = `mrag`/`mrag`/`mrag_secret`, port `5433:5432`, named volume `postgres_data`
- **Qdrant latest**: container `rag-qdrant`, port `6333:6333`, named volume `qdrant_storage`

---

## 4. CORE DOMAIN MODELS (`core/elements.py`)

Use Python `@dataclass` (NOT Pydantic) for all element types:

```python
@dataclass
class BaseElement:
    element_id: str = field(default_factory=lambda: str(uuid4()))
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["BaseElement"] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    def has_embedding(self) -> bool: ...
    def to_dict(self) -> Dict[str, Any]: ...

@dataclass
class TextElement(BaseElement):
    content: str = ""

@dataclass
class ImageElement(BaseElement):
    content: Optional[Image.Image] = None   # PIL live object
    base64_data: Optional[str] = None

@dataclass
class TableElement(BaseElement):
    content: str = ""  # Markdown string
```

---

## 5. CONFIGURATION (`core/config.py`)

Two classes:

**`Settings(BaseSettings)`** — loaded from `.env`, never sent to frontend:
- `groq_api_key` (required)
- `qdrant_host`, `qdrant_port`, `qdrant_api_key`
- `database_url` (asyncpg postgres)
- `default_llm_model`, `default_vision_model`, `default_embedding_model`, `default_embedding_type`, `default_collection`

**`PipelineConfig(BaseModel)`** — per-request, safe to expose:
- Image: `describe_images: bool = True`, `image_description_prompt: str`
- Chunking: `chunking_method: Literal["recursive","hierarchical","semantic"] = "recursive"`, `chunk_size=1000`, `chunk_overlap=100`, `parent_chunk_size=2000`, `child_chunk_size=800`, `semantic_breakpoint=0.8`
- Embedding: `embedding_type: Literal["text","image","multimodal"] = "text"`, `embedding_model = "all-MiniLM-L6-v2"`
- Storage: `collection_name = "mrag_default"`
- LLMs: `llm_model`, `vision_model`
- Class method `from_settings(settings) -> PipelineConfig`

`core/settings.py` exposes `get_settings()` via `@lru_cache`.

---

## 6. INGESTION STATE (`core/state.py`)

```python
class IngestionState(TypedDict, total=False):
    file_path: str
    document_id: str
    elements: Annotated[List[BaseElement], _replace]  # REPLACE reducer
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    status: str
    error: Optional[str]
```

The `_replace` reducer returns `right` if not None, otherwise keeps `left` (full ownership semantics).

---

## 7. LOADERS (`core/loaders/`)

All loaders implement `BaseLoader` with a static `load(file_path) -> List[BaseElement]`.

- **`TextLoader`**: reads file, returns single `TextElement`
- **`ImageLoader`**: reads bytes → `Image.open().convert("RGB")` → stores `content` (PIL) and `base64_data` (PNG base64). Has `process_raw_bytes(data, metadata)` as a static helper used by PDFLoader.
- **`PDFLoader`**: uses `fitz` (PyMuPDF). Per page: extract text → `TextElement`, extract images via `page.get_images(full=True)` → delegate to `ImageLoader.process_raw_bytes()`.

Loader registry dict: `{".pdf": PDFLoader, ".txt": TextLoader, ".md": TextLoader, ".png": ImageLoader, ".jpg": ImageLoader, ".jpeg": ImageLoader}`. Factory: `get_loader_for_file(file_path)`.

---

## 8. CHUNKERS (`core/chunkers/`)

All implement `BaseChunker.chunk(elements) -> List[BaseElement]`. Non-text elements pass through unchanged.

- **`RecursiveChunker`**: uses `langchain_text_splitters.RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`. Each split gets `chunk_index` and `parent_id` in metadata.
- **`HierarchicalChunker`**: creates parent chunks (default 2000 chars) then child chunks (default 800 chars) from each parent. Children are tagged with `chunk_type: "child"`, `parent_id`, `child_index`. Parents tagged with `chunk_type: "parent"`, `child_count`. Children stored in `parent.children` list — the ingestion `node_chunk` flattens these before embedding.
- **`SemanticChunker`**: splits text to sentences via regex `(?<=[.!?]) +`, embeds all sentences using `TextEmbedder`, computes cosine similarity between adjacent sentences, breaks clusters when similarity drops below `breakpoint_threshold` (default 0.8). Since TextEmbedder normalizes, similarity = dot product.

Chunker registry + `get_chunker(method, **kwargs)` factory.

---

## 9. EMBEDDERS (`core/embedders/`)

All implement `BaseEmbedder` with `embed(items) -> List[List[float]]`, `dimension: int`, `model_name: str`.

- **`TextEmbedder`**: `SentenceTransformer("all-MiniLM-L6-v2")`, batch=32, normalize=True, returns `.tolist()`
- **`ImageEmbedder`**: `SentenceTransformer("clip-ViT-B-32")`, moves to CUDA if available, encodes `List[PIL.Image]`, batch=16
- **`MultimodalEmbedder`**: same CLIP model, accepts `List[Union[str, PIL.Image]]` — text and images share the same vector space

Embedder registry + `get_embedder(type, **kwargs)` factory.

---

## 10. IMAGE DESCRIBER (`core/processors/image_describer.py`)

`ImageDescriber(prompt, vision_model)` class:
- Constructor optionally sets `os.environ["VISION_MODEL"]` to override the model globally
- `describe(image_elements) -> List[TextElement]`: for each `ImageElement`, calls `call_vision_llm(img.content, prompt)` from `core/llm_src.py`, wraps result in `TextElement` preserving `element_id` and metadata plus `original_type: "image"` and `description_method: "vision_llm"`

---

## 11. LLM FACTORY (`core/llm_src.py`)

- `get_llm()` → `@lru_cache` `ChatGroq` with model from `LLM_MODEL` env var, `temperature=0.0`, `max_tokens=2048`
- `get_vision_llm()` → `@lru_cache` `ChatGroq` with model from `VISION_MODEL` env var, `max_tokens=512`
- `call_vision_llm(image: PIL.Image, prompt: str) -> str`: converts PIL → JPEG → base64, wraps in `HumanMessage` with `image_url` content block, invokes vision LLM, returns `.content.strip()`

---

## 12. QDRANT STORER (`core/storers/qdrant.py`)

`QdrantStorer(host, port, api_key)`:
- Connects with `https=False`
- Single named vector field: `VECTOR_NAME = "default"`
- `_ensure_collection(name, vector_dim)`: creates if missing; if exists, warns on dimension mismatch (do NOT recreate — just warn)
- `store(elements, collection_name) -> int`: filters elements with embeddings, derives `vector_dim` from first element, calls `_ensure_collection`, builds `PointStruct` list (payload includes `type`, `metadata`, `content` for TextElement or `base64_data` for ImageElement), upserts, returns stored count

---

## 13. INGESTION GRAPH (`ingestion/`)

### `nodes.py`

Five nodes, each decorated with `@log_node` (timing + item count logger):

1. **`node_load`**: calls `get_loader_for_file(file_path).load(file_path)`, updates metrics with `load_time` and `raw_elements`

2. **`node_describe_images`**: filters `ImageElement`s, skips if none. Creates `ImageDescriber` from config. Replaces image elements using ID-set subtraction: `image_ids = {img.element_id for img in image_elements}`, rebuilds list as `non_image + described_texts`. Updates metrics.

3. **`node_chunk`**: reads `chunking_method` from config, builds kwargs conditionally (recursive gets `chunk_size`, hierarchical gets `parent_chunk_size`/`child_chunk_size`, semantic gets `breakpoint_threshold`). After chunking, calls **`_flatten_elements(chunks)`** which walks `el.children`, appends them to a flat list, and clears `el.children = []`. Returns flat list.

4. **`node_embed`**: reads `embedding_type` and `embedding_model` from config. Builds items_to_embed: text content for TextElements, description fallback for ImageElements. Calls `embedder.embed(items)`, assigns `el.embedding = emb` and adds `embedding_model`/`embedding_dim` to metadata.

5. **`node_store`**: creates `QdrantStorer` from config (falls back to `get_settings()`), calls `storer.store(elements, collection_name)`, sets `status = "success"`.

Helper `_merge_metrics(state, new)` merges dicts non-destructively.

### `graph.py`

Routing: `_route_after_load(state)` → `"describe_images"` if `config.describe_images=True` AND any `ImageElement` in elements, else `"chunk"`.

Graph: `load → (conditional) → describe_images → chunk → embed → store → END` (describe_images is optional branch).

`run_ingestion(file_path, config, run_name)` is an async function that:
- Fills default config from `PipelineConfig.from_settings(settings)` if not provided
- Sets `document_id = os.path.basename(file_path)` (sanitized)
- Optionally wires `langfuse.langchain.CallbackHandler` (skips gracefully if not installed)
- Returns `final_state` dict or error dict on crash

---

## 14. RAG STATE (`rag/state.py`)

```python
class RAGState(TypedDict, total=False):
    # Inputs
    original_query: str
    collection_name: str
    config: Dict[str, Any]
    # Loop controls
    iteration: int
    max_iterations: int
    status: str  # "running"|"success"|"failed"|"max_iterations"|"no_store"
    # Query evolution
    active_queries: Annotated[List[str], _replace]
    all_queries_tried: Annotated[List[str], _append]
    # Retrieval
    retrieved_chunks: Annotated[List[RetrievedChunk], _replace]
    all_chunks: Annotated[List[RetrievedChunk], _append]
    selected_chunks: Annotated[List[RetrievedChunk], _replace]
    # Memory
    notepad: Annotated[List[NotepadEntry], _append]
    notepad_summary: str
    # Current iteration
    chosen_strategy: str
    current_task: str
    # Answer
    answer_draft: str
    answer_confidence: float
    final_answer: str
    answer_sources: List[str]
    # Store probe
    store_ready: bool
    vector_name: Optional[str]
    probe_diagnostics: str
    error: Optional[str]
```

`RetrievedChunk` TypedDict: `id, content, score, metadata`. `NotepadEntry` TypedDict: `iteration, strategy, queries_used, chunks_found, key_findings`.

---

## 15. RAG CONFIG (`rag/config.py`)

`RAGConfig(BaseModel)`:
- `collection_name = "mrag_default"`
- `strategy: StrategyName = "auto"` (Literal of all 7 strategy names + "auto")
- `enabled_strategies: List[str]` (all 7 by default)
- `top_k = 5`, `score_threshold = 0.25`
- `rerank_mode: Literal["llm","score"] = "llm"`, `rerank_top_n = 4`
- `max_iterations = 5`, `confidence_threshold = 0.75`
- `llm_model = "openai/gpt-oss-120b"`, `temperature = 0.0`, `max_tokens_per_call = 512`
- `embedding_type = "text"`, `embedding_model = "all-MiniLM-L6-v2"`
- `include_sources = True`, `answer_language = "English"`

---

## 16. QUERY STRATEGIES (`rag/strategies.py`)

7 strategies, each takes `(llm, query, notepad_summary) -> List[str]`. All use a shared `_SYSTEM` prompt instructing "one per line, no preamble" and a `_parse_lines(raw, max_items)` helper.

1. **`query_expansion`**: 3 expanded versions with synonyms/domain terms
2. **`query_rewriting`**: 2 phrasings optimized for dense retrieval (noun phrases, fewer stop words)
3. **`query_decomposition`**: 2-4 independent sub-questions
4. **`step_back_prompting`**: 1-2 abstract background questions + always appends original query
5. **`hyde`**: generates 2-sentence hypothetical document excerpt, returns `[excerpt, original_query]`
6. **`multi_query`**: 4 diverse phrasings
7. **`sub_query`**: 2-3 atomic facts still needed; uses last 200 chars of `notepad_summary` as memory hint

Registry dict `STRATEGY_FN`, `ALL_STRATEGIES` list, `run_strategy(name, llm, query, summary)` dispatcher.

---

## 17. RETRIEVER (`rag/retriever.py`)

`MultiQueryRetriever(host, port, api_key, embedder_type, embedder_model, top_k, score_threshold)`:
- Connects to Qdrant with `https=False`
- Loads embedder via `core.embedders.get_embedder`
- `_detect_vector_name(collection_name)`: introspects `info.config.params.vectors` — if dict, returns `"default"` if present else first key; if not dict (unnamed), returns `None`
- `_search_safe(embedding, collection_name, vector_name, limit, threshold)`: tries `client.query_points()` first (Qdrant 1.10+ Universal Query API), falls back to legacy `client.search()` if `AttributeError`
- `probe_collection(collection_name, raw_query) -> StoreProbeResult`: checks existence, counts points, detects vector name, runs original query verbatim, returns `StoreProbeResult` dataclass with `collection_exists, point_count, vector_name, initial_chunks, diagnostics`
- `retrieve(queries, collection_name) -> List[Dict]`: embeds all queries, runs `_search_safe` for each, deduplicates by ID keeping highest score, sorts descending, caps at `top_k * 2`
- `_hits_to_chunks(hits)`: converts Qdrant hits to `[{id, content, score, metadata}]` dicts

---

## 18. RERANKER (`rag/reranker.py`)

`rerank(llm, query, chunks, mode, top_n) -> List[RetrievedChunk]`:
- `mode="score"`: return `chunks[:top_n]` (passthrough)
- `mode="llm"`: takes up to 5 candidates, builds compact snippet string (ID[:8] + content[:120]), asks LLM to return JSON array of IDs ranked best-first via `_RERANK_SYSTEM` prompt. Parses JSON with regex fallback. Maps short IDs back, appends unmentioned chunks, returns `[:top_n]`. Falls back to score order on parse failure.

`format_context(chunks, max_chars=1600) -> str`: builds `[N|src:X] content` lines, hard-truncates at `max_chars`.

---

## 19. NOTEPAD (`rag/notepad.py`)

Token-efficient memory system. Max 150 tokens per summary.

- `compress_notepad(entries, original_query) -> str`: takes last 4 entries, formats as `[iter N] strategy | N chunks | findings`, adds cumulative 60-word summary
- `make_notepad_entry(iteration, strategy, queries_used, chunks_found, key_findings) -> NotepadEntry`
- `build_minimal_prompt(system_role, notepad_summary, current_task, context_snippet=None) -> str`: assembles `[ROLE]`, `[MEMORY]`, `[TASK]`, `[CONTEXT]` sections. Context hard-truncated to 1600 chars. Target: ≤700 tokens total.

---

## 20. RAG NODES (`rag/nodes.py`)

Seven nodes. All use `_llm(cfg)` (cached `ChatGroq` via `@lru_cache`) and `_build_retriever(cfg)`.

1. **`node_init`**: initializes all state fields to defaults (iteration=0, status="running", empty lists, etc.)

2. **`node_probe`**: builds retriever, calls `probe_collection()`. If collection missing or empty: returns `{store_ready: False, status: "no_store"}`. If ready: returns `{store_ready: True, vector_name, retrieved_chunks: initial_chunks, all_chunks: initial_chunks, notepad_summary: probe_summary}`.

3. **`node_plan`**: increments `iteration`. If `strategy != "auto"`, uses fixed strategy. If "auto": builds `build_minimal_prompt` with available (not-yet-tried) strategies listed, asks LLM to pick ONE by name, parses with `next((s for s in remaining if s in raw), remaining[0])`.

4. **`node_transform`**: calls `run_strategy(chosen_strategy, llm, query, notepad_summary)`, deduplicates against `all_queries_tried`.

5. **`node_retrieve`**: calls `retriever.retrieve(active_queries, collection_name)`. If returns empty, falls back to `state["retrieved_chunks"]` (probe baseline) with a warning log.

6. **`node_rerank`**: calls `rerank(llm, query, chunks, mode, top_n)`.

7. **`node_evaluate`**: builds minimal prompt with `format_context(selected, max_chars=1200)`. LLM must return answer draft, then `CONFIDENCE: 0.X`, then `KEY_FINDING: ...`. Parses with regex. Clamps confidence to [0.0, 1.0]. Creates `NotepadEntry`, appends to notepad, compresses to new summary.

8. **`node_synthesize`**: if `status == "no_store"`, returns error answer immediately. Otherwise builds final prompt with context (max 1400 chars) and answer draft. LLM writes complete answer optionally ending with `Sources: <list>`. Extracts sources from `selected_chunks` metadata.

---

## 21. RAG GRAPH (`rag/graph.py`)

Routing:
- `_route_after_probe(state)`: returns `"plan"` if `store_ready`, else `"synthesize"`
- `_should_continue(state)`: returns `"synthesize"` if `confidence >= threshold` OR `iteration >= max_iterations` (sets `status = "max_iterations"`), else `"loop"`

Graph topology:
```
init → probe → (conditional: plan OR synthesize)
plan → transform → retrieve → rerank → evaluate → (conditional: plan/loop OR synthesize)
synthesize → END
```

`run_rag(query, config) -> Dict` is async:
- Fills Qdrant connection from env if not in config
- Builds initial `RAGState` with all fields
- Optionally wires Langfuse callback
- Returns `final` state dict or error dict

---

## 22. FASTAPI BACKEND (`api/`)

### `main.py`

- FastAPI app with CORS allow all
- Serves `./uploads/` directory as static files at `/uploads`
- Mounts `rag_router` at `/rag`
- In-memory `_jobs` dict (swap for Redis in production)
- **`POST /ingest`** (multipart): validates extension (`.pdf/.txt/.md/.png/.jpg/.jpeg`), saves to `uploads/{uuid}_{safe_filename}` (never deleted), creates job, runs `ingestion.graph.run_ingestion` as `BackgroundTask`
- **`GET /ingest/{job_id}`**: polls job status returning `JobStatus(job_id, status, document_id, metrics, error)`
- **`GET /health`**, **`GET /config/defaults`**, **`GET /collections`** (lists Qdrant collections)

### `rag_router.py`

Prefix `/rag`:
- **`POST /rag/query`** (202): queues job, runs `rag.graph.run_rag` as background task
- **`GET /rag/query/{job_id}`**: polls returning `QueryResult` with all fields including `notepad, probe_diagnostics, store_ready`
- **`POST /rag/query/sync`**: awaits `run_rag` with 120s timeout
- **`GET /rag/strategies`**: returns `ALL_STRATEGIES` list

---

## 23. CLI (`cli/main.py`)

Typer app `m-rag`:
- Global `--log` option sets logging level (silences `transformers`, `sentence_transformers`, `httpx`)
- **`ingest PATH`** command: runs `run_ingestion` with hardcoded config defaults, shows Rich spinner
- **`status`** command: checks Qdrant, Postgres, Langfuse health; renders Rich Table with colored status

---

## 24. STREAMLIT FRONTEND (`frontend/app.py`)

Two tabs:

**Tab 1 — Ingest**:
- Left column: all `PipelineConfig` fields as Streamlit widgets (toggle, sliders, selectboxes). Chunking method controls show/hide relevant size sliders. Embedder type controls model dropdown.
- Right column: `st.file_uploader(accept_multiple_files=True)`. On submit, loops through files, POSTs each to `/ingest`, polls job status with progress bar, shows metrics as `st.metric`.

**Tab 2 — Agentic RAG**:
- Left column: collection selector, strategy mode (auto/fixed), per-strategy enable checkboxes, retrieval/rerank params, agent loop params, LLM params
- Estimated token budget display: `max_iter × ~1500 t/iter + synthesis`
- Right column: text area for question, "Run Agent" button
- On run: POSTs to `/rag/query`, polls with 2s sleep. Shows:
  - Store probe result as success/error banner
  - Live progress bar with iter count and confidence %
  - Expanding notepad trace (one expander per iteration showing queries used + key finding)
  - Final answer section with confidence badge
  - Sources section: images displayed via `st.image`, text chunks in expanders with `st.markdown`

---

## 25. KEY DESIGN DECISIONS

**Token efficiency**: Each LLM call targets ≤700 tokens: ~80 system role + ~150 notepad summary + ~80 task + ~400 context. Full conversation history is never sent raw to the LLM.

**Probe-first architecture**: Before any strategy loop, `node_probe` verifies the collection exists, has points, and runs the raw query. If empty/missing, the graph short-circuits directly to `node_synthesize` with a clear error. This avoids wasting 5 iterations against an empty store.

**Hierarchical chunk flattening**: `_flatten_elements()` in `node_chunk` expands `parent.children` into the flat element list before embedding, so both parents (broad context) and children (precise retrieval) get independently embedded and stored.

**Dynamic Qdrant schema**: `QdrantStorer` reads `vector_dim` from `len(embedded[0].embedding)` at store time. Swapping from `all-MiniLM-L6-v2` (384-d) to `clip-ViT-B-32` (512-d) just works — use a different `collection_name` to avoid mixing dimensions.

**Universal Query API**: Retriever uses `client.query_points()` (Qdrant 1.10+) with fallback to legacy `client.search()`. Named vector field `"default"` is used consistently between ingestion and retrieval.

**REPLACE vs APPEND reducers**: `elements` in ingestion state uses REPLACE (node owns the full list). `all_chunks`, `all_queries_tried`, `notepad` in RAG state use APPEND (accumulate across iterations). `retrieved_chunks`, `selected_chunks`, `active_queries` use REPLACE (only current iteration's data).

---

## 26. ENVIRONMENT VARIABLES (`.env`)

```env
GROQ_API_KEY=gsk_...
LLM_MODEL=openai/gpt-oss-120b
VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_EMBEDDING_TYPE=text
DEFAULT_COLLECTION=mrag_default
QDRANT_HOST=localhost
QDRANT_PORT=6333
DATABASE_URL=postgresql+asyncpg://mrag:mrag_secret@localhost:5433/mrag
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=http://localhost:3000
```

---

## 27. STARTUP SEQUENCE

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Install dependencies
uv sync

# 3. Run API
uvicorn api.main:app --reload --port 8000

# 4. Run frontend
streamlit run frontend/app.py

# 5. CLI ingestion
m-rag ingest path/to/document.pdf --collection my_collection

# 6. CLI status check
m-rag status
```
