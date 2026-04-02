# Enterprise Research Analyst

**Agentic Multi-Document RAG powered by hipVS on AMD Instinct GPUs**

An AI research analyst that ingests a document corpus and answers complex
questions by decomposing them into sub-queries, running parallel
GPU-accelerated retrieval via hipVS, and synthesising cited answers.

---

## Table of Contents

1. [Features](#features)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
6. [Technical Deep Dive](#technical-deep-dive)
   - [Document Ingestion Pipeline](#document-ingestion-pipeline)
   - [Text Chunking Strategy](#text-chunking-strategy)
   - [Embedding Model](#embedding-model)
   - [Embedding Cache](#embedding-cache)
   - [hipVS Vector Index](#hipvs-vector-index)
   - [Agentic RAG Engine](#agentic-rag-engine)
   - [LLM Backends](#llm-backends)
   - [Performance Tracking](#performance-tracking)
7. [hipVS API Usage](#hipvs-api-usage)
8. [UI Components](#ui-components)
9. [Network Access](#network-access)
10. [Configuration Tuning Guide](#configuration-tuning-guide)
11. [Troubleshooting](#troubleshooting)

---

## Features

| Feature | Description |
|---------|-------------|
| **Agentic RAG** | Complex questions are decomposed into focused sub-queries via LLM, each executed against the hipVS index, then synthesised into a unified answer with citations. |
| **Algorithm Arena** | Compare CAGRA, IVF-Flat, IVF-PQ, and Brute-Force side-by-side on the same query — see latency, results, and relevance differences live. |
| **Live Performance Dashboard** | Real-time metrics: last search latency, average latency, total queries, vectors indexed, document count, embedding dimension. |
| **Source Citations** | Every claim links back to the exact source passage with relevance scores and page numbers. |
| **Multi-Format Ingestion** | PDF (via PyMuPDF), Markdown, reStructuredText, plain text, and web URLs. Upload additional documents via the UI at runtime. |
| **Auto-Downloaded Corpus** | 11 curated ROCm blog posts (hipVS, MLPerf, robotics, VLMs, diffusion models, etc.) downloaded on first run — no static data checked in. |
| **Configurable LLM** | Ollama (local GPU inference), OpenAI (API), or retrieval-only mode (no LLM needed). |
| **Embedding Cache** | SHA-256-keyed cache avoids re-embedding unchanged corpora on restart. |
| **Heuristic Decomposition** | Even without an LLM, complex queries are split using keyword-based heuristics (comparison detection, clause splitting). |

---

### Data Flow (per query)

```
User Question
     │
     ▼                                                     CPU / GPU
[1] Query Decomposition ─── LLM splits into 2-4 sub-      ── CPU
     │                       queries via Ollama (llama3.2)
     │                       or heuristic keyword splitting
     ▼
[2] Embedding ───────────── Sub-query → 384-dim vector     ── GPU (default)
     │                       via sentence-transformers
     │                       (PyTorch 2.9 + ROCm 7.2)
     ▼
[3] GPU Vector Search ───── amd-cupy cp.asarray → hipVS    ── GPU
     │                       search (CAGRA/IVF/BF)
     │                       inner product, top-K
     ▼
[4] Transfer to CPU ─────── amd-cupy cp.asnumpy            ── GPU → CPU
     │                       (neighbors + distances)
     ▼
[5] Deduplication ───────── Merge results across            ── CPU
     │                       sub-queries by chunk_id
     ▼
[6] Synthesis ───────────── LLM generates answer with       ── CPU
     │                       [N] citations via Ollama
     ▼
[7] Format Response ─────── Reasoning trace table           ── CPU
                             + answer + sources
```

---

## System Requirements

| Component | Minimum | Tested on |
|-----------|---------|-----------|
| **GPU** | Any AMD GPU with ROCm support | AMD Instinct MI210 |
| **ROCm** | 7.2+ | 7.2 |
| **VRAM** | 4 GB | 64 GB |
| **Python** | 3.10 | 3.10 |
| **RAM** | 8 GB | 32+ GB |

### Python Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `amd-hipvs` | GPU-accelerated vector search (hipVS) | Yes |
| `amd-cupy` | GPU array management (CuPy for ROCm) | Yes |
| `torch` (ROCm) | PyTorch for AMD GPUs | Yes |
| `sentence-transformers` | Text embedding (MiniLM-L6-v2) | Yes |
| `numpy` | CPU array operations | Yes |
| `gradio` | Web UI framework | Yes |
| `pymupdf` (fitz) | PDF document parsing | Optional |
| `ollama` (server) | Local LLM inference | Optional |

---

## Installation

```bash
# Activate your hipVS environment
micromamba activate hipvs

# Install all dependencies
pip install -r requirements.txt

# Optional: Local LLM via Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

---

## Quick Start

### Retrieval-only mode (no LLM required)

```bash
python enterprise_research_analyst.py
```

Downloads 11 ROCm blog articles on first run, returns retrieved passages with
scores. No LLM needed — query decomposition uses keyword heuristics.

### Full agentic RAG with Ollama

```bash
# Start Ollama server in the background (if not already running)
ollama serve &

# Pull the default model (first time only)
ollama pull llama3.2:3b

# Launch demo
python enterprise_research_analyst.py --llm ollama
```

LLM-powered decomposition, parallel GPU retrieval, synthesised answers with
citations.

### CPU-only embeddings

By default the embedding model runs on the AMD GPU. To fall back to CPU
(e.g. if no free GPU is available):

```bash
python enterprise_research_analyst.py --llm ollama --embed-device cpu
```

### With OpenAI

```bash
export OPENAI_API_KEY="sk-..."
python enterprise_research_analyst.py --llm openai --openai-model gpt-4o-mini
```
---

## CLI Reference

```
python enterprise_research_analyst.py [OPTIONS]

Document options:
  --chunk-size N          Chunk size in characters (default: 800)
  --chunk-overlap N       Chunk overlap in characters (default: 150)

Embedding options:
  --embed-model MODEL     sentence-transformers model (default: all-MiniLM-L6-v2)
  --embed-device {cpu,gpu} Device for embedding model (default: gpu)

Index options:
  --algorithms ALG [ALG]  Algorithms to build: cagra ivf_flat ivf_pq brute_force
                          (default: all four)
  --top-k N               Results per sub-query (default: 5)

LLM options:
  --llm {ollama,openai,retrieval}
                          LLM backend (default: retrieval)
  --ollama-model MODEL    Ollama model name (default: llama3.2:3b)
  --ollama-url URL        Ollama API URL (default: http://localhost:11434)
  --openai-model MODEL    OpenAI model name (default: gpt-4o-mini)

Server options:
  --port N                Gradio server port (default: 7863)
```

---

## Technical Deep Dive

### Document Ingestion Pipeline

**Class: `DocumentParser`** (lines 355-422)

Handles multi-format parsing with per-page metadata preservation:

| Format | Parser | Output |
|--------|--------|--------|
| `.pdf` | PyMuPDF (`fitz`) | One segment per page, with page numbers |
| `.md`, `.rst` | Plain text read | Single segment per file |
| `.txt`, `.text` | Plain text read | Single segment per file |
| Web URLs | `urllib` + HTML stripping | Single segment per URL |

Each parsed segment is a dict:
```python
{"text": "...", "source": "filename.pdf", "page": 3}  # page is None for non-PDF
```

**Directory scanning** uses `pathlib.Path.rglob("*")` to recursively find all
supported files. The HTML stripper uses Python's built-in `HTMLParser` to
remove tags from fetched web content.

---

### Text Chunking Strategy

**Class: `TextChunker`** (lines 430-462)

Implements a sliding-window chunker with configurable size and overlap:

```
Document text:  [===========================================]
                 |--- chunk 1 (800 chars) ---|
                              |--- overlap (150) ---|
                                   |--- chunk 2 (800 chars) ---|
```

- **chunk_size** (default: 800 chars): Target size per chunk. Chosen to fit
  ~2-3 paragraphs — large enough for semantic coherence, small enough for
  precise retrieval.
- **overlap** (default: 150 chars): Ensures sentences split at chunk
  boundaries are still retrievable. Roughly one paragraph of context bleed.
- Each chunk carries metadata: `source`, `page`, `chunk_id`, `start_char`.

**Why character-based, not token-based?** Simpler, faster, and the embedding
model (MiniLM) has a 256 word-piece token limit which is rarely exceeded at
800 characters.

---

### Embedding Model

**Class: `TextEmbedder`** (lines 470-513)

Uses `sentence-transformers/all-MiniLM-L6-v2`:

| Property | Value |
|----------|-------|
| Architecture | 6-layer BERT (MiniLM distillation) |
| Embedding dimension | 384 |
| Max sequence length | 256 tokens |
| Normalisation | L2-normalised (unit vectors) |
| Similarity metric | Inner product = cosine similarity (since normalised) |

**Batch encoding** processes chunks in batches of 64 with progress reporting
(chunks/second). Embeddings are L2-normalised at encode time
(`normalize_embeddings=True`), which means inner product distance is
equivalent to cosine similarity — this is the metric used by all hipVS
indexes in this demo.

**Query encoding** uses `encode_query()` which returns a `(1, 384)` float32
numpy array, also normalised.

---

### Embedding Cache

**Functions**: `_embedding_cache_key`, `save_embedding_cache`,
`load_embedding_cache` (lines 521-548)

The cache key is a SHA-256 hash of:
```
abs_path(docs_dir) | embed_model_name | chunk_size_overlap
```

Cached artifacts are stored in `./data/cache/<hash16>/`:
- `embeddings.npy` — numpy array of shape `(N, 384)`
- `chunks.json` — full chunk metadata list

The cache is invalidated automatically if the number of chunks changes
(detects added/removed documents).

---

### hipVS Vector Index

**Class: `HipVSIndex`** (lines 556-667)

This is the core integration with hipVS. It wraps the `cuvs.neighbors` Python
API and manages four concurrent index types on the AMD GPU.

#### GPU Data Transfer

```python
gpu_data = cp.asarray(embeddings, dtype=cp.float32)
```

All embeddings are transferred to GPU memory as a CuPy float32 array once.
This array is shared across all index builds.

#### Index Build Parameters

| Algorithm | Key Parameters | Notes |
|-----------|---------------|-------|
| **CAGRA** | `graph_degree=min(32, N-1)`, `intermediate_graph_degree=min(64, N-1)`, `metric="inner_product"` | Graph-based ANN. Requires N >= 4 vectors. Graph degree adapts to corpus size. |
| **IVF-Flat** | `n_lists=min(100, N/10)`, `metric="inner_product"` | Inverted file with full vectors. Number of clusters scales with corpus size. |
| **IVF-PQ** | `n_lists=min(100, N/10)`, `pq_dim=min(dim, dim/4)`, `pq_bits=8`, `metric="inner_product"` | Inverted file with product quantization. Compresses vectors to reduce memory. |
| **Brute-Force** | `metric="inner_product"` | Exact k-NN. No approximation. Baseline for accuracy comparison. |

Each index is built via `cuvs.<algo>.build(params, gpu_data, resources=resources)`
and synchronised with `resources.sync()`.

#### Search

```python
gpu_query = cp.asarray(query_embedding, dtype=cp.float32)
distances, neighbors = cagra.search(search_params, index, gpu_query, k, resources=resources)
resources.sync()
ids = cp.asnumpy(cp.asarray(neighbors))[0].tolist()
```

Search workflow:
1. Query embedding (numpy) → CuPy GPU array
2. Algorithm-specific search with `SearchParams`
3. GPU sync to ensure results are ready
4. Transfer neighbor IDs and distances back to CPU
5. Return `(ids, distances, latency_ms)`

**Search parameters by algorithm:**

| Algorithm | SearchParams |
|-----------|-------------|
| CAGRA | Default (auto-tuned itopk, search width) |
| IVF-Flat | `n_probes=50` |
| IVF-PQ | `n_probes=max(1, sqrt(N)/5)` |
| Brute-Force | None (exhaustive) |

---

### Agentic RAG Engine

**Class: `ResearchAnalyst`** (lines 803-1103)

The central orchestrator that implements the agentic RAG pipeline.

#### Step 1: Query Decomposition

**With LLM** (`_llm_decompose`): Sends a prompt to the LLM asking it to break
the question into 2-4 focused sub-queries, returned as a JSON array. The
response is parsed with robust fallback logic (JSON array → code block
extraction → line-by-line extraction).

**Without LLM** (`_heuristic_decompose`): Keyword-based splitting:
1. Detect comparison queries ("compare", "vs", "versus", "difference", etc.)
2. Split on conjunctions (`and`, `or`, commas)
3. Split on question words (`how`, `what`, `why`, `when`, `which`)
4. Fall back to using the original query as-is

#### Step 2: Multi-Retrieve

**Method: `multi_retrieve(sub_queries, algorithm)`**

For each sub-query:
1. Embed with `TextEmbedder.encode_query()`
2. Search hipVS index with `HipVSIndex.search()`
3. Map neighbor IDs back to chunk metadata
4. Deduplicate across sub-queries by `chunk_id`
5. Sort merged results by descending similarity score

Returns `(unique_chunks, per_query_stats, total_latency_ms)` where
`per_query_stats` is a list of dicts with per-sub-query metrics.

#### Step 3: Synthesis

**Method: `synthesise(question, sub_queries, chunks)`**

Constructs a prompt with:
- The original question
- List of sub-queries investigated
- Up to 15 retrieved passages with source, page, and score metadata
- Instructions to cite with `[N]` notation and answer only from provided passages

The LLM generates a structured answer grounded in the retrieved evidence.

#### Step 4: Format Response

**Method: `_format_response(...)`**

Always displays (even in retrieval-only mode):
- Mode label ("Agentic RAG Analysis" or "Multi-Retrieval Analysis")
- Decomposition table: sub-query text, chunks found, per-query latency
- Retrieval summary: unique passages, source documents, total time, algorithm, vectors indexed
- Synthesised answer (if LLM) or retrieved passages with scores (if retrieval-only)
- Source citations with excerpts

#### Dynamic Document Addition

**Method: `add_documents(file_paths, algorithms)`**

Allows adding documents at runtime via the UI:
1. Parse new files with `DocumentParser`
2. Chunk with `TextChunker`
3. Assign new `chunk_id` values (offset from existing)
4. Embed new chunks
5. Concatenate with existing embeddings (`np.vstack`)
6. Rebuild all hipVS indexes from scratch

---

### LLM Backends

Three interchangeable backends, all implementing `LLMBackend.generate(prompt) → str`:

| Backend | Class | API | Config |
|---------|-------|-----|--------|
| **Ollama** | `OllamaBackend` | `POST /api/generate` | `temperature=0.3`, `num_predict=1024`, `timeout=180s` |
| **OpenAI** | `OpenAIBackend` | `POST /v1/chat/completions` | `temperature=0.3`, `max_tokens=1024`, `timeout=180s` |
| **Retrieval-only** | `RetrievalOnlyBackend` | None | Returns empty string; skips synthesis |

All backends use Python's built-in `urllib.request` — no external HTTP
libraries required.

---

### Performance Tracking

**Class: `PerformanceTracker`** (lines 675-704)

Records every search query with:
- Timestamp
- Algorithm used
- Retrieval latency (ms)
- Number of results returned
- Number of sub-queries

Exposes `avg_latency()`, `last_latency()`, and `total_queries()` for the
live dashboard.

---

## hipVS API Usage

This demo exercises the following hipVS (cuvs) Python APIs:

```python
from cuvs.common import Resources
from cuvs.neighbors import brute_force, cagra, ivf_flat, ivf_pq

# Shared GPU resource handle
resources = Resources()

# Build (embeddings already on GPU via CuPy)
gpu_data = cp.asarray(embeddings, dtype=cp.float32)

# CAGRA
cagra_params = cagra.IndexParams(metric="inner_product", graph_degree=32,
                                  intermediate_graph_degree=64)
cagra_index = cagra.build(cagra_params, gpu_data, resources=resources)

# IVF-Flat
ivf_flat_params = ivf_flat.IndexParams(n_lists=10, metric="inner_product")
ivf_flat_index = ivf_flat.build(ivf_flat_params, gpu_data, resources=resources)

# IVF-PQ
ivf_pq_params = ivf_pq.IndexParams(n_lists=10, metric="inner_product",
                                    pq_dim=96, pq_bits=8)
ivf_pq_index = ivf_pq.build(ivf_pq_params, gpu_data, resources=resources)

# Brute-Force
bf_index = brute_force.build(gpu_data, metric="inner_product", resources=resources)

resources.sync()

# Search
gpu_query = cp.asarray(query_vec, dtype=cp.float32)  # shape (1, 384)

distances, neighbors = cagra.search(cagra.SearchParams(), cagra_index,
                                     gpu_query, k=5, resources=resources)
resources.sync()

# Transfer results to CPU
ids = cp.asnumpy(neighbors)[0].tolist()
scores = cp.asnumpy(distances)[0].tolist()
```

---

## UI Components

### Tab 1: Research Chat

- **Chatbot**: Multi-turn conversation with the research analyst
- **Algorithm selector**: Dropdown to choose CAGRA, IVF-Flat, IVF-PQ, or Brute-Force
- **Top-K slider**: Adjust results per sub-query (1-20)
- **Example queries**: Pre-built questions to demonstrate capabilities
- **Performance metrics panel**: Live-updating HTML grid showing last latency, avg latency, total queries, vectors indexed, document count, embedding dimension

### Tab 2: Algorithm Arena

- Enter a query and run it against **all** built algorithms simultaneously
- Side-by-side comparison cards showing:
  - Per-algorithm latency
  - Top-5 retrieved passages with source and score
- Latency summary bar at the top

### Tab 3: Knowledge Base

- **File upload**: Drag-and-drop PDF, Markdown, or text files
- **Corpus overview table**: Lists all indexed documents with chunk counts
- Documents are parsed, chunked, embedded, and all indexes are rebuilt on upload

---

## Network Access

The server binds to `0.0.0.0` by default, making it accessible from any
machine on the same network.

```bash
# Find your machine's IP
hostname -I

# Colleagues access via:
# http://<your-ip>:<port>
# http://<hostname>:<port>
```

---

## Configuration Tuning Guide

### For larger corpora (>10K documents)

```bash
python enterprise_research_analyst.py \
    --chunk-size 512 \
    --chunk-overlap 100 \
    --top-k 10 \
    --algorithms cagra ivf_pq \
    --llm ollama
```

- Smaller chunks improve retrieval precision
- More top-K results give the LLM more context for synthesis
- CAGRA and IVF-PQ are most efficient at scale (skip brute-force)

### For maximum accuracy

```bash
python enterprise_research_analyst.py \
    --embed-model all-mpnet-base-v2 \
    --chunk-size 600 \
    --chunk-overlap 200 \
    --top-k 15 \
    --algorithms brute_force \
    --llm openai --openai-model gpt-4o
```

- `all-mpnet-base-v2` produces 768-dim embeddings (higher quality than MiniLM)
- Brute-force gives exact nearest neighbors (no approximation)
- GPT-4o for best synthesis quality

### For lowest latency

```bash
python enterprise_research_analyst.py \
    --chunk-size 1000 \
    --top-k 3 \
    --algorithms cagra \
    --llm retrieval
```

- Fewer, larger chunks = fewer vectors to index
- Low top-K = less data to process
- CAGRA only = fastest ANN algorithm
- Retrieval-only = no LLM latency

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ImportError: ROCM_HOME is not set` | ROCm environment not configured | `export ROCM_HOME=/opt/rocm` and `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH` |
| `OSError: Cannot find empty port` | Port already in use | Kill the process: `fuser -k <port>/tcp` or use `--port <other>` |
| PDF upload fails silently | PyMuPDF not installed | `pip install pymupdf` |
| Ollama connection refused | Ollama server not running | Start with `ollama serve &`, pull the model with `ollama pull llama3.2:3b`, then retry |
| `--embed-device gpu` hangs | GPU contention from other processes | Check `rocm-smi` for idle GPUs and set `export HIP_VISIBLE_DEVICES=<id>` to pin to a free device |
| `CAGRA: too few vectors` | Corpus has < 4 chunks | Add more documents or use `--algorithms ivf_flat brute_force` |
| Slow first query | GPU/model warmup | Expected — subsequent queries are fast. The demo runs automatic warmup at startup. |
| Embedding cache not loading | Chunk config changed | Cache key includes chunk_size and overlap. Change either → cache miss → re-embeds automatically. |
| LLM synthesis is empty | Running in retrieval-only mode | Use `--llm ollama` or `--llm openai` for full agentic experience |
