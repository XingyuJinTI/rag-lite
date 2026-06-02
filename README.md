# RAG-Lite

A lightweight, privacy-first Retrieval-Augmented Generation (RAG) system for knowledge-based question answering.

## Overview

RAG-Lite provides a modular RAG pipeline with hybrid search capabilities, running entirely on local infrastructure.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Data Loader │ ──▶ │  Vector DB   │ ──▶ │  Retrieval  │ ──▶ │ Generation │
└─────────────┘     │  (pgvector)  │     └─────────────┘     │  (Ollama)  │
                    └──────────────┘                         └────────────┘
                           │                    │
                    Embeddings (BGE)    Hybrid Search + Rerank
```

**Components:**

| Module | Description |
|--------|-------------|
| `data_loader` | Text file ingestion with UTF-8 encoding |
| `vector_db` | PostgreSQL + pgvector storage (HNSW + tsvector) |
| `retrieval` | Semantic search, tsvector FTS, RRF fusion, cross-encoder reranking |
| `generation` | Context-aware response generation |
| `rag_pipeline` | Orchestration layer |
| `config` | Environment-based configuration |

- **Embeddings**: sentence-transformers (local, HuggingFace models)
- **LLM Generation**: Ollama (local LLM inference)

## Quickstart — run inference yourself

The fastest path to a working `/query` is Docker Compose, which provisions Postgres +
pgvector and the API for you. The only external dependency is Ollama (the LLM), which
runs on your host.

**Prerequisites**

- [Docker](https://docs.docker.com/get-docker/) (with Compose v2)
- [Ollama](https://ollama.ai/) installed on the host

**1. Start Ollama and pull the generation model** (one-time download):

```bash
ollama serve                                                   # in its own terminal
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

**2. Build and start the stack** (first build downloads the embedding + reranker
models and bakes them into the image; subsequent runs need no internet):

```bash
docker compose up --build
```

**3. Confirm it's healthy:**

```bash
curl localhost:8000/healthz   # liveness: {"status":"ok"}
curl localhost:8000/readyz    # readiness: {"status":"ok","database":"ok","collection":"rag_lite","documents":0}
```

**4. Ingest documents** (raw text chunks; duplicates are skipped):

```bash
curl -X POST localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"documents": ["Cats sleep 12 to 16 hours per day.", "A group of cats is a clowder."]}'
```

**5. Ask a question** — retrieval + grounded generation:

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How long do cats sleep?", "use_hybrid_search": true, "use_reranking": true}'
```

Interactive API docs (try every endpoint from the browser): http://localhost:8000/docs

> **No Docker?** See [Installation](#installation) to run Postgres + pgvector and the
> API directly with `pip install -e .[service]` and `uvicorn api.main:app`. For a
> terminal-only experience with no HTTP layer, use the [CLI](#usage).
>
> **Air-gapped:** after the one-time build (and `ollama pull`), nothing needs the
> internet — both ML models are baked into the image with `HF_HUB_OFFLINE=1`.

## Requirements

- Python 3.8+
- PostgreSQL 16+ with pgvector extension
- [Ollama](https://ollama.ai/) running locally
- 4GB+ RAM (model dependent)
- ~500MB disk space for embedding model (downloaded on first run)

## Installation

```bash
git clone https://github.com/XingyuJinTI/rag-lite.git
cd rag-lite
pip install -r requirements.txt
```

**Set up PostgreSQL + pgvector:**

```bash
# macOS
brew install postgresql@16
brew services start postgresql@16
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

# Build and install pgvector extension
cd /tmp && git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config make
PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config make install

# Create database
createdb rag_lite
psql rag_lite -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Pull LLM model:**

```bash
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

The embedding model (`BAAI/bge-base-en-v1.5`) downloads automatically from HuggingFace on first run.

## Configuration

All settings are configured via environment variables:

**Models:**

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model (dense, 1024-d, long-context) |
| `LANGUAGE_MODEL` | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` | Ollama model for generation |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model for reranking |

**Storage:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_DSN` | `postgresql://localhost/rag_lite` | PostgreSQL connection string |
| `PG_COLLECTION` | `rag_lite` | Collection name (stored as a column) |
| `EMBEDDING_DIM` | `1024` | Embedding dimension — must match the model (bge-m3 → 1024, bge-base → 768) |
| `DATA_FILE` | `cat-facts.txt` | Input data file path |

**Retrieval:**

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVE_TOP_N` | `3` | Final results count |
| `RETRIEVE_K` | `50` | Candidates per search method |
| `FUSION_K` | `20` | Candidates after RRF fusion |
| `USE_HYBRID_SEARCH` | `false` | Enable hybrid search (semantic + tsvector + RRF) |
| `USE_RERANKING` | `false` | Enable cross-encoder reranking |
| `RRF_K` | `60` | RRF smoothing constant |
| `RRF_WEIGHT` | `0.7` | Semantic weight in RRF; tsvector gets 1 - RRF_WEIGHT |

## Usage

**CLI:**

```bash
export PG_DSN="postgresql://localhost/rag_lite"
ollama serve

# Default (cat-facts dataset)
python main.py

# RAGQArena Tech dataset (28k+ tech documents)
python main.py --dataset ragqa

# Custom text file
python main.py --file path/to/data.txt

# With hybrid search
python main.py --dataset ragqa --hybrid

# With cross-encoder reranking
python main.py --dataset ragqa --hybrid --rerank
```

**Programmatic:**

```python
from rag_lite import Config, ModelConfig, RAGPipeline
from rag_lite.data_loader import load_text_file

config = Config.from_env()
config.retrieval.use_reranking = True
config.model.reranker_model = ModelConfig.RERANKER_BGE_BASE

pipeline = RAGPipeline(config)
pipeline.index_documents(load_text_file("your-data.txt"))

results, response = pipeline.query("Your question here", stream=False)
print("".join(response))
```

**HTTP API:**

The service wraps the pipeline in a FastAPI app backed by a pgvector connection pool,
so it is safe to share across concurrent requests. The pipeline (embedding model,
reranker, pool) is built once at startup.

Run with Docker (brings up Postgres + pgvector + the API; expects Ollama on the host):

```bash
docker compose up --build
curl localhost:8000/healthz
```

The image bakes in both the embedding and reranker models and sets
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so once built the container runs
**fully air-gapped** — only the one-time build (and the Ollama model pull on the host)
needs internet.

Or run locally against an existing Postgres:

```bash
pip install -e .[service]
export PG_DSN="postgresql://localhost/rag_lite"
uvicorn api.main:app --reload
```

Endpoints (interactive docs at `/docs`):

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/healthz` | Liveness — process is up (no DB check; safe for restart probes) |
| `GET`  | `/readyz` | Readiness — pings the pool, reports doc count (503 if DB down) |
| `POST` | `/search` | Retrieve chunks (no generation) |
| `POST` | `/query` | Retrieve + generate an answer (JSON) |
| `POST` | `/query/stream` | Retrieve + stream the answer as Server-Sent Events |
| `POST` | `/ingest` | Index pre-chunked text (strings or `{text, source, title, uri, metadata}`) |
| `POST` | `/ingest/file` | Upload + parse + chunk a document (PDF/DOCX/MD/TXT) |
| `GET` | `/documents` | List indexed sources and chunk counts |
| `DELETE` | `/documents?source=<name>` | Delete all chunks for one source |
| `DELETE` | `/collections/{name}` | Clear the served collection |

```bash
# Upload a real document — it's parsed, token-chunked, and indexed with provenance
curl -F file=@handbook.pdf localhost:8000/ingest/file

# Or index raw text directly
curl -X POST localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"documents": ["Cats sleep 12-16 hours a day.", "A group of cats is a clowder."]}'

# Answers come back with source provenance (source/title/page) in `sources`
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How long do cats sleep?", "use_hybrid_search": true}'

curl "localhost:8000/documents"                       # list sources
curl -X DELETE "localhost:8000/documents?source=handbook.pdf"
```

Documents are parsed locally (PDF via `pypdf`, DOCX via `python-docx`) and split with
token-aware chunking (`CHUNK_MAX_TOKENS`/`CHUNK_OVERLAP`); each chunk keeps its
`source`, `title`, and (for PDFs) `page` for citation. Chunks are deduplicated by
`source` + content, so re-uploading the same file is idempotent.

**Citations & abstention.** The answer cites supporting context inline with bracketed
markers (`[1]`, `[2]`) whose numbers map to the order of the `sources` array (each with
`title`/`page`). Markers are validated server-side: any pointing to a non-existent
source are stripped from `answer`, and the response's `citations` field lists the
1-based source indices actually cited. (Inline markers are model-generated and
best-effort; the `sources` array itself is always reliable — it comes straight from
retrieval.) If retrieval is too weak to ground an answer, the service abstains — it
returns `"I don't have enough information…"` with `"abstained": true` and **does not
call the LLM**. Abstention is controlled by `ABSTAIN_THRESHOLD` (default `0` = off);
set it to a positive value, most meaningfully with reranking enabled (cross-encoder
scores are sigmoid relevance probabilities in `(0,1)`; raw RRF scores are ~`0.01–0.02`).

> **Upgrading an existing index:** the chunk-id scheme now includes the source, and
> older rows have no provenance. For clean citations, clear and re-ingest:
> `curl -X DELETE localhost:8000/collections/rag_lite` then re-upload your documents.

Set `API_KEY` to require an `X-API-Key` header on every request. Service-layer
settings (`API_HOST`, `API_PORT`, `API_KEY`, `CORS_ORIGINS`, `POOL_MIN_SIZE`,
`POOL_MAX_SIZE`, `OLLAMA_TIMEOUT`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP`,
`ABSTAIN_THRESHOLD`) are read from the environment / `.env`. Every
response carries an `X-Request-ID` (generated if not supplied) that appears in the
structured access logs, so a single request can be traced end-to-end.

> **Scope:** Phase 1 serves a single collection (`PG_COLLECTION`). Multi-collection
> tenancy, per-document access control, and richer ingestion (PDF/DOCX, citations)
> are planned in later phases — see [PRODUCTION.md](PRODUCTION.md).

## Retrieval

**Hybrid Search with RRF** (`USE_HYBRID_SEARCH=true`):

```
Query
  │
  ├──▶ Semantic Search (pgvector HNSW) ──▶ top 50
  │                                           │
  └──▶ tsvector Full-text Search ────────▶ top 50
                                              │
                                        Weighted RRF Fusion
                                         (semantic 0.7 / FTS 0.3)
                                              │
                                        top 20 (fusion_k)
                                              │
                               (optional cross-encoder rerank)
                                              │
                                        top 3 (top_n)
```

- **Semantic**: pgvector HNSW index (O(log n), persistent on disk)
- **Full-text**: PostgreSQL tsvector with GIN index — GENERATED ALWAYS column, always in sync
- **Fusion**: Weighted RRF combines rankings without raw score normalisation

**Semantic Only** (default, `USE_HYBRID_SEARCH=false`):
- pgvector HNSW only — best results on technical/code datasets where embeddings capture terminology better than keyword matching

**Cross-Encoder Reranking** (optional, disabled by default):
- Uses `BAAI/bge-reranker-base` — jointly encodes (query, chunk) pairs
- More accurate than bi-encoder cosine similarity at the cost of ~200ms latency
- Recommended when retrieval quality is critical
- Enable with `USE_RERANKING=true` or `--rerank` flag

## Storage

PostgreSQL + pgvector provides persistent vector storage:

- **Single store**: vectors (HNSW), full-text (tsvector GIN), and metadata in one table
- **ACID guarantees**: no sync drift between vector and keyword indexes
- **Deduplication**: `ON CONFLICT DO NOTHING` at the DB level
- **Multi-collection**: collections share one table via a `collection` column

Reset a collection:

```python
pipeline.vector_db.clear()
```

## Evaluation & Benchmarking

RAG-Lite includes an evaluation suite for measuring retrieval performance with standard IR metrics (Recall@K, MRR, NDCG, Hit Rate).

```bash
pip install -e .[eval]

# Full benchmark (all 28k docs)
python -m evaluation.run_benchmark

# Faster run with limited eval examples
python -m evaluation.run_benchmark --max-eval 300

# Compare retrieval strategies side-by-side
python -m evaluation.run_benchmark --compare --max-eval 100

# Compare with custom config file
python -m evaluation.run_benchmark --config-file configs/rrf_weight_sweep.json --max-eval 100
```

See [evaluation/README.md](evaluation/README.md) for datasets, metrics, config files, and CLI options.

**TODO:**

- [ ] Add document chunking with sentence boundaries
- [ ] Add Semantic F1 metric (LLM-as-judge)
- [x] Add cross-encoder reranking
- [x] pgvector migration (replaced ChromaDB + SQLite FTS5)
- [ ] Compare LLM models
- [ ] Compare embedding models

## Security

- **Local Processing**: Embeddings via sentence-transformers, LLM via Ollama
- **No External APIs**: Data stays local (models downloaded once, cached locally)
- **Local Persistence**: PostgreSQL on local filesystem

**Production next steps:**

1. Run Ollama behind firewall/VPN
2. Set PostgreSQL access controls (`pg_hba.conf`, strong `PG_DSN` credentials)
3. Implement input validation
4. Monitor resource usage

## License

MIT License
