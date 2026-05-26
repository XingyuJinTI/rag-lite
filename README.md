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
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `LANGUAGE_MODEL` | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` | Ollama model for generation |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model for reranking |

**Storage:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_DSN` | `postgresql://localhost/rag_lite` | PostgreSQL connection string |
| `PG_COLLECTION` | `rag_lite` | Collection name (stored as a column) |
| `EMBEDDING_DIM` | `768` | Embedding dimension — must match the model |
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
