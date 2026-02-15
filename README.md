# RAG-Lite

A lightweight, privacy-first Retrieval-Augmented Generation (RAG) system for knowledge-based question answering.

## Overview

RAG-Lite provides a modular RAG pipeline with hybrid search capabilities, running entirely on local infrastructure via Ollama. No data leaves your environment.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Data Loader │ ──▶ │  Vector DB   │ ──▶ │  Retrieval  │ ──▶ │ Generation │
└─────────────┘     │  (ChromaDB)  │     └─────────────┘     └────────────┘
                    └──────────────┘
                           │                    │
                    Embeddings (BGE)    Hybrid Search + Rerank
```

**Components:**

| Module | Description |
|--------|-------------|
| `data_loader` | Text file ingestion with UTF-8 encoding |
| `vector_db` | ChromaDB-backed persistent vector storage |
| `retrieval` | Semantic search, BM25, RRF fusion, LLM reranking |
| `generation` | Context-aware response generation |
| `rag_pipeline` | Orchestration layer |
| `config` | Environment-based configuration |

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally
- 4GB+ RAM (model dependent)

## Installation

```bash
# Clone repository
git clone https://github.com/XingyuJinTI/rag-lite.git
cd rag-lite

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

**Pull models:**

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Configuration

All settings are configured via environment variables:

**Models:**

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` | Embedding model |
| `LANGUAGE_MODEL` | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` | Generation model |
| `OLLAMA_BASE_URL` | `None` | Custom Ollama endpoint |

**Storage:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CHROMA_COLLECTION` | `rag_lite` | ChromaDB collection name |
| `DATA_FILE` | `cat-facts.txt` | Input data file path |

**Retrieval:**

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVE_TOP_N` | `3` | Final results count |
| `RETRIEVE_K` | `50` | Candidates per search method |
| `FUSION_K` | `20` | Candidates after RRF fusion |
| `USE_HYBRID_SEARCH` | `true` | Enable hybrid search (semantic + BM25 + RRF) |
| `USE_RERANKING` | `false` | Enable LLM-based reranking |
| `RRF_K` | `60` | RRF constant (when hybrid enabled) |
| `BM25_K1` | `1.5` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 length normalization |

## Usage

**CLI:**

```bash
# Ensure Ollama is running
ollama serve

# Default (cat-facts dataset)
python main.py

# RAGQArena Tech dataset (28k+ tech documents)
python main.py --dataset ragqa

# Custom text file
python main.py --file path/to/data.txt

# With hybrid search
python main.py --dataset ragqa --hybrid --rrf-weight 0.7
```

**Programmatic:**

```python
from rag_lite.config import Config
from rag_lite.data_loader import load_text_file
from rag_lite.rag_pipeline import RAGPipeline

config = Config.from_env()
pipeline = RAGPipeline(config)

documents = load_text_file("your-data.txt")
pipeline.index_documents(documents)

results, response = pipeline.query("Your question here", stream=False)
print("".join(response))
```

## Retrieval

**Hybrid Search with RRF** (default, `USE_HYBRID_SEARCH=true`):

```
Query
  │
  ├──▶ Semantic Search (ChromaDB HNSW) ──▶ top 50
  │                                           │
  └──▶ BM25 Keyword Search ───────────▶ top 50
                                              │
                                        RRF Fusion
                                              │
                                        top 20 (fusion_k)
                                              │
                                   (optional LLM rerank)
                                              │
                                        top 3 (top_n)
```

- **Semantic**: ChromaDB's HNSW index (O(log n))
- **Keyword**: BM25 (industry standard)
- **Fusion**: RRF combines rankings without weight tuning

**Semantic Only** (`USE_HYBRID_SEARCH=false`):
- Faster, uses only ChromaDB's HNSW index
- Good for smaller datasets or when keyword matching isn't needed

**Reranking** (optional):
- LLM-based relevance scoring
- Enable with `USE_RERANKING=true`

## Storage

ChromaDB provides persistent vector storage:

- **Automatic Persistence**: Data survives restarts
- **Deduplication**: Duplicate documents skipped
- **Batch Embedding**: Efficient bulk indexing
- **Scalability**: Up to ~1M documents

Reset database:

```python
pipeline.vector_db.clear()
```

## Evaluation & Benchmarking

RAG-Lite includes an evaluation suite for measuring retrieval performance with standard IR metrics (Recall@K, MRR, NDCG, Hit Rate).

```bash
# Install evaluation dependencies
pip install -e .[eval]

# Full benchmark (all 28k docs, all eval examples)
python -m evaluation.run_benchmark

# Faster run with limited eval examples
python -m evaluation.run_benchmark --max-eval 300

# Compare retrieval configurations
python -m evaluation.run_benchmark --config-file configs/rrf_weight_sweep.json --max-eval 100
```

See [evaluation/README.md](evaluation/README.md) for detailed documentation on datasets, metrics, config files, and CLI options.

## Security

- **Local Processing**: All inference via Ollama
- **No External APIs**: Data stays local
- **Local Persistence**: ChromaDB on local filesystem
- **Telemetry Disabled**: ChromaDB analytics off

**Production - Next steps:**

1. Run Ollama behind firewall/VPN
2. Set file permissions on `CHROMA_PERSIST_DIR`
3. Implement input validation
4. Monitor resource usage

## License

MIT License
