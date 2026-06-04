# RAG-Lite

A lightweight, privacy-first Retrieval-Augmented Generation (RAG) service for
question answering over your own documents. Runs entirely on local infrastructure.

## Architecture

```
Documents ─▶ Loaders + token chunking ─▶ pgvector (HNSW + tsvector)
                                              │
                       Retrieval: semantic (+ optional hybrid FTS / rerank)
                                              │
                              Generation (Ollama) ─▶ cited answer
```

- **Embeddings**: `BAAI/bge-m3` via sentence-transformers (local, 1024-d dense, long-context)
- **Store**: PostgreSQL + pgvector — vectors (HNSW), full-text (tsvector), and metadata in one table
- **LLM**: Ollama (local inference); model set via `LANGUAGE_MODEL`
- **API**: FastAPI over a shared connection pool; static test console at `/ui`

## Quickstart (Docker)

The stack (API + Postgres/pgvector) comes up with one command. The only external
dependency is **Ollama on the host** for answer generation.

```bash
# 1. Start Ollama and pull the generation model (one-time)
ollama serve
ollama pull qwen2.5:14b   # best reasoning for contract/doc Q&A; use qwen2.5:7b for speed

# 2. Build and run (first build bakes the embedding + reranker models into the image)
docker compose up --build
```

Then:

```bash
curl localhost:8000/healthz                       # liveness
curl localhost:8000/readyz                         # readiness + doc count

# Upload a document — parsed, token-chunked, indexed with provenance
curl -F file=@handbook.pdf localhost:8000/ingest/file

# Ask — grounded answer with citations + sources
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "How long do cats sleep?"}'
```

- **Test console**: http://localhost:8000/ui   ·   **API docs**: http://localhost:8000/docs
- **Air-gapped:** after the one-time build (and `ollama pull`), nothing needs the
  internet — both ML models are baked into the image with `HF_HUB_OFFLINE=1`.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/healthz` | Liveness — process up (no DB check; safe for restart probes) |
| `GET`  | `/readyz` | Readiness — pings the pool, reports doc count (503 if DB down) |
| `POST` | `/search` | Retrieve chunks (no generation) |
| `POST` | `/query` | Retrieve + generate a cited answer |
| `POST` | `/query/stream` | Same, streamed as Server-Sent Events |
| `POST` | `/ingest` | Index pre-chunked text (`["str", ...]` or `{text, source, title, uri, metadata}`) |
| `POST` | `/ingest/file` | Upload + parse + chunk a document (PDF / DOCX / MD / TXT) |
| `GET` | `/documents` | List indexed sources + chunk counts |
| `DELETE` | `/documents?source=<name>` | Delete all chunks for one source |
| `DELETE` | `/collections/{name}` | Clear the served collection |

**Provenance & citations.** Every retrieved chunk carries `source`, `title`, and (for
PDFs) `page`. `/query` returns an `answer`, the `sources` it drew on, and a `citations`
list of the source indices actually cited. Inline `[n]` markers are model-generated
(best-effort) and validated server-side — out-of-range markers are stripped; the
`sources` array is always reliable (it comes straight from retrieval).

**Abstention.** If the top retrieval score is below `ABSTAIN_THRESHOLD` (default `0` =
off), the service returns "I don't have enough information…" with `"abstained": true`
and does **not** call the LLM. Most meaningful with reranking on (scores are absolute).

**Observability.** Every response carries an `X-Request-ID` (generated if absent) that
appears in the structured access logs, so a request can be traced end to end.

## Configuration

All via environment variables (or a `.env` file). Most-used:

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_DSN` | `postgresql://localhost/rag_lite` | PostgreSQL connection string |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | sentence-transformers embedding model |
| `EMBEDDING_DIM` | `1024` | Must match the model (bge-m3 → 1024, bge-base → 768) |
| `TABLE_NAME` | `chunks` | Physical table; dim is fixed per table, so different-dim models need different tables |
| `LANGUAGE_MODEL` | `qwen2.5:14b` | Ollama model (best doc-QA reasoning; `qwen2.5:7b` ≈½ latency for simple lookups) |
| `USE_HYBRID_SEARCH` | `false` | Add tsvector keyword search fused with semantic (RRF) |
| `USE_RERANKING` | `false` | Cross-encoder rerank (`BAAI/bge-reranker-base`) |
| `ABSTAIN_THRESHOLD` | `0` | Min top score to answer; `0` disables abstention |
| `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP` | `256` / `48` | Token-aware chunking |
| `OLLAMA_TIMEOUT` | `60` | Seconds before an LLM call fails (→ 504) |
| `API_KEY` | _(unset)_ | If set, require `X-API-Key` on every request |

Service/pool knobs: `API_HOST`, `API_PORT`, `CORS_ORIGINS`, `POOL_MIN_SIZE`, `POOL_MAX_SIZE`.

> Changing `EMBEDDING_DIM` needs a fresh table (the embedding column's dimension is
> fixed). Use a new `TABLE_NAME`, or reset with `docker compose down -v`.

## Local dev (without Docker)

Requires Python 3.10+, a PostgreSQL 16 with the `vector` extension, and Ollama.

```bash
pip install -e .[service]
export PG_DSN="postgresql://localhost/rag_lite"
uvicorn api.main:app --reload
```

A terminal-only CLI (no HTTP) is also available: `python main.py --help`.

## Evaluation

Retrieval quality is measured against ground truth with standard IR metrics
(Recall@K, MRR, NDCG):

```bash
pip install -e .[eval]
python -m evaluation.run_benchmark --compare --max-eval 100   # index + compare configs
python -m evaluation.run_eval_only                            # re-score an indexed corpus (fast)
```

`evaluation/compare_embeddings.py` and `compare_generation.py` compare embedding models
and LLMs respectively.

## Roadmap

Phases 1–2 (serving, hardening, ingestion + citations) are done. Access control,
multi-collection tenancy, and groundedness eval are next — see [PRODUCTION.md](PRODUCTION.md).

## License

MIT
