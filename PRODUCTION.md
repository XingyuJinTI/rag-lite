# Productionization Plan — Internal Enterprise RAG

Target: a RAG service deployed **inside a company** to answer questions over private
documents (wikis, PDFs, policies, tickets, code docs). Stays on-prem / in-VPC for
privacy. The differentiators that matter here are **document fidelity, citations,
access control, and operability** — not billing or public multi-tenancy.

This builds on the existing engine: pgvector (HNSW + tsvector) → weighted RRF hybrid
search → cross-encoder rerank → LLM generation, plus the evaluation harness.

---

## Honest state today

**Strong:** clean module separation, good hybrid-retrieval engine, real eval harness
(Recall@k / MRR / NDCG / config comparison), lazy model loading.

**Blocking for enterprise use:**
- No API — only an interactive CLI. Nothing can consume it.
- Single shared `psycopg2` connection — not concurrency-safe.
- Ingestion is one-line-per-doc + a **hard 500-char truncation that silently drops
  the tail of long documents** (`VectorDB._truncate_text`). Real docs are lost.
- No metadata: chunks have no `source`, `title`, `page` → **citations are impossible**.
- No access control — `collection` is just an unguarded column.
- All-or-nothing reindex (`clear()` + re-add); no per-document update/delete.
- Ollama-only generation, hardcoded.

---

## Phase 1 — Serving & data-layer foundation
*Goal: make it callable and safe under concurrent load.*

- **FastAPI service** (`api/`) with: `POST /query` (retrieve+generate, streaming),
  `POST /search` (retrieve only), `POST /ingest`, `DELETE /documents/{source}`,
  `GET /healthz`.
- **Connection pooling**: migrate `VectorDB` to `psycopg` (v3) + `psycopg_pool`,
  remove the single long-lived connection and manual commits.
- **Config via Pydantic Settings** with validation (fail fast on bad DSN / dim mismatch).
- **Docker + docker-compose**: app + Postgres-w/-pgvector + (optional) Ollama, one
  `docker compose up`. Move the 26MB corpus and `chroma_db/` out of the repo.

## Phase 2 — Document fidelity & citations
*Goal: answers a person can trust and verify. The core enterprise differentiator.*

- **Real document loaders**: PDF, DOCX, HTML, Markdown, plain text (e.g. `unstructured`
  or `pymupdf` + `markdownify`). Extract text + structure.
- **Token-aware chunking** with overlap (replace char truncation). Configurable size/
  overlap; never silently drop content.
- **Metadata schema**: add `source`, `title`, `page`/`section`, `created_at`, `uri`,
  `content_hash` columns. Embed-time metadata travels with each chunk.
- **Citations in output**: generation returns answer **plus** the source chunks with
  title/page/uri so the UI can show "according to <doc> p.4". Add inline citation
  markers in the prompt contract.
- **"I don't know" calibration**: if top rerank score is below a threshold, answer
  abstains rather than hallucinating.

## Phase 3 — Access control & lifecycle
*Goal: safe for multiple teams over sensitive docs.*

- **Per-document ACLs**: `allowed_groups`/`owner` metadata; retrieval filters by the
  caller's identity (`WHERE allowed_groups && :user_groups`). Index the filter columns.
- **Auth**: API-key or OIDC/JWT at the gateway; pass identity into retrieval.
- **Incremental ingestion**: upsert-by-`source` using `content_hash` to skip unchanged
  chunks; `DELETE /documents/{source}` removes a doc's chunks cleanly.
- **Filtered retrieval API**: metadata filters (date range, source type, tags) on both
  semantic and FTS paths, fused consistently.

## Phase 4 — Quality, observability, ops
*Goal: keep it accurate and debuggable in production.*

- **Groundedness/faithfulness eval**: extend the harness beyond retrieval to score
  whether the generated answer is supported by retrieved context (LLM-as-judge or NLI).
- **Query observability**: structured per-query logs (latency by stage, retrieved
  chunk ids + scores, tokens, abstain/answer). Persist for offline analysis.
- **Pluggable LLM backend**: abstract `generation` so Ollama / OpenAI / Anthropic /
  vLLM are swappable via config (keeps on-prem default, allows approved cloud).
- **Tests + CI**: unit tests for chunking, RRF, retrieval, ingestion idempotency;
  GitHub Actions running them + the eval smoke test.
- **Minimal admin UI** (optional): upload docs, run a query, see cited sources.

---

## Suggested sequencing
Each phase is independently shippable. Phase 1 → 2 delivers the most enterprise value
fastest (callable service + trustworthy cited answers). Phase 3 unlocks multi-team
rollout; Phase 4 keeps it healthy.
