"""
RAG-Lite HTTP API.

A thin FastAPI layer over the existing `RAGPipeline`. The pipeline (embedding model,
reranker, and a pgvector connection pool) is built once at startup and shared across
requests; endpoints are defined as sync functions so FastAPI runs them in its worker
threadpool, keeping CPU-bound embedding / DB / LLM calls off the event loop.

Scope (Phase 1): the service operates on a single collection (`PG_COLLECTION`).
Multi-collection / per-tenant access control is a later phase.
"""

import contextvars
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag_lite.config import Config
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite.generation import validate_citations
from rag_lite.service_settings import ServiceSettings
from rag_lite.types import IngestChunk, RetrievedChunk
from rag_lite.loaders import SUPPORTED_EXTENSIONS, UnsupportedFormatError

from .schemas import (
    ChatRequest,
    ChatResponse,
    Chunk,
    DeleteResponse,
    FileIngestResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LivenessResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SourceInfo,
    SourceListResponse,
)

# Cap conversation history fed back into condensation/generation (turns, not exchanges).
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))


def _to_chunk(c: RetrievedChunk) -> Chunk:
    """Map a RetrievedChunk (domain type) to the API Chunk schema."""
    return Chunk(
        content=c.content,
        score=c.score,
        source=c.source,
        title=c.title,
        uri=c.uri,
        page=c.page,
        metadata=c.metadata or {},
    )


# Returned (and cited as the answer) when retrieval is too weak to ground a response.
ABSTAIN_MESSAGE = "I don't have enough information in the provided context to answer that."


def _should_abstain(retrieved: list, pipeline: RAGPipeline) -> bool:
    """Abstain when nothing was retrieved, or the top score is below the threshold.

    Threshold 0.0 disables abstention. Compares against the top result's score,
    whose scale depends on the active pipeline (normalized [0,1] with reranking,
    small RRF values otherwise) — see RetrievalConfig.abstain_threshold.
    """
    threshold = pipeline.config.retrieval.abstain_threshold
    if not retrieved:
        return True
    if threshold <= 0:
        return False
    return retrieved[0].score < threshold

logger = logging.getLogger(__name__)

settings = ServiceSettings()

# Carries the current request's ID so every log line emitted while handling it can
# be correlated — the basis of tracing one request across modules.
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Inject the current request_id into every log record on the root handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def _generation_http_error(exc: Exception) -> HTTPException:
    """Map an LLM-call failure to an appropriate gateway status code."""
    if isinstance(exc, httpx.TimeoutException):
        return HTTPException(status_code=504, detail=f"LLM timed out: {exc}")
    if isinstance(exc, (httpx.ConnectError, httpx.TransportError)):
        return HTTPException(status_code=502, detail=f"LLM unreachable: {exc}")
    return HTTPException(status_code=502, detail=f"LLM error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the shared pipeline on startup, release the pool on shutdown."""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s - %(message)s",
    )
    # Attach the request-id filter to the root handler so it populates every record.
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIdFilter())
    config = Config.from_env()
    logger.info("Initializing RAG pipeline (collection=%s)…", config.storage.collection_name)
    app.state.pipeline = RAGPipeline(config)
    app.state.config = config
    try:
        yield
    finally:
        logger.info("Shutting down — closing connection pool.")
        app.state.pipeline.close()


app = FastAPI(
    title="RAG-Lite",
    description="Retrieval-Augmented Generation over private documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign/propagate a request ID, time the request, and emit an access log line."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id_ctx.set(rid)
    start = time.perf_counter()
    status = 500  # assume failure until proven otherwise (covers unhandled exceptions)
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "access method=%s path=%s status=%s duration_ms=%.1f",
            request.method, request.url.path, status, duration_ms,
        )
        request_id_ctx.reset(token)


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Enforce the API key when one is configured. No-op in open (dev) mode."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def get_pipeline() -> RAGPipeline:
    return app.state.pipeline


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------

@app.get("/healthz", response_model=LivenessResponse, tags=["ops"])
def healthz() -> LivenessResponse:
    """
    Liveness: is the process up and serving? Deliberately does NOT touch the DB —
    a transient database blip should not cause an orchestrator to kill and restart
    an otherwise-healthy API (which would turn a brief hiccup into a crash loop).
    """
    return LivenessResponse(status="ok")


@app.get("/readyz", response_model=HealthResponse, tags=["ops"])
def readyz() -> HealthResponse:
    """
    Readiness: can this instance serve traffic right now? Pings the DB pool and
    reports the doc count. Returns 503 if the database is unreachable so the load
    balancer stops routing here until it recovers.
    """
    pipeline: RAGPipeline = app.state.pipeline
    try:
        pipeline.vector_db.ping()
        count = pipeline.vector_db.size()
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        logger.error("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")

    return HealthResponse(
        status="ok",
        database="ok",
        collection=pipeline.vector_db.collection_name,
        documents=count,
    )


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------

@app.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
    tags=["retrieval"],
)
def search(req: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> SearchResponse:
    """Retrieve relevant chunks without generating an answer (precise child passages)."""
    results = pipeline.retrieve(
        query=req.query,
        top_n=req.top_n,
        use_hybrid_search=req.use_hybrid_search,
        use_reranking=req.use_reranking,
        expand_parents=False,  # /search returns the precise matched passages
    )
    return SearchResponse(
        query=req.query,
        results=[_to_chunk(c) for c in results],
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key)],
    tags=["retrieval"],
)
def query(req: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    """Retrieve context and generate an answer (non-streaming)."""
    retrieved = pipeline.retrieve(
        query=req.query,
        top_n=req.top_n,
        use_hybrid_search=req.use_hybrid_search,
        use_reranking=req.use_reranking,
    )

    # Abstain (skip the LLM entirely) when retrieval is too weak to ground an answer.
    if _should_abstain(retrieved, pipeline):
        logger.info("Abstaining: weak/no retrieval for query")
        return QueryResponse(
            query=req.query,
            answer=ABSTAIN_MESSAGE,
            sources=[_to_chunk(c) for c in retrieved],
            abstained=True,
        )

    try:
        answer = "".join(pipeline.generate(req.query, retrieved, stream=False))
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        raise _generation_http_error(exc)

    # Strip any citation markers the model invented that point to non-existent
    # sources, and report which valid sources were actually cited.
    answer, citations = validate_citations(answer, len(retrieved))
    return QueryResponse(
        query=req.query,
        answer=answer,
        sources=[_to_chunk(c) for c in retrieved],
        abstained=False,
        citations=citations,
    )


@app.post(
    "/query/stream",
    dependencies=[Depends(require_api_key)],
    tags=["retrieval"],
)
def query_stream(req: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> StreamingResponse:
    """
    Retrieve context and stream the generated answer as Server-Sent Events.

    Event sequence:
      event: sources   → JSON array of source chunks (with provenance)
      event: token     → JSON {text} per generated chunk
      event: citations → JSON {citations: [n,...]} validated indices actually cited
      event: done      → empty
    """
    retrieved = pipeline.retrieve(
        query=req.query,
        top_n=req.top_n,
        use_hybrid_search=req.use_hybrid_search,
        use_reranking=req.use_reranking,
    )

    abstain = _should_abstain(retrieved, pipeline)

    def event_stream():
        sources = [_to_chunk(c).model_dump() for c in retrieved]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        # Abstain: emit the canned answer as a single token, no LLM call.
        if abstain:
            logger.info("Abstaining (stream): weak/no retrieval for query")
            yield f"event: token\ndata: {json.dumps({'text': ABSTAIN_MESSAGE})}\n\n"
            yield "event: done\ndata: {\"abstained\": true}\n\n"
            return

        parts = []
        try:
            for token in pipeline.generate(req.query, retrieved, stream=True):
                parts.append(token)
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
        except Exception as exc:  # surface generation errors to the client stream
            # Headers are already sent, so we can't change the status code — report
            # the failure as a terminal stream event instead.
            detail = _generation_http_error(exc).detail
            logger.error("Generation failed mid-stream: %s", detail)
            yield f"event: error\ndata: {json.dumps({'detail': detail})}\n\n"
            return

        # Validate citations against the streamed answer (markers can't be stripped
        # from already-sent tokens, but we report which sources were validly cited).
        _, citations = validate_citations("".join(parts), len(retrieved))
        yield f"event: citations\ndata: {json.dumps({'citations': citations})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
    tags=["retrieval"],
)
def chat(req: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> ChatResponse:
    """
    Multi-turn QA. The client sends the full conversation in `messages` (stateless);
    the last message must be the new user turn. The follow-up is condensed with the
    history into a standalone question for retrieval, then answered with the history
    in context. Returns the answer, its sources, and the condensed question.
    """
    if req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="The last message must be a user turn.")
    question = req.messages[-1].content
    history = [(m.role, m.content) for m in req.messages[:-1]][-MAX_HISTORY_TURNS:]

    standalone, retrieved, response = pipeline.chat(
        history, question, stream=False,
        top_n=req.top_n, use_hybrid_search=req.use_hybrid_search,
        use_reranking=req.use_reranking,
    )

    # Abstain (skip the LLM) when retrieval is too weak. `response` is lazy, so not
    # consuming it means no generation call happens.
    if _should_abstain(retrieved, pipeline):
        return ChatResponse(
            answer=ABSTAIN_MESSAGE, sources=[_to_chunk(c) for c in retrieved],
            standalone_question=standalone, abstained=True,
        )
    try:
        answer = "".join(response)
    except Exception as exc:
        logger.error("Chat generation failed: %s", exc)
        raise _generation_http_error(exc)

    answer, citations = validate_citations(answer, len(retrieved))
    return ChatResponse(
        answer=answer, sources=[_to_chunk(c) for c in retrieved],
        standalone_question=standalone, citations=citations, abstained=False,
    )


# ----------------------------------------------------------------------
# Ingestion / admin
# ----------------------------------------------------------------------

@app.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
    tags=["admin"],
)
def ingest(req: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    """
    Index pre-chunked text into the served collection. Each document may be a bare
    string or an object with provenance ({text, source, title, uri, metadata}).
    Duplicates (same source+text) are skipped. For files, use POST /ingest/file.
    """
    chunks = []
    for d in req.documents:
        if isinstance(d, str):
            chunks.append(IngestChunk(text=d))
        else:
            chunks.append(IngestChunk(
                text=d.text,
                source=d.source or "inline",
                title=d.title,
                uri=d.uri,
                metadata=d.metadata or {},
            ))
    inserted = pipeline.vector_db.add_chunks(chunks, show_progress=False)
    return IngestResponse(
        received=len(req.documents),
        inserted=inserted,
        collection=pipeline.vector_db.collection_name,
        total_in_collection=pipeline.vector_db.size(),
    )


@app.post(
    "/ingest/file",
    response_model=FileIngestResponse,
    dependencies=[Depends(require_api_key)],
    tags=["admin"],
)
def ingest_file(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> FileIngestResponse:
    """
    Upload and index a document (PDF, DOCX, Markdown, or plain text). The file is
    parsed, token-chunked with provenance, and indexed under source=<filename>.
    """
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Persist under the *original* filename in a temp dir so the loader derives a
    # sensible title from the real name (not a random temp stem).
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, os.path.basename(filename))
            with open(tmp_path, "wb") as out:
                out.write(file.file.read())
            inserted = pipeline.ingest_file(tmp_path, source=filename, uri=filename)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except Exception as exc:
        logger.error("File ingestion failed for %s: %s", filename, exc)
        raise HTTPException(status_code=422, detail=f"Could not ingest '{filename}': {exc}")

    return FileIngestResponse(
        source=filename,
        inserted=inserted,
        collection=pipeline.vector_db.collection_name,
        total_in_collection=pipeline.vector_db.size(),
    )


@app.get(
    "/documents",
    response_model=SourceListResponse,
    dependencies=[Depends(require_api_key)],
    tags=["admin"],
)
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)) -> SourceListResponse:
    """List indexed source documents and their chunk counts."""
    sources = pipeline.vector_db.list_sources()
    return SourceListResponse(
        collection=pipeline.vector_db.collection_name,
        sources=[SourceInfo(source=s, chunks=n) for s, n in sources],
    )


@app.delete(
    "/documents",
    response_model=DeleteResponse,
    dependencies=[Depends(require_api_key)],
    tags=["admin"],
)
def delete_document(
    source: str = Query(..., description="Source document to delete (e.g. the filename)."),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DeleteResponse:
    """Delete all chunks belonging to one source document."""
    deleted = pipeline.vector_db.delete_by_source(source)
    return DeleteResponse(
        collection=pipeline.vector_db.collection_name,
        deleted=deleted > 0,
        deleted_chunks=deleted,
    )


@app.delete(
    "/collections/{name}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_api_key)],
    tags=["admin"],
)
def delete_collection(name: str, pipeline: RAGPipeline = Depends(get_pipeline)) -> DeleteResponse:
    """
    Clear all documents in a collection.

    Phase 1 serves a single collection, so only the configured collection may be
    cleared; other names return 404.
    """
    served = pipeline.vector_db.collection_name
    if name != served:
        raise HTTPException(
            status_code=404,
            detail=f"This service is bound to collection '{served}'.",
        )
    pipeline.vector_db.clear()
    return DeleteResponse(collection=name, deleted=True)


# ----------------------------------------------------------------------
# Test console (static UI)
# ----------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the bare root to the test console."""
    return RedirectResponse(url="/ui/")


# Mounted last so it never shadows the API routes above. Serves the single-page
# console at /ui (index.html). html=True makes /ui/ resolve to index.html.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")
