"""
RAG-Lite HTTP API.

A thin FastAPI layer over the existing `RAGPipeline`. The pipeline (embedding model,
reranker, and a pgvector connection pool) is built once at startup and shared across
requests; endpoints are defined as sync functions so FastAPI runs them in its worker
threadpool, keeping CPU-bound embedding / DB / LLM calls off the event loop.

Scope (Phase 1): the service operates on a single collection (`PG_COLLECTION`).
Multi-collection / per-tenant access control is a later phase.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag_lite.config import Config
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite.service_settings import ServiceSettings

from .schemas import (
    Chunk,
    DeleteResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

settings = ServiceSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the shared pipeline on startup, release the pool on shutdown."""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
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


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Enforce the API key when one is configured. No-op in open (dev) mode."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def get_pipeline() -> RAGPipeline:
    return app.state.pipeline


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    """Liveness + readiness: confirms the DB pool answers and reports doc count."""
    pipeline: RAGPipeline = app.state.pipeline
    try:
        pipeline.vector_db.ping()
        db_status = "ok"
        count = pipeline.vector_db.size()
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        logger.error("Health check failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")

    return HealthResponse(
        status="ok",
        database=db_status,
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
    """Retrieve relevant chunks without generating an answer."""
    results = pipeline.retrieve(
        query=req.query,
        top_n=req.top_n,
        use_hybrid_search=req.use_hybrid_search,
        use_reranking=req.use_reranking,
    )
    return SearchResponse(
        query=req.query,
        results=[Chunk(content=c, score=s) for c, s in results],
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
    answer = "".join(pipeline.generate(req.query, retrieved, stream=False))
    return QueryResponse(
        query=req.query,
        answer=answer,
        sources=[Chunk(content=c, score=s) for c, s in retrieved],
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
      event: sources  → JSON array of {content, score}
      event: token    → JSON {text} per generated chunk
      event: done     → empty
    """
    retrieved = pipeline.retrieve(
        query=req.query,
        top_n=req.top_n,
        use_hybrid_search=req.use_hybrid_search,
        use_reranking=req.use_reranking,
    )

    def event_stream():
        sources = [{"content": c, "score": s} for c, s in retrieved]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        try:
            for token in pipeline.generate(req.query, retrieved, stream=True):
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
        except Exception as exc:  # surface generation errors to the client stream
            logger.error("Generation failed mid-stream: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
    """Index raw text chunks into the served collection. Duplicates are skipped."""
    inserted = pipeline.vector_db.add_chunks(req.documents, show_progress=False)
    return IngestResponse(
        received=len(req.documents),
        inserted=inserted,
        collection=pipeline.vector_db.collection_name,
        total_in_collection=pipeline.vector_db.size(),
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
