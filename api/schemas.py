"""Request/response models for the RAG-Lite HTTP API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query.")
    top_n: Optional[int] = Field(None, ge=1, le=100, description="Results to return.")
    use_hybrid_search: Optional[bool] = Field(
        None, description="Override hybrid (semantic + FTS) search."
    )
    use_reranking: Optional[bool] = Field(
        None, description="Override cross-encoder reranking."
    )


class Chunk(BaseModel):
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[Chunk]


class QueryRequest(SearchRequest):
    """A query that also generates an answer from the retrieved context."""


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Chunk]


class IngestRequest(BaseModel):
    documents: List[str] = Field(..., min_length=1, description="Raw text chunks to index.")


class IngestResponse(BaseModel):
    received: int
    inserted: int
    collection: str
    total_in_collection: int


class DeleteResponse(BaseModel):
    collection: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    database: str
    collection: str
    documents: int
