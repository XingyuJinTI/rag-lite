"""Request/response models for the RAG-Lite HTTP API."""

from typing import Any, Dict, List, Optional, Union

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
    source: Optional[str] = None
    title: Optional[str] = None
    uri: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: List[Chunk]


class QueryRequest(SearchRequest):
    """A query that also generates an answer from the retrieved context."""


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Chunk]
    abstained: bool = False


class IngestDocument(BaseModel):
    """A single text chunk to index, with optional provenance."""
    text: str = Field(..., min_length=1)
    source: Optional[str] = None
    title: Optional[str] = None
    uri: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    # Accept either bare strings (convenience) or structured documents with metadata.
    documents: List[Union[str, IngestDocument]] = Field(
        ..., min_length=1, description="Text chunks to index (strings or objects)."
    )


class IngestResponse(BaseModel):
    received: int
    inserted: int
    collection: str
    total_in_collection: int


class FileIngestResponse(BaseModel):
    source: str
    inserted: int
    collection: str
    total_in_collection: int


class SourceInfo(BaseModel):
    source: str
    chunks: int


class SourceListResponse(BaseModel):
    collection: str
    sources: List[SourceInfo]


class DeleteResponse(BaseModel):
    collection: str
    deleted: bool
    deleted_chunks: Optional[int] = None


class LivenessResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    database: str
    collection: str
    documents: int
