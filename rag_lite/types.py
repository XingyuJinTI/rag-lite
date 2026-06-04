"""
Shared data types for the RAG pipeline.

Kept in a dependency-free module so both the storage layer (`vector_db`) and the
retrieval layer (`retrieval`) can import them without creating an import cycle.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class IngestChunk:
    """A unit of text to index, with its provenance.

    `page` is 1-based and only set for paginated formats (PDF); it is None for
    formats without a fixed page model (DOCX, Markdown, plain text).
    """

    text: str
    source: str = "inline"            # filename or logical origin; basis for delete-by-source
    title: Optional[str] = None       # human-readable document title
    uri: Optional[str] = None         # path/URL to the original document
    page: Optional[int] = None        # 1-based page number, when applicable
    chunk_index: int = 0              # ordinal of this chunk within its source document
    parent_id: Optional[str] = None   # id of the larger parent block (small-to-big retrieval)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Parent:
    """A larger context block, retrieved (not embedded) when a child chunk matches."""

    parent_id: str
    content: str
    source: str = "inline"
    title: Optional[str] = None
    uri: Optional[str] = None
    page: Optional[int] = None


@dataclass
class RetrievedChunk:
    """A retrieval result carrying its score and the provenance of its source chunk."""

    content: str
    score: float
    chunk_id: str = ""
    source: Optional[str] = None
    title: Optional[str] = None
    uri: Optional[str] = None
    page: Optional[int] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
