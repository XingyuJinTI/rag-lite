"""
Main RAG pipeline that orchestrates the entire retrieval and generation process.

This module provides a high-level interface for the complete RAG workflow.
"""

import logging
from typing import List, Tuple, Iterator, Optional, Union

from .config import Config
from .vector_db import VectorDB
from .retrieval import retrieve, expand_query
from .generation import generate_response
from .types import IngestChunk, RetrievedChunk
from . import loaders, chunking

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates retrieval and generation.
    
    This class provides a clean interface for:
    1. Loading and indexing documents
    2. Retrieving relevant context (semantic + BM25 with RRF fusion)
    3. Generating responses
    """

    def __init__(self, config: Config):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.vector_db = VectorDB(
            embedding_model=config.model.embedding_model,
            pg_dsn=config.storage.pg_dsn,
            collection_name=config.storage.collection_name,
            max_chunk_chars=config.storage.max_chunk_chars,
            embedding_dim=config.storage.embedding_dim,
            pool_min_size=config.storage.pool_min_size,
            pool_max_size=config.storage.pool_max_size,
            table_name=config.storage.table_name,
        )

    def index_documents(
        self,
        documents: Union[List[str], List[IngestChunk]],
        show_progress: bool = True,
    ) -> int:
        """
        Index documents/chunks by creating embeddings and storing them.

        Args:
            documents: list of raw strings or IngestChunk objects
            show_progress: Whether to log progress
        Returns:
            Number of chunks actually inserted
        """
        logger.info(f"Indexing {len(documents)} documents...")
        inserted = self.vector_db.add_chunks(documents, show_progress=show_progress)
        logger.info(f"Indexed {inserted} new chunks (collection size: {self.vector_db.size()})")
        return inserted

    def ingest_file(
        self,
        path: str,
        *,
        source: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> int:
        """
        Parse a document (PDF/DOCX/MD/TXT), chunk it, and index it with provenance.

        Args:
            path: filesystem path to the document
            source: override the source label (defaults to the filename)
            uri: override the stored URI (defaults to the absolute path)
        Returns:
            Number of chunks inserted
        """
        segments, meta = loaders.load_document(path, uri=uri)
        chunks = chunking.chunk_segments(
            segments,
            source=source or meta["source"],
            title=meta["title"],
            uri=meta["uri"],
            embedding_model=self.config.model.embedding_model,
            max_tokens=self.config.chunking.max_tokens,
            overlap=self.config.chunking.overlap,
        )
        if not chunks:
            logger.warning(f"No text extracted from '{path}'")
            return 0
        return self.index_documents(chunks, show_progress=True)

    def retrieve(
        self,
        query: str,
        top_n: Optional[int] = None,
        use_hybrid_search: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        retrieve_k: Optional[int] = None,
        fusion_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        rrf_weight: Optional[float] = None,
        reranker_model: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Hybrid pipeline (default):
            1. pgvector HNSW semantic search
            2. PostgreSQL tsvector full-text search
            3. Weighted RRF fusion (semantic=0.7, tsvector=0.3)
            4. Optional cross-encoder reranking

        Semantic-only (use_hybrid_search=False):
            1. pgvector HNSW semantic search
            2. Optional cross-encoder reranking

        All parameters override their corresponding config values when provided.
        """
        top_n = top_n if top_n is not None else self.config.retrieval.top_n
        hybrid = use_hybrid_search if use_hybrid_search is not None else self.config.retrieval.use_hybrid_search
        rerank = use_reranking if use_reranking is not None else self.config.retrieval.use_reranking
        retrieve_k = retrieve_k if retrieve_k is not None else self.config.retrieval.retrieve_k
        fusion_k = fusion_k if fusion_k is not None else self.config.retrieval.fusion_k
        rrf_k = rrf_k if rrf_k is not None else self.config.retrieval.rrf_k
        rrf_weight = rrf_weight if rrf_weight is not None else self.config.retrieval.rrf_weight
        reranker_model = reranker_model if reranker_model is not None else self.config.model.reranker_model

        return retrieve(
            query=query,
            vector_db=self.vector_db,
            language_model=self.config.model.language_model,
            top_n=top_n,
            retrieve_k=retrieve_k,
            fusion_k=fusion_k,
            use_hybrid_search=hybrid,
            use_reranking=rerank,
            rrf_k=rrf_k,
            rrf_weight=rrf_weight,
            reranker_model=reranker_model,
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        stream: bool = True
    ) -> Iterator[str]:
        """
        Generate a response using retrieved context.

        Args:
            query: User query
            retrieved_chunks: List of RetrievedChunk from retrieval
            stream: Whether to stream the response

        Yields:
            Response text chunks (if streaming)
        """
        # Pass full RetrievedChunks so the prompt can render citation labels
        # ([n] (Title, p.X)); generation maps citation index n → this list's order.
        return generate_response(
            query,
            retrieved_chunks,
            self.config.model.language_model,
            stream=stream,
            timeout=self.config.model.request_timeout,
        )

    def query(self, query: str, stream: bool = True) -> Tuple[List[RetrievedChunk], Iterator[str]]:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            query: User query
            stream: Whether to stream the response
            
        Returns:
            Tuple of (retrieved_chunks, response_iterator)
        """
        retrieved = self.retrieve(query)
        response = self.generate(query, retrieved, stream=stream)
        return retrieved, response

    def close(self) -> None:
        """Release the underlying connection pool."""
        self.vector_db.close()
