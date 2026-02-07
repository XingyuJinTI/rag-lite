"""
Main RAG pipeline that orchestrates the entire retrieval and generation process.

This module provides a high-level interface for the complete RAG workflow.
"""

import logging
from typing import List, Tuple, Iterator, Optional

from .config import Config
from .vector_db import VectorDB
from .retrieval import retrieve, retrieve_hybrid_rrf, expand_query
from .generation import generate_response

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates retrieval and generation.
    
    This class provides a clean interface for:
    1. Loading and indexing documents
    2. Retrieving relevant context
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
            persist_directory=config.storage.persist_directory,
            collection_name=config.storage.collection_name,
            ollama_base_url=config.ollama_base_url
        )

    def index_documents(self, documents: List[str], show_progress: bool = True) -> None:
        """
        Index documents by creating embeddings and storing them.
        
        Args:
            documents: List of document chunks to index
            show_progress: Whether to log progress
        """
        logger.info(f"Indexing {len(documents)} documents...")
        self.vector_db.add_chunks(documents, show_progress=show_progress)
        logger.info(f"Indexed {self.vector_db.size()} documents")

    def retrieve(
        self, 
        query: str,
        top_n: Optional[int] = None,
        use_query_expansion: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        use_rrf: Optional[bool] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Supports two hybrid search modes:
        - RRF (Reciprocal Rank Fusion): Uses ChromaDB native search + BM25, fused with RRF
        - Weighted: Legacy mode with weighted combination of semantic + keyword scores
        
        Args:
            query: User query
            top_n: Number of results to return (overrides config)
            use_query_expansion: Whether to use query expansion (overrides config)
            use_reranking: Whether to use reranking (overrides config)
            use_rrf: Whether to use RRF fusion (overrides config)
            
        Returns:
            List of (chunk, score) tuples
        """
        top_n = top_n or self.config.retrieval.top_n
        use_expansion = use_query_expansion if use_query_expansion is not None else self.config.retrieval.use_query_expansion
        use_rerank = use_reranking if use_reranking is not None else self.config.retrieval.use_reranking
        use_rrf_mode = use_rrf if use_rrf is not None else self.config.retrieval.use_rrf
        
        # Use RRF-based hybrid search (recommended)
        if use_rrf_mode and self.config.retrieval.use_hybrid_search:
            logger.debug("Using RRF-based hybrid search")
            return retrieve_hybrid_rrf(
                query=query,
                vector_db=self.vector_db,
                language_model=self.config.model.language_model,
                top_n=top_n,
                retrieve_k=self.config.retrieval.retrieve_k,
                fusion_k=self.config.retrieval.fusion_k,
                use_reranking=use_rerank,
                keyword_method=self.config.retrieval.keyword_method,
                bm25_k1=self.config.retrieval.bm25_k1,
                bm25_b=self.config.retrieval.bm25_b,
                rrf_k=self.config.retrieval.rrf_k,
                rerank_weight=self.config.retrieval.rerank_weight,
                original_score_weight=self.config.retrieval.original_score_weight,
            )
        
        # Legacy weighted hybrid search
        if use_expansion:
            # Expand query and retrieve from all variations
            expanded_queries = expand_query(
                query,
                self.config.model.language_model,
                num_alternatives=2
            )
            logger.info(f"Expanded queries: {expanded_queries}")
            
            all_results = []
            for q in expanded_queries:
                results = retrieve(
                    q,
                    self.vector_db.get_all(),
                    self.config.model.embedding_model,
                    self.config.model.language_model,
                    top_n=top_n * 2,  # Get more results per query
                    retrieve_k=self.config.retrieval.retrieve_k,
                    use_reranking=use_rerank,
                    use_hybrid=self.config.retrieval.use_hybrid_search,
                    semantic_weight=self.config.retrieval.semantic_weight,
                    keyword_weight=self.config.retrieval.keyword_weight,
                    rerank_weight=self.config.retrieval.rerank_weight,
                    original_score_weight=self.config.retrieval.original_score_weight,
                    keyword_method=self.config.retrieval.keyword_method,
                    bm25_k1=self.config.retrieval.bm25_k1,
                    bm25_b=self.config.retrieval.bm25_b,
                )
                all_results.extend(results)
            
            # Deduplicate and take top results
            seen = set()
            unique_results = []
            for chunk, score in all_results:
                if chunk not in seen:
                    seen.add(chunk)
                    unique_results.append((chunk, score))
            unique_results.sort(key=lambda x: x[1], reverse=True)
            return unique_results[:top_n]
        else:
            return retrieve(
                query,
                self.vector_db.get_all(),
                self.config.model.embedding_model,
                self.config.model.language_model,
                top_n=top_n,
                retrieve_k=self.config.retrieval.retrieve_k,
                use_reranking=use_rerank,
                use_hybrid=self.config.retrieval.use_hybrid_search,
                semantic_weight=self.config.retrieval.semantic_weight,
                keyword_weight=self.config.retrieval.keyword_weight,
                rerank_weight=self.config.retrieval.rerank_weight,
                original_score_weight=self.config.retrieval.original_score_weight,
                keyword_method=self.config.retrieval.keyword_method,
                bm25_k1=self.config.retrieval.bm25_k1,
                bm25_b=self.config.retrieval.bm25_b,
            )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Tuple[str, float]],
        stream: bool = True
    ) -> Iterator[str]:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: List of (chunk, score) tuples from retrieval
            stream: Whether to stream the response
            
        Yields:
            Response text chunks (if streaming)
        """
        context_chunks = [chunk for chunk, _ in retrieved_chunks]
        return generate_response(
            query,
            context_chunks,
            self.config.model.language_model,
            stream=stream
        )

    def query(self, query: str, stream: bool = True) -> Tuple[List[Tuple[str, float]], Iterator[str]]:
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
