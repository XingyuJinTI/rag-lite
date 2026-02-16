"""
Retrieval module for semantic search and reranking.

This module implements retrieval strategies including:
- Semantic search using ChromaDB's HNSW index
- BM25 keyword search
- Hybrid search with RRF fusion
- Cross-encoder reranking (using sentence-transformers)
- Query expansion
"""

import logging
import math
import re
from collections import Counter
from typing import List, Tuple, Optional, Dict

import numpy as np
import ollama

from .utils import get_device

# Lazy load cross-encoder to avoid import overhead when not using reranking
_cross_encoder_model = None
_cross_encoder_model_name = None

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r'\b\w+\b', text.lower())


class BM25Scorer:
    """BM25 scorer with pre-computed corpus statistics."""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize scorer with corpus statistics.
        
        Args:
            corpus: List of document texts
            k1: Term frequency saturation (typically 1.2-2.0)
            b: Document length normalization (0=none, 1=full)
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        
        # Pre-compute corpus statistics
        self.N = len(corpus)
        self.df: Counter = Counter()  # Document frequency
        total_len = 0
        
        for doc in corpus:
            doc_tokens = _tokenize(doc)
            total_len += len(doc_tokens)
            for term in set(doc_tokens):
                self.df[term] += 1
        
        self.avgdl = total_len / self.N if self.N > 0 else 1.0
        
        # Pre-compute IDF values using BM25 formula
        self.idf = {
            term: math.log((self.N - df_count + 0.5) / (df_count + 0.5) + 1)
            for term, df_count in self.df.items()
        }
    
    def score(self, query: str, chunk: str) -> float:
        """
        Calculate BM25 score for a query-chunk pair.
        
        Args:
            query: Query text
            chunk: Document chunk text
            
        Returns:
            BM25 score (normalized to 0-1 range)
        """
        query_tokens = _tokenize(query)
        chunk_tokens = _tokenize(chunk)
        
        if not query_tokens or not chunk_tokens:
            return 0.0
        
        doc_len = len(chunk_tokens)
        chunk_tf = Counter(chunk_tokens)
        
        score = 0.0
        for term in set(query_tokens):
            if term in chunk_tf:
                f = chunk_tf[term]
                term_idf = self.idf.get(term, math.log(2))
                
                # BM25 formula
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += term_idf * (numerator / denominator)
        
        # Normalize to 0-1 range
        max_term_score = (self.k1 + 1)
        max_idf = max(self.idf.values()) if self.idf else math.log(2)
        max_possible = len(set(query_tokens)) * max_term_score * max_idf
        
        if max_possible > 0:
            return min(1.0, score / max_possible)
        return 0.0


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    rrf_weight: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Combine ranked lists using Weighted Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: List of (chunk, score) tuples per ranking method
        k: RRF constant (default 60)
        rrf_weight: Weight for first list (default 0.7); second gets 1 - rrf_weight
        
    Returns:
        List of (chunk, rrf_score) tuples, sorted descending
    """
    rrf_scores: Dict[str, float] = {}
    
    # Calculate weights: first list gets rrf_weight, second gets (1 - rrf_weight)
    # For >2 lists, remaining lists split the second weight equally
    if len(ranked_lists) == 2:
        weights = [rrf_weight, 1 - rrf_weight]
    else:
        weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
    
    for list_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[list_idx] if list_idx < len(weights) else weights[-1]
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            if chunk not in rrf_scores:
                rrf_scores[chunk] = 0.0
            rrf_scores[chunk] += weight * (1.0 / (k + rank))
    
    results = [(chunk, score) for chunk, score in rrf_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def retrieve_bm25(
    query: str,
    corpus: List[str],
    top_k: int = 50,
    k1: float = 1.5,
    b: float = 0.75
) -> List[Tuple[str, float]]:
    """
    Retrieve documents using BM25 keyword search.
    
    Args:
        query: Search query
        corpus: List of document texts
        top_k: Number of top results to return
        k1: BM25 k1 parameter
        b: BM25 b parameter
        
    Returns:
        List of (chunk, score) tuples sorted by score descending
    """
    scorer = BM25Scorer(corpus=corpus, k1=k1, b=b)
    scored = [(chunk, scorer.score(query, chunk)) for chunk in corpus]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def expand_query(
    query: str, 
    language_model: str, 
    num_alternatives: int = 2
) -> List[str]:
    """
    Expand query with alternative phrasings using LLM.
    
    Args:
        query: Original query
        language_model: Name of the language model
        num_alternatives: Number of alternatives to generate
        
    Returns:
        List of queries including original and alternatives
    """
    expansion_prompt = f"""Given the following question, generate 2-3 alternative phrasings or related questions that would help find relevant information. 
Return only the alternative questions, one per line, without numbering or bullets.

Original question: {query}

Alternative questions:"""
    
    try:
        response = ollama.chat(
            model=language_model,
            messages=[{'role': 'user', 'content': expansion_prompt}],
        )
        alternatives = [
            line.strip() 
            for line in response['message']['content'].strip().split('\n') 
            if line.strip()
        ]
        return [query] + alternatives[:num_alternatives]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}, using original query")
        return [query]


def _get_cross_encoder(model_name: str):
    """
    Get or create a cross-encoder model (lazy loading with caching).
    
    Args:
        model_name: Name of the cross-encoder model (e.g., 'BAAI/bge-reranker-base')
        
    Returns:
        CrossEncoder model instance
    """
    global _cross_encoder_model, _cross_encoder_model_name
    
    if _cross_encoder_model is None or _cross_encoder_model_name != model_name:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install it with: pip install sentence-transformers"
            )
        
        device = get_device()
        logger.info(f"Loading cross-encoder model: {model_name} on {device}")
        _cross_encoder_model = CrossEncoder(model_name, device=device)
        _cross_encoder_model_name = model_name
        logger.info(f"Cross-encoder model loaded successfully on {device}")
    
    return _cross_encoder_model


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Tuple[str, float]],
    reranker_model: str,
) -> List[Tuple[str, float]]:
    """
    Rerank candidates using a cross-encoder model.
    
    Args:
        query: User query
        candidates: List of (chunk, score) tuples from first-stage retrieval
        reranker_model: Name of the cross-encoder model
        
    Returns:
        List of (chunk, score) tuples, sorted by relevance
    """
    if not candidates:
        return []
    
    # Get cross-encoder model
    cross_encoder = _get_cross_encoder(reranker_model)
    
    # Prepare query-document pairs for scoring
    candidate_texts = [chunk.strip() for chunk, _ in candidates]
    pairs = [(query, text) for text in candidate_texts]
    
    # Get relevance scores
    logger.debug(f"Reranking {len(candidates)} candidates...")
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    # Normalize scores to 0-1 range
    min_score, max_score = float(np.min(scores)), float(np.max(scores))
    if max_score > min_score:
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
    else:
        normalized_scores = [0.5] * len(scores)
    
    # Build result with cross-encoder scores only
    reranked = [(chunk, normalized_scores[i]) for i, (chunk, _) in enumerate(candidates)]
    
    # Sort by score descending
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def retrieve(
    query: str,
    vector_db,  # VectorDB instance
    language_model: str,
    top_n: int = 3,
    retrieve_k: int = 50,
    fusion_k: int = 20,
    use_hybrid_search: bool = True,
    use_reranking: bool = False,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    rrf_k: int = 60,
    rrf_weight: float = 0.7,
    reranker_model: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Retrieve relevant chunks using semantic search with optional hybrid (BM25 + RRF).
    
    Args:
        query: User query
        vector_db: VectorDB instance
        language_model: Language model (kept for API compatibility)
        top_n: Final results to return
        retrieve_k: Candidates from each search method
        fusion_k: Candidates after fusion (rerank pool)
        use_hybrid_search: Use semantic + BM25 with RRF fusion (False = semantic only)
        use_reranking: Enable cross-encoder reranking
        bm25_k1: BM25 k1 parameter
        bm25_b: BM25 b parameter
        rrf_k: RRF constant
        rrf_weight: Weight for semantic in RRF (default 0.7); BM25 gets 1 - rrf_weight (0.3)
        reranker_model: Cross-encoder model for reranking (e.g., 'BAAI/bge-reranker-base')
        
    Returns:
        List of (chunk, score) tuples
    """
    if use_hybrid_search:
        # Step 1: Semantic search
        logger.debug(f"Semantic search for: {query[:50]}...")
        semantic_results = vector_db.search(query, n_results=retrieve_k)
        logger.debug(f"Semantic: {len(semantic_results)} results")
        
        # Step 2: FTS5 keyword search (fast, uses pre-built index)
        logger.debug("FTS5 keyword search...")
        keyword_results = vector_db.search_fts(query=query, n_results=retrieve_k)
        logger.debug(f"FTS5: {len(keyword_results)} results")
        
        # Step 3: RRF fusion (weighted: semantic gets rrf_weight, BM25 gets 1-rrf_weight)
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[semantic_results, keyword_results],
            k=rrf_k,
            rrf_weight=rrf_weight
        )
        logger.debug(f"RRF fusion: {len(fused_results)} unique results")
        
        top_candidates = fused_results[:fusion_k]
    else:
        # Semantic only - fetch only what we need (fusion_k for reranking, or top_n if no reranking)
        fetch_k = fusion_k if use_reranking else top_n
        logger.debug(f"Semantic search for: {query[:50]}...")
        semantic_results = vector_db.search(query, n_results=fetch_k)
        logger.debug(f"Semantic: {len(semantic_results)} results")
        top_candidates = semantic_results
    
    # Step 4: Optional cross-encoder reranking
    if use_reranking and len(top_candidates) > top_n and reranker_model:
        logger.debug(f"Reranking {len(top_candidates)} candidates with {reranker_model}...")
        reranked = rerank_with_cross_encoder(query, top_candidates, reranker_model)
        return reranked[:top_n]
    
    return top_candidates[:top_n]
