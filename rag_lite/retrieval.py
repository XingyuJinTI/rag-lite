"""
Retrieval module for semantic search and reranking.

Implements:
- Semantic search via pgvector HNSW index
- Full-text search via PostgreSQL tsvector
- Hybrid search with Weighted RRF fusion
- Cross-encoder reranking (sentence-transformers)
- Query expansion
"""

import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import ollama

from .types import RetrievedChunk
from .utils import get_device

_cross_encoder_model = None
_cross_encoder_model_name = None

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievedChunk]],
    k: int = 60,
    rrf_weight: float = 0.7
) -> List[RetrievedChunk]:
    """
    Combine ranked lists using Weighted Reciprocal Rank Fusion.

    Score = weight * 1/(k + rank). Using ranks rather than raw scores makes
    fusion robust to incompatible score scales across retrieval methods. Chunks
    are deduplicated by chunk_id (identity), so the same chunk surfacing in two
    lists accumulates both contributions.

    Args:
        ranked_lists: List of RetrievedChunk lists, one per retrieval method
        k: Smoothing constant (higher = more uniform distribution)
        rrf_weight: Weight for the first list; second gets 1 - rrf_weight.
                    For >2 lists, weight is distributed equally.
    Returns:
        List of RetrievedChunk (with fused rrf score), sorted descending
    """
    rrf_scores: Dict[str, float] = {}
    chunks_by_id: Dict[str, RetrievedChunk] = {}

    weights = (
        [rrf_weight, 1 - rrf_weight]
        if len(ranked_lists) == 2
        else [1.0 / len(ranked_lists)] * len(ranked_lists)
    )

    for list_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[list_idx] if list_idx < len(weights) else weights[-1]
        for rank, chunk in enumerate(ranked_list, start=1):
            key = chunk.chunk_id or chunk.content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + weight * (1.0 / (k + rank))
            chunks_by_id.setdefault(key, chunk)

    fused = []
    for key, score in rrf_scores.items():
        chunk = chunks_by_id[key]
        chunk.score = score
        fused.append(chunk)
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


def expand_query(
    query: str,
    language_model: str,
    num_alternatives: int = 2
) -> List[str]:
    """
    Expand query with alternative phrasings using the LLM.

    Args:
        query: Original query
        language_model: Ollama model name
        num_alternatives: Number of alternatives to generate
    Returns:
        List containing the original query followed by alternatives
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
    """Lazy-load and cache the cross-encoder model."""
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
        logger.info(f"Cross-encoder model loaded on {device}")

    return _cross_encoder_model


def rerank_with_cross_encoder(
    query: str,
    candidates: List[RetrievedChunk],
    reranker_model: str,
) -> List[RetrievedChunk]:
    """
    Rerank candidates using a cross-encoder model.

    Unlike bi-encoder cosine similarity, the cross-encoder jointly encodes
    (query, chunk) pairs — much more accurate for relevance judgement at the
    cost of ~O(n) inference calls. Provenance is preserved; only the score and
    ordering change.

    Args:
        query: User query
        candidates: First-stage RetrievedChunks
        reranker_model: HuggingFace cross-encoder model name
    Returns:
        Candidates re-sorted by cross-encoder score, normalized to [0, 1]
    """
    if not candidates:
        return []

    cross_encoder = _get_cross_encoder(reranker_model)
    pairs = [(query, c.content.strip()) for c in candidates]

    logger.debug(f"Reranking {len(candidates)} candidates...")
    scores = cross_encoder.predict(pairs, show_progress_bar=False)

    # Sigmoid of the raw logit gives an *absolute* relevance probability in (0, 1):
    # ~0.5 is the decision boundary, high logits → ~1. Unlike min-max normalization
    # (which forces the top candidate to 1.0 regardless of how relevant it is), this
    # is comparable across queries — so it can drive an abstention threshold.
    probs = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))

    for c, s in zip(candidates, probs):
        c.score = float(s)
    reranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    return reranked


def retrieve(
    query: str,
    vector_db,
    language_model: str,
    top_n: int = 3,
    retrieve_k: int = 50,
    fusion_k: int = 20,
    use_hybrid_search: bool = True,
    use_reranking: bool = False,
    rrf_k: int = 60,
    rrf_weight: float = 0.7,
    reranker_model: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve relevant chunks for a query.

    Hybrid pipeline (use_hybrid_search=True):
        1. pgvector HNSW semantic search  → top retrieve_k
        2. PostgreSQL tsvector FTS        → top retrieve_k
        3. Weighted RRF fusion            → top fusion_k
        4. Optional cross-encoder rerank  → top top_n

    Semantic-only (use_hybrid_search=False):
        1. pgvector HNSW semantic search
        2. Optional cross-encoder rerank  → top top_n

    Args:
        query: User query
        vector_db: VectorDB instance
        language_model: Language model name (reserved for query expansion)
        top_n: Final results to return
        retrieve_k: Candidates per retrieval method
        fusion_k: Pool size after RRF, fed into reranker
        use_hybrid_search: Enable tsvector + semantic fusion
        use_reranking: Enable cross-encoder reranking
        rrf_k: RRF smoothing constant
        rrf_weight: Semantic weight in RRF (0.7); tsvector gets 1 - rrf_weight
        reranker_model: Cross-encoder model name
    Returns:
        List of RetrievedChunk
    """
    if use_hybrid_search:
        semantic_results = vector_db.search(query, n_results=retrieve_k)
        logger.debug(f"Semantic: {len(semantic_results)} results")

        # Step 2: tsvector full-text search (GENERATED column, always in sync)
        keyword_results = vector_db.search_fts(query=query, n_results=retrieve_k)
        logger.debug(f"tsvector: {len(keyword_results)} results")

        # Step 3: Weighted RRF — semantic gets rrf_weight, tsvector gets 1-rrf_weight
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[semantic_results, keyword_results],
            k=rrf_k,
            rrf_weight=rrf_weight
        )
        logger.debug(f"RRF fusion: {len(fused_results)} unique results")

        top_candidates = fused_results[:fusion_k]
    else:
        # Fetch fusion_k candidates if reranking will run, otherwise just top_n
        fetch_k = fusion_k if use_reranking else top_n
        semantic_results = vector_db.search(query, n_results=fetch_k)
        logger.debug(f"Semantic: {len(semantic_results)} results")
        top_candidates = semantic_results

    if use_reranking and len(top_candidates) > top_n and reranker_model:
        reranked = rerank_with_cross_encoder(query, top_candidates, reranker_model)
        return reranked[:top_n]

    return top_candidates[:top_n]
