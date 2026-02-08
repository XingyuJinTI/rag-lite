"""
Retrieval metrics for RAG evaluation.

This module provides standard IR metrics:
- Recall@K: Fraction of relevant documents retrieved in top K
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant result
- NDCG: Normalized Discounted Cumulative Gain
- Hit Rate: Fraction of queries with at least one relevant result
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    
    # Core metrics
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    
    mrr: float = 0.0  # Mean Reciprocal Rank
    hit_rate: float = 0.0  # Fraction with at least one hit
    
    # Optional metrics
    ndcg_at_10: float = 0.0  # Normalized DCG
    
    # Statistics
    total_queries: int = 0
    queries_with_hits: int = 0
    avg_retrieval_time_ms: float = 0.0
    
    # Per-query details (optional)
    per_query_results: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "ndcg@10": self.ndcg_at_10,
            "total_queries": self.total_queries,
            "queries_with_hits": self.queries_with_hits,
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
        }
    
    def __str__(self) -> str:
        return (
            f"RetrievalMetrics(\n"
            f"  Recall@1:  {self.recall_at_1:.4f}\n"
            f"  Recall@3:  {self.recall_at_3:.4f}\n"
            f"  Recall@5:  {self.recall_at_5:.4f}\n"
            f"  Recall@10: {self.recall_at_10:.4f}\n"
            f"  MRR:       {self.mrr:.4f}\n"
            f"  Hit Rate:  {self.hit_rate:.4f}\n"
            f"  NDCG@10:   {self.ndcg_at_10:.4f}\n"
            f"  Queries:   {self.total_queries} ({self.queries_with_hits} with hits)\n"
            f"  Avg Time:  {self.avg_retrieval_time_ms:.1f}ms\n"
            f")"
        )


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return ' '.join(text.lower().split())


def text_overlap_score(retrieved: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Calculate text overlap between retrieved and ground truth.
    
    Uses token overlap (Jaccard-like) for fuzzy matching.
    
    Args:
        retrieved: Retrieved text
        ground_truth: Ground truth text
        threshold: Minimum overlap ratio to consider a match
        
    Returns:
        Overlap score (0-1)
    """
    retrieved_tokens = set(normalize_text(retrieved).split())
    truth_tokens = set(normalize_text(ground_truth).split())
    
    if not truth_tokens:
        return 0.0
    
    intersection = len(retrieved_tokens & truth_tokens)
    union = len(retrieved_tokens | truth_tokens)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def answer_in_context(answer: str, context: str) -> bool:
    """Check if the answer text appears in the context."""
    return normalize_text(answer) in normalize_text(context)


def calculate_recall_at_k(
    retrieved_chunks: List[str],
    ground_truth: str,
    k: int,
    match_fn: Optional[Callable[[str, str], bool]] = None
) -> float:
    """
    Calculate Recall@K for a single query.
    
    Args:
        retrieved_chunks: List of retrieved chunk texts
        ground_truth: Ground truth context that should be retrieved
        k: Number of top results to consider
        match_fn: Optional custom matching function (default: substring match)
        
    Returns:
        1.0 if ground truth found in top K, 0.0 otherwise
    """
    if match_fn is None:
        # Default: check if ground truth is contained in or contains retrieved chunk
        match_fn = lambda r, g: (
            normalize_text(g) in normalize_text(r) or 
            normalize_text(r) in normalize_text(g) or
            text_overlap_score(r, g) > 0.5
        )
    
    for chunk in retrieved_chunks[:k]:
        if match_fn(chunk, ground_truth):
            return 1.0
    
    return 0.0


def calculate_mrr(
    retrieved_chunks: List[str],
    ground_truth: str,
    match_fn: Optional[Callable[[str, str], bool]] = None
) -> float:
    """
    Calculate Mean Reciprocal Rank for a single query.
    
    Args:
        retrieved_chunks: List of retrieved chunk texts
        ground_truth: Ground truth context
        match_fn: Optional custom matching function
        
    Returns:
        1/rank of first relevant result, or 0 if none found
    """
    if match_fn is None:
        match_fn = lambda r, g: (
            normalize_text(g) in normalize_text(r) or 
            normalize_text(r) in normalize_text(g) or
            text_overlap_score(r, g) > 0.5
        )
    
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if match_fn(chunk, ground_truth):
            return 1.0 / rank
    
    return 0.0


def calculate_ndcg_at_k(
    retrieved_chunks: List[str],
    ground_truth: str,
    k: int,
    match_fn: Optional[Callable[[str, str], bool]] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    For binary relevance (relevant or not), this simplifies to:
    DCG = rel_i / log2(i + 1) for each position i
    IDCG = 1 / log2(2) = 1.0 (best case: relevant at position 1)
    
    Args:
        retrieved_chunks: List of retrieved chunks
        ground_truth: Ground truth context
        k: Number of results to consider
        match_fn: Optional custom matching function
        
    Returns:
        NDCG score (0-1)
    """
    if match_fn is None:
        match_fn = lambda r, g: (
            normalize_text(g) in normalize_text(r) or 
            normalize_text(r) in normalize_text(g) or
            text_overlap_score(r, g) > 0.5
        )
    
    dcg = 0.0
    for i, chunk in enumerate(retrieved_chunks[:k]):
        if match_fn(chunk, ground_truth):
            # Binary relevance: 1 if match, 0 otherwise
            dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed
    
    # Ideal DCG: relevant document at position 1
    idcg = 1.0 / math.log2(2)  # = 1.0
    
    return dcg / idcg if idcg > 0 else 0.0
