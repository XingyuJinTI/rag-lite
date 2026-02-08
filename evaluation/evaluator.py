"""
RAG Evaluator for measuring retrieval performance.

This module provides the RAGEvaluator class for running evaluations
and comparing different retrieval configurations.
"""

import logging
import time
from typing import List, Dict, Optional

from .datasets import BaseDataset
from .metrics import (
    RetrievalMetrics,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    normalize_text,
    text_overlap_score,
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluator for RAG retrieval performance.
    
    Runs evaluation queries against a RAG pipeline and computes metrics.
    
    Example:
        >>> from rag_lite import RAGPipeline, Config
        >>> from evaluation import load_dataset, RAGEvaluator
        >>>
        >>> config = Config.default()
        >>> pipeline = RAGPipeline(config)
        >>> 
        >>> dataset = load_dataset("cuad", max_examples=100)
        >>> pipeline.index_documents(dataset.get_chunks())
        >>>
        >>> evaluator = RAGEvaluator(pipeline, dataset)
        >>> metrics = evaluator.evaluate(top_k=10)
        >>> evaluator.print_report()
    """
    
    def __init__(
        self,
        pipeline,  # RAGPipeline instance
        dataset: BaseDataset,
        match_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            pipeline: RAGPipeline instance (must have retrieve() method)
            dataset: Dataset with evaluation examples
            match_threshold: Minimum text overlap for a match
            verbose: Print progress during evaluation
        """
        self.pipeline = pipeline
        self.dataset = dataset
        self.match_threshold = match_threshold
        self.verbose = verbose
        self.metrics: Optional[RetrievalMetrics] = None
        
        # Custom match function with configurable threshold
        self.match_fn = lambda r, g: (
            normalize_text(g) in normalize_text(r) or 
            normalize_text(r) in normalize_text(g) or
            text_overlap_score(r, g) > self.match_threshold
        )
    
    def evaluate(
        self,
        top_k: int = 10,
        max_examples: Optional[int] = None,
        use_hybrid_search: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        # Advanced retrieval parameters (passed to pipeline.retrieve())
        retrieve_k: Optional[int] = None,
        fusion_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        rrf_weight: Optional[float] = None,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
        rerank_weight: Optional[float] = None,
    ) -> RetrievalMetrics:
        """
        Run evaluation on the dataset.
        
        Args:
            top_k: Number of results to retrieve per query
            max_examples: Maximum examples to evaluate (None = all)
            use_hybrid_search: Override pipeline's hybrid search setting
            use_reranking: Override pipeline's reranking setting
            retrieve_k: Candidates from each search method (overrides config)
            fusion_k: Candidates after RRF fusion (overrides config)
            rrf_k: RRF constant, higher = more uniform ranking (default 60)
            rrf_weight: Semantic weight in RRF; BM25 gets 1 - rrf_weight (default 0.7)
            bm25_k1: BM25 term frequency saturation (default 1.5)
            bm25_b: BM25 document length normalization (default 0.75)
            rerank_weight: Weight for rerank score vs original (default 0.6)
            
        Returns:
            RetrievalMetrics with all computed metrics
        """
        eval_examples = self.dataset.get_eval_examples()
        
        if not eval_examples:
            logger.warning("Dataset has no evaluation examples!")
            return RetrievalMetrics()
        
        if max_examples:
            eval_examples = eval_examples[:max_examples]
        
        # Build retrieval kwargs (only non-None values)
        retrieve_kwargs = {}
        if use_hybrid_search is not None:
            retrieve_kwargs["use_hybrid_search"] = use_hybrid_search
        if use_reranking is not None:
            retrieve_kwargs["use_reranking"] = use_reranking
        if retrieve_k is not None:
            retrieve_kwargs["retrieve_k"] = retrieve_k
        if fusion_k is not None:
            retrieve_kwargs["fusion_k"] = fusion_k
        if rrf_k is not None:
            retrieve_kwargs["rrf_k"] = rrf_k
        if rrf_weight is not None:
            retrieve_kwargs["rrf_weight"] = rrf_weight
        if bm25_k1 is not None:
            retrieve_kwargs["bm25_k1"] = bm25_k1
        if bm25_b is not None:
            retrieve_kwargs["bm25_b"] = bm25_b
        if rerank_weight is not None:
            retrieve_kwargs["rerank_weight"] = rerank_weight
        
        # Accumulators
        recall_1 = []
        recall_3 = []
        recall_5 = []
        recall_10 = []
        mrr_scores = []
        ndcg_scores = []
        retrieval_times = []
        per_query_results = []
        
        if self.verbose:
            print(f"\nEvaluating {len(eval_examples)} examples...")
        
        for i, example in enumerate(eval_examples):
            # Time the retrieval
            start_time = time.time()
            
            # Retrieve with all configured parameters
            results = self.pipeline.retrieve(
                query=example.question,
                top_n=top_k,
                **retrieve_kwargs,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            retrieval_times.append(elapsed_ms)
            
            # Extract chunk texts
            retrieved_chunks = [chunk for chunk, _ in results]
            ground_truth = example.context
            
            # Calculate metrics for this query
            r1 = calculate_recall_at_k(retrieved_chunks, ground_truth, 1, self.match_fn)
            r3 = calculate_recall_at_k(retrieved_chunks, ground_truth, 3, self.match_fn)
            r5 = calculate_recall_at_k(retrieved_chunks, ground_truth, 5, self.match_fn)
            r10 = calculate_recall_at_k(retrieved_chunks, ground_truth, min(10, top_k), self.match_fn)
            mrr = calculate_mrr(retrieved_chunks, ground_truth, self.match_fn)
            ndcg = calculate_ndcg_at_k(retrieved_chunks, ground_truth, min(10, top_k), self.match_fn)
            
            recall_1.append(r1)
            recall_3.append(r3)
            recall_5.append(r5)
            recall_10.append(r10)
            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)
            
            # Store per-query details
            per_query_results.append({
                "example_id": example.example_id,
                "question": example.question[:100],
                "recall@1": r1,
                "recall@3": r3,
                "mrr": mrr,
                "retrieval_time_ms": elapsed_ms,
                "hit": r10 > 0,
            })
            
            # Progress
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(eval_examples)} examples...")
        
        # Compute aggregate metrics
        n = len(eval_examples)
        self.metrics = RetrievalMetrics(
            recall_at_1=sum(recall_1) / n,
            recall_at_3=sum(recall_3) / n,
            recall_at_5=sum(recall_5) / n,
            recall_at_10=sum(recall_10) / n,
            mrr=sum(mrr_scores) / n,
            hit_rate=sum(1 for r in recall_10 if r > 0) / n,
            ndcg_at_10=sum(ndcg_scores) / n,
            total_queries=n,
            queries_with_hits=sum(1 for r in recall_10 if r > 0),
            avg_retrieval_time_ms=sum(retrieval_times) / n,
            per_query_results=per_query_results,
        )
        
        return self.metrics
    
    def print_report(self, detailed: bool = False):
        """
        Print evaluation report.
        
        Args:
            detailed: If True, print per-query results
        """
        if not self.metrics:
            print("No evaluation results. Run evaluate() first.")
            return
        
        print("\n" + "=" * 60)
        print("RAG RETRIEVAL EVALUATION REPORT")
        print("=" * 60)
        print(f"Dataset: {self.dataset.name}")
        print(f"Total Queries: {self.metrics.total_queries}")
        print(f"Queries with Hits: {self.metrics.queries_with_hits}")
        print("-" * 60)
        print("\nRETRIEVAL METRICS:")
        print(f"  Recall@1:   {self.metrics.recall_at_1:.4f}")
        print(f"  Recall@3:   {self.metrics.recall_at_3:.4f}")
        print(f"  Recall@5:   {self.metrics.recall_at_5:.4f}")
        print(f"  Recall@10:  {self.metrics.recall_at_10:.4f}")
        print(f"  MRR:        {self.metrics.mrr:.4f}")
        print(f"  Hit Rate:   {self.metrics.hit_rate:.4f}")
        print(f"  NDCG@10:    {self.metrics.ndcg_at_10:.4f}")
        print("-" * 60)
        print("\nPERFORMANCE:")
        print(f"  Avg Retrieval Time: {self.metrics.avg_retrieval_time_ms:.1f}ms")
        print("=" * 60)
        
        if detailed and self.metrics.per_query_results:
            print("\nPER-QUERY RESULTS:")
            print("-" * 60)
            
            # Sort by MRR to show worst performers first
            sorted_results = sorted(
                self.metrics.per_query_results,
                key=lambda x: x['mrr']
            )
            
            for result in sorted_results[:20]:  # Show top 20 worst
                hit_marker = "✓" if result['hit'] else "✗"
                print(f"  {hit_marker} MRR={result['mrr']:.2f} | {result['question'][:60]}...")
    
    def compare_configurations(
        self,
        configurations: List[Dict],
        top_k: int = 10,
        max_examples: Optional[int] = None
    ) -> Dict[str, RetrievalMetrics]:
        """
        Compare multiple retrieval configurations.
        
        Args:
            configurations: List of config dicts. Each dict should have a "name" key
                and any retrieval parameters to override:
                
                - use_hybrid_search: bool - Enable/disable hybrid search
                - use_reranking: bool - Enable/disable LLM reranking
                - retrieve_k: int - Candidates from each search method
                - fusion_k: int - Candidates after RRF fusion
                - rrf_k: int - RRF constant (higher = more uniform ranking)
                - rrf_weight: float - Semantic weight in RRF (BM25 = 1 - rrf_weight)
                - bm25_k1: float - BM25 term frequency saturation
                - bm25_b: float - BM25 document length normalization
                - rerank_weight: float - Weight for rerank score vs original
                
            top_k: Number of results to retrieve
            max_examples: Maximum examples to evaluate
            
        Returns:
            Dictionary mapping config names to their metrics
            
        Example:
            >>> # Compare different RRF weights
            >>> configs = [
            ...     {"name": "rrf_k=30", "rrf_k": 30},
            ...     {"name": "rrf_k=60", "rrf_k": 60},
            ...     {"name": "rrf_k=100", "rrf_k": 100},
            ... ]
            >>> results = evaluator.compare_configurations(configs)
            
            >>> # Compare semantic vs hybrid with different BM25 tuning
            >>> configs = [
            ...     {"name": "semantic_only", "use_hybrid_search": False},
            ...     {"name": "hybrid_default", "use_hybrid_search": True},
            ...     {"name": "hybrid_bm25_tuned", "use_hybrid_search": True, "bm25_k1": 2.0, "bm25_b": 0.5},
            ... ]
            >>> results = evaluator.compare_configurations(configs)
        """
        results = {}
        
        for config in configurations:
            name = config.pop("name", f"config_{len(results)}")
            print(f"\n--- Evaluating: {name} ---")
            
            metrics = self.evaluate(
                top_k=top_k,
                max_examples=max_examples,
                **config
            )
            results[name] = metrics
        
        # Print comparison
        print("\n" + "=" * 80)
        print("CONFIGURATION COMPARISON")
        print("=" * 80)
        print(f"{'Config':<25} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'MRR':>8} {'Time':>10}")
        print("-" * 80)
        
        for name, metrics in results.items():
            print(
                f"{name:<25} "
                f"{metrics.recall_at_1:>8.4f} "
                f"{metrics.recall_at_3:>8.4f} "
                f"{metrics.recall_at_5:>8.4f} "
                f"{metrics.mrr:>8.4f} "
                f"{metrics.avg_retrieval_time_ms:>8.1f}ms"
            )
        
        print("=" * 80)
        
        return results


def quick_evaluate(
    pipeline,
    dataset: BaseDataset,
    top_k: int = 10,
    max_examples: int = 50
) -> RetrievalMetrics:
    """
    Quick evaluation helper function.
    
    Args:
        pipeline: RAGPipeline instance
        dataset: Dataset with evaluation examples
        top_k: Number of results to retrieve
        max_examples: Maximum examples to evaluate
        
    Returns:
        RetrievalMetrics
    """
    evaluator = RAGEvaluator(pipeline, dataset)
    return evaluator.evaluate(top_k=top_k, max_examples=max_examples)
