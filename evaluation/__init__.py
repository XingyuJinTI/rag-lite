"""
RAG-Lite Evaluation Package

This package provides evaluation tools for RAG systems, including:
- Dataset loaders (CUAD, SQuAD, custom datasets)
- Retrieval metrics (Recall@K, MRR, NDCG, Hit Rate)
- Evaluation runner with comparison capabilities

This is a separate package from the core rag_lite library to keep
dependencies optional (requires `datasets` package from HuggingFace).

Installation:
    pip install rag-lite[eval]
    # or
    pip install datasets

Quick Start:
    >>> from rag_lite import RAGPipeline, Config
    >>> from evaluation import load_dataset, RAGEvaluator
    >>>
    >>> # Load dataset
    >>> dataset = load_dataset("cuad", max_examples=100)
    >>>
    >>> # Setup pipeline
    >>> config = Config.default()
    >>> pipeline = RAGPipeline(config)
    >>> pipeline.index_documents(dataset.get_chunks())
    >>>
    >>> # Evaluate
    >>> evaluator = RAGEvaluator(pipeline, dataset)
    >>> metrics = evaluator.evaluate()
    >>> evaluator.print_report()
"""

from .datasets import (
    load_dataset,
    list_available_datasets,
    BaseDataset,
    Document,
    QAExample,
    CatFactsDataset,
    CUADDataset,
    HuggingFaceDataset,
    CustomDataset,
)

from .metrics import (
    RetrievalMetrics,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    normalize_text,
    text_overlap_score,
)

from .evaluator import (
    RAGEvaluator,
    quick_evaluate,
)

__all__ = [
    # Dataset loading
    "load_dataset",
    "list_available_datasets",
    "BaseDataset",
    "Document",
    "QAExample",
    "CatFactsDataset",
    "CUADDataset",
    "HuggingFaceDataset",
    "CustomDataset",
    # Metrics
    "RetrievalMetrics",
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "normalize_text",
    "text_overlap_score",
    # Evaluator
    "RAGEvaluator",
    "quick_evaluate",
]
