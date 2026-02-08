#!/usr/bin/env python3
"""
RAG Evaluation Benchmark

This script demonstrates how to:
1. Swap between different datasets (cat-facts, CUAD, SQuAD, etc.)
2. Run retrieval evaluation with metrics
3. Compare different retrieval configurations

Usage:
    # Run from project root
    python -m evaluation.run_benchmark
    
    # Or with arguments
    python -m evaluation.run_benchmark --dataset cuad --max-docs 50 --max-eval 20
    
    # Compare retrieval strategies
    python -m evaluation.run_benchmark --compare
    
    # Quick test with cat facts
    python -m evaluation.run_benchmark --dataset cat_facts --file cat-facts.txt

Requirements:
    pip install rag-lite[eval]
    # or
    pip install datasets
"""

import argparse
import logging
import sys
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAG-Lite core
from rag_lite import RAGPipeline, Config

# Import evaluation tools
from evaluation import (
    load_dataset,
    list_available_datasets,
    RAGEvaluator,
)


def run_evaluation(
    dataset_name: str = "cuad",
    max_documents: Optional[int] = None,
    max_eval_examples: int = 50,
    top_k: int = 10,
    file_path: Optional[str] = None,
    compare_configs: bool = False,
    use_hybrid: bool = True,
    use_reranking: bool = False,
):
    """
    Run RAG evaluation on a dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        max_documents: Maximum documents to index
        max_eval_examples: Maximum evaluation examples
        top_k: Number of results to retrieve per query
        file_path: File path for cat_facts or custom datasets
        compare_configs: Whether to compare multiple configurations
        use_hybrid: Enable hybrid search (BM25 + semantic)
        use_reranking: Enable LLM reranking
    """
    print("\n" + "=" * 70)
    print("RAG-LITE EVALUATION BENCHMARK")
    print("=" * 70)
    
    # Show available datasets
    print(f"\nAvailable datasets: {list_available_datasets()}")
    print(f"Selected dataset: {dataset_name}")
    
    # Load dataset
    print(f"\n--- Loading Dataset: {dataset_name} ---")
    
    dataset_kwargs = {
        "max_examples": max_documents,
        "max_eval_examples": max_eval_examples,
    }
    
    if dataset_name == "cat_facts" and file_path:
        dataset_kwargs["file_path"] = file_path
    
    try:
        dataset = load_dataset(dataset_name, **dataset_kwargs)
    except ImportError as e:
        print(f"\nError: {e}")
        print("Install required dependencies: pip install rag-lite[eval]")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(dataset)} documents")
    print(f"Evaluation examples: {len(dataset.get_eval_examples())}")
    
    if not dataset.has_ground_truth():
        print("\nWarning: This dataset has no ground truth Q&A pairs.")
        print("Evaluation metrics require datasets with labeled questions/answers.")
        print("Try: --dataset cuad or --dataset squad")
        return
    
    # Show sample document and Q&A
    print("\n--- Sample Data ---")
    docs = dataset.get_documents()
    if docs:
        print(f"Sample document ({len(docs[0].text)} chars):")
        print(f"  {docs[0].text[:200]}...")
    
    eval_examples = dataset.get_eval_examples()
    if eval_examples:
        print(f"\nSample Q&A:")
        print(f"  Q: {eval_examples[0].question}")
        print(f"  A: {eval_examples[0].answer[:100]}...")
    
    # Initialize pipeline
    print("\n--- Initializing RAG Pipeline ---")
    config = Config.default()
    config.retrieval.use_hybrid_search = use_hybrid
    config.retrieval.use_reranking = use_reranking
    
    # Update collection name to be unique per dataset
    config.storage.collection_name = f"eval_{dataset_name}"
    
    pipeline = RAGPipeline(config)
    
    # Index documents
    print("\n--- Indexing Documents ---")
    chunks = dataset.get_chunks()
    pipeline.index_documents(chunks, show_progress=True)
    
    # Create evaluator
    evaluator = RAGEvaluator(pipeline, dataset, verbose=True)
    
    if compare_configs:
        # Compare multiple configurations
        print("\n--- Comparing Retrieval Configurations ---")
        configurations = [
            {"name": "semantic_only", "use_hybrid_search": False, "use_reranking": False},
            {"name": "hybrid_bm25+semantic", "use_hybrid_search": True, "use_reranking": False},
        ]
        
        # Only add reranking if it's fast enough (small dataset)
        if len(eval_examples) <= 30:
            configurations.append(
                {"name": "hybrid+reranking", "use_hybrid_search": True, "use_reranking": True}
            )
        
        evaluator.compare_configurations(
            configurations=configurations,
            top_k=top_k,
            max_examples=max_eval_examples,
        )
    else:
        # Single evaluation
        print(f"\n--- Running Evaluation (top_k={top_k}) ---")
        print(f"Hybrid search: {use_hybrid}")
        print(f"Reranking: {use_reranking}")
        
        metrics = evaluator.evaluate(
            top_k=top_k,
            max_examples=max_eval_examples,
        )
        
        evaluator.print_report(detailed=True)
    
    # Interactive query mode
    print("\n--- Interactive Query Mode ---")
    print("Type a question to test retrieval, or 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if not query:
                continue
            
            results = pipeline.retrieve(query, top_n=3)
            print("\nRetrieved chunks:")
            for i, (chunk, score) in enumerate(results, 1):
                print(f"  {i}. (score: {score:.4f}) {chunk[:150]}...")
            print()
            
        except KeyboardInterrupt:
            print("\n")
            break
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Lite Evaluation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate CUAD (legal contracts)
  python -m evaluation.run_benchmark --dataset cuad --max-docs 50 --max-eval 30
  
  # Evaluate SQuAD (reading comprehension)
  python -m evaluation.run_benchmark --dataset squad --max-docs 200 --max-eval 50
  
  # Compare retrieval configurations
  python -m evaluation.run_benchmark --dataset cuad --compare
  
  # Quick test with cat facts (no evaluation, just indexing)
  python -m evaluation.run_benchmark --dataset cat_facts --file cat-facts.txt
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="cuad",
        choices=["cat_facts", "cuad", "squad", "hotpot_qa"],
        help="Dataset to use (default: cuad)"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="File path for cat_facts or custom dataset"
    )
    
    parser.add_argument(
        "--max-docs", "-m",
        type=int,
        default=None,
        help="Maximum documents to index (default: all)"
    )
    
    parser.add_argument(
        "--max-eval", "-e",
        type=int,
        default=50,
        help="Maximum evaluation examples (default: 50)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare multiple retrieval configurations"
    )
    
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search (use semantic only)"
    )
    
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable LLM reranking"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        dataset_name=args.dataset,
        max_documents=args.max_docs,
        max_eval_examples=args.max_eval,
        top_k=args.top_k,
        file_path=args.file,
        compare_configs=args.compare,
        use_hybrid=not args.no_hybrid,
        use_reranking=args.rerank,
    )


if __name__ == "__main__":
    main()
