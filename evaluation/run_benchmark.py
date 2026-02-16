#!/usr/bin/env python3
"""
RAG Evaluation Benchmark

This script demonstrates how to:
1. Swap between different datasets (cat-facts, RAGQArena)
2. Run retrieval evaluation with metrics
3. Compare different retrieval configurations

Supported Datasets:
- cat_facts: Simple text file (no ground truth Q&A, for testing indexing)
- ragqa_arena: RAGQArena Tech - 28k+ tech docs with Q&A pairs (recommended)

Usage:
    # Full benchmark (all 28k docs, all eval examples)
    python -m evaluation.run_benchmark
    
    # Faster run with limited eval examples
    python -m evaluation.run_benchmark --max-eval 300
    
    # Compare retrieval strategies
    python -m evaluation.run_benchmark --compare --max-eval 100
    
    # Compare with config file
    python -m evaluation.run_benchmark --config-file configs/rrf_sweep.json --max-eval 100
    
    # Quick test with cat facts
    python -m evaluation.run_benchmark --dataset cat_facts --file cat-facts.txt

Config File Format (JSON):
    {
      "configurations": [
        {"name": "semantic_only", "use_hybrid_search": false},
        {"name": "hybrid_default", "rrf_weight": 0.7},
        {"name": "bm25_heavy", "rrf_weight": 0.3}
      ]
    }

Requirements:
    pip install rag-lite[eval]
    # or
    pip install requests orjson
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAG-Lite core
from rag_lite import RAGPipeline, Config, ModelConfig

# Import evaluation tools
from evaluation import (
    load_dataset,
    list_available_datasets,
    RAGEvaluator,
)


def load_config_file(config_path: str) -> List[Dict]:
    """
    Load configurations from a JSON file.
    
    Expected format:
        {
          "configurations": [
            {"name": "config1", "use_hybrid_search": false},
            {"name": "config2", "rrf_weight": 0.5},
            ...
          ]
        }
    
    Or simply a list:
        [
          {"name": "config1", "use_hybrid_search": false},
          {"name": "config2", "rrf_weight": 0.5}
        ]
    
    Available config parameters:
        - name: str (required) - Display name for the configuration
        - use_hybrid_search: bool - Enable BM25 + semantic fusion
        - use_reranking: bool - Enable cross-encoder reranking
        - reranker_model: str - Cross-encoder model ('bge' or 'ms-marco')
        - rrf_k: int - RRF constant (higher = more uniform ranking)
        - rrf_weight: float - Semantic weight (0.0=all BM25, 1.0=all semantic)
        - retrieve_k: int - Candidates from each search method
        - fusion_k: int - Candidates after RRF fusion
        - bm25_k1: float - BM25 term frequency saturation
        - bm25_b: float - BM25 document length normalization
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        List of configuration dictionaries
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Support both {"configurations": [...]} and direct list format
    if isinstance(data, list):
        configs = data
    elif isinstance(data, dict) and "configurations" in data:
        configs = data["configurations"]
    else:
        raise ValueError(
            "Config file must be a JSON list or object with 'configurations' key"
        )
    
    # Validate each config has a name
    for i, cfg in enumerate(configs):
        if "name" not in cfg:
            cfg["name"] = f"config_{i}"
    
    return configs


def run_evaluation(
    dataset_name: str = "ragqa_arena",
    max_eval_examples: Optional[int] = None,
    top_k: int = 10,
    file_path: Optional[str] = None,
    compare_configs: bool = False,
    config_file: Optional[str] = None,
    use_hybrid: bool = True,
    use_reranking: bool = False,
    reranker_model: str = "bge",
    interactive: bool = True,
):
    """
    Run RAG evaluation on a dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        max_eval_examples: Maximum evaluation examples (None = all)
        top_k: Number of results to retrieve per query
        file_path: File path for cat_facts dataset
        compare_configs: Whether to compare multiple configurations
        config_file: Path to JSON config file with configurations to compare
        use_hybrid: Enable hybrid search (BM25 + semantic)
        use_reranking: Enable cross-encoder reranking
        reranker_model: Reranker model ('bge' or 'ms-marco')
        interactive: Run interactive query mode after evaluation
    """
    print("\n" + "=" * 70)
    print("RAG-LITE EVALUATION BENCHMARK")
    print("=" * 70)
    
    # Show available datasets
    print(f"\nAvailable datasets: {list_available_datasets()}")
    print(f"Selected dataset: {dataset_name}")
    
    # Load dataset
    print(f"\n--- Loading Dataset: {dataset_name} ---")
    
    dataset_kwargs = {}
    
    if dataset_name == "cat_facts":
        if file_path:
            dataset_kwargs["file_path"] = file_path
    elif dataset_name == "ragqa_arena":
        if max_eval_examples is not None:
            dataset_kwargs["max_eval"] = max_eval_examples
    
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
        print("Try: --dataset ragqa_arena")
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
    
    # Set reranker model
    if reranker_model == "ms-marco":
        config.model.reranker_model = ModelConfig.RERANKER_MS_MARCO
    else:
        config.model.reranker_model = ModelConfig.RERANKER_BGE_BASE
    
    # Update collection name to be unique per dataset
    config.storage.collection_name = f"eval_{dataset_name}"
    
    pipeline = RAGPipeline(config)
    
    # Index documents
    print("\n--- Indexing Documents ---")
    chunks = dataset.get_chunks()
    pipeline.index_documents(chunks, show_progress=True)
    
    # Create evaluator
    evaluator = RAGEvaluator(pipeline, dataset, verbose=True)
    
    # Determine configurations to run
    configurations = None
    
    if config_file:
        # Load configurations from file
        print(f"\n--- Loading Configurations from: {config_file} ---")
        try:
            configurations = load_config_file(config_file)
            print(f"Loaded {len(configurations)} configurations:")
            for cfg in configurations:
                print(f"  - {cfg['name']}: {cfg}")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    elif compare_configs:
        # Use default comparison configurations
        print("\n--- Using Default Comparison Configurations ---")
        configurations = [
            {"name": "semantic_only", "use_hybrid_search": False, "use_reranking": False},
            {"name": "hybrid_bm25+semantic", "use_hybrid_search": True, "use_reranking": False},
        ]
        
        # Add reranking configurations (cross-encoder is much faster than LLM-based)
        if len(eval_examples) <= 100:
            configurations.append(
                {"name": "hybrid+rerank_bge", "use_hybrid_search": True, "use_reranking": True, "reranker_model": "bge"}
            )
            configurations.append(
                {"name": "hybrid+rerank_msmarco", "use_hybrid_search": True, "use_reranking": True, "reranker_model": "ms-marco"}
            )
    
    if configurations:
        # Compare configurations
        print(f"\n--- Comparing {len(configurations)} Retrieval Configurations ---")
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
    if interactive:
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
  # Full benchmark (all 28k docs, all eval examples) - recommended
  python -m evaluation.run_benchmark
  
  # Faster run with limited eval examples
  python -m evaluation.run_benchmark --max-eval 300
  
  # Compare retrieval configurations
  python -m evaluation.run_benchmark --compare --max-eval 100
  
  # Compare with custom config file
  python -m evaluation.run_benchmark --config-file configs/rrf_sweep.json --max-eval 100
  
  # Quick test with cat facts (no evaluation, just indexing)
  python -m evaluation.run_benchmark --dataset cat_facts --file cat-facts.txt

Config File Format (JSON):
  {
    "configurations": [
      {"name": "semantic_only", "use_hybrid_search": false},
      {"name": "rrf_0.3", "rrf_weight": 0.3},
      {"name": "rrf_0.5", "rrf_weight": 0.5},
      {"name": "rrf_0.7", "rrf_weight": 0.7}
    ]
  }
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="ragqa_arena",
        choices=["cat_facts", "ragqa_arena"],
        help="Dataset to use (default: ragqa_arena)"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="File path for cat_facts dataset"
    )
    
    parser.add_argument(
        "--max-eval", "-e",
        type=int,
        default=None,
        help="Maximum evaluation examples (default: all)"
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
        "--config-file", "-C",
        type=str,
        default=None,
        help="Path to JSON config file with configurations to compare"
    )
    
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid search (use semantic only)"
    )
    
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking"
    )
    
    parser.add_argument(
        "--reranker",
        type=str,
        choices=["bge", "ms-marco"],
        default="bge",
        help="Reranker model: 'bge' (BAAI/bge-reranker-base) or 'ms-marco' (cross-encoder/ms-marco-MiniLM-L-6-v2)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive query mode after evaluation"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        dataset_name=args.dataset,
        max_eval_examples=args.max_eval,
        top_k=args.top_k,
        file_path=args.file,
        compare_configs=args.compare,
        config_file=args.config_file,
        use_hybrid=not args.no_hybrid,
        use_reranking=args.rerank,
        reranker_model=args.reranker,
        interactive=not args.no_interactive,
    )


if __name__ == "__main__":
    main()
