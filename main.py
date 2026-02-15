#!/usr/bin/env python3
"""
Main entry point for RAG-Lite CLI application.

This script provides a command-line interface for the RAG system.

Usage:
    # Use default cat-facts dataset
    python main.py

    # Use RAGQArena Tech dataset (28k+ tech documents)
    python main.py --dataset ragqa

    # Use custom text file
    python main.py --file path/to/your/data.txt
"""

import argparse
import logging
import sys
from pathlib import Path

from rag_lite.config import Config
from rag_lite.data_loader import load_text_file
from rag_lite.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ragqa_dataset(cache_dir: str = ".") -> tuple:
    """Load RAGQArena Tech dataset for Q&A.
    
    Returns:
        Tuple of (chunks list, dataset name for collection)
    """
    try:
        from evaluation.datasets import RAGQArenaDataset
    except ImportError:
        logger.error("RAGQArena dataset requires evaluation module.")
        logger.info("Install with: pip install requests orjson")
        sys.exit(1)
    
    logger.info("Loading RAGQArena Tech dataset (28k+ documents)...")
    dataset = RAGQArenaDataset(cache_dir=cache_dir)
    dataset.load()
    
    # Return list of document texts and dataset name
    return dataset.get_chunks(), dataset.name


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="RAG-Lite: Knowledge-Based Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Use default cat-facts
  python main.py --dataset ragqa      # Use RAGQArena Tech (28k docs)
  python main.py --file mydata.txt    # Use custom file
        """
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["cat_facts", "ragqa"],
        default=None,
        help="Dataset to use: cat_facts (default) or ragqa (RAGQArena Tech)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to custom text file (overrides --dataset)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".",
        help="Directory to cache downloaded datasets (default: current dir)"
    )
    parser.add_argument(
        "--hybrid", 
        action="store_true",
        default=False,
        help="Use hybrid search (BM25 + semantic). Default is semantic-only."
    )
    parser.add_argument(
        "--rrf-weight",
        type=float,
        default=1.0,
        help="RRF weight for hybrid search (0=BM25 only, 1=semantic only, default: 1.0)"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        default=False,
        help="Force re-indexing even if data already exists"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config.from_env()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config = Config.default()
        logger.info("Using default configuration")
    
    # Override config with CLI args
    if args.hybrid:
        config.retrieval.use_hybrid_search = True
        config.retrieval.rrf_weight = args.rrf_weight
    else:
        # Pure semantic search by default (best for RAGQArena)
        config.retrieval.use_hybrid_search = False
    
    # Load data based on arguments
    collection_suffix = None  # For dataset-specific collection naming
    
    if args.file:
        # Custom file
        data_file = Path(args.file)
        if not data_file.is_absolute():
            project_root = Path(__file__).parent
            data_file = project_root / data_file
        
        try:
            dataset = load_text_file(str(data_file))
            dataset_name = data_file.name
            collection_suffix = data_file.stem  # Use filename without extension
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load data file: {e}")
            sys.exit(1)
    
    elif args.dataset == "ragqa":
        # RAGQArena Tech dataset
        dataset, ds_name = load_ragqa_dataset(cache_dir=args.cache_dir)
        dataset_name = "RAGQArena Tech"
        collection_suffix = ds_name  # "ragqa_arena" - matches evaluation
    
    else:
        # Default: cat-facts
        data_file = Path(config.data_file)
        if not data_file.is_absolute():
            project_root = Path(__file__).parent
            data_file = project_root / data_file
        
        try:
            dataset = load_text_file(str(data_file))
            dataset_name = "cat-facts"
            collection_suffix = "cat_facts"
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_file}")
            logger.info("Please ensure the data file exists or set DATA_FILE environment variable")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load data file: {e}")
            sys.exit(1)
    
    # Use dataset-specific collection (matches evaluation naming: eval_{dataset_name})
    if collection_suffix:
        config.storage.collection_name = f"eval_{collection_suffix}"
    
    logger.info(f"Loaded {len(dataset)} documents from {dataset_name}")
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Check if we need to index
    existing_count = pipeline.vector_db.size()
    
    if existing_count > 0 and not args.reindex:
        logger.info(f"Using existing index with {existing_count:,} documents (use --reindex to rebuild)")
    else:
        if args.reindex and existing_count > 0:
            logger.info(f"Re-indexing {len(dataset):,} documents (clearing {existing_count:,} existing)...")
            pipeline.vector_db.clear()
        
        # Index documents
        try:
            pipeline.index_documents(dataset, show_progress=True)
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            sys.exit(1)
    
    # Interactive query loop
    search_mode = "hybrid" if config.retrieval.use_hybrid_search else "semantic"
    indexed_count = pipeline.vector_db.size()
    print("\n" + "="*60)
    print(f"RAG-Lite: {dataset_name}")
    print(f"Documents: {indexed_count:,} | Search: {search_mode}")
    print("="*60)
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            query = input("Ask me a question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Retrieve and generate
            retrieved, response_stream = pipeline.query(query, stream=True)
            
            # Display retrieved knowledge
            print('\nRetrieved knowledge:')
            for chunk, score in retrieved:
                # Truncate long chunks for display
                display_text = chunk.strip()
                if len(display_text) > 200:
                    display_text = display_text[:200] + "..."
                print(f' - (score: {score:.3f}) {display_text}')
            
            # Display response
            print('\nChatbot response:')
            for chunk in response_stream:
                print(chunk, end='', flush=True)
            print()  # New line at the end
            print()  # Extra spacing
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"An error occurred: {e}\n")


if __name__ == "__main__":
    main()
