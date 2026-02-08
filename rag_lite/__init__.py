"""
RAG-Lite: A lightweight RAG implementation for knowledge-based question answering.

This is the core library with minimal dependencies. For evaluation tools,
install with: pip install rag-lite[eval]

Quick Start:
    >>> from rag_lite import RAGPipeline, Config
    >>> from rag_lite.data_loader import load_text_file
    >>>
    >>> # Load data
    >>> documents = load_text_file("my-data.txt")
    >>>
    >>> # Initialize pipeline
    >>> config = Config.default()
    >>> pipeline = RAGPipeline(config)
    >>> pipeline.index_documents(documents)
    >>>
    >>> # Query
    >>> results, response = pipeline.query("What is...?")
    >>> for chunk in response:
    ...     print(chunk, end="")
"""

__version__ = "0.1.0"

from .config import Config, ModelConfig, RetrievalConfig, StorageConfig
from .rag_pipeline import RAGPipeline
from .data_loader import load_text_file

__all__ = [
    # Config
    "Config",
    "ModelConfig", 
    "RetrievalConfig",
    "StorageConfig",
    # Pipeline
    "RAGPipeline",
    # Data loading
    "load_text_file",
]
