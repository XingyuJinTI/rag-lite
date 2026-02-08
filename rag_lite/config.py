"""
Configuration management for RAG-Lite.

This module handles all configuration settings including model names,
retrieval parameters, and feature flags.

Configuration Precedence (highest to lowest):
1. Environment variables
2. Dataclass defaults

"""

import os
from dataclasses import dataclass
from typing import Optional


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.getenv(key, default)


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    embedding_model: str = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
    language_model: str = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create ModelConfig from environment variables."""
        return cls(
            embedding_model=_get_env_str("EMBEDDING_MODEL", cls.embedding_model),
            language_model=_get_env_str("LANGUAGE_MODEL", cls.language_model),
        )


@dataclass
class StorageConfig:
    """Configuration for vector storage."""
    persist_directory: str = "./chroma_db"
    collection_name: str = "rag_lite"
    max_chunk_chars: int = 500  # Safety limit (~250-333 tokens for very dense text)

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create StorageConfig from environment variables."""
        return cls(
            persist_directory=_get_env_str("CHROMA_PERSIST_DIR", cls.persist_directory),
            collection_name=_get_env_str("CHROMA_COLLECTION", cls.collection_name),
            max_chunk_chars=_get_env_int("MAX_CHUNK_CHARS", cls.max_chunk_chars),
        )


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""
    # Result counts
    top_n: int = 3                # Final results to return
    retrieve_k: int = 50          # Candidates from each search method
    fusion_k: int = 20            # Candidates after RRF fusion (rerank pool)
    
    # Feature flags
    use_hybrid_search: bool = True   # Use semantic + BM25 with RRF (False = semantic only)
    use_reranking: bool = False
    
    # RRF parameters
    rrf_k: int = 60               # RRF constant (standard value)
    rrf_weight: float = 0.7       # Semantic weight in RRF; BM25 gets 1 - rrf_weight (0.3)
    
    # Reranking weights
    rerank_weight: float = 0.8
    original_score_weight: float = 0.2
    
    # BM25 parameters
    bm25_k1: float = 1.5          # Term frequency saturation (1.2-2.0)
    bm25_b: float = 0.75          # Document length normalization (0-1)

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create RetrievalConfig from environment variables."""
        return cls(
            top_n=_get_env_int("RETRIEVE_TOP_N", cls.top_n),
            retrieve_k=_get_env_int("RETRIEVE_K", cls.retrieve_k),
            fusion_k=_get_env_int("FUSION_K", cls.fusion_k),
            use_hybrid_search=_get_env_bool("USE_HYBRID_SEARCH", cls.use_hybrid_search),
            use_reranking=_get_env_bool("USE_RERANKING", cls.use_reranking),
            rrf_k=_get_env_int("RRF_K", cls.rrf_k),
            rrf_weight=_get_env_float("RRF_WEIGHT", cls.rrf_weight),
            rerank_weight=_get_env_float("RERANK_WEIGHT", cls.rerank_weight),
            original_score_weight=_get_env_float("ORIGINAL_SCORE_WEIGHT", cls.original_score_weight),
            bm25_k1=_get_env_float("BM25_K1", cls.bm25_k1),
            bm25_b=_get_env_float("BM25_B", cls.bm25_b),
        )


@dataclass
class Config:
    """
    Main configuration class for RAG-Lite.
    
    Example:
        >>> config = Config.default()
        >>> config.retrieval.use_hybrid_search = False  # Semantic only
        >>> pipeline = RAGPipeline(config)
    """
    model: ModelConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    data_file: str = "cat-facts.txt"
    ollama_base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            model=ModelConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            storage=StorageConfig.from_env(),
            data_file=_get_env_str("DATA_FILE", cls.data_file),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        )

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            retrieval=RetrievalConfig(),
            storage=StorageConfig(),
        )
    
    @classmethod
    def semantic_only(cls) -> "Config":
        """Create configuration with semantic search only (no BM25)."""
        config = cls.default()
        config.retrieval.use_hybrid_search = False
        return config
    
    @classmethod
    def with_reranking(cls) -> "Config":
        """Create configuration with reranking enabled."""
        config = cls.default()
        config.retrieval.use_reranking = True
        return config
