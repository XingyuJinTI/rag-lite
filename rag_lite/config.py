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

from rag_lite.retrieval import KeywordMethod


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable.
    
    Accepts: 'true', 'True', '1', 'yes', 'on' (case-insensitive)
    """
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


def _get_keyword_method(key: str, default: "KeywordMethod") -> "KeywordMethod":
    """Get KeywordMethod from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return KeywordMethod(value.lower())
    except ValueError:
        return default


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

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create StorageConfig from environment variables."""
        return cls(
            persist_directory=_get_env_str("CHROMA_PERSIST_DIR", cls.persist_directory),
            collection_name=_get_env_str("CHROMA_COLLECTION", cls.collection_name),
        )


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""
    top_n: int = 3
    retrieve_k: int = 50  # Candidates from each search method (semantic/keyword)
    fusion_k: int = 20    # Candidates after RRF fusion (rerank pool)
    use_reranking: bool = False
    use_hybrid_search: bool = True
    use_query_expansion: bool = False
    # RRF (Reciprocal Rank Fusion) - recommended for hybrid search
    use_rrf: bool = True  # Use RRF fusion instead of weighted scoring
    rrf_k: int = 60  # RRF constant (standard value from original paper)
    # Legacy weighted scoring (used when use_rrf=False)
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    rerank_weight: float = 0.8
    original_score_weight: float = 0.2
    # Keyword search method: "jaccard", "tfidf", or "bm25"
    keyword_method: KeywordMethod = KeywordMethod.BM25
    # BM25 parameters (only used when keyword_method is BM25)
    bm25_k1: float = 1.5  # Term frequency saturation (typically 1.2-2.0)
    bm25_b: float = 0.75  # Document length normalization (0=none, 1=full)

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create RetrievalConfig from environment variables."""
        return cls(
            top_n=_get_env_int("RETRIEVE_TOP_N", cls.top_n),
            retrieve_k=_get_env_int("RETRIEVE_K", cls.retrieve_k),
            fusion_k=_get_env_int("FUSION_K", cls.fusion_k),
            use_reranking=_get_env_bool("USE_RERANKING", cls.use_reranking),
            use_hybrid_search=_get_env_bool("USE_HYBRID_SEARCH", cls.use_hybrid_search),
            use_query_expansion=_get_env_bool("USE_QUERY_EXPANSION", cls.use_query_expansion),
            use_rrf=_get_env_bool("USE_RRF", cls.use_rrf),
            rrf_k=_get_env_int("RRF_K", cls.rrf_k),
            semantic_weight=_get_env_float("SEMANTIC_WEIGHT", cls.semantic_weight),
            keyword_weight=_get_env_float("KEYWORD_WEIGHT", cls.keyword_weight),
            rerank_weight=_get_env_float("RERANK_WEIGHT", cls.rerank_weight),
            original_score_weight=_get_env_float("ORIGINAL_SCORE_WEIGHT", cls.original_score_weight),
            keyword_method=_get_keyword_method("KEYWORD_METHOD", cls.keyword_method),
            bm25_k1=_get_env_float("BM25_K1", cls.bm25_k1),
            bm25_b=_get_env_float("BM25_B", cls.bm25_b),
        )


@dataclass
class Config:
    """Main configuration class.
    
    Configuration precedence (highest to lowest):
    1. Environment variables (via from_env())
    2. Dataclass field defaults
    3. Programmatic assignment
    """
    model: ModelConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    data_file: str = "cat-facts.txt"
    ollama_base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables.
        
        This method respects the dataclass defaults as fallbacks,
        ensuring a single source of truth for default values.
        """
        return cls(
            model=ModelConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            storage=StorageConfig.from_env(),
            data_file=_get_env_str("DATA_FILE", cls.data_file),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),  # None is valid, so no default
        )

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration using dataclass defaults.
        
        This is the same as creating instances directly, but provides
        a clear API for getting defaults.
        """
        return cls(
            model=ModelConfig(),
            retrieval=RetrievalConfig(),
            storage=StorageConfig(),
        )
