"""
Token-based text chunking for RAG pipelines.

This module provides robust chunking that respects embedding model token limits
"""

import logging
from typing import List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class TokenChunker:
    """
    Token-aware text chunker that guarantees chunks fit within model context limits.
    
    Uses the same tokenizer as the embedding model to ensure accurate token counts.
    Handles dense text (legal, code) where char/token ratio can be ~1:1.
    
    Example:
        >>> chunker = TokenChunker(model_name="BAAI/bge-base-en-v1.5", max_tokens=480)
        >>> chunks = chunker.chunk_text(long_document)
        >>> # All chunks guaranteed to be <= 480 tokens
    """
    
    # Cache tokenizers to avoid repeated loading
    _tokenizer_cache = {}
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        max_tokens: int = 480,
        overlap_tokens: int = 80,
    ):
        """
        Initialize the token chunker.
        
        Args:
            model_name: HuggingFace model name for tokenizer (should match embedding model)
            max_tokens: Maximum tokens per chunk (default 480 for 512-token models)
            overlap_tokens: Token overlap between chunks for context continuity
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Lazy-load tokenizer on first use."""
        if self._tokenizer is None:
            self._tokenizer = self._get_tokenizer(self.model_name)
        return self._tokenizer
    
    @classmethod
    def _get_tokenizer(cls, model_name: str):
        """Get or create cached tokenizer."""
        if model_name not in cls._tokenizer_cache:
            try:
                from transformers import AutoTokenizer
                logger.info(f"Loading tokenizer for {model_name}...")
                cls._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
                logger.warning("Falling back to simple whitespace tokenizer")
                cls._tokenizer_cache[model_name] = None
        return cls._tokenizer_cache[model_name]
    
    def token_count(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer is None:
            # Fallback: estimate ~4 chars per token (conservative for dense text: use 3)
            return len(text) // 3
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into token-limited chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks, each <= max_tokens
        """
        if not text or not text.strip():
            return []
        
        # Fast path: text fits in one chunk
        if self.token_count(text) <= self.max_tokens:
            return [text.strip()]
        
        if self.tokenizer is None:
            # Fallback to char-based chunking with conservative estimate
            return self._chunk_by_chars(text)
        
        return self._chunk_by_tokens(text)
    
    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Chunk text using actual tokenizer, preserving original text."""
        # Get token offsets to slice original text (preserves casing, spacing)
        encoding = self.tokenizer(
            text, 
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        token_ids = encoding['input_ids']
        offsets = encoding['offset_mapping']  # [(start_char, end_char), ...]
        
        if not token_ids:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start_token = 0
        
        while start_token < len(token_ids):
            end_token = min(start_token + self.max_tokens, len(token_ids))
            
            # Get character positions from offsets
            char_start = offsets[start_token][0]
            char_end = offsets[end_token - 1][1]
            
            # Slice original text to preserve casing and spacing
            chunk_text = text[char_start:char_end].strip()
            
            if chunk_text:
                chunks.append(chunk_text)
            
            if end_token >= len(token_ids):
                break
            
            # Move start with overlap
            start_token = end_token - self.overlap_tokens
            if start_token <= 0:
                start_token = end_token  # Prevent infinite loop
        
        return chunks
    
    def _chunk_by_chars(self, text: str) -> List[str]:
        """Fallback char-based chunking (conservative: 3 chars/token)."""
        max_chars = self.max_tokens * 3  # Conservative estimate
        overlap_chars = self.overlap_tokens * 3
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start + max_chars - 200, start), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end >= len(text):
                break
            
            start = end - overlap_chars
            if start <= 0:
                start = end
        
        return chunks
    
    def chunk_texts(self, texts: List[str]) -> List[str]:
        """
        Chunk multiple texts, flattening results.
        
        Args:
            texts: List of texts to chunk
            
        Returns:
            Flat list of all chunks from all texts
        """
        all_chunks = []
        for text in texts:
            all_chunks.extend(self.chunk_text(text))
        return all_chunks


def guardrail_texts(
    texts: List[str],
    max_tokens: int = 480,
    model_name: str = "BAAI/bge-base-en-v1.5"
) -> List[str]:
    """
    Ensure all texts fit within token limit, splitting if necessary.
    
    Use this as a safety guardrail right before embedding.
    
    Args:
        texts: List of texts to check/split
        max_tokens: Maximum tokens per text
        model_name: Tokenizer model name
        
    Returns:
        List of texts, all guaranteed to be <= max_tokens
    """
    chunker = TokenChunker(model_name=model_name, max_tokens=max_tokens, overlap_tokens=50)
    safe_texts = []
    
    for text in texts:
        if chunker.token_count(text) <= max_tokens:
            safe_texts.append(text)
        else:
            # Split oversized text
            logger.debug(f"Splitting oversized text ({chunker.token_count(text)} tokens)")
            safe_texts.extend(chunker.chunk_text(text))
    
    return safe_texts


# Convenience function for quick token counting
def count_tokens(text: str, model_name: str = "BAAI/bge-base-en-v1.5") -> int:
    """Count tokens in text using the specified model's tokenizer."""
    chunker = TokenChunker(model_name=model_name)
    return chunker.token_count(text)
