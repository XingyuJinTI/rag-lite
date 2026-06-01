"""
Token-aware text chunking.

Splits documents into chunks sized to the embedding model's token budget, packing
whole sentences so chunks stay coherent and never exceed the model's max sequence
length (beyond which the model would silently truncate at embed time).

Defaults (chunk_max_tokens=256, chunk_overlap=48) are starting points — they are
config-driven so they can be swept against the evaluation harness per corpus. See
the rationale in the Phase 2 plan / README.
"""

import logging
import re
from typing import List, Optional, Tuple

from .types import IngestChunk

logger = logging.getLogger(__name__)

# Cache one tokenizer per model name, mirroring retrieval._get_cross_encoder.
_tokenizers: dict = {}

# bge-base max sequence length; chunks must never exceed this.
MODEL_MAX_TOKENS = 512

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _get_tokenizer(model_name: str):
    """Lazy-load and cache the embedding model's HF tokenizer."""
    if model_name not in _tokenizers:
        from transformers import AutoTokenizer  # pulled in by sentence-transformers
        logger.info(f"Loading tokenizer for chunking: {model_name}")
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizers[model_name]


def _split_sentences(text: str) -> List[str]:
    """Split on sentence terminators and blank lines; keep non-empty pieces."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _pack_text(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap: int,
) -> List[str]:
    """
    Greedily pack sentences into windows of <= max_tokens tokens, carrying ~`overlap`
    tokens of trailing sentences into the next window. A single sentence longer than
    max_tokens is hard-split on token boundaries as a last resort.
    """
    sentences = _split_sentences(text)
    windows: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            windows.append(" ".join(current))

    for sent in sentences:
        sent_tokens = _token_len(tokenizer, sent)

        # A lone over-long sentence: hard-split on token boundaries.
        if sent_tokens > max_tokens:
            flush()
            current, current_tokens = [], 0
            ids = tokenizer.encode(sent, add_special_tokens=False)
            for i in range(0, len(ids), max_tokens):
                piece = tokenizer.decode(ids[i : i + max_tokens]).strip()
                if piece:
                    windows.append(piece)
            continue

        if current_tokens + sent_tokens > max_tokens:
            flush()
            # Build the overlap tail: keep trailing sentences up to `overlap` tokens.
            tail: List[str] = []
            tail_tokens = 0
            for s in reversed(current):
                t = _token_len(tokenizer, s)
                if tail_tokens + t > overlap:
                    break
                tail.insert(0, s)
                tail_tokens += t
            current = tail
            current_tokens = tail_tokens

        current.append(sent)
        current_tokens += sent_tokens

    flush()
    return windows


def chunk_segments(
    segments: List[Tuple[str, Optional[int]]],
    *,
    source: str,
    embedding_model: str,
    title: Optional[str] = None,
    uri: Optional[str] = None,
    max_tokens: int = 256,
    overlap: int = 48,
) -> List[IngestChunk]:
    """
    Turn (text, page) segments into sized IngestChunks.

    Chunking happens *within* each segment, so a chunk never spans a page boundary —
    the `page` attached to each chunk is therefore accurate for citations. Chunks are
    numbered with a document-global, monotonically increasing `chunk_index`.

    Args:
        segments: list of (text, page) — page is 1-based or None (non-paginated formats)
        source: filename / logical origin (delete + dedup key)
        embedding_model: HF model name whose tokenizer defines the token budget
        title, uri: document-level provenance
        max_tokens: target tokens per chunk (capped at the model max)
        overlap: tokens of trailing context carried between consecutive chunks
    """
    max_tokens = min(max_tokens, MODEL_MAX_TOKENS)
    overlap = max(0, min(overlap, max_tokens // 2))
    tokenizer = _get_tokenizer(embedding_model)

    chunks: List[IngestChunk] = []
    idx = 0
    for text, page in segments:
        if not text or not text.strip():
            continue
        for window in _pack_text(text, tokenizer, max_tokens, overlap):
            chunks.append(
                IngestChunk(
                    text=window,
                    source=source,
                    title=title,
                    uri=uri,
                    page=page,
                    chunk_index=idx,
                    metadata={},
                )
            )
            idx += 1

    logger.info(f"Chunked '{source}' into {len(chunks)} chunks (max_tokens={max_tokens}, overlap={overlap})")
    return chunks
