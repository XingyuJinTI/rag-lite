"""
Token-aware text chunking.

Splits documents into chunks sized to the embedding model's token budget, packing
whole sentences so chunks stay coherent and never exceed the model's max sequence
length (beyond which the model would silently truncate at embed time).

Defaults (chunk_max_tokens=256, chunk_overlap=48) are starting points — they are
config-driven so they can be swept against the evaluation harness per corpus. See
the rationale in the Phase 2 plan / README.
"""

import hashlib
import logging
import re
from typing import List, Optional, Tuple

from .types import IngestChunk, Parent

logger = logging.getLogger(__name__)

# Cache one tokenizer per model name, mirroring retrieval._get_cross_encoder.
_tokenizers: dict = {}

# Absolute ceiling on chunk size regardless of model. Some models report an
# enormous model_max_length (e.g. 1e30 sentinel); cap to a sane long-context value.
HARD_MAX_TOKENS = 8192
# Used when a tokenizer doesn't report a usable model_max_length.
FALLBACK_MAX_TOKENS = 512

# Block separators: blank lines, and the start of a markdown heading or list item —
# so a bulleted/numbered list or heading isn't glued into one giant "sentence".
_BLOCK_RE = re.compile(r"\n\s*\n|\n(?=\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}\s))")
# Sentence boundary: terminator + whitespace, only when the next token looks like a
# new sentence start (capital / digit / opening quote-or-paren).
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(\[])")
# Tokens ending a fragment that usually do NOT end a sentence (abbreviations/initials).
_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "cf.", "al.", "approx.", "vol.", "no.", "fig.",
    "eq.", "inc.", "ltd.", "co.", "dr.", "mr.", "mrs.", "ms.", "st.", "jr.", "sr.",
}
_INITIAL_RE = re.compile(r"^[A-Za-z]\.$")  # single-letter initial like "A."


def _get_tokenizer(model_name: str):
    """Lazy-load and cache the embedding model's HF tokenizer."""
    if model_name not in _tokenizers:
        from transformers import AutoTokenizer  # pulled in by sentence-transformers
        logger.info(f"Loading tokenizer for chunking: {model_name}")
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizers[model_name]


def _model_max_tokens(tokenizer) -> int:
    """The model's usable max sequence length, clamped to a sane range.

    Derived from the tokenizer (bge-base→512, bge-m3→8192) rather than hardcoded, so
    chunking tracks the active model and a larger CHUNK_MAX_TOKENS can exploit a
    long-context model instead of being silently capped at 512.
    """
    m = getattr(tokenizer, "model_max_length", None)
    if not isinstance(m, int) or m <= 0 or m > HARD_MAX_TOKENS:
        return FALLBACK_MAX_TOKENS if not (isinstance(m, int) and m > 0) else HARD_MAX_TOKENS
    return m


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-ish units, robust to markdown blocks, abbreviations,
    and decimals. Whitespace within a block is normalized so hard-wrapped lines join."""
    out: List[str] = []
    for block in _BLOCK_RE.split(text.strip()):
        block = " ".join(block.split())  # collapse internal whitespace/newlines
        if not block:
            continue
        buf = ""
        for piece in _SENT_RE.split(block):
            buf = f"{buf} {piece}".strip() if buf else piece
            tail = buf.split()[-1].lower() if buf.split() else ""
            # Keep buffering if the boundary was a false positive (abbreviation/initial).
            if tail in _ABBREVIATIONS or _INITIAL_RE.match(tail):
                continue
            out.append(buf)
            buf = ""
        if buf.strip():
            out.append(buf.strip())
    return out


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
    tokenizer = _get_tokenizer(embedding_model)
    max_tokens = min(max_tokens, _model_max_tokens(tokenizer))
    overlap = max(0, min(overlap, max_tokens // 2))

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


def _parent_id(source: str, text: str) -> str:
    return hashlib.sha256(f"{source}\x00{text}".encode()).hexdigest()[:16]


def chunk_hierarchical(
    segments: List[Tuple[str, Optional[int]]],
    *,
    source: str,
    embedding_model: str,
    title: Optional[str] = None,
    uri: Optional[str] = None,
    parent_max_tokens: int = 1024,
    child_max_tokens: int = 256,
    child_overlap: int = 48,
) -> Tuple[List[IngestChunk], List[Parent]]:
    """
    Two-level chunking for small-to-big retrieval.

    Each segment (page) is partitioned into non-overlapping **parent** windows
    (~parent_max_tokens), and each parent is split into overlapping **child** chunks
    (~child_max_tokens). Children carry their parent_id. Children are embedded/indexed;
    parents are returned for separate (text-only) storage and fetched at answer time.

    Returns (children, parents).
    """
    tokenizer = _get_tokenizer(embedding_model)
    model_cap = _model_max_tokens(tokenizer)
    parent_max = min(parent_max_tokens, max(model_cap, child_max_tokens))
    child_max = min(child_max_tokens, parent_max)
    child_overlap = max(0, min(child_overlap, child_max // 2))

    children: List[IngestChunk] = []
    parents: List[Parent] = []
    idx = 0
    for text, page in segments:
        if not text or not text.strip():
            continue
        # Parent windows partition the page (no overlap); a short page = one parent.
        for parent_text in _pack_text(text, tokenizer, parent_max, overlap=0):
            pid = _parent_id(source, parent_text)
            parents.append(Parent(parent_id=pid, content=parent_text, source=source,
                                  title=title, uri=uri, page=page))
            for window in _pack_text(parent_text, tokenizer, child_max, child_overlap):
                children.append(IngestChunk(
                    text=window, source=source, title=title, uri=uri, page=page,
                    chunk_index=idx, parent_id=pid, metadata={},
                ))
                idx += 1

    logger.info(f"Chunked '{source}' → {len(children)} children / {len(parents)} parents "
                f"(parent={parent_max}, child={child_max})")
    return children, parents
