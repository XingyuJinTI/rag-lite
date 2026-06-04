"""
Response generation module for creating answers from retrieved context.

This module handles the generation of responses using the language model
with retrieved context in a RAG (Retrieval-Augmented Generation) pipeline.
"""

import logging
import re
from functools import lru_cache
from typing import List, Iterator, Tuple, Union

import ollama

from .types import RetrievedChunk

logger = logging.getLogger(__name__)

# Matches a citation marker referencing one or more 1-based source indices:
# [1], [12], or [1, 2]. Used to validate the LLM's inline citations against the
# sources actually provided.
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def validate_citations(answer: str, num_sources: int) -> Tuple[str, List[int]]:
    """
    Validate the inline [n] citation markers in an answer against the available
    sources, deterministically — independent of whether the LLM behaved.

    The LLM is instructed to cite sources as [n], but a (small) model can emit an
    index that doesn't exist (e.g. [4] when only 3 sources were given). Such markers
    are misleading, so we strip out-of-range ones; markers that reference real
    sources are kept and reported.

    Args:
        answer: the generated answer text
        num_sources: number of sources provided (valid indices are 1..num_sources)
    Returns:
        (cleaned_answer, cited_indices) — cleaned_answer has invalid markers removed;
        cited_indices is the sorted, deduplicated set of valid 1-based indices cited.
    """
    used: set = set()

    def _replace(match: "re.Match") -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", match.group(1))]
        valid = [n for n in nums if 1 <= n <= num_sources]
        used.update(valid)
        if not valid:
            return ""  # drop a marker that references nothing real
        return "[" + ", ".join(str(n) for n in valid) + "]"

    cleaned = _CITATION_RE.sub(_replace, answer)
    # Tidy whitespace/punctuation left behind by removed markers.
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, sorted(used)

# Default seconds to wait on the LLM. Without a bound, a hung Ollama would pin a
# server worker thread indefinitely and eventually exhaust the threadpool.
DEFAULT_REQUEST_TIMEOUT = 60.0

# A context item is either a RetrievedChunk (with provenance) or a bare string.
ContextItem = Union[RetrievedChunk, str]


@lru_cache(maxsize=8)
def _get_client(timeout: float) -> "ollama.Client":
    """Cached Ollama client with a fixed timeout (host comes from OLLAMA_HOST)."""
    return ollama.Client(timeout=timeout)


def _source_label(chunk: RetrievedChunk) -> str:
    """Human-readable provenance label, e.g. '(Handbook, p.4)' or '(notes.md)'."""
    name = chunk.title or chunk.source
    if not name:
        return ""
    label = name if chunk.page is None else f"{name}, p.{chunk.page}"
    return f" ({label})"


def format_context(chunks: List[ContextItem]) -> str:
    """
    Format retrieved chunks into a numbered context block. Each line is prefixed
    with a citation index `[n]` and, when provenance is available, a source label —
    so the model can cite `[n]` and the caller can map `n` back to its source.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, RetrievedChunk):
            # Prefer the expanded parent block (context_content) for the LLM; the
            # chunk's own content remains the precise, cited passage.
            text = (chunk.context_content or chunk.content).strip()
            label = _source_label(chunk)
        else:
            text, label = str(chunk).strip(), ""
        lines.append(f"[{i}]{label} {text}")
    return "\n".join(lines)


def create_prompt(query: str, context_chunks: List[ContextItem]) -> str:
    """
    Create a prompt for the language model with numbered, citable context.

    Args:
        query: User query
        context_chunks: List of RetrievedChunk (preferred) or strings

    Returns:
        Formatted prompt string
    """
    context_text = format_context(context_chunks)

    prompt = f'''You are a helpful and accurate chatbot that answers questions based on provided context.

Each context item below is numbered with a citation marker like [1], [2].

Context information:
{context_text}

Instructions:
- Answer the question using ONLY the information provided in the context above
- Cite the supporting context item(s) inline using their bracketed numbers, e.g. [1] or [2][3]
- If the context doesn't contain enough information to answer, reply exactly: "I don't have enough information in the provided context to answer that."
- Do not make up or infer information that isn't in the context
- Be concise but complete in your answer

Question: {query}

Answer:'''

    return prompt


def generate_response(
    query: str,
    context_chunks: List[ContextItem],
    language_model: str,
    stream: bool = True,
    system_message: str = "You are a helpful assistant that provides accurate answers based on given context.",
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Iterator[str]:
    """
    Generate a response using the language model with retrieved context.

    Args:
        query: User query
        context_chunks: List of retrieved context chunks
        language_model: Name of the language model to use
        stream: Whether to stream the response
        system_message: System message for the LLM
        timeout: Seconds to wait on the LLM before raising

    Yields:
        Response text chunks (if streaming) or complete response
    """
    prompt = create_prompt(query, context_chunks)

    try:
        response = _get_client(timeout).chat(
            model=language_model,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt},
            ],
            stream=stream,
        )
        
        if stream:
            for chunk in response:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    yield content
        else:
            yield response['message']['content']
            
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        raise


def generate_response_string(
    query: str,
    context_chunks: List[ContextItem],
    language_model: str,
    system_message: str = "You are a helpful assistant that provides accurate answers based on given context.",
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Generate a complete response as a string (non-streaming).
    
    Args:
        query: User query
        context_chunks: List of retrieved context chunks
        language_model: Name of the language model to use
        system_message: System message for the LLM
        
    Returns:
        Complete response string
    """
    response_gen = generate_response(
        query,
        context_chunks,
        language_model,
        stream=False,
        system_message=system_message,
        timeout=timeout,
    )
    return next(response_gen)


# ----------------------------------------------------------------------
# Multi-turn (conversational) support
# ----------------------------------------------------------------------

# A turn is (role, text) with role in {"user", "assistant"}.
Turn = Tuple[str, str]


def condense_question(
    history: List[Turn],
    question: str,
    language_model: str,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """
    Rewrite a follow-up into a standalone question using the conversation.

    A follow-up like "and how do I give it?" is meaningless to a retriever on its
    own. Condensing (history + follow-up) → a self-contained question is what makes
    retrieval work across turns. Returns the question unchanged when there's no
    history or on any failure.
    """
    if not history:
        return question
    convo = "\n".join(f"{'User' if r == 'user' else 'Assistant'}: {t}" for r, t in history)
    prompt = (
        "Given the conversation and a follow-up question, rewrite the follow-up as a "
        "STANDALONE question that includes any context needed to understand it on its "
        "own (resolve pronouns and references). If it is already standalone, return it "
        "unchanged. Output ONLY the standalone question.\n\n"
        f"Conversation:\n{convo}\n\nFollow-up question: {question}\nStandalone question:"
    )
    try:
        resp = _get_client(timeout).chat(
            model=language_model, messages=[{"role": "user", "content": prompt}]
        )
        rewritten = resp["message"]["content"].strip()
        return rewritten or question
    except Exception as e:
        logger.warning(f"Question condensation failed: {e}; using original question")
        return question


def generate_chat_response(
    history: List[Turn],
    question: str,
    context_chunks: List[ContextItem],
    language_model: str,
    stream: bool = True,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Iterator[str]:
    """
    Generate an answer with prior turns in the message history so the model can
    resolve references and stay coherent, grounded in the retrieved context.
    """
    context_text = format_context(context_chunks)
    system = (
        "You are a helpful, accurate assistant in a multi-turn conversation. Answer the "
        "user's latest question using ONLY the context provided in that message, citing "
        "supporting items inline as [n]. If the context is insufficient, say so."
    )
    messages = [{"role": "system", "content": system}]
    for role, text in history:
        messages.append({"role": "assistant" if role == "assistant" else "user", "content": text})
    messages.append({
        "role": "user",
        "content": f"Context information:\n{context_text}\n\nQuestion: {question}\n\nAnswer:",
    })
    try:
        resp = _get_client(timeout).chat(model=language_model, messages=messages, stream=stream)
        if stream:
            for chunk in resp:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        else:
            yield resp["message"]["content"]
    except Exception as e:
        logger.error(f"Failed to generate chat response: {e}")
        raise
