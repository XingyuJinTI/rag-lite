"""
Response generation module for creating answers from retrieved context.

This module handles the generation of responses using the language model
with retrieved context in a RAG (Retrieval-Augmented Generation) pipeline.
"""

import logging
from functools import lru_cache
from typing import List, Iterator, Union

import ollama

from .types import RetrievedChunk

logger = logging.getLogger(__name__)

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
            text, label = chunk.content.strip(), _source_label(chunk)
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
