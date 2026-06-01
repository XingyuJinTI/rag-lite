"""
Response generation module for creating answers from retrieved context.

This module handles the generation of responses using the language model
with retrieved context in a RAG (Retrieval-Augmented Generation) pipeline.
"""

import logging
from functools import lru_cache
from typing import List, Iterator

import ollama

logger = logging.getLogger(__name__)

# Default seconds to wait on the LLM. Without a bound, a hung Ollama would pin a
# server worker thread indefinitely and eventually exhaust the threadpool.
DEFAULT_REQUEST_TIMEOUT = 60.0


@lru_cache(maxsize=8)
def _get_client(timeout: float) -> "ollama.Client":
    """Cached Ollama client with a fixed timeout (host comes from OLLAMA_HOST)."""
    return ollama.Client(timeout=timeout)


def format_context(chunks: List[str]) -> str:
    """
    Format retrieved chunks into context text.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Formatted context string
    """
    return '\n'.join([f'{i+1}. {chunk.strip()}' for i, chunk in enumerate(chunks)])


def create_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Create a prompt for the language model with context and query.
    
    Args:
        query: User query
        context_chunks: List of retrieved context chunks
        
    Returns:
        Formatted prompt string
    """
    context_text = format_context(context_chunks)
    
    prompt = f'''You are a helpful and accurate chatbot that answers questions based on provided context.

Context information:
{context_text}

Instructions:
- Answer the question using ONLY the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Do not make up or infer information that isn't in the context
- Be concise but complete in your answer
- Cite which fact(s) you're using when relevant

Question: {query}

Answer:'''
    
    return prompt


def generate_response(
    query: str,
    context_chunks: List[str],
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
    context_chunks: List[str],
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
