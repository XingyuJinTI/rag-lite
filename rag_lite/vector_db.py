"""
Vector database implementation using ChromaDB for persistent storage.

This module provides a ChromaDB-backed vector database for storing document chunks
and their corresponding embeddings with automatic persistence.
"""

import logging
from typing import List, Tuple, Optional
import hashlib

import chromadb
from chromadb.config import Settings
import ollama

logger = logging.getLogger(__name__)


class VectorDB:
    """
    ChromaDB-backed vector database for storing document chunks and embeddings.
    
    Provides persistent storage with automatic embedding generation via Ollama.
    Data persists across restarts when using a persist_directory.
    """

    def __init__(
        self,
        embedding_model: str,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_lite",
        ollama_base_url: Optional[str] = None
    ):
        """
        Initialize the vector database with ChromaDB backend.
        
        Args:
            embedding_model: Name of the Ollama embedding model to use
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            ollama_base_url: Optional base URL for Ollama API
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Configure Ollama client if base URL is provided
        if ollama_base_url:
            ollama.Client(host=ollama_base_url)
        
        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(
            f"Initialized ChromaDB at '{persist_directory}' "
            f"with collection '{collection_name}' ({self._collection.count()} documents)"
        )

    def _generate_id(self, chunk: str) -> str:
        """Generate a deterministic ID for a chunk."""
        return hashlib.sha256(chunk.encode()).hexdigest()[:16]

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = ollama.embed(model=self.embedding_model, input=text)
        return result['embeddings'][0]

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single call."""
        result = ollama.embed(model=self.embedding_model, input=texts)
        return result['embeddings']

    def add_chunk(self, chunk: str) -> None:
        """
        Add a chunk to the database with its embedding.
        
        Args:
            chunk: Text chunk to add
        """
        chunk_id = self._generate_id(chunk)
        
        # Check if already exists
        existing = self._collection.get(ids=[chunk_id])
        if existing['ids']:
            logger.debug(f"Chunk already exists: {chunk_id}")
            return
        
        try:
            embedding = self._get_embedding(chunk)
            self._collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk]
            )
        except Exception as e:
            logger.error(f"Failed to add chunk to database: {e}")
            raise

    def add_chunks(self, chunks: List[str], show_progress: bool = True, batch_size: int = 50) -> None:
        """
        Add multiple chunks to the database with batch embedding.
        
        Args:
            chunks: List of text chunks to add
            show_progress: Whether to log progress
            batch_size: Number of chunks to embed per batch
        """
        # Filter out chunks that already exist
        chunk_ids = [self._generate_id(chunk) for chunk in chunks]
        existing = self._collection.get(ids=chunk_ids)
        existing_ids = set(existing['ids'])
        
        new_chunks = [
            (chunk, chunk_id) 
            for chunk, chunk_id in zip(chunks, chunk_ids) 
            if chunk_id not in existing_ids
        ]
        
        if not new_chunks:
            logger.info("All chunks already exist in database")
            return
        
        logger.info(f"Adding {len(new_chunks)} new chunks ({len(existing_ids)} already exist)")
        
        # Process in batches
        total = len(new_chunks)
        for i in range(0, total, batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_chunks = [chunk for chunk, _ in batch]
            batch_ids = [chunk_id for _, chunk_id in batch]
            
            try:
                embeddings = self._get_embeddings_batch(batch_chunks)
                self._collection.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_chunks
                )
                
                if show_progress:
                    processed = min(i + batch_size, total)
                    logger.info(f"Added chunks {processed}/{total} to the database")
                    
            except Exception as e:
                logger.error(f"Failed to add batch to database: {e}")
                raise

    def get_all(self) -> List[Tuple[str, List[float]]]:
        """
        Get all chunks and their embeddings.
        
        Returns:
            List of (chunk, embedding) tuples
        """
        results = self._collection.get(include=['documents', 'embeddings'])
        
        if not results['documents']:
            return []
        
        return list(zip(results['documents'], results['embeddings']))

    def search(
        self,
        query: str,
        n_results: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using ChromaDB's native search.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        query_embedding = self._get_embedding(query)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # ChromaDB returns distances, convert to similarity (1 - distance for cosine)
        chunks = results['documents'][0]
        distances = results['distances'][0]
        similarities = [1 - d for d in distances]
        
        return list(zip(chunks, similarities))

    def size(self) -> int:
        """Get the number of chunks in the database."""
        return self._collection.count()

    def clear(self) -> None:
        """Clear all data from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector database cleared")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection '{self.collection_name}'")
