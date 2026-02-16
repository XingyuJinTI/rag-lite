"""
Vector database implementation using ChromaDB for persistent storage.

This module provides a ChromaDB-backed vector database for storing document chunks
and their corresponding embeddings with automatic persistence.

Uses sentence-transformers for fast, efficient local embeddings.
Also includes SQLite FTS5 for fast full-text keyword search with BM25 ranking.
"""

import logging
import os
import sqlite3
from typing import List, Tuple, Optional
import hashlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .utils import get_device

logger = logging.getLogger(__name__)


class VectorDB:
    """
    ChromaDB-backed vector database for storing document chunks and embeddings.
    
    Provides persistent storage with automatic embedding generation via sentence-transformers.
    Data persists across restarts when using a persist_directory.
    """

    # Max chars per chunk (~250-333 tokens, fits 512-token models)
    MAX_CHUNK_CHARS = 500

    def __init__(
        self,
        embedding_model: str,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_lite",
        max_chunk_chars: Optional[int] = None
    ):
        """
        Initialize the vector database with ChromaDB backend.
        
        Args:
            embedding_model: HuggingFace model name (e.g., "BAAI/bge-base-en-v1.5")
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            max_chunk_chars: Maximum chars per chunk before truncation (default 500)
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.max_chunk_chars = max_chunk_chars or self.MAX_CHUNK_CHARS
        
        # Lazy load embedding model (loaded on first use)
        self._model = None
        
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
        
        # Initialize FTS5 for keyword search with persistent connection
        self._fts_db_path = os.path.join(persist_directory, f"{collection_name}_fts.db")
        self._fts_conn = sqlite3.connect(self._fts_db_path, check_same_thread=False)
        self._init_fts()
        
        # Check if FTS index needs rebuilding (out of sync with ChromaDB)
        self._sync_fts_if_needed()
        
        logger.info(
            f"Initialized ChromaDB at '{persist_directory}' "
            f"with collection '{collection_name}' ({self._collection.count()} documents)"
        )

    def _init_fts(self) -> None:
        """Initialize SQLite FTS5 database for keyword search."""
        cursor = self._fts_conn.cursor()
        
        # Create FTS5 virtual table with BM25 support
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                content,
                tokenize='porter unicode61'
            )
        """)
        
        # Create a regular table to track indexed chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexed_chunks (
                chunk_id TEXT PRIMARY KEY
            )
        """)
        
        self._fts_conn.commit()
    
    def _sync_fts_if_needed(self) -> None:
        """Check if FTS index is in sync with ChromaDB and rebuild if needed."""
        chroma_count = self._collection.count()
        
        if chroma_count == 0:
            return
        
        cursor = self._fts_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM indexed_chunks")
        fts_count = cursor.fetchone()[0]
        
        # If counts differ significantly, rebuild
        if fts_count < chroma_count * 0.9:  # Allow 10% tolerance
            logger.info(f"FTS index out of sync ({fts_count} vs {chroma_count}), rebuilding...")
            self.rebuild_fts_index(show_progress=True)
    
    def _add_to_fts(self, chunk_ids: List[str], chunks: List[str]) -> None:
        """Add chunks to FTS5 index."""
        if not chunk_ids:
            return
            
        cursor = self._fts_conn.cursor()
        
        # Check which chunks are already indexed
        placeholders = ','.join('?' * len(chunk_ids))
        cursor.execute(
            f"SELECT chunk_id FROM indexed_chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids
        )
        existing = {row[0] for row in cursor.fetchall()}
        
        # Insert only new chunks
        new_data = [
            (cid, chunk) for cid, chunk in zip(chunk_ids, chunks)
            if cid not in existing
        ]
        
        if new_data:
            cursor.executemany(
                "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
                new_data
            )
            cursor.executemany(
                "INSERT INTO indexed_chunks (chunk_id) VALUES (?)",
                [(cid,) for cid, _ in new_data]
            )
            self._fts_conn.commit()
    
    def _generate_id(self, chunk: str) -> str:
        """Generate a deterministic ID for a chunk."""
        return hashlib.sha256(chunk.encode()).hexdigest()[:16]

    def _get_model(self) -> SentenceTransformer:
        """Get embedding model (lazy loaded on first use)."""
        if self._model is None:
            device = get_device()
            logger.info(f"Loading embedding model: {self.embedding_model} on {device}")
            self._model = SentenceTransformer(self.embedding_model, device=device)
            logger.info(f"Loaded embedding model with dimension {self._model.get_sentence_embedding_dimension()}")
        return self._model

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_chunk_chars, breaking at word boundary."""
        if len(text) <= self.max_chunk_chars:
            return text
        
        # Find last space before limit
        truncated = text[:self.max_chunk_chars]
        last_space = truncated.rfind(' ')
        if last_space > self.max_chunk_chars * 0.8:  # Only use if > 80% of max
            truncated = truncated[:last_space]
        
        logger.debug(f"Truncated chunk from {len(text)} to {len(truncated)} chars")
        return truncated

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        text = self._truncate_text(text)
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in parallel."""
        texts = [self._truncate_text(t) for t in texts]
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

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
            # Also add to FTS5 index
            self._add_to_fts([chunk_id], [chunk])
        except Exception as e:
            logger.error(f"Failed to add chunk to database: {e}")
            raise

    def add_chunks(self, chunks: List[str], show_progress: bool = True, batch_size: int = 256) -> None:
        """
        Add multiple chunks to the database with batch embedding.
        
        Args:
            chunks: List of text chunks to add
            show_progress: Whether to log progress
            batch_size: Number of chunks to embed per batch
        """
        # Generate IDs and deduplicate (same text = same ID)
        chunk_id_map = {}  # id -> chunk (keeps first occurrence)
        for chunk in chunks:
            chunk_id = self._generate_id(chunk)
            if chunk_id not in chunk_id_map:
                chunk_id_map[chunk_id] = chunk
        
        # Check which IDs already exist in the database (batch to avoid SQL variable limit)
        unique_ids = list(chunk_id_map.keys())
        existing_ids = set()
        
        # SQLite has a limit of ~999 variables per query, use 500 to be safe
        check_batch_size = 500
        for i in range(0, len(unique_ids), check_batch_size):
            batch_ids = unique_ids[i:i + check_batch_size]
            existing = self._collection.get(ids=batch_ids)
            existing_ids.update(existing['ids'])
        
        # Filter to only new chunks
        new_chunks = [
            (chunk_id_map[chunk_id], chunk_id) 
            for chunk_id in unique_ids
            if chunk_id not in existing_ids
        ]
        
        duplicates_in_input = len(chunks) - len(unique_ids)
        if duplicates_in_input > 0:
            logger.info(f"Deduplicated {duplicates_in_input} duplicate chunks from input")
        
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
                
                # Also add to FTS5 index
                self._add_to_fts(batch_ids, batch_chunks)
                
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

    def search_fts(
        self,
        query: str,
        n_results: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Search for chunks using FTS5 full-text search with BM25 ranking.
        
        This is much faster than in-memory BM25 as the index is pre-built.
        
        Args:
            query: Search query (supports FTS5 syntax like AND, OR, NOT, "phrases")
            n_results: Maximum number of results to return
            
        Returns:
            List of (chunk, bm25_score) tuples sorted by relevance
        """
        cursor = self._fts_conn.cursor()
        
        # Escape special FTS5 characters and build query
        # Split into words, escape each, join with implicit AND
        words = query.split()
        if not words:
            return []
        
        # Escape special characters for FTS5
        escaped_words = []
        for word in words:
            # Remove FTS5 special chars, keep alphanumeric
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word:
                escaped_words.append(f'"{clean_word}"')
        
        if not escaped_words:
            return []
        
        # Use OR for better recall (AND is too strict for short queries)
        fts_query = ' OR '.join(escaped_words)
        
        try:
            # BM25 returns negative scores (more negative = more relevant)
            # We negate to get positive scores where higher = better
            cursor.execute("""
                SELECT content, -bm25(chunks_fts) as score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
            """, (fts_query, n_results))
            
            results = [(row[0], row[1]) for row in cursor.fetchall()]
            
        except sqlite3.OperationalError as e:
            # If query syntax is invalid, try simpler approach
            logger.debug(f"FTS5 query failed: {e}, trying simpler query")
            simple_query = ' OR '.join(f'"{w}"' for w in words if w.isalnum())
            if simple_query:
                cursor.execute("""
                    SELECT content, -bm25(chunks_fts) as score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY score DESC
                    LIMIT ?
                """, (simple_query, n_results))
                results = [(row[0], row[1]) for row in cursor.fetchall()]
            else:
                results = []
        
        # Normalize scores to 0-1 range
        if results:
            max_score = max(score for _, score in results) if results else 1.0
            if max_score > 0:
                results = [(chunk, score / max_score) for chunk, score in results]
        
        return results

    def size(self) -> int:
        """Get the number of chunks in the database."""
        return self._collection.count()

    def clear(self) -> None:
        """Clear all data from the collection and FTS index."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear FTS5 index
        cursor = self._fts_conn.cursor()
        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute("DELETE FROM indexed_chunks")
        self._fts_conn.commit()
        
        logger.info("Vector database and FTS index cleared")

    def rebuild_fts_index(self, show_progress: bool = True) -> None:
        """
        Rebuild FTS5 index from ChromaDB data.
        
        Useful if FTS index is missing, corrupted, or out of sync.
        
        Args:
            show_progress: Whether to log progress
        """
        # Get all documents from ChromaDB (without embeddings for speed)
        results = self._collection.get(include=['documents'])
        
        if not results['documents']:
            logger.info("No documents to index")
            return
        
        chunks = results['documents']
        chunk_ids = results['ids']
        
        logger.info(f"Rebuilding FTS index for {len(chunks)} documents...")
        
        # Clear existing FTS data
        cursor = self._fts_conn.cursor()
        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute("DELETE FROM indexed_chunks")
        self._fts_conn.commit()
        
        # Add all chunks in batches
        batch_size = 500
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]
            self._add_to_fts(batch_ids, batch_chunks)
            
            if show_progress:
                processed = min(i + batch_size, total)
                logger.info(f"FTS indexed {processed}/{total} documents")
        
        logger.info("FTS index rebuild complete")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection '{self.collection_name}'")
