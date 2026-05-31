"""
Vector database implementation using PostgreSQL + pgvector.

Uses a single Postgres table — pgvector HNSW for ANN search, a tsvector GENERATED
column for full-text search. One store, one set of indexes, ACID guarantees.

Connections come from a `psycopg_pool.ConnectionPool` (psycopg 3) so the store is
safe to share across concurrent requests in the HTTP service: each operation borrows
a connection for the duration of its transaction and returns it to the pool.
"""

import logging
import hashlib
from typing import List, Tuple, Optional

import numpy as np
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import tuple_row
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from .utils import get_device

logger = logging.getLogger(__name__)


class VectorDB:
    """
    PostgreSQL + pgvector vector database.

    Schema
    ------
    chunks (chunk_id, collection, content, embedding vector(N), content_tsv tsvector)
      - HNSW index on embedding (cosine)
      - GIN  index on content_tsv (full-text)

    The tsvector column is GENERATED ALWAYS, so FTS is always in sync — no separate
    sidecar, no manual rebuild.
    """

    MAX_CHUNK_CHARS = 500

    def __init__(
        self,
        embedding_model: str,
        pg_dsn: str,
        collection_name: str = "rag_lite",
        max_chunk_chars: Optional[int] = None,
        embedding_dim: int = 768,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
    ):
        """
        Args:
            embedding_model: HuggingFace model name (e.g. "BAAI/bge-base-en-v1.5")
            pg_dsn:          PostgreSQL connection string
                             e.g. "postgresql://user:pass@localhost:5432/rag"
            collection_name: Logical namespace stored as a column — multiple
                             collections share one table.
            max_chunk_chars: Hard truncation limit per chunk (default 500).
            embedding_dim:   Dimension of the embedding model output.
                             Must match the model. Default 768 (bge-base-en-v1.5).
            pool_min_size:   Idle connections the pool keeps open.
            pool_max_size:   Hard ceiling on concurrent connections.
        """
        self.embedding_model = embedding_model
        self.pg_dsn = pg_dsn
        self.collection_name = collection_name
        self.max_chunk_chars = max_chunk_chars or self.MAX_CHUNK_CHARS
        self.embedding_dim = embedding_dim

        self._model: Optional[SentenceTransformer] = None

        # The `vector` extension must exist before any connection registers the
        # pgvector adapters (register_vector looks up the type OID). On a fresh
        # database the extension is absent, so create it once on a standalone
        # connection *before* the pool opens — otherwise every pooled connection
        # fails its configure step and the pool times out.
        self._ensure_vector_extension(pg_dsn)

        # `configure` runs once per physical connection, registering the pgvector
        # type adapters so numpy arrays bind directly to `vector` columns.
        self._pool = ConnectionPool(
            conninfo=pg_dsn,
            min_size=pool_min_size,
            max_size=pool_max_size,
            configure=self._configure_connection,
            open=True,
        )
        self._init_schema()

        logger.info(
            f"pgvector DB ready — collection='{collection_name}', "
            f"dim={embedding_dim}, pool={pool_min_size}-{pool_max_size}, "
            f"docs={self.size()}"
        )

    @staticmethod
    def _ensure_vector_extension(dsn: str) -> None:
        """Create the pgvector extension if absent (idempotent, runs once at init)."""
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    @staticmethod
    def _configure_connection(conn) -> None:
        register_vector(conn)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create extension, table, and indexes idempotently."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id    TEXT                    NOT NULL,
                        collection  TEXT                    NOT NULL,
                        content     TEXT                    NOT NULL,
                        embedding   vector({self.embedding_dim}) NOT NULL,
                        content_tsv tsvector GENERATED ALWAYS AS
                                    (to_tsvector('english', content)) STORED,
                        PRIMARY KEY (chunk_id, collection)
                    )
                """)

                # HNSW index: fast approximate nearest-neighbour with cosine distance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                """)

                # GIN index for full-text search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx
                    ON chunks USING gin (content_tsv)
                """)
            # `with conn` commits on clean exit.

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            device = get_device()
            logger.info(f"Loading embedding model: {self.embedding_model} on {device}")
            self._model = SentenceTransformer(self.embedding_model, device=device)
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self.embedding_dim:
                raise ValueError(
                    f"Model produces {actual_dim}-dim embeddings but "
                    f"embedding_dim={self.embedding_dim} was configured. "
                    "Update StorageConfig.embedding_dim or clear the database."
                )
            logger.info(f"Embedding model loaded (dim={actual_dim})")
        return self._model

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.max_chunk_chars:
            return text
        truncated = text[:self.max_chunk_chars]
        last_space = truncated.rfind(" ")
        if last_space > self.max_chunk_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated

    def _generate_id(self, chunk: str) -> str:
        return hashlib.sha256(chunk.encode()).hexdigest()[:16]

    def _embed(self, text: str) -> np.ndarray:
        text = self._truncate_text(text)
        return self._get_model().encode(text, convert_to_numpy=True, show_progress_bar=False)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        texts = [self._truncate_text(t) for t in texts]
        return self._get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: str) -> None:
        """Add a single chunk (skipped if duplicate)."""
        chunk_id = self._generate_id(chunk)
        embedding = self._embed(chunk)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, collection, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chunk_id, collection) DO NOTHING
                    """,
                    (chunk_id, self.collection_name, chunk, embedding),
                )

    def add_chunks(
        self,
        chunks: List[str],
        show_progress: bool = True,
        batch_size: int = 256,
    ) -> int:
        """
        Batch-embed and insert chunks; duplicates are silently skipped.

        Returns the number of rows actually inserted (excludes ON CONFLICT skips).
        """
        # Deduplicate within the incoming list (same text → same SHA256 id)
        chunk_id_map: dict = {}
        for chunk in chunks:
            cid = self._generate_id(chunk)
            if cid not in chunk_id_map:
                chunk_id_map[cid] = chunk

        dropped = len(chunks) - len(chunk_id_map)
        if dropped:
            logger.info(f"Deduplicated {dropped} duplicate chunks from input")

        items = list(chunk_id_map.items())  # [(id, text), ...]
        if not items:
            logger.info("All chunks already exist in the database")
            return 0

        logger.info(f"Inserting up to {len(items)} chunks into pgvector…")
        total = len(items)
        inserted_total = 0

        for i in range(0, total, batch_size):
            batch = items[i : i + batch_size]
            batch_ids = [cid for cid, _ in batch]
            batch_texts = [text for _, text in batch]
            embeddings = self._embed_batch(batch_texts)

            rows = [
                (cid, self.collection_name, text, emb)
                for cid, text, emb in zip(batch_ids, batch_texts, embeddings)
            ]

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chunks (chunk_id, collection, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (chunk_id, collection) DO NOTHING
                        """,
                        rows,
                    )
                    inserted_total += cur.rowcount

            if show_progress:
                processed = min(i + batch_size, total)
                logger.info(f"Processed {processed}/{total} chunks ({inserted_total} inserted)")

        return inserted_total

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_all(self) -> List[Tuple[str, List[float]]]:
        """Return all (content, embedding) pairs for this collection."""
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    "SELECT content, embedding FROM chunks WHERE collection = %s",
                    (self.collection_name,),
                )
                return [(row[0], list(row[1])) for row in cur.fetchall()]

    def search(self, query: str, n_results: int = 10) -> List[Tuple[str, float]]:
        """
        Semantic search using pgvector HNSW (cosine).

        Returns list of (chunk, similarity) sorted by descending similarity.
        Similarity = 1 - cosine_distance, so 1.0 is identical.
        """
        q_emb = self._embed(query)
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    """
                    SELECT content,
                           1 - (embedding <=> %s) AS similarity
                    FROM chunks
                    WHERE collection = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (q_emb, self.collection_name, q_emb, n_results),
                )
                return [(row[0], float(row[1])) for row in cur.fetchall()]

    def search_fts(self, query: str, n_results: int = 50) -> List[Tuple[str, float]]:
        """
        Full-text search using PostgreSQL tsvector + ts_rank.

        Scores are normalized to [0, 1] to match the contract expected by
        the RRF fusion layer.
        """
        if not query.strip():
            return []

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    """
                    SELECT content,
                           ts_rank(content_tsv, plainto_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE collection = %s
                      AND content_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, self.collection_name, query, n_results),
                )
                results = [(row[0], float(row[1])) for row in cur.fetchall()]

        if results:
            max_score = max(s for _, s in results)
            if max_score > 0:
                results = [(chunk, s / max_score) for chunk, s in results]

        return results

    # ------------------------------------------------------------------
    # Admin operations
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Count chunks in this collection."""
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE collection = %s",
                    (self.collection_name,),
                )
                return cur.fetchone()[0]

    def clear(self) -> None:
        """Delete all chunks in this collection."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE collection = %s",
                    (self.collection_name,),
                )
        logger.info(f"Cleared collection '{self.collection_name}'")

    def delete_collection(self) -> None:
        """Alias for clear() — removes all rows for this collection."""
        self.clear()
        logger.info(f"Deleted collection '{self.collection_name}'")

    def ping(self) -> bool:
        """Lightweight liveness check — borrows a connection and runs SELECT 1."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True

    def rebuild_fts_index(self, show_progress: bool = True) -> None:
        """
        No-op: the tsvector column is GENERATED ALWAYS, so it is always
        in sync with content — no manual rebuild is ever needed.
        """
        logger.info(
            "FTS index is a GENERATED column in PostgreSQL — always in sync, "
            "no rebuild required."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the connection pool."""
        pool = getattr(self, "_pool", None)
        if pool is not None and not pool.closed:
            pool.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
