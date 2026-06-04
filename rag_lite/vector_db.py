"""
Vector database implementation using PostgreSQL + pgvector.

Uses a single Postgres table — pgvector HNSW for ANN search, a tsvector GENERATED
column for full-text search. One store, one set of indexes, ACID guarantees.

Connections come from a `psycopg_pool.ConnectionPool` (psycopg 3) so the store is
safe to share across concurrent requests in the HTTP service: each operation borrows
a connection for the duration of its transaction and returns it to the pool.
"""

import json
import logging
import hashlib
import re
from typing import List, Tuple, Optional, Union

import numpy as np
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import tuple_row
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from .types import IngestChunk, Parent, RetrievedChunk
from .utils import get_device

logger = logging.getLogger(__name__)

# Columns selected for retrieval, in order. Centralized so search/search_fts stay in sync.
_RETRIEVE_COLS = "chunk_id, content, source, title, uri, page, parent_id, metadata"


def _row_to_retrieved(row: tuple, score: float) -> RetrievedChunk:
    """Build a RetrievedChunk from a row selected with _RETRIEVE_COLS (+ score)."""
    chunk_id, content, source, title, uri, page, parent_id, metadata = row
    return RetrievedChunk(
        content=content,
        score=score,
        chunk_id=chunk_id,
        source=source,
        title=title,
        uri=uri,
        page=page,
        parent_id=parent_id,
        metadata=metadata or {},
    )


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
        table_name: str = "chunks",
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
            table_name:      Physical table name. The embedding column's dimension is
                             fixed per table, so models of different dimensionality
                             (e.g. bge-base=768 vs bge-large=1024) must use separate
                             tables. Validated to a safe identifier.
        """
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError(f"Invalid table_name: {table_name!r}")
        self.embedding_model = embedding_model
        self.pg_dsn = pg_dsn
        self.collection_name = collection_name
        self.max_chunk_chars = max_chunk_chars or self.MAX_CHUNK_CHARS
        self.embedding_dim = embedding_dim
        self.table_name = table_name

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
        """Create extension, table, indexes, and metadata columns idempotently."""
        t = self.table_name
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {t} (
                        chunk_id    TEXT                    NOT NULL,
                        collection  TEXT                    NOT NULL,
                        content     TEXT                    NOT NULL,
                        embedding   vector({self.embedding_dim}) NOT NULL,
                        content_tsv tsvector GENERATED ALWAYS AS
                                    (to_tsvector('english', content)) STORED,
                        PRIMARY KEY (chunk_id, collection)
                    )
                """)

                # Provenance columns — added via ALTER so existing tables migrate in
                # place. All nullable/defaulted, so pre-Phase-2 rows remain valid.
                cur.execute(f"""
                    ALTER TABLE {t}
                        ADD COLUMN IF NOT EXISTS source      TEXT,
                        ADD COLUMN IF NOT EXISTS title       TEXT,
                        ADD COLUMN IF NOT EXISTS uri         TEXT,
                        ADD COLUMN IF NOT EXISTS page        INT,
                        ADD COLUMN IF NOT EXISTS chunk_index INT,
                        ADD COLUMN IF NOT EXISTS parent_id   TEXT,
                        ADD COLUMN IF NOT EXISTS metadata    JSONB DEFAULT '{{}}'::jsonb,
                        ADD COLUMN IF NOT EXISTS created_at  TIMESTAMPTZ DEFAULT now()
                """)

                # Parent blocks for small-to-big retrieval (text only, not embedded).
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {t}_parents (
                        parent_id  TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        source     TEXT,
                        title      TEXT,
                        uri        TEXT,
                        page       INT,
                        content    TEXT NOT NULL,
                        PRIMARY KEY (parent_id, collection)
                    )
                """)

                # HNSW index: fast approximate nearest-neighbour with cosine distance
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {t}_embedding_hnsw_idx
                    ON {t} USING hnsw (embedding vector_cosine_ops)
                """)

                # GIN index for full-text search
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {t}_tsv_gin_idx
                    ON {t} USING gin (content_tsv)
                """)

                # Index supporting metadata filters and delete-by-source.
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {t}_source_idx
                    ON {t} (collection, source)
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

    @staticmethod
    def _generate_id(text: str, source: str) -> str:
        """Content+source hash. Including source keeps identical text from different
        documents distinct, so each retains its own provenance and citation."""
        key = f"{source}\x00{text}".encode()
        return hashlib.sha256(key).hexdigest()[:16]

    def _embed(self, text: str) -> np.ndarray:
        # No char truncation here: the chunker sizes inputs to the model's token
        # budget. The model still token-truncates as a final safety net.
        return self._get_model().encode(text, convert_to_numpy=True, show_progress_bar=False)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        return self._get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: Union[List[str], List[IngestChunk]],
        show_progress: bool = True,
        batch_size: int = 256,
    ) -> int:
        """
        Batch-embed and insert chunks with their provenance; duplicates are skipped.

        Accepts either plain strings (wrapped as inline chunks for backward
        compatibility) or IngestChunk objects. Returns rows actually inserted
        (excludes ON CONFLICT skips).
        """
        # Normalize to IngestChunk
        norm: List[IngestChunk] = [
            c if isinstance(c, IngestChunk) else IngestChunk(text=c) for c in chunks
        ]

        # Deduplicate within the incoming batch by (source, text) hash.
        by_id: dict = {}
        for c in norm:
            cid = self._generate_id(c.text, c.source)
            by_id.setdefault(cid, c)

        dropped = len(norm) - len(by_id)
        if dropped:
            logger.info(f"Deduplicated {dropped} duplicate chunks from input")

        items = list(by_id.items())  # [(id, IngestChunk), ...]
        if not items:
            logger.info("No chunks to insert")
            return 0

        logger.info(f"Inserting up to {len(items)} chunks into pgvector…")
        total = len(items)
        inserted_total = 0

        for i in range(0, total, batch_size):
            batch = items[i : i + batch_size]
            embeddings = self._embed_batch([c.text for _, c in batch])

            rows = [
                (
                    cid, self.collection_name, c.text, emb,
                    c.source, c.title, c.uri, c.page, c.chunk_index, c.parent_id,
                    json.dumps(c.metadata or {}),
                )
                for (cid, c), emb in zip(batch, embeddings)
            ]

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"""
                        INSERT INTO {self.table_name}
                            (chunk_id, collection, content, embedding,
                             source, title, uri, page, chunk_index, parent_id, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id, collection) DO NOTHING
                        """,
                        rows,
                    )
                    inserted_total += cur.rowcount

            if show_progress:
                processed = min(i + batch_size, total)
                logger.info(f"Processed {processed}/{total} chunks ({inserted_total} inserted)")

        return inserted_total

    def add_parents(self, parents: List[Parent]) -> int:
        """Store parent blocks (text only, not embedded). Duplicates skipped."""
        if not parents:
            return 0
        rows = [
            (p.parent_id, self.collection_name, p.source, p.title, p.uri, p.page, p.content)
            for p in {p.parent_id: p for p in parents}.values()
        ]
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {self.table_name}_parents
                        (parent_id, collection, source, title, uri, page, content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (parent_id, collection) DO NOTHING
                    """,
                    rows,
                )
        return len(rows)

    def get_parents(self, parent_ids: List[str]) -> dict:
        """Fetch parents by id → {parent_id: Parent}. Missing ids are simply absent."""
        if not parent_ids:
            return {}
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    f"""
                    SELECT parent_id, content, source, title, uri, page
                    FROM {self.table_name}_parents
                    WHERE collection = %s AND parent_id = ANY(%s)
                    """,
                    (self.collection_name, list(parent_ids)),
                )
                return {
                    row[0]: Parent(parent_id=row[0], content=row[1], source=row[2],
                                   title=row[3], uri=row[4], page=row[5])
                    for row in cur.fetchall()
                }

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_all(self) -> List[Tuple[str, List[float]]]:
        """Return all (content, embedding) pairs for this collection."""
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    f"SELECT content, embedding FROM {self.table_name} WHERE collection = %s",
                    (self.collection_name,),
                )
                return [(row[0], list(row[1])) for row in cur.fetchall()]

    def search(self, query: str, n_results: int = 10, source: Optional[str] = None) -> List[RetrievedChunk]:
        """
        Semantic search using pgvector HNSW (cosine).

        Returns RetrievedChunks (with provenance) sorted by descending similarity.
        Similarity = 1 - cosine_distance, so 1.0 is identical.

        `source` optionally restricts the search to one source document.
        """
        q_emb = self._embed(query)
        src_clause = "AND source = %s" if source else ""
        params = [q_emb, self.collection_name]
        if source:
            params.append(source)
        params += [q_emb, n_results]
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    f"""
                    SELECT {_RETRIEVE_COLS},
                           1 - (embedding <=> %s) AS similarity
                    FROM {self.table_name}
                    WHERE collection = %s {src_clause}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    params,
                )
                return [_row_to_retrieved(row[:-1], float(row[-1])) for row in cur.fetchall()]

    def search_fts(self, query: str, n_results: int = 50) -> List[RetrievedChunk]:
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
                    f"""
                    SELECT {_RETRIEVE_COLS},
                           ts_rank(content_tsv, plainto_tsquery('english', %s)) AS score
                    FROM {self.table_name}
                    WHERE collection = %s
                      AND content_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, self.collection_name, query, n_results),
                )
                results = [_row_to_retrieved(row[:-1], float(row[-1])) for row in cur.fetchall()]

        if results:
            max_score = max(r.score for r in results)
            if max_score > 0:
                for r in results:
                    r.score = r.score / max_score

        return results

    # ------------------------------------------------------------------
    # Admin operations
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Count chunks in this collection."""
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.table_name} WHERE collection = %s",
                    (self.collection_name,),
                )
                return cur.fetchone()[0]

    def clear(self) -> None:
        """Delete all chunks in this collection."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE collection = %s",
                    (self.collection_name,),
                )
                cur.execute(
                    f"DELETE FROM {self.table_name}_parents WHERE collection = %s",
                    (self.collection_name,),
                )
        logger.info(f"Cleared collection '{self.collection_name}'")

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks for one source document. Returns rows deleted."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE collection = %s AND source = %s",
                    (self.collection_name, source),
                )
                deleted = cur.rowcount
                cur.execute(
                    f"DELETE FROM {self.table_name}_parents WHERE collection = %s AND source = %s",
                    (self.collection_name, source),
                )
        logger.info(f"Deleted {deleted} chunks for source '{source}'")
        return deleted

    def list_sources(self) -> List[Tuple[str, int]]:
        """Return (source, chunk_count) for each distinct source in this collection."""
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(
                    f"""
                    SELECT source, COUNT(*) AS n
                    FROM {self.table_name}
                    WHERE collection = %s AND source IS NOT NULL
                    GROUP BY source
                    ORDER BY source
                    """,
                    (self.collection_name,),
                )
                return [(row[0], int(row[1])) for row in cur.fetchall()]

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
