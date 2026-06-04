"""
Retrieval-precision comparison on real contracts.

The contract experiment's outright failures were *retrieval misses* (the relevant
clause was never retrieved), not context problems. This harness compares retrieval
configurations and prints, per question, the top chunks each surfaces — so we can
judge which configuration actually pulls the right clause.

Configs: semantic-only · +cross-encoder rerank · +query-expansion · +both.
Output is retrieval (pages + snippets), not generation — directly measures precision.

    docker compose exec -T api python -m evaluation.compare_retrieval_precision
"""

import glob
import logging
import os

logging.basicConfig(level=logging.WARNING)

from rag_lite.config import Config, ModelConfig
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite import loaders, chunking
from rag_lite.retrieval import rerank_with_cross_encoder, expand_query

CONTRACTS_DIR = os.getenv("CONTRACTS_DIR", "/app/contracts")
MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
RERANKER = ModelConfig.RERANKER_BGE_BASE
EXPANSION_LLM = os.getenv("EXPANSION_LLM", "qwen2.5:7b")
TABLE = "parentcmp"          # reuse the already-indexed contracts
COLLECTION = "contracts"
TOP_N = 4
POOL = 20                    # candidate pool for rerank / expansion

QUESTIONS = {
    "lease": [
        "Under what conditions, and with how much notice, can I end the tenancy early?",
        "Can I sublet or assign the property to someone else?",
    ],
    "employment": [
        "What am I restricted from doing after I leave, and for how long?",
        "Am I entitled to a bonus, and how is it determined?",
    ],
    "mortgage": [
        "What are the early repayment charges?",
    ],
}


def doc_type(name: str) -> str:
    n = name.lower()
    if "mortgage" in n:
        return "mortgage"
    if "lease" in n or "apartment" in n or "renewal" in n or "tenan" in n:
        return "lease"
    if "offer" in n:
        return "employment"
    return "employment"


def snip(c) -> str:
    return f"p.{c.page} «{' '.join(c.content.split())[:70]}»"


def _merge(lists):
    """Union candidates from multiple queries, keep best score per chunk_id."""
    best = {}
    for lst in lists:
        for c in lst:
            if c.chunk_id not in best or c.score > best[c.chunk_id].score:
                best[c.chunk_id] = c
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def main() -> None:
    cfg = Config.from_env()
    cfg.model.embedding_model = MODEL
    cfg.storage.embedding_dim = DIM
    cfg.storage.table_name = TABLE
    cfg.storage.collection_name = COLLECTION
    pipe = RAGPipeline(cfg)
    vdb = pipe.vector_db

    files = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.pdf")))
    docs = []
    for path in files:
        name = os.path.basename(path)
        if not vdb.search("x", n_results=1, source=name):
            continue  # already indexed from prior runs
        docs.append((name, doc_type(name)))
    # ensure list of (name, type) even if all pre-indexed
    if not docs:
        for path in files:
            name = os.path.basename(path)
            docs.append((name, doc_type(name)))
    print(f"docs={len(docs)} | reranker={RERANKER} | expansion={EXPANSION_LLM}\n")

    for name, dtype in docs:
        print("#" * 90)
        print(f"# {name}  ({dtype})")
        print("#" * 90)
        for q in QUESTIONS.get(dtype, []):
            print(f"\nQ: {q}")
            semantic = vdb.search(q, n_results=POOL, source=name)
            reranked = rerank_with_cross_encoder(q, semantic, RERANKER)
            variants = expand_query(q, EXPANSION_LLM, num_alternatives=2)
            qexp = _merge([vdb.search(v, n_results=POOL, source=name) for v in variants])
            qexp_rr = rerank_with_cross_encoder(q, qexp, RERANKER)
            for label, res in (("semantic ", semantic), ("+rerank  ", reranked),
                               ("+qexp    ", qexp), ("+qexp+rr ", qexp_rr)):
                tops = " | ".join(snip(c) for c in res[:TOP_N])
                print(f"  {label}: {tops}")

    pipe.close()


if __name__ == "__main__":
    main()
