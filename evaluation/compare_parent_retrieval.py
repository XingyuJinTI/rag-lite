"""
Child-only vs parent-document (small-to-big) retrieval on real long documents.

Holds matching + model fixed; only the *context unit* changes: the precise child
chunks vs their expanded parent blocks. Tests whether parent context produces more
complete, better-grounded answers on cross-referencing legal text.

    docker compose exec -T api python -m evaluation.compare_parent_retrieval
"""

import glob
import logging
import os
import time

logging.basicConfig(level=logging.WARNING)

from rag_lite.config import Config
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite import loaders, chunking
from rag_lite.generation import generate_response

CONTRACTS_DIR = os.getenv("CONTRACTS_DIR", "/app/contracts")
MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
LLM = os.getenv("LANGUAGE_MODEL", "qwen2.5:14b")
TABLE = "parentcmp"
COLLECTION = "contracts"
TOP_N = 3

# Context-dependent questions: a single child chunk is often insufficient because
# the answer relies on surrounding clauses / defined terms.
QUESTIONS = {
    "lease": [
        "What are the consequences if I pay the rent late?",
        "Under what conditions, and with how much notice, can I end the tenancy early?",
    ],
    "employment": [
        "Exactly what am I restricted from doing after I leave, and for how long?",
        "What happens to my pay and benefits if I'm terminated during probation?",
    ],
    "offer": [
        "What is my total potential compensation, and which parts are guaranteed?",
    ],
    "mortgage": [
        "What fees and charges might I pay over the life of this mortgage?",
    ],
}


def doc_type(name: str) -> str:
    n = name.lower()
    if "mortgage" in n:
        return "mortgage"
    if "lease" in n or "apartment" in n or "renewal" in n or "tenan" in n:
        return "lease"
    if "offer" in n:
        return "offer"
    return "employment"


def short(t: str, n: int = 300) -> str:
    return " ".join(t.split())[:n]


def main() -> None:
    cfg = Config.from_env()
    cfg.model.embedding_model = MODEL
    cfg.storage.embedding_dim = DIM
    cfg.storage.table_name = TABLE
    cfg.storage.collection_name = COLLECTION
    pipe = RAGPipeline(cfg)

    files = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.pdf")))
    loaded = []
    for path in files:
        name = os.path.basename(path)
        try:
            segs, meta = loaders.load_document(path, uri=name)
        except Exception as e:
            print(f"skip {name[:40]} ({type(e).__name__})")
            continue
        loaded.append((name, doc_type(name)))
        if pipe.vector_db.search("x", n_results=1, source=name):
            continue
        children, parents = chunking.chunk_hierarchical(
            segs, source=name, title=meta["title"], uri=name, embedding_model=MODEL,
            parent_max_tokens=cfg.chunking.parent_max_tokens,
            child_max_tokens=cfg.chunking.max_tokens, child_overlap=cfg.chunking.overlap,
        )
        pipe.vector_db.add_parents(parents)
        pipe.vector_db.add_chunks(children, show_progress=False)
    print(f"Indexed {pipe.vector_db.size()} children from {len(loaded)} docs | LLM: {LLM}\n")

    for name, dtype in loaded:
        print("#" * 92)
        print(f"# {name}  ({dtype})")
        print("#" * 92)
        for q in QUESTIONS.get(dtype, []):
            # Same scoped children for both modes; parent mode expands them.
            children = pipe.vector_db.search(q, n_results=TOP_N * 3, source=name)
            child_ctx = children[:TOP_N]
            parent_ctx = pipe._expand_to_parents(children, TOP_N)
            print(f"\nQ: {q}")
            for label, ctx in (("child ", child_ctx), ("parent", parent_ctx)):
                t0 = time.time()
                try:
                    ans = "".join(generate_response(q, ctx, LLM, stream=False, timeout=300))
                except Exception as e:
                    ans = f"<error: {e}>"
                dt = time.time() - t0
                toks = sum(len(c.content) for c in ctx) // 4
                print(f"  [{label} ctx≈{toks:>4}tok {dt:5.1f}s] {short(ans)}")

    pipe.close()


if __name__ == "__main__":
    main()
