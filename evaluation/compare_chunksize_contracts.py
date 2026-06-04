"""
Qualitative chunk-size comparison on real long documents (contracts).

Ingests each document at several chunk sizes into separate tables, then runs the same
questions through retrieve + generate so the answers can be compared side by side.
There is no ground truth here — the point is to eyeball whether bigger chunks give
better-grounded, more complete answers on long, cross-referencing legal text (the
case that motivates parent-document retrieval).

    docker compose exec -T api python -m evaluation.compare_chunksize_contracts
"""

import glob
import logging
import os

logging.basicConfig(level=logging.WARNING)

from rag_lite.config import Config
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite import loaders, chunking

CONTRACTS_DIR = os.getenv("CONTRACTS_DIR", "/app/contracts")
SIZES = [256, 512, 1024]            # overlap is ~12% of size
TOP_N = 4
MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Generic but dangling-reference-prone questions, matched to doc type by filename hint.
QUESTIONS = {
    "lease": [
        "What is the monthly or annual rent, and when is it payable?",
        "What is the notice period to terminate, and how must notice be given?",
        "Who is responsible for repairs and maintenance?",
    ],
    "employment": [
        "What is the notice period for termination by either party?",
        "Are there any non-compete or restrictive covenants after employment ends?",
        "What is the salary or remuneration?",
    ],
    "offer": [
        "What is the salary and any bonus or equity?",
        "What is the start date and probation period?",
        "What are the conditions of the offer?",
    ],
    "mortgage": [
        "What is the loan amount and the interest rate?",
        "What is the term of the mortgage and the monthly repayment?",
        "What are the early repayment charges?",
    ],
}


def doc_type(name: str) -> str:
    n = name.lower()
    if "lease" in n or "tenan" in n or "apartment" in n or "renewal" in n:
        return "lease"
    if "employment" in n or "contract" in n:
        return "employment"
    if "offer" in n:
        return "offer"
    if "mortgage" in n:
        return "mortgage"
    return "employment"


def short(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def main() -> None:
    files = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.pdf")))
    print(f"Documents: {len(files)} | sizes: {SIZES} | model: {MODEL}\n")

    # Build one pipeline per chunk size; index every readable doc into it.
    pipes = {}
    loaded = []  # (filename, doc_type)
    for size in SIZES:
        cfg = Config.from_env()
        cfg.model.embedding_model = MODEL
        cfg.storage.embedding_dim = DIM
        cfg.storage.table_name = f"contracts_{size}"
        cfg.storage.collection_name = "contracts"
        cfg.retrieval.use_hybrid_search = False
        cfg.retrieval.use_reranking = False
        pipe = RAGPipeline(cfg)
        pipes[size] = pipe

    for path in files:
        name = os.path.basename(path)
        try:
            segs, meta = loaders.load_document(path)
        except Exception as e:
            print(f"  skip {name[:45]} ({type(e).__name__})")
            continue
        loaded.append((name, doc_type(name)))
        for size in SIZES:
            pipe = pipes[size]
            chunks = chunking.chunk_segments(
                segs, source=name, title=meta["title"], uri=name,
                embedding_model=MODEL, max_tokens=size, overlap=max(16, size // 8),
            )
            pipe.vector_db.add_chunks(chunks, show_progress=False)

    for size in SIZES:
        print(f"[size {size}] indexed {pipes[size].vector_db.size()} chunks")
    print()

    # Run each doc's questions against each chunk size.
    for name, dtype in loaded:
        print("#" * 88)
        print(f"# {name}  ({dtype})")
        print("#" * 88)
        for q in QUESTIONS.get(dtype, []):
            print(f"\nQ: {q}")
            for size in SIZES:
                pipe = pipes[size]
                # Restrict retrieval to this document by filtering after retrieve.
                results = [r for r in pipe.retrieve(q, top_n=TOP_N * 3,
                                                    use_hybrid_search=False, use_reranking=False)
                           if r.source == name][:TOP_N]
                if not results:
                    print(f"  [{size:>4}] (no retrieval)")
                    continue
                try:
                    ans = "".join(pipe.generate(q, results, stream=False))
                except Exception as e:
                    ans = f"<gen error: {e}>"
                pages = ",".join(str(r.page) for r in results if r.page)
                print(f"  [{size:>4}] p.{pages or '?'}  {short(ans)}")

    for pipe in pipes.values():
        pipe.close()


if __name__ == "__main__":
    main()
