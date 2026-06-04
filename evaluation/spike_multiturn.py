"""
Spike: multi-turn (conversational) RAG.

Runs scripted conversations whose later turns are follow-ups that only make sense
with the prior turns ("and what evidence do I need?"). For each turn it shows:
  - the raw follow-up,
  - the CONDENSED standalone question (what retrieval actually runs on),
  - the answer + cited source pages.

The point is to see whether condensation resolves references so retrieval + answers
stay on-topic across turns.

    docker compose exec -T api python -m evaluation.spike_multiturn
"""

import glob
import logging
import os

logging.basicConfig(level=logging.WARNING)

from rag_lite.config import Config
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite import loaders, chunking

CONTRACTS_DIR = os.getenv("CONTRACTS_DIR", "/app/contracts")
MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
TABLE = "parentcmp"
COLLECTION = "contracts"

# (filename-substring, [turns]) — later turns deliberately reference earlier ones.
CONVERSATIONS = [
    ("Argo", [
        "Can I end my tenancy early?",
        "What evidence do I need to provide for that?",          # "that" = early termination
        "And how much notice is required?",                       # elliptical follow-up
        "What if I just want to move to a nicer flat?",           # tests the conditions
    ]),
    ("Employment Contract", [
        "Am I restricted from working for competitors after I leave?",
        "For how long?",                                          # bare follow-up
        "Does that also stop me from contacting clients?",        # "that" = the restriction
    ]),
    ("Mortgage", [
        "What's my interest rate?",
        "What happens to it after the fixed period?",             # "it" = the rate
        "And if I overpay before then, is there a charge?",       # references the fixed period
    ]),
]


def find_doc(sub: str):
    for p in sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.pdf"))):
        if sub.lower() in os.path.basename(p).lower():
            return p
    return None


def main() -> None:
    cfg = Config.from_env()
    cfg.model.embedding_model = MODEL
    cfg.storage.embedding_dim = DIM
    cfg.storage.table_name = TABLE
    cfg.storage.collection_name = COLLECTION
    pipe = RAGPipeline(cfg)

    # Ensure docs are indexed (reuse if already present).
    for sub, _ in CONVERSATIONS:
        path = find_doc(sub)
        if not path:
            continue
        name = os.path.basename(path)
        if pipe.vector_db.search("x", n_results=1, source=name):
            continue
        segs, meta = loaders.load_document(path, uri=name)
        children, parents = chunking.chunk_hierarchical(
            segs, source=name, title=meta["title"], uri=name, embedding_model=MODEL,
            parent_max_tokens=cfg.chunking.parent_max_tokens,
            child_max_tokens=cfg.chunking.max_tokens, child_overlap=cfg.chunking.overlap,
        )
        pipe.vector_db.add_parents(parents)
        pipe.vector_db.add_chunks(children, show_progress=False)
    print(f"Indexed {pipe.vector_db.size()} children | model: {cfg.model.language_model}\n")

    for sub, turns in CONVERSATIONS:
        print("#" * 92)
        print(f"# Conversation over: {sub}")
        print("#" * 92)
        history = []
        for q in turns:
            standalone, retrieved, resp = pipe.chat(history, q, stream=False)
            answer = "".join(resp)
            pages = ",".join(str(c.page) for c in retrieved if c.page)
            condensed = "(no change)" if standalone.strip() == q.strip() else standalone
            print(f"\nUSER: {q}")
            print(f"   ↳ condensed: {condensed}")
            print(f"   ↳ sources p.{pages or '?'}")
            print(f"   BOT: {' '.join(answer.split())[:300]}")
            history.append(("user", q))
            history.append(("assistant", answer))

    pipe.close()


if __name__ == "__main__":
    main()
