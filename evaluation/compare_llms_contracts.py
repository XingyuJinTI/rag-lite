"""
Generation-model comparison on real contracts.

Retrieval is held fixed (bge-m3, chunk 256, scoped to the target document); only the
LLM changes. So this isolates answer quality: given the same retrieved clauses, which
model produces the most faithful, complete answer — and how fast on this hardware.

    docker compose exec -T api python -m evaluation.compare_llms_contracts
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
TABLE = "llmcmp"
COLLECTION = "contracts"
CHUNK = 256
TOP_N = 4

import os as _os
_ROUND = _os.getenv("LLM_ROUND", "1")
if _ROUND == "3":
    LLMS = [  # survivors, for the hard tiebreaker
        ("qwen2.5-7b", "qwen2.5:7b"),
        ("gemma3-12b", "gemma3:12b"),
        ("qwen2.5-14b", "qwen2.5:14b"),
        ("qwen3-14b", "qwen3:14b"),
    ]
elif _ROUND == "2":
    LLMS = [
        ("qwen2.5-7b", "qwen2.5:7b"),     # carry-forward leaders
        ("qwen2.5-14b", "qwen2.5:14b"),
        ("qwen3-14b", "qwen3:14b"),       # alternatives
        ("phi4-14b", "phi4"),
        ("gemma3-12b", "gemma3:12b"),
    ]
else:
    LLMS = [
        ("qwen2.5-7b", "qwen2.5:7b"),
        ("llama3.1-8b", "llama3.1:8b"),
        ("qwen2.5-14b", "qwen2.5:14b"),
    ]

# qwen3 reasons by default; disable for a fair, fast extraction comparison.
_NO_THINK = {"qwen3:14b"}

# Harder questions: numerical application, statutory reasoning, false-positives,
# synthesis, and deliberately-absent topics (to test abstention).
HARD_QUESTIONS = {
    "lease": [
        "Am I responsible for repairing the heating system, or is the landlord?",
        "What happens if I don't pay the rent on time?",
        "Is keeping a pet allowed under this agreement?",   # likely absent → abstain
    ],
    "employment": [
        "How much notice must the Company give to terminate someone who has worked 5 years?",
        "Immediately after I leave, can I join a direct competitor, and for how long am I restricted?",
        "Does the contract mention anything about remote or hybrid working?",  # likely absent
    ],
    "offer": [
        "What is my guaranteed total compensation, and is the bonus guaranteed?",
        "Under what conditions can this offer be withdrawn?",
    ],
    "mortgage": [
        "If I repay £100,000 early in 2027, how much is the early repayment charge?",  # 2% = £2,000
        "After the fixed-rate period ends, what happens to my interest rate and monthly payment?",
    ],
}

QUESTIONS = {
    "lease": [
        "What is the rent, and when is it payable?",
        "What is the notice period to terminate, and how must notice be given?",
        "Who is responsible for repairs and maintenance?",
    ],
    "employment": [
        "What is the notice period for termination by either party?",
        "Are there non-compete or restrictive covenants after employment ends?",
        "What is the salary?",
    ],
    "offer": [
        "What is the salary and any bonus?",
        "What is the start date and probation period?",
    ],
    "mortgage": [
        "What is the loan amount and interest rate?",
        "What are the early repayment charges?",
    ],
}


def doc_type(name: str) -> str:
    n = name.lower()
    if "mortgage" in n:
        return "mortgage"
    if "lease" in n or "tenan" in n or "apartment" in n or "renewal" in n:
        return "lease"
    if "offer" in n:
        return "offer"
    if "employment" in n or "contract" in n:
        return "employment"
    return "employment"


def short(text: str, n: int = 260) -> str:
    return " ".join(text.split())[:n]


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
            segs, meta = loaders.load_document(path)
        except Exception as e:
            print(f"skip {name[:40]} ({type(e).__name__})")
            continue
        if pipe.vector_db.search("x", n_results=1, source=name):
            loaded.append((name, doc_type(name)))
            continue
        chunks = chunking.chunk_segments(segs, source=name, title=meta["title"], uri=name,
                                         embedding_model=MODEL, max_tokens=CHUNK, overlap=32)
        pipe.vector_db.add_chunks(chunks, show_progress=False)
        loaded.append((name, doc_type(name)))
    print(f"Indexed {pipe.vector_db.size()} chunks from {len(loaded)} docs | models: {[m for m,_ in LLMS]}\n")

    timings = {label: [] for label, _ in LLMS}
    for name, dtype in loaded:
        print("#" * 90)
        print(f"# {name}  ({dtype})")
        print("#" * 90)
        qset = HARD_QUESTIONS if _ROUND == "3" else QUESTIONS
        for q in qset.get(dtype, []):
            # Same retrieved context for every model: scoped to this document.
            ctx = pipe.vector_db.search(q, n_results=TOP_N, source=name)
            pages = ",".join(str(c.page) for c in ctx if c.page)
            print(f"\nQ: {q}   (ctx p.{pages or '?'})")
            for label, model in LLMS:
                t0 = time.time()
                qq = q + " /no_think" if model in _NO_THINK else q
                try:
                    ans = "".join(generate_response(qq, ctx, model, stream=False, timeout=300))
                except Exception as e:
                    ans = f"<error: {e}>"
                dt = time.time() - t0
                timings[label].append(dt)
                print(f"  [{label:<12} {dt:5.1f}s] {short(ans)}")

    print("\n" + "=" * 60)
    print("Avg latency per answer:")
    for label, _ in LLMS:
        ts = timings[label]
        if ts:
            print(f"  {label:<14} {sum(ts)/len(ts):5.1f}s  (n={len(ts)})")
    pipe.close()


if __name__ == "__main__":
    main()
