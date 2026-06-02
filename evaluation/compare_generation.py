"""
Generation-model comparison.

Retrieval is identical across models, so this isolates *generation quality*: given
the SAME retrieved context, how well does each LLM produce a grounded, correctly-cited
answer — and how fast on this hardware.

For each (model x question) it records:
  - correctness  : does the answer contain the expected fact (keyword match)
  - cited        : did it emit >=1 valid [n] marker (validated against sources)
  - declined     : for the unanswerable question, did it decline rather than fabricate
  - latency      : seconds, and approx tokens/sec

Run inside the API container (has deps + reaches host Ollama):
    docker compose exec -T api python -m evaluation.compare_generation
Assumes the collection already has the sample docs indexed.
"""

import time
from typing import List, Optional, Tuple

from rag_lite.config import Config
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite.generation import generate_response, validate_citations

MODELS = [
    ("Llama-3.2-1B (current)", "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"),
    ("Llama-3.1-8B",           "llama3.1:8b"),
    ("Qwen2.5-7B",             "qwen2.5:7b"),
]

# (question, expected_keywords, answerable). Answerable from the indexed sample docs.
QUESTIONS: List[Tuple[str, List[str], bool]] = [
    ("How many toes does a cat have on each back paw?", ["four", "4"], True),
    ("How many hours per day does a typical adult cat sleep?", ["twelve", "12", "sixteen", "16"], True),
    ("What is a group of cats called?", ["clowder"], True),
    ("How fast can a domestic cat run over short distances?", ["thirty", "30"], True),
    ("What lets cats see well at night?", ["tapetum"], True),
    ("What is the capital of France?", [], False),  # unanswerable from context → should decline
]

TIMEOUT = 240.0  # generous: 7-8B models on Mac CPU are slow


def _contains_any(text: str, keywords: List[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def _declined(text: str) -> bool:
    markers = ["don't have enough", "do not have enough", "not contain", "no information",
               "isn't in the context", "cannot answer", "can't answer", "context does not"]
    low = text.lower()
    return any(m in low for m in markers)


def main() -> None:
    config = Config.from_env()
    config.retrieval.use_hybrid_search = True
    config.retrieval.use_reranking = True
    config.retrieval.top_n = 3
    pipe = RAGPipeline(config)

    # Retrieve once per question — identical context for every model.
    retrieved = {q: pipe.retrieve(q) for q, _, _ in QUESTIONS}
    print(f"Indexed chunks in collection: {pipe.vector_db.size()}\n")

    results = {}
    samples = {}
    for label, model in MODELS:
        print(f"=== {label} ({model}) ===")
        hits = cited = 0
        answerable = sum(1 for _, _, a in QUESTIONS if a)
        declined_ok = None
        latencies = []
        tps = []
        for q, kws, ans_ok in QUESTIONS:
            ctx = retrieved[q]
            t0 = time.time()
            try:
                text = "".join(generate_response(q, ctx, model, stream=False, timeout=TIMEOUT))
            except Exception as e:
                text = f"<error: {e}>"
            dt = time.time() - t0
            latencies.append(dt)
            tps.append((len(text) / 4) / dt if dt > 0 else 0)  # ~4 chars/token

            clean, cites = validate_citations(text, len(ctx))
            if ans_ok:
                if _contains_any(text, kws):
                    hits += 1
                if cites:
                    cited += 1
            else:
                declined_ok = _declined(text)

            if q not in samples:
                samples[q] = {}
            samples[q][label] = (clean[:200], cites)
            mark = "✓" if (ans_ok and _contains_any(text, kws)) else ("—" if not ans_ok else "✗")
            print(f"  {mark} [{dt:5.1f}s] {q[:48]:48} cites={cites}")

        results[label] = {
            "correct": f"{hits}/{answerable}",
            "cited": f"{cited}/{answerable}",
            "declined_unanswerable": declined_ok,
            "avg_latency_s": sum(latencies) / len(latencies),
            "avg_tok_s": sum(tps) / len(tps),
        }
        print()

    # Summary table
    print("=" * 78)
    print(f"{'Model':<26}{'Correct':>9}{'Cited':>8}{'Declined':>10}{'Latency':>10}{'tok/s':>8}")
    print("-" * 78)
    for label, _ in MODELS:
        r = results[label]
        print(f"{label:<26}{r['correct']:>9}{r['cited']:>8}{str(r['declined_unanswerable']):>10}"
              f"{r['avg_latency_s']:>9.1f}s{r['avg_tok_s']:>8.1f}")
    print("=" * 78)

    # One qualitative sample (the toes question)
    sample_q = QUESTIONS[0][0]
    print(f"\nSample answers — \"{sample_q}\":")
    for label, _ in MODELS:
        txt, cites = samples[sample_q][label]
        print(f"\n[{label}] cites={cites}\n  {txt}")

    pipe.close()


if __name__ == "__main__":
    main()
