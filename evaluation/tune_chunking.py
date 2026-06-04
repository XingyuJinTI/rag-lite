"""
Tune chunk size / overlap (and calibrate the abstention threshold) on bge-m3.

Unlike run_benchmark (which indexes whole docs), this token-chunks the corpus through
the production chunker, so it measures the real ingest path. For each (max_tokens,
overlap) it re-embeds a capped corpus into its own table and scores semantic-only
retrieval (Recall@k / MRR / NDCG). It also dumps the distribution of the top hit vs
top miss score so a sensible ABSTAIN_THRESHOLD can be picked from data.

Capped corpus (all gold docs + a deterministic sample) keeps it tractable on CPU; the
comparison across configs is fair (identical doc set, only chunking differs).

    docker compose exec -T api python -m evaluation.tune_chunking
"""

import logging
import os
import time

logging.basicConfig(level=logging.WARNING)

from rag_lite.config import Config
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite import chunking
from evaluation import load_dataset, RAGEvaluator
from evaluation.metrics import normalize_text, text_overlap_score

MAX_EVAL = int(os.getenv("TUNE_MAX_EVAL", "150"))
CORPUS_CAP = int(os.getenv("TUNE_CORPUS_CAP", "1500"))
MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
COLLECTION = "tune_chunk"
# (max_tokens, overlap) — overlap ~15-19% of size.
SWEEP = [(128, 24), (256, 48), (512, 96)]


def build_chunks(dataset, model, max_tokens, overlap):
    docs = dataset.get_documents()
    gold = set()
    for ex in dataset.get_eval_examples():
        gold.update(g for g in ex.gold_doc_ids if g < len(docs))
    selected = set(gold)
    for i in range(len(docs)):
        if len(selected) >= CORPUS_CAP:
            break
        selected.add(i)
    chunks = []
    for i in sorted(selected):
        chunks.extend(chunking.chunk_segments(
            [(docs[i].text, None)], source=docs[i].doc_id,
            embedding_model=model, max_tokens=max_tokens, overlap=overlap,
        ))
    return chunks, len(selected)


def _match(retrieved_text: str, gold: str, thresh: float = 0.5) -> bool:
    a, b = normalize_text(retrieved_text), normalize_text(gold)
    return b in a or a in b or text_overlap_score(retrieved_text, gold) > thresh


def main() -> None:
    dataset = load_dataset("ragqa_arena", max_eval=MAX_EVAL)
    examples = dataset.get_eval_examples()
    print(f"model={MODEL} dim={DIM} | eval={len(examples)} | corpus cap={CORPUS_CAP}\n")

    rows = []
    pipes = {}
    for mt, ov in SWEEP:
        table = f"tune_{mt}_{ov}"
        cfg = Config.from_env()
        cfg.model.embedding_model = MODEL
        cfg.storage.embedding_dim = DIM
        cfg.storage.table_name = table
        cfg.storage.collection_name = COLLECTION
        cfg.retrieval.use_hybrid_search = False
        cfg.retrieval.use_reranking = False
        pipe = RAGPipeline(cfg)

        if pipe.vector_db.size() == 0:
            chunks, ndocs = build_chunks(dataset, MODEL, mt, ov)
            t0 = time.time()
            pipe.index_documents(chunks, show_progress=False)
            print(f"[{mt}/{ov}] {len(chunks)} chunks / {ndocs} docs — indexed {time.time()-t0:.0f}s")

        ev = RAGEvaluator(pipe, dataset, verbose=False)
        m = ev.evaluate(top_k=10, max_examples=MAX_EVAL,
                        use_hybrid_search=False, use_reranking=False)
        rows.append((mt, ov, m))
        pipes[(mt, ov)] = pipe
        print(f"[{mt}/{ov}] R@1={m.recall_at_1:.3f} R@5={m.recall_at_5:.3f} "
              f"MRR={m.mrr:.3f} NDCG@10={m.ndcg_at_10:.3f}")

    print("\n" + "=" * 72)
    print(f"{'max_tok/overlap':<18}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'NDCG@10':>9}")
    print("-" * 72)
    for mt, ov, m in rows:
        print(f"{f'{mt}/{ov}':<18}{m.recall_at_1:>8.3f}{m.recall_at_3:>8.3f}"
              f"{m.recall_at_5:>8.3f}{m.mrr:>8.3f}{m.ndcg_at_10:>9.3f}")
    print("=" * 72)

    # Abstain calibration on the best-MRR config: top-1 score for hits vs misses.
    best = max(rows, key=lambda r: r[2].mrr)
    pipe = pipes[(best[0], best[1])]
    hit_scores, miss_scores = [], []
    for ex in examples:
        res = pipe.retrieve(ex.question, top_n=5, use_hybrid_search=False, use_reranking=False)
        if not res:
            continue
        top = res[0].score
        hit = any(_match(r.content, ex.context) for r in res)
        (hit_scores if hit else miss_scores).append(top)

    def stats(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        return f"n={len(xs)} min={xs[0]:.3f} med={xs[len(xs)//2]:.3f} max={xs[-1]:.3f} mean={sum(xs)/len(xs):.3f}"

    print(f"\nAbstain calibration (best config {best[0]}/{best[1]}, top-1 cosine score):")
    print(f"  hits : {stats(hit_scores)}")
    print(f"  miss : {stats(miss_scores)}")
    for pipe in pipes.values():
        pipe.close()


if __name__ == "__main__":
    main()
