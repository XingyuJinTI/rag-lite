"""
Embedding-model comparison for retrieval (semantic-only).

Answers "which embedding model retrieves best on our corpus" by re-embedding the SAME
token-chunked corpus with each model and scoring Recall@k / MRR / NDCG against the
ragqa ground truth. Generation is irrelevant here — this isolates the embedding model.

Design notes:
- Token chunks the corpus via the production chunker (chunking.chunk_segments), so it
  reflects the real ingest path (unlike run_benchmark, which indexes whole docs and lets
  the model truncate long ones at 512 tokens).
- Each model writes to its OWN table (embedding dim is fixed per table) so dims of 768
  (bge-base) and 1024 (bge-large / bge-m3) can coexist.
- To stay tractable on CPU, the corpus is CAPPED: all gold docs for the eval questions
  are always included, plus a deterministic sample of other docs up to CORPUS_CAP. This
  makes absolute recall optimistic, but the comparison ACROSS models is fair (identical
  corpus + chunks for every model).

Run inside the container with HF offline disabled so new models can download:
    docker compose exec -T -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
        api python -m evaluation.compare_embeddings
"""

import logging
import os
import time

logging.basicConfig(level=logging.WARNING)

from rag_lite.config import Config
from rag_lite.rag_pipeline import RAGPipeline
from rag_lite import chunking
from rag_lite.types import IngestChunk
from evaluation import load_dataset, RAGEvaluator

# (label, hf_model, dim, table). Dim must match the model's output.
MODELS = [
    ("bge-base-en-v1.5",  "BAAI/bge-base-en-v1.5",  768,  "emb_bge_base"),
    ("bge-large-en-v1.5", "BAAI/bge-large-en-v1.5", 1024, "emb_bge_large"),
    ("bge-m3",            "BAAI/bge-m3",            1024, "emb_bge_m3"),
]

MAX_EVAL = int(os.getenv("CMP_MAX_EVAL", "300"))
CORPUS_CAP = int(os.getenv("CMP_CORPUS_CAP", "6000"))
COLLECTION = "emb_cmp"


def build_corpus_chunks(dataset, embedding_model: str):
    """All gold docs for the eval set + a sample of others, token-chunked. The chunk
    set is identical for every model (only the embedding differs)."""
    docs = dataset.get_documents()
    gold = set()
    for ex in dataset.get_eval_examples():
        gold.update(gid for gid in ex.gold_doc_ids if gid < len(docs))

    selected = set(gold)
    for i in range(len(docs)):           # deterministic fill to the cap
        if len(selected) >= CORPUS_CAP:
            break
        selected.add(i)

    chunks = []
    for i in sorted(selected):
        segs = [(docs[i].text, None)]
        chunks.extend(chunking.chunk_segments(
            segs, source=docs[i].doc_id, embedding_model=embedding_model,
            max_tokens=256, overlap=48,
        ))
    return chunks, len(selected), len(gold)


def main() -> None:
    dataset = load_dataset("ragqa_arena", max_eval=MAX_EVAL)
    print(f"Eval questions: {len(dataset.get_eval_examples())} | corpus cap: {CORPUS_CAP}\n")

    rows = []
    for label, model, dim, table in MODELS:
        print(f"=== {label} ({model}, dim={dim}) ===")
        config = Config.from_env()
        config.model.embedding_model = model
        config.storage.embedding_dim = dim
        config.storage.table_name = table
        config.storage.collection_name = COLLECTION
        config.retrieval.use_hybrid_search = False   # semantic-only (net takeaway)
        config.retrieval.use_reranking = False

        pipe = RAGPipeline(config)

        # Index once per model (idempotent: skip if already populated).
        existing = pipe.vector_db.size()
        if existing == 0:
            chunks, ndocs, ngold = build_corpus_chunks(dataset, model)
            print(f"  indexing {len(chunks)} chunks from {ndocs} docs ({ngold} gold)…")
            t0 = time.time()
            pipe.index_documents(chunks, show_progress=False)
            index_s = time.time() - t0
            print(f"  indexed in {index_s:.0f}s")
        else:
            index_s = 0.0
            print(f"  reusing {existing} existing chunks")

        evaluator = RAGEvaluator(pipe, dataset, verbose=False)
        m = evaluator.evaluate(top_k=10, max_examples=MAX_EVAL,
                               use_hybrid_search=False, use_reranking=False)
        rows.append((label, dim, m, index_s))
        print(f"  R@1={m.recall_at_1:.3f} R@5={m.recall_at_5:.3f} MRR={m.mrr:.3f} "
              f"NDCG@10={m.ndcg_at_10:.3f} {m.avg_retrieval_time_ms:.0f}ms/q\n")
        pipe.close()

    print("=" * 86)
    print(f"{'Embedding model':<22}{'dim':>5}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}"
          f"{'NDCG@10':>9}{'ms/q':>8}{'index':>8}")
    print("-" * 86)
    for label, dim, m, index_s in rows:
        print(f"{label:<22}{dim:>5}{m.recall_at_1:>8.3f}{m.recall_at_3:>8.3f}"
              f"{m.recall_at_5:>8.3f}{m.mrr:>8.3f}{m.ndcg_at_10:>9.3f}"
              f"{m.avg_retrieval_time_ms:>7.0f}{index_s:>7.0f}s")
    print("=" * 86)


if __name__ == "__main__":
    main()
