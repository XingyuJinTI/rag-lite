"""
Evaluate retrieval configs against an ALREADY-INDEXED collection (no re-indexing).

Use after the corpus has been indexed once (e.g. by run_benchmark): this only runs
the eval queries, so iterating on configurations is fast.

    docker compose exec -T api python -m evaluation.run_eval_only
"""

import logging

logging.basicConfig(level=logging.WARNING)

from rag_lite import RAGPipeline, Config, ModelConfig
from evaluation import load_dataset, RAGEvaluator

MAX_EVAL = 100
COLLECTION = "eval_ragqa_arena"


def main() -> None:
    dataset = load_dataset("ragqa_arena", max_eval=MAX_EVAL)

    config = Config.from_env()
    config.retrieval.use_hybrid_search = True
    config.model.reranker_model = ModelConfig.RERANKER_BGE_BASE
    config.storage.collection_name = COLLECTION
    pipe = RAGPipeline(config)
    print(f"Evaluating against '{COLLECTION}' ({pipe.vector_db.size()} chunks), "
          f"{len(dataset.get_eval_examples())} eval questions\n")

    evaluator = RAGEvaluator(pipe, dataset, verbose=False)
    evaluator.compare_configurations(
        configurations=[
            {"name": "semantic_only", "use_hybrid_search": False, "use_reranking": False},
            {"name": "hybrid", "use_hybrid_search": True, "use_reranking": False},
            {"name": "hybrid+rerank", "use_hybrid_search": True, "use_reranking": True, "reranker_model": "bge"},
        ],
        top_k=10,
        max_examples=MAX_EVAL,
    )
    pipe.close()


if __name__ == "__main__":
    main()
