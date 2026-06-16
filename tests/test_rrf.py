"""
Tests for weighted Reciprocal Rank Fusion (rag_lite.retrieval.reciprocal_rank_fusion).

RRF is the heart of hybrid retrieval: it must dedupe by chunk identity, accumulate
contributions when a chunk appears in multiple lists, and honour the semantic/keyword
weight split — all independent of the underlying score scales.
"""

from rag_lite.retrieval import reciprocal_rank_fusion
from rag_lite.types import RetrievedChunk


def _chunk(cid: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(content=f"text-{cid}", score=score, chunk_id=cid)


def test_dedupes_by_chunk_id_and_accumulates():
    # Chunk "a" is rank 1 in both lists → it should accumulate both contributions
    # and outrank chunks that appear only once.
    semantic = [_chunk("a"), _chunk("b")]
    keyword = [_chunk("a"), _chunk("c")]
    fused = reciprocal_rank_fusion([semantic, keyword], k=60, rrf_weight=0.7)

    ids = [c.chunk_id for c in fused]
    assert ids[0] == "a"               # appears top of both lists
    assert set(ids) == {"a", "b", "c"}  # deduped, union of inputs
    assert len(fused) == 3


def test_sorted_descending_by_fused_score():
    fused = reciprocal_rank_fusion(
        [[_chunk("a"), _chunk("b"), _chunk("c")]], k=60, rrf_weight=0.7
    )
    scores = [c.score for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_weighting_favours_semantic_list():
    # Same chunk at the same rank in each single-item list: the one weighted higher
    # (semantic, 0.7) must win over the keyword-only one (0.3).
    semantic = [_chunk("sem")]
    keyword = [_chunk("kw")]
    fused = reciprocal_rank_fusion([semantic, keyword], k=60, rrf_weight=0.7)
    assert fused[0].chunk_id == "sem"
    assert fused[0].score > fused[1].score


def test_score_matches_weighted_formula():
    # Single list, weight 1.0 for a lone list → score == 1/(k+rank).
    k = 60
    fused = reciprocal_rank_fusion([[_chunk("a"), _chunk("b")]], k=k, rrf_weight=0.7)
    by_id = {c.chunk_id: c.score for c in fused}
    assert by_id["a"] == 1.0 / (k + 1)
    assert by_id["b"] == 1.0 / (k + 2)


def test_empty_lists_yield_empty():
    assert reciprocal_rank_fusion([[], []], k=60, rrf_weight=0.7) == []


def test_three_lists_use_equal_weights():
    # With >2 lists, weights are distributed equally (1/3 each), so a chunk that is
    # rank 1 in two of three lists beats one that's rank 1 in only one.
    twice = [[_chunk("x")], [_chunk("x")], [_chunk("y")]]
    fused = reciprocal_rank_fusion(twice, k=60)
    assert fused[0].chunk_id == "x"
