"""
Tests for token-aware chunking (rag_lite.chunking).

Covers the pure-logic pieces that don't need a real model:
  - _model_max_tokens clamping
  - _split_sentences (markdown blocks, abbreviations, decimals)
  - chunk_segments sizing / overlap / provenance invariants
  - chunk_hierarchical parent/child relationship + sizing

A whitespace FakeTokenizer (see conftest) makes "tokens" == words, so sizing
assertions are exact and offline.
"""

from rag_lite.chunking import (
    _model_max_tokens,
    _split_sentences,
    chunk_segments,
    chunk_hierarchical,
    HARD_MAX_TOKENS,
    FALLBACK_MAX_TOKENS,
)
from tests.conftest import FakeTokenizer


# ---------------------------------------------------------------------------
# _model_max_tokens
# ---------------------------------------------------------------------------

def test_model_max_tokens_uses_reported_value():
    assert _model_max_tokens(FakeTokenizer(model_max_length=512)) == 512


def test_model_max_tokens_clamps_huge_sentinel():
    # Some tokenizers report an absurd model_max_length; clamp to the hard ceiling.
    assert _model_max_tokens(FakeTokenizer(model_max_length=10**9)) == HARD_MAX_TOKENS


def test_model_max_tokens_falls_back_on_nonpositive():
    assert _model_max_tokens(FakeTokenizer(model_max_length=0)) == FALLBACK_MAX_TOKENS


def test_model_max_tokens_falls_back_on_missing():
    class NoAttr:
        pass
    assert _model_max_tokens(NoAttr()) == FALLBACK_MAX_TOKENS


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

def test_splits_on_sentence_boundaries():
    assert _split_sentences("First sentence. Second sentence.") == [
        "First sentence.",
        "Second sentence.",
    ]


def test_does_not_split_on_abbreviation():
    # "Dr." must not end a sentence.
    out = _split_sentences("Dr. Smith went home. He slept.")
    assert out == ["Dr. Smith went home.", "He slept."]


def test_does_not_split_on_decimal():
    # No whitespace inside "3.14", so it stays a single sentence.
    out = _split_sentences("Pi is about 3.14 in value. Done.")
    assert out == ["Pi is about 3.14 in value.", "Done."]


def test_markdown_list_items_are_separate_blocks():
    out = _split_sentences("Items:\n- one\n- two")
    assert out == ["Items:", "- one", "- two"]


def test_collapses_hard_wrapped_lines():
    # A single logical sentence split across hard-wrapped lines rejoins.
    out = _split_sentences("This is a single\nwrapped sentence.")
    assert out == ["This is a single wrapped sentence."]


# ---------------------------------------------------------------------------
# chunk_segments
# ---------------------------------------------------------------------------

def _tokens(text: str) -> int:
    return len(text.split())


def test_chunk_segments_respects_max_tokens_and_overlap(patch_tokenizer):
    patch_tokenizer(model_max_length=512)
    # Capitalised starts so the splitter yields real sentences (lowercase starts
    # would be treated as one sentence and hard-split, bypassing overlap).
    text = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta."
    chunks = chunk_segments(
        [(text, 1)],
        source="doc.txt",
        embedding_model="fake",
        max_tokens=4,
        overlap=2,
    )
    assert len(chunks) >= 2
    # Every chunk fits the budget.
    for c in chunks:
        assert _tokens(c.text) <= 4
    # Consecutive chunks overlap (share at least one word).
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.text.split()) & set(b.text.split())


def test_chunk_segments_sets_provenance_and_indices(patch_tokenizer):
    patch_tokenizer(model_max_length=512)
    chunks = chunk_segments(
        [("alpha beta. gamma delta. epsilon zeta.", 7)],
        source="handbook.pdf",
        embedding_model="fake",
        title="Handbook",
        uri="/docs/handbook.pdf",
        max_tokens=2,
        overlap=0,
    )
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.source == "handbook.pdf"
        assert c.title == "Handbook"
        assert c.uri == "/docs/handbook.pdf"
        assert c.page == 7


def test_chunk_segments_does_not_span_pages(patch_tokenizer):
    patch_tokenizer(model_max_length=512)
    chunks = chunk_segments(
        [("page one text.", 1), ("page two text.", 2)],
        source="doc.pdf",
        embedding_model="fake",
        max_tokens=50,
        overlap=0,
    )
    pages = {c.page for c in chunks}
    assert pages == {1, 2}
    # chunk_index is document-global and monotonic across pages.
    assert [c.chunk_index for c in chunks] == sorted(c.chunk_index for c in chunks)


def test_chunk_segments_skips_empty_segments(patch_tokenizer):
    patch_tokenizer(model_max_length=512)
    chunks = chunk_segments(
        [("", 1), ("   ", 2), ("real content here.", 3)],
        source="doc.txt",
        embedding_model="fake",
        max_tokens=50,
        overlap=0,
    )
    assert len(chunks) == 1
    assert chunks[0].page == 3


def test_chunk_segments_caps_max_tokens_at_model_limit(patch_tokenizer):
    # Asking for more tokens than the model supports must clamp to the model max.
    patch_tokenizer(model_max_length=3)
    text = "one two three four five six seven eight."
    chunks = chunk_segments(
        [(text, 1)],
        source="doc.txt",
        embedding_model="fake",
        max_tokens=999,
        overlap=0,
    )
    for c in chunks:
        assert _tokens(c.text) <= 3


# ---------------------------------------------------------------------------
# chunk_hierarchical
# ---------------------------------------------------------------------------

def test_hierarchical_children_reference_existing_parents(patch_tokenizer):
    patch_tokenizer(model_max_length=512)
    text = "a b. c d. e f. g h. i j. k l."
    children, parents = chunk_hierarchical(
        [(text, 1)],
        source="doc.txt",
        embedding_model="fake",
        parent_max_tokens=4,
        child_max_tokens=2,
        child_overlap=0,
    )
    assert children and parents
    parent_ids = {p.parent_id for p in parents}
    # Every child points at a real parent.
    for c in children:
        assert c.parent_id in parent_ids


def test_hierarchical_size_invariants(patch_tokenizer):
    patch_tokenizer(model_max_length=512)
    text = "a b. c d. e f. g h. i j. k l."
    children, parents = chunk_hierarchical(
        [(text, 1)],
        source="doc.txt",
        embedding_model="fake",
        parent_max_tokens=4,
        child_max_tokens=2,
        child_overlap=0,
    )
    for p in parents:
        assert _tokens(p.content) <= 4
    for c in children:
        assert _tokens(c.text) <= 2


def test_hierarchical_parent_ids_are_stable_hashes(patch_tokenizer):
    # Same (source, parent text) → same parent_id across runs (content-addressed).
    patch_tokenizer(model_max_length=512)
    args = dict(source="doc.txt", embedding_model="fake",
               parent_max_tokens=4, child_max_tokens=2, child_overlap=0)
    _, parents_a = chunk_hierarchical([("a b. c d. e f.", 1)], **args)
    patch_tokenizer(model_max_length=512)
    _, parents_b = chunk_hierarchical([("a b. c d. e f.", 1)], **args)
    assert [p.parent_id for p in parents_a] == [p.parent_id for p in parents_b]
