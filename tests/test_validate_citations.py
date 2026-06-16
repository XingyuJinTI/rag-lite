"""
Tests for the citation-validation guardrail (rag_lite.generation.validate_citations).

This is pure logic and the last line of defence against a model emitting citation
markers that don't correspond to a real source — so it's worth pinning precisely.
"""

from rag_lite.generation import validate_citations


def test_keeps_valid_single_markers():
    answer = "The cap is £2,000 [1] and notice is 5 weeks [2]."
    cleaned, cited = validate_citations(answer, num_sources=2)
    assert cleaned == "The cap is £2,000 [1] and notice is 5 weeks [2]."
    assert cited == [1, 2]


def test_strips_out_of_range_marker():
    # Only 2 sources were provided; [4] references nothing real.
    answer = "Holiday accrues monthly [1] per the schedule [4]."
    cleaned, cited = validate_citations(answer, num_sources=2)
    assert "[4]" not in cleaned
    assert cited == [1]
    # Whitespace before the removed marker is tidied, punctuation preserved.
    assert cleaned == "Holiday accrues monthly [1] per the schedule."


def test_multi_index_marker_partially_valid():
    answer = "See clauses [1, 3, 5]."
    cleaned, cited = validate_citations(answer, num_sources=3)
    # 5 is dropped; 1 and 3 are kept and re-rendered.
    assert cleaned == "See clauses [1, 3]."
    assert cited == [1, 3]


def test_multi_index_marker_with_spacing():
    answer = "Refer to [2 , 1]."
    cleaned, cited = validate_citations(answer, num_sources=3)
    # Order within the marker is preserved as written; cited set is sorted.
    assert cleaned == "Refer to [2, 1]."
    assert cited == [1, 2]


def test_dedupes_repeated_citations():
    answer = "First [1]. Again [1]. And [2]."
    cleaned, cited = validate_citations(answer, num_sources=2)
    assert cited == [1, 2]


def test_no_markers_returns_text_unchanged():
    answer = "A plain answer with no citations."
    cleaned, cited = validate_citations(answer, num_sources=3)
    assert cleaned == "A plain answer with no citations."
    assert cited == []


def test_all_markers_invalid_yields_empty_citations():
    answer = "Everything cites [9] and [10]."
    cleaned, cited = validate_citations(answer, num_sources=2)
    assert cited == []
    assert "[9]" not in cleaned and "[10]" not in cleaned


def test_zero_and_negative_like_indices_rejected():
    # [0] is below the 1-based valid range and must be stripped.
    answer = "Bad ref [0] but good ref [1]."
    cleaned, cited = validate_citations(answer, num_sources=1)
    assert cited == [1]
    assert "[0]" not in cleaned


def test_num_sources_zero_strips_everything():
    answer = "Citing [1] when nothing was retrieved."
    cleaned, cited = validate_citations(answer, num_sources=0)
    assert cited == []
    assert "[1]" not in cleaned
