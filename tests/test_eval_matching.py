"""Tests for the alias-aware gold matcher."""
from __future__ import annotations

from llm_survey.eval.matching import normalize, relationship_matches


def test_normalize_lowercases_lemmatizes() -> None:
    out = normalize("Job Crafting and stressing demands")
    assert "job" in out
    assert "craft" in out  # crafting -> craft
    assert "stress" in out  # stressing -> stress
    assert "demand" in out  # demands -> demand


def test_normalize_strips_punctuation() -> None:
    out = normalize("Hello, world! Variable_name v1.2")
    # Underscores split tokens; punctuation is dropped.
    assert "hello" in out and "world" in out
    assert "_" not in "".join(out)
    assert "!" not in "".join(out)


def test_alias_match_basic() -> None:
    rel = {
        "chunk_id": "respondent_1_chunk_0",
        "from_variable": "Workload",
        "to_variable": "Stress",
    }
    gold = {
        "respondent_hint": "respondent_1",
        "from_aliases": ["workload", "task demands"],
        "to_aliases": ["stress", "anxiety"],
    }
    assert relationship_matches(rel, gold) is True


def test_alias_match_lemmatized_synonyms() -> None:
    rel = {
        "chunk_id": "respondent_1_chunk_0",
        "from_variable": "Stressing deadlines",
        "to_variable": "Anxieties",
    }
    gold = {
        "respondent_hint": "respondent_1",
        "from_aliases": ["stress", "deadline"],
        "to_aliases": ["anxiety"],
    }
    # Lemmatized: stressing -> stress, anxieties -> anxiety.
    assert relationship_matches(rel, gold) is True


def test_word_boundary_avoids_false_match() -> None:
    """The naive substring matcher would say 'job' matches 'side job', overcounting."""
    rel = {
        "chunk_id": "respondent_1_chunk_0",
        "from_variable": "side job",
        "to_variable": "income",
    }
    gold_too_broad = {
        "respondent_hint": "respondent_1",
        "from_aliases": ["job satisfaction"],  # multi-word phrase, not present
        "to_aliases": ["income"],
    }
    # Old matcher: True (because 'job' substring is in 'side job').
    # New matcher: False (because the full phrase 'job satisfaction' is not a contiguous subsequence).
    assert relationship_matches(rel, gold_too_broad) is False


def test_respondent_hint_must_match_chunk_id() -> None:
    rel = {
        "chunk_id": "respondent_5_chunk_0",
        "from_variable": "workload",
        "to_variable": "stress",
    }
    gold = {
        "respondent_hint": "respondent_1",
        "from_aliases": ["workload"],
        "to_aliases": ["stress"],
    }
    assert relationship_matches(rel, gold) is False


def test_legacy_substrings_field_still_works() -> None:
    """Backward compat: gold files using `from_substrings` / `to_substrings`."""
    rel = {
        "chunk_id": "respondent_1_chunk_0",
        "from_variable": "deadline pressure",
        "to_variable": "feeling overwhelmed",
    }
    gold = {
        "respondent_hint": "respondent_1",
        "from_substrings": ["deadline", "workload"],
        "to_substrings": ["overwhelm", "stress"],
    }
    assert relationship_matches(rel, gold) is True


def test_aliases_take_precedence_over_substrings() -> None:
    rel = {
        "chunk_id": "respondent_1_chunk_0",
        "from_variable": "workload",
        "to_variable": "stress",
    }
    gold = {
        "respondent_hint": "respondent_1",
        "from_aliases": ["something_completely_different"],
        "from_substrings": ["workload"],  # would match if used
        "to_aliases": ["stress"],
    }
    # When both are present, *_aliases wins; the substrings are ignored.
    assert relationship_matches(rel, gold) is False


def test_empty_terms_returns_false() -> None:
    rel = {"chunk_id": "x", "from_variable": "a", "to_variable": "b"}
    assert relationship_matches(rel, {}) is False
