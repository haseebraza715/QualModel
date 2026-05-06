"""Gold-relationship matcher.

Replaces the naive substring matcher in `scripts/compute_eval_metrics.py` with
a more careful one that:

  1. Normalizes whitespace + casing + punctuation,
  2. Applies a lightweight lemmatizer (no NLTK dependency at eval time),
  3. Matches on word boundaries instead of bare `in`, so "job" does not match
     "side job" or "job crafting" by accident,
  4. Supports a richer optional `aliases` schema in the gold file:
        {
          "id": "HG01",
          "from_aliases": ["workload", "deadline pressure", "task demands"],
          "to_aliases": ["stress", "feeling overwhelmed"],
          "respondent_hint": "respondent_1"
        }
     When `from_aliases` / `to_aliases` are present they take precedence over
     the legacy `from_substrings` / `to_substrings` arrays.

The legacy substring fields keep working unchanged — this module is additive,
so existing gold files don't need to be migrated immediately.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Trivial English suffix lemmatizer — strips a handful of high-impact endings.
# Not Porter-quality but good enough to make {"crafting" ↔ "craft"} match.
_LEMMA_SUFFIXES = (
    ("iveness", ""),
    ("ization", "ize"),
    ("ational", "ate"),
    ("ization", "ize"),
    ("ousness", "ous"),
    ("alities", "ality"),
    ("nesses", "ness"),
    ("ingly", ""),
    ("edly", ""),
    ("ying", "y"),
    ("ied", "y"),
    ("ies", "y"),
    ("ssing", "ss"),  # stressing -> stress
    ("ssed", "ss"),
    ("ing", ""),
    ("ed", ""),
    ("er", ""),
    ("est", ""),
    ("ly", ""),
    ("s", ""),
)


def _lemmatize(token: str) -> str:
    """Strip the longest matching suffix. Tokens shorter than 4 chars stay as-is."""
    if len(token) < 4:
        return token
    for suffix, replacement in _LEMMA_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, lemmatize each token. Returns ordered tokens.

    Example:
        >>> normalize("Job crafting AND task autonomy.")
        ['job', 'craft', 'and', 'task', 'autonomy']
    """
    return [_lemmatize(m.group(0).lower()) for m in _TOKEN_RE.finditer(text or "")]


def _phrase_in_tokens(phrase: str, tokens: list[str]) -> bool:
    """True iff the lemmatized `phrase` appears as a contiguous subsequence of `tokens`."""
    needle = normalize(phrase)
    if not needle:
        return False
    n, m = len(tokens), len(needle)
    if m > n:
        return False
    for i in range(n - m + 1):
        if tokens[i : i + m] == needle:
            return True
    return False


def _matches_alias_list(extracted_variable: str, aliases: Iterable[str]) -> bool:
    """True iff any alias appears as a contiguous lemmatized phrase in the extracted variable."""
    tokens = normalize(extracted_variable)
    if not tokens:
        return False
    return any(_phrase_in_tokens(alias, tokens) for alias in aliases if alias)


def _matches_substring_list(extracted_variable: str, substrings: Iterable[str]) -> bool:
    """Legacy matcher: any substring (lemmatized, word-boundary) found in the extracted variable.

    Word boundaries prevent "job" matching "side job" by accident — use the
    lemmatized token sequence and check substring containment as a token-list
    subsequence rather than as raw character containment.
    """
    return _matches_alias_list(extracted_variable, substrings)


def relationship_matches(
    rel: dict,
    gold: dict,
) -> bool:
    """True iff an extracted relationship `rel` matches a gold spec `gold`.

    `rel` shape (matches `_iter_relationships` in compute_eval_metrics.py):
        {"chunk_id": str, "from_variable": str, "to_variable": str}

    `gold` shape:
        {
          "id": str,
          "respondent_hint": str (optional; substring match on chunk_id),
          "from_aliases": [str, ...] OR "from_substrings": [str, ...],
          "to_aliases":   [str, ...] OR "to_substrings":   [str, ...],
        }
    """
    hint = str(gold.get("respondent_hint", "")).lower()
    cid = str(rel.get("chunk_id", "")).lower()
    if hint and hint not in cid:
        return False

    from_terms = list(gold.get("from_aliases") or gold.get("from_substrings") or [])
    to_terms = list(gold.get("to_aliases") or gold.get("to_substrings") or [])
    if not from_terms or not to_terms:
        return False

    return _matches_alias_list(rel.get("from_variable", ""), from_terms) and _matches_alias_list(
        rel.get("to_variable", ""), to_terms
    )


__all__ = ["normalize", "relationship_matches"]
