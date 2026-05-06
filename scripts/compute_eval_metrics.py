#!/usr/bin/env python3
"""
Compare flattened extraction relationships to hand-coded gold (substring rules).

Default: fixture extraction vs fixture-scoped gold (reproducible).

  python3 scripts/compute_eval_metrics.py

Live pipeline output:

  python3 scripts/compute_eval_metrics.py \\
    --extractions outputs/extracted_models.json \\
    --gold docs/evaluation_gold.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Make `llm_survey` importable when running this script from a source checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from llm_survey.eval.stats import bootstrap_prf1, per_document_variance
except Exception:  # pragma: no cover - bootstrap stats are optional
    bootstrap_prf1 = None  # type: ignore[assignment]
    per_document_variance = None  # type: ignore[assignment]

try:
    from llm_survey.eval.matching import relationship_matches as _smart_match
except Exception:  # pragma: no cover - matcher is optional
    _smart_match = None  # type: ignore[assignment]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_relationships(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("success") or not isinstance(row.get("model"), dict):
            continue
        cid = str(row.get("chunk_id", ""))
        for rel in row["model"].get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            out.append(
                {
                    "chunk_id": cid,
                    "from_variable": str(rel.get("from_variable", "")),
                    "to_variable": str(rel.get("to_variable", "")),
                }
            )
    return out


def _matches(rel: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    """Match an extracted relationship against a gold spec.

    Prefers `llm_survey.eval.matching.relationship_matches` (lemmatized,
    word-boundary aware, supports the new `*_aliases` schema). Falls back to
    the legacy raw-substring matcher if the eval module isn't importable.
    """
    if _smart_match is not None:
        return _smart_match(rel, gold)
    # Legacy fallback (kept verbatim for backward compat).
    hint = str(gold.get("respondent_hint", "")).lower()
    cid = rel["chunk_id"].lower()
    if hint and hint not in cid:
        return False
    fv = rel["from_variable"].lower()
    tv = rel["to_variable"].lower()
    from_ok = any(sub.lower() in fv for sub in gold.get("from_substrings", []))
    to_ok = any(sub.lower() in tv for sub in gold.get("to_substrings", []))
    return from_ok and to_ok


def evaluate(extractions: List[Dict[str, Any]], gold_doc: Dict[str, Any]) -> Dict[str, Any]:
    gold_items: List[Dict[str, Any]] = list(gold_doc.get("relationships", []))
    rels = _iter_relationships(extractions)

    matched_gold: Set[str] = set()
    false_positives: List[Dict[str, Any]] = []

    for rel in rels:
        hit: str | None = None
        for g in gold_items:
            gid = str(g.get("id", ""))
            if gid in matched_gold:
                continue
            if _matches(rel, g):
                hit = gid
                break
        if hit:
            matched_gold.add(hit)
        else:
            false_positives.append(rel)

    tp = len(matched_gold)
    fp = len(false_positives)
    fn = len(gold_items) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    result: Dict[str, Any] = {
        "gold_items": len(gold_items),
        "extracted_relationships": len(rels),
        "true_positives_matched_gold": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_examples": false_positives[:8],
    }

    # Bootstrap CIs over a per-item outcome list. Each gold item contributes one
    # row (matched=True for TPs, False for FNs), and each unmatched extraction
    # contributes one row (gold=False, predicted=True, match=False). Resampling
    # this list with replacement gives the standard non-parametric CI.
    if bootstrap_prf1 is not None:
        outcome_rows: List[Dict[str, Any]] = []
        for g in gold_items:
            gid = str(g.get("id", ""))
            outcome_rows.append({"gold": True, "predicted": gid in matched_gold, "match": gid in matched_gold})
        for _ in false_positives:
            outcome_rows.append({"gold": False, "predicted": True, "match": False})
        result["bootstrap_ci_95"] = bootstrap_prf1(outcome_rows, n_resamples=1000, seed=20260101)

    # Per-chunk variance: group outcomes by chunk_id and compute precision/recall per chunk.
    if per_document_variance is not None:
        per_chunk: Dict[str, Dict[str, int]] = {}
        for rel in rels:
            cid = rel["chunk_id"] or "(unknown)"
            d = per_chunk.setdefault(cid, {"tp": 0, "fp": 0, "fn": 0})
            matched = False
            for g in gold_items:
                if str(g.get("id", "")) in matched_gold and _matches(rel, g):
                    matched = True
                    break
            if matched:
                d["tp"] += 1
            else:
                d["fp"] += 1
        # Distribute FNs across the chunks named in their respondent_hint, if any.
        for g in gold_items:
            if str(g.get("id", "")) in matched_gold:
                continue
            hint = str(g.get("respondent_hint", "")) or "(unknown)"
            d = per_chunk.setdefault(hint, {"tp": 0, "fp": 0, "fn": 0})
            d["fn"] += 1
        per_doc_rows: List[Dict[str, float]] = []
        for cid, counts in per_chunk.items():
            tp_c, fp_c, fn_c = counts["tp"], counts["fp"], counts["fn"]
            p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
            r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
            f = (2 * p * r / (p + r)) if (p + r) else 0.0
            per_doc_rows.append({"precision": p, "recall": r, "f1": f})
        result["per_chunk_variance"] = per_document_variance(per_doc_rows)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractions", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gold_path = args.gold or (root / "docs" / "fixtures" / "evaluation_gold_fixture_subset.json")
    ext_path = args.extractions or (root / "docs" / "fixtures" / "extracted_models_eval_fixture.json")

    gold_doc = _load_json(gold_path)
    extractions = _load_json(ext_path)
    if not isinstance(extractions, list):
        raise SystemExit("Extractions file must be a JSON list of chunk results.")

    metrics = evaluate(extractions, gold_doc)
    metrics["gold_path"] = str(gold_path.relative_to(root))
    metrics["extractions_path"] = str(ext_path.relative_to(root))

    out_path = root / "docs" / "evaluation_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
