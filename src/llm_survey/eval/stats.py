"""Bootstrap confidence intervals and paired significance tests for eval metrics.

Plain stdlib only (uses `random`/`statistics`), so no extra dep is required to
report CIs alongside the existing precision/recall/F1 numbers in
`docs/evaluation_metrics.json`.

Conventions:
  - All bootstrap routines accept `seed` for determinism.
  - CIs are reported as 95% percentile by default (2.5 / 97.5).
  - "Per-document" variance helpers expect a list of per-doc metric dicts.

The McNemar test here is the exact mid-p variant for small N; for large N a
chi-square approximation would be cheaper but unnecessary at our gold-set
scale.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Callable, Iterable, Sequence


Number = float


@dataclass
class BootstrapResult:
    point: float
    lo: float
    hi: float
    n_resamples: int
    confidence: float = 0.95
    samples_summary: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point": round(self.point, 4),
            "ci_lo": round(self.lo, 4),
            "ci_hi": round(self.hi, 4),
            "ci_width": round(self.hi - self.lo, 4),
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
            **{k: round(v, 4) for k, v in self.samples_summary.items()},
        }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo]) * (1 - frac) + float(sorted_values[hi]) * frac


def bootstrap_metric(
    items: Sequence,
    metric_fn: Callable[[Sequence], float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> BootstrapResult:
    """Generic bootstrap. Resamples `items` with replacement and applies `metric_fn`.

    `metric_fn` must accept a list of items and return a single scalar metric.
    """
    if not items:
        return BootstrapResult(point=0.0, lo=0.0, hi=0.0, n_resamples=0, confidence=confidence)
    rng = random.Random(seed)
    n = len(items)
    samples: list[float] = []
    for _ in range(n_resamples):
        resample = [items[rng.randrange(n)] for _ in range(n)]
        samples.append(float(metric_fn(resample)))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = _percentile(samples, alpha)
    hi = _percentile(samples, 1.0 - alpha)
    point = float(metric_fn(items))
    summary = {"mean": float(mean(samples)), "std": float(pstdev(samples)) if len(samples) > 1 else 0.0}
    return BootstrapResult(
        point=point,
        lo=lo,
        hi=hi,
        n_resamples=n_resamples,
        confidence=confidence,
        samples_summary=summary,
    )


def bootstrap_prf1(
    matches: Sequence[dict],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap precision / recall / F1 over a sequence of per-item outcomes.

    Each item is a dict with boolean fields:
      - `gold`: True iff the item is a gold relationship.
      - `predicted`: True iff the item was extracted (positive prediction).
      - `match`: True iff this is a true positive.
    For "extraction-only" rows (FPs), set `gold=False, predicted=True, match=False`.
    For "gold-only" rows (FNs), set `gold=True, predicted=False, match=False`.
    For TPs, set all three True.

    Returns a dict { "precision": {...}, "recall": {...}, "f1": {...} } where each
    value is `BootstrapResult.as_dict()`.
    """

    def _prf(rows: Sequence[dict]) -> tuple[float, float, float]:
        tp = sum(1 for r in rows if r.get("gold") and r.get("predicted") and r.get("match"))
        fp = sum(1 for r in rows if r.get("predicted") and not r.get("match"))
        fn = sum(1 for r in rows if r.get("gold") and not r.get("match"))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rcl = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * rcl / (p + rcl)) if (p + rcl) else 0.0
        return p, rcl, f1

    return {
        "precision": bootstrap_metric(
            matches, lambda rows: _prf(rows)[0], n_resamples=n_resamples, confidence=confidence, seed=seed
        ).as_dict(),
        "recall": bootstrap_metric(
            matches, lambda rows: _prf(rows)[1], n_resamples=n_resamples, confidence=confidence, seed=seed + 1
        ).as_dict(),
        "f1": bootstrap_metric(
            matches, lambda rows: _prf(rows)[2], n_resamples=n_resamples, confidence=confidence, seed=seed + 2
        ).as_dict(),
    }


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> dict[str, float | int]:
    """Paired bootstrap for `mean(a) - mean(b)`. Returns point, CI, and one-sided p-value.

    `a` and `b` must be paired (same length, indices align). Useful when comparing
    two pipeline variants on the same set of documents/chunks.
    """
    if len(a) != len(b):
        raise ValueError(f"Paired bootstrap requires equal lengths, got {len(a)} vs {len(b)}")
    if not a:
        return {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_resamples": 0, "p_value_a_gt_b": 1.0}
    rng = random.Random(seed)
    n = len(a)
    diffs = [float(a[i]) - float(b[i]) for i in range(n)]
    point = float(mean(diffs))
    samples: list[float] = []
    favor = 0
    for _ in range(n_resamples):
        s = 0.0
        for _i in range(n):
            j = rng.randrange(n)
            s += diffs[j]
        avg = s / n
        samples.append(avg)
        if avg > 0:
            favor += 1
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": round(point, 4),
        "ci_lo": round(_percentile(samples, alpha), 4),
        "ci_hi": round(_percentile(samples, 1.0 - alpha), 4),
        "n_resamples": n_resamples,
        # Fraction of bootstrap means with a > b. NOT a real p-value, but a useful
        # "how often does A beat B under resampling" summary. Two-sided p ~= 2*min(p, 1-p).
        "frac_a_beats_b": round(favor / n_resamples, 4),
    }


def mcnemar_exact(b: int, c: int) -> dict[str, float | int]:
    """Exact two-sided binomial test for paired binary outcomes (McNemar).

    `b` = items A got right, B got wrong. `c` = items A got wrong, B got right.
    Diagonal cells (both right / both wrong) cancel out.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_disagreements": 0, "p_value": 1.0}
    k = min(b, c)
    # P(X<=k) under Binomial(n, 0.5)
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n, i)
    p_one_sided = cum / (2 ** n)
    p_two_sided = min(1.0, 2 * p_one_sided)
    return {"b": b, "c": c, "n_disagreements": n, "p_value": round(p_two_sided, 4)}


def per_document_variance(per_doc_metrics: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Given an iterable of {metric_name: value} dicts, one per document, return mean/std/min/max per metric."""
    rows = list(per_doc_metrics)
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [float(row[k]) for row in rows if k in row]
        if not vals:
            continue
        out[k] = {
            "n": len(vals),
            "mean": round(mean(vals), 4),
            "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }
    return out


__all__ = [
    "BootstrapResult",
    "bootstrap_metric",
    "bootstrap_prf1",
    "paired_bootstrap_diff",
    "mcnemar_exact",
    "per_document_variance",
]
