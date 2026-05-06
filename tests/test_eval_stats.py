"""Unit tests for `llm_survey.eval.stats` — bootstrap CIs, McNemar, paired bootstrap."""
from __future__ import annotations

from llm_survey.eval.stats import (
    bootstrap_metric,
    bootstrap_prf1,
    mcnemar_exact,
    paired_bootstrap_diff,
    per_document_variance,
)


def test_bootstrap_metric_ci_contains_point_estimate() -> None:
    items = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]  # mean = 0.7
    res = bootstrap_metric(items, lambda xs: sum(xs) / len(xs), n_resamples=500, seed=42)
    assert res.lo <= res.point <= res.hi
    assert abs(res.point - 0.7) < 1e-9


def test_bootstrap_metric_deterministic_with_seed() -> None:
    items = list(range(20))
    a = bootstrap_metric(items, lambda xs: sum(xs) / len(xs), n_resamples=300, seed=7)
    b = bootstrap_metric(items, lambda xs: sum(xs) / len(xs), n_resamples=300, seed=7)
    assert a.lo == b.lo and a.hi == b.hi and a.point == b.point


def test_bootstrap_metric_handles_empty() -> None:
    res = bootstrap_metric([], lambda xs: 0.0, n_resamples=100)
    assert res.point == 0.0 and res.lo == 0.0 and res.hi == 0.0 and res.n_resamples == 0


def test_bootstrap_prf1_perfect_classifier() -> None:
    # 10 TPs only — precision/recall/F1 must all be 1.0 with degenerate CI.
    rows = [{"gold": True, "predicted": True, "match": True}] * 10
    out = bootstrap_prf1(rows, n_resamples=200)
    assert out["precision"]["point"] == 1.0
    assert out["recall"]["point"] == 1.0
    assert out["f1"]["point"] == 1.0
    # Standard deviation should be 0 — all resamples produce identical metrics.
    assert out["precision"]["std"] == 0.0


def test_bootstrap_prf1_realistic_mix() -> None:
    rows = (
        [{"gold": True, "predicted": True, "match": True}] * 8
        + [{"gold": True, "predicted": False, "match": False}] * 2  # FNs
        + [{"gold": False, "predicted": True, "match": False}] * 1  # FP
    )
    out = bootstrap_prf1(rows, n_resamples=300)
    # P = 8/9 ≈ 0.889, R = 8/10 = 0.8 (point is rounded to 4 decimals in as_dict).
    assert abs(out["precision"]["point"] - 8 / 9) < 1e-3
    assert abs(out["recall"]["point"] - 0.8) < 1e-6
    # CIs must contain the point estimate.
    for metric in ("precision", "recall", "f1"):
        m = out[metric]
        assert m["ci_lo"] <= m["point"] <= m["ci_hi"]


def test_paired_bootstrap_zero_diff_on_identical_inputs() -> None:
    a = [0.7, 0.8, 0.9, 0.6]
    res = paired_bootstrap_diff(a, a, n_resamples=300, seed=1)
    # Mean of zero diffs is exactly zero.
    assert res["point"] == 0.0
    # CI must contain zero.
    assert res["ci_lo"] <= 0.0 <= res["ci_hi"]


def test_paired_bootstrap_detects_clear_advantage() -> None:
    # A is uniformly higher than B — favor count should be very high.
    a = [0.9, 0.8, 0.85, 0.95, 0.88]
    b = [0.5, 0.4, 0.45, 0.55, 0.48]
    res = paired_bootstrap_diff(a, b, n_resamples=500, seed=3)
    assert res["point"] > 0.3
    assert res["frac_a_beats_b"] > 0.95


def test_mcnemar_no_disagreement_returns_p1() -> None:
    res = mcnemar_exact(b=0, c=0)
    assert res["p_value"] == 1.0
    assert res["n_disagreements"] == 0


def test_mcnemar_symmetric_disagreement_p_high() -> None:
    res = mcnemar_exact(b=5, c=5)
    # b == c is the most-likely outcome under H0 -> p should be high.
    assert res["p_value"] > 0.5


def test_mcnemar_strongly_asymmetric_p_low() -> None:
    res = mcnemar_exact(b=15, c=1)
    assert res["p_value"] < 0.01


def test_per_document_variance_basic_stats() -> None:
    rows = [
        {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        {"precision": 0.5, "recall": 0.5, "f1": 0.5},
        {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    ]
    out = per_document_variance(rows)
    assert out["precision"]["n"] == 3
    assert abs(out["precision"]["mean"] - 0.5) < 1e-6
    assert out["precision"]["min"] == 0.0
    assert out["precision"]["max"] == 1.0
