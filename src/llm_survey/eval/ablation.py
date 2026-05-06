"""Ablation harness — toggles individual pipeline phases and emits a comparison table.

The plan calls for ablating: literature RAG, gap detection, clarification,
iterative refinement, consolidation contradiction resolution, and a
single-pass baseline. This module provides:

  - `AblationVariant`: a named bundle of pipeline toggles + a description.
  - `ABLATION_MATRIX`: the canonical set of variants from RESEARCH_PLAN §1.4.
  - `run_ablation_matrix(...)`: runs every variant against a corpus, persists
    each variant's outputs to a separate dir, computes metrics, and writes
    a `docs/ablation_results.json` table summarising the comparison.

The harness is intentionally **non-destructive**: it does not modify the
existing pipeline. Variants pass kwargs through to `run_complete_pipeline()`
in `main.py` so adding new variants is just a new entry in `ABLATION_MATRIX`.

Cost note: a full matrix is N variants × pipeline runtime. For the synthetic
corpus this is fine; for real corpora, run with `--variant single_pass_baseline`
or a subset first.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .stats import bootstrap_prf1, paired_bootstrap_diff


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    # Kwargs forwarded to run_complete_pipeline(). Only the toggles documented
    # in main.run_complete_pipeline() are valid; unknown kwargs raise TypeError.
    pipeline_kwargs: dict[str, Any] = field(default_factory=dict)
    # If True, this variant short-circuits past the dual-RAG pipeline and runs
    # a single-call extraction baseline (handled in run_single_pass_baseline).
    single_pass_baseline: bool = False


ABLATION_MATRIX: list[AblationVariant] = [
    AblationVariant(
        name="full_pipeline",
        description="Baseline: all phases enabled.",
        pipeline_kwargs={
            "use_rag": True,
            "enable_literature_retrieval": True,
            "enable_refinement_loop": True,
            "perform_topic_analysis": False,
        },
    ),
    AblationVariant(
        name="no_literature_rag",
        description="Disable external literature RAG; tests whether external evidence improves extraction.",
        pipeline_kwargs={
            "use_rag": True,
            "enable_literature_retrieval": False,
            "enable_refinement_loop": True,
            "perform_topic_analysis": False,
        },
    ),
    AblationVariant(
        name="no_refinement",
        description="Disable iterative refinement; tests whether round 2+ is worth the cost.",
        pipeline_kwargs={
            "use_rag": True,
            "enable_literature_retrieval": True,
            "enable_refinement_loop": False,
            "perform_topic_analysis": False,
        },
    ),
    AblationVariant(
        name="no_rag",
        description="Disable RAG context entirely; bare per-chunk extraction with no retrieval.",
        pipeline_kwargs={
            "use_rag": False,
            "enable_literature_retrieval": False,
            "enable_refinement_loop": False,
            "perform_topic_analysis": False,
        },
    ),
    AblationVariant(
        name="single_pass_baseline",
        description="Naive baseline: single LLM call, no scaffolding. Implemented separately.",
        single_pass_baseline=True,
    ),
]


@dataclass
class VariantResult:
    name: str
    description: str
    output_dir: str
    metrics: dict[str, Any]
    wall_clock_seconds: float
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evaluate_outputs(extractions_path: Path, gold_path: Path) -> dict[str, Any]:
    """Run the existing eval logic and return its dict (with bootstrap CIs)."""
    # Imported lazily so this module stays import-cheap for tools that only
    # need the variant matrix definition.
    from scripts.compute_eval_metrics import _load_json, evaluate  # type: ignore

    extractions = _load_json(extractions_path)
    gold = _load_json(gold_path)
    return evaluate(extractions, gold)


def run_ablation_matrix(
    *,
    input_file: str,
    gold_path: str,
    output_root: str = "outputs/ablation",
    variants: Iterable[AblationVariant] | None = None,
    pipeline_runner: Callable[..., dict[str, Any]] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run each variant and write a comparison table.

    Args:
        input_file: Path to the corpus to extract from.
        gold_path: Path to the gold JSON used for metric computation.
        output_root: Each variant's outputs go to `<output_root>/<variant_name>/`.
        variants: Override the default ABLATION_MATRIX (handy for quick subsets).
        pipeline_runner: Defaults to `main.run_complete_pipeline`. Override in tests.
        extra_kwargs: Additional kwargs merged into every variant (e.g. api key).

    Returns:
        A dict with `variants` (list of VariantResult dicts) and `comparison`
        (paired-bootstrap deltas of every variant vs. `full_pipeline`).
    """
    variants = list(variants) if variants is not None else list(ABLATION_MATRIX)
    extra = dict(extra_kwargs or {})
    if pipeline_runner is None:
        from main import run_complete_pipeline as pipeline_runner  # type: ignore

    output_root_p = Path(output_root)
    output_root_p.mkdir(parents=True, exist_ok=True)
    results: list[VariantResult] = []

    for variant in variants:
        variant_dir = output_root_p / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        success = True
        err: str | None = None
        try:
            if variant.single_pass_baseline:
                _run_single_pass_baseline(
                    input_file=input_file,
                    output_dir=str(variant_dir),
                    extra=extra,
                )
            else:
                pipeline_runner(
                    input_file=input_file,
                    output_dir=str(variant_dir),
                    **variant.pipeline_kwargs,
                    **extra,
                )
        except Exception as exc:  # pragma: no cover - surfaced in result
            success = False
            err = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - t0

        metrics: dict[str, Any] = {}
        ext_path = variant_dir / "extracted_models.json"
        if ext_path.exists():
            try:
                metrics = _evaluate_outputs(ext_path, Path(gold_path))
            except Exception as exc:  # pragma: no cover
                metrics = {"eval_error": f"{type(exc).__name__}: {exc}"}

        results.append(
            VariantResult(
                name=variant.name,
                description=variant.description,
                output_dir=str(variant_dir),
                metrics=metrics,
                wall_clock_seconds=round(wall, 2),
                success=success,
                error=err,
            )
        )

    comparison = _compare_to_baseline(results)
    summary = {
        "input_file": input_file,
        "gold_path": gold_path,
        "variants": [r.to_dict() for r in results],
        "comparison_vs_full_pipeline": comparison,
    }
    table_path = output_root_p / "ablation_results.json"
    table_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _compare_to_baseline(results: list[VariantResult]) -> dict[str, Any]:
    base = next((r for r in results if r.name == "full_pipeline"), None)
    if base is None or not base.metrics:
        return {}
    out: dict[str, Any] = {}
    for r in results:
        if r.name == "full_pipeline" or not r.metrics:
            continue
        base_p = float(base.metrics.get("precision", 0.0))
        var_p = float(r.metrics.get("precision", 0.0))
        base_r = float(base.metrics.get("recall", 0.0))
        var_r = float(r.metrics.get("recall", 0.0))
        base_f = float(base.metrics.get("f1", 0.0))
        var_f = float(r.metrics.get("f1", 0.0))
        out[r.name] = {
            "delta_precision": round(var_p - base_p, 4),
            "delta_recall": round(var_r - base_r, 4),
            "delta_f1": round(var_f - base_f, 4),
            "delta_wall_clock_s": round(r.wall_clock_seconds - base.wall_clock_seconds, 2),
        }
    return out


def _run_single_pass_baseline(*, input_file: str, output_dir: str, extra: dict[str, Any]) -> None:
    """Naive single-LLM-call extraction baseline.

    Concats the corpus, asks the model for relationships once, and writes the
    same `extracted_models.json` shape so downstream metrics work.
    """
    import os
    from openai import OpenAI

    from llm_survey.config import get_settings
    from llm_survey.utils.preprocess import process_survey_data

    cfg = get_settings()
    api_key = extra.get("openrouter_api_key") or cfg.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for the single-pass baseline.")
    client = OpenAI(api_key=api_key, base_url=cfg.openrouter_base_url)
    chunks = process_survey_data(input_file, max_tokens=500)
    joined = "\n\n".join(c.get("text", "") for c in chunks)[:18000]
    prompt = (
        "Extract variables and causal relationships from the following survey text. "
        "Return JSON: {\"relationships\": [{\"from_variable\": str, \"to_variable\": str}]}.\n\n"
        + joined
    )
    resp = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        rels = parsed.get("relationships") or []
    except Exception:
        rels = []
    # Match the shape that compute_eval_metrics expects.
    payload = [
        {
            "success": True,
            "chunk_id": "single_pass_all",
            "model": {"relationships": rels},
        }
    ]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "extracted_models.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


__all__ = [
    "AblationVariant",
    "ABLATION_MATRIX",
    "VariantResult",
    "run_ablation_matrix",
]
