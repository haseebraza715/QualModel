#!/usr/bin/env python3
"""Run the ablation matrix from the CLI.

Examples:

  # Full matrix on the synthetic corpus, using the bundled fixture gold:
  python3 scripts/run_ablation.py \\
    --input data/raw/synthetic_workplace_survey.csv \\
    --gold docs/evaluation_gold.json

  # Just one variant (handy when iterating):
  python3 scripts/run_ablation.py --variant no_refinement
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    from llm_survey.eval.ablation import ABLATION_MATRIX, run_ablation_matrix

    parser = argparse.ArgumentParser(description="Run the pipeline ablation matrix.")
    parser.add_argument(
        "--input",
        default=str(_REPO_ROOT / "data" / "raw" / "synthetic_workplace_survey.csv"),
        help="Corpus to extract from.",
    )
    parser.add_argument(
        "--gold",
        default=str(_REPO_ROOT / "docs" / "evaluation_gold.json"),
        help="Gold relationships JSON used for metrics.",
    )
    parser.add_argument("--output-root", default=str(_REPO_ROOT / "outputs" / "ablation"))
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Run only this variant (repeatable). Defaults to the full matrix.",
    )
    args = parser.parse_args()

    if args.variant:
        names = set(args.variant)
        variants = [v for v in ABLATION_MATRIX if v.name in names]
        if not variants:
            raise SystemExit(f"No matching variants. Known: {[v.name for v in ABLATION_MATRIX]}")
    else:
        variants = list(ABLATION_MATRIX)

    summary = run_ablation_matrix(
        input_file=args.input,
        gold_path=args.gold,
        output_root=args.output_root,
        variants=variants,
    )
    print(json.dumps(summary["comparison_vs_full_pipeline"], indent=2))
    print(f"\nFull table: {Path(args.output_root) / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
