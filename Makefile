# Reproducibility entrypoint for the LLM Survey research pipeline.
#
# Targets:
#   make install      Install the package + dev tooling in editable mode.
#   make lint         Run ruff + black --check.
#   make typecheck    Run mypy over src/.
#   make test         Run the offline unit tests.
#   make eval         Recompute docs/evaluation_metrics.json with bootstrap CIs.
#   make ablation     Run the ablation matrix on the synthetic corpus.
#   make reproduce    Full reproduction pipeline (install + eval + ablation + figures).
#
# All targets are idempotent and safe to re-run.

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

.PHONY: install lint format typecheck test eval ablation reproduce clean

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	ruff check src tests scripts
	black --check src tests scripts

format:
	ruff check --fix src tests scripts
	black src tests scripts

typecheck:
	mypy src

test:
	pytest -q -m "not live_api"

eval:
	$(PYTHON) scripts/compute_eval_metrics.py
	@echo ""
	@echo "Wrote docs/evaluation_metrics.json"

ablation:
	$(PYTHON) scripts/run_ablation.py \
		--input data/raw/synthetic_workplace_survey.csv \
		--gold docs/evaluation_gold.json \
		--variant full_pipeline --variant no_refinement --variant no_literature_rag

reproduce: install eval
	@echo ""
	@echo "=== Reproduction complete ==="
	@echo "  Metrics: docs/evaluation_metrics.json"
	@echo "  Re-run with OPENROUTER_API_KEY set to also rebuild raw extractions."

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache .ruff_cache
