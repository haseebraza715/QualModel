# Reproducing this work

This file documents how to reproduce every number in `docs/evaluation_metrics.json`
and the variants in `outputs/ablation/ablation_results.json`.

## Hardware / runtime expectations

| Stage | Wall clock (synthetic, 20 rows) | Cost (USD, indicative) |
|---|---|---|
| `make install` | ~2 min (cold) | $0 |
| `make eval` (offline; no LLM calls) | < 5 sec | $0 |
| `make ablation` (3 variants, default model) | ~3–6 min | < $0.05 |
| Full pipeline + topic analysis | ~5–8 min | < $0.10 |

The offline `make eval` target works without an API key — it recomputes
metrics + bootstrap CIs over the bundled fixture extractions in
`docs/fixtures/`. The ablation and full-pipeline targets require
`OPENROUTER_API_KEY`.

## Environment

```bash
git clone <repo>
cd llm-survey-model-specification

# Editable install pulls dev tooling (ruff, black, mypy, pytest, hypothesis).
make install

# Offline reproduction of metrics from bundled fixtures:
make eval
```

This regenerates `docs/evaluation_metrics.json`. If the result differs from
the committed version, the determinism CI job (see `.github/workflows/ci.yml`)
will fail — that's the trip-wire.

## End-to-end reproduction (requires API key)

```bash
export OPENROUTER_API_KEY=sk-or-...
make reproduce            # install + offline eval
python3 main.py -i data/raw/synthetic_workplace_survey.csv  # full pipeline run
make ablation             # 3-variant ablation matrix
```

The ablation script writes a comparison table to
`outputs/ablation/ablation_results.json` with per-variant precision/recall/F1
deltas vs. `full_pipeline`.

## What pins what

| File | What it freezes |
|---|---|
| `pyproject.toml` | Direct dependency pins, build metadata, lint/format/type config |
| `requirements.lock` | Currently mirrors `requirements.txt`. To freeze transitive deps too, run `pip-compile --output-file=requirements.lock pyproject.toml`. |
| `src/llm_survey/prompts/registry/v1.0/*.md` | Versioned prompt files with sha256 hashes attached to every run |
| `Settings.seed` (default `20260101`) | RNG seed for bootstrap CIs and any future stochastic sampling |
| `Settings.llm_temperature` (default `0.0`) | LLM decoding determinism |

## Run-log provenance

Pipeline runs that use `llm_survey.eval.runlog.RunLog` write a `runlog.json`
alongside outputs. The run log captures: prompt sha256s, model + temperature
+ seed, embedding model, requirements.lock hash, git commit, dirty flag,
Python version, and start/end timestamps. To compare two runs:

```bash
diff -u outputs/run_A/runlog.json outputs/run_B/runlog.json
```

Any non-trivial diff indicates a reproducibility risk.

## Known limitations

- Transitive deps are still floating until `pip-compile` is re-run inside CI.
  See `requirements.lock` header for the command.
- Reranker / embedding-model comparisons in RESEARCH_PLAN §3.3 are not yet
  implemented — the harness scaffolding exists but the alternative backends
  haven't been wired in.
- Multi-corpus ingestion (RESEARCH_PLAN §1.1) requires you to obtain and
  license each corpus individually; the repo only ships the synthetic one.
