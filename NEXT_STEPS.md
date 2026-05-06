# Next Steps — Improvements to the Pipeline

Honest list of weaknesses in the current codebase (post-Round-2 implementation),
ranked by how much each one would actually move the needle, not by how much
code it would take. Companion to [RESEARCH_PLAN.md](RESEARCH_PLAN.md): the
research plan is the strategic roadmap, this file is the tactical punch list.

Each item below has a **Why it matters**, a **Fix sketch**, and a **Cost**
estimate. Items 1–4 unblock real research; items 5–10 clean up debt I shipped
in the last two implementation rounds.

---

## 1. The gold standard is too thin to learn from

**Why it matters.** 15 hand-labeled relationships across 20 synthetic rows means
the bootstrap CIs are basically noise. Recall CI on the bundled fixture is
`[1.0, 1.0]` because every gold item matches — there is no headroom for the
metric to detect a regression. Every other improvement on this list is
invisible until the gold set grows.

**Fix sketch.** Hand-label 100–200 relationships across 2–3 small real
corpora — even single-annotator, no IRR. That's bad for a paper but fine for
catching regressions during development. Inter-rater work (RESEARCH_PLAN §1.2)
can come later when annotators are available.

**Cost.** 1–2 days of focused labeling. No code changes.

---

## 2. Token counts in `cost_report.json` are estimates, not truth

**Why it matters.** `RAGModelExtractor._record_call_estimate` tiktoken-estimates
prompt + completion tokens from string lengths because the `instructor`
structured-output path does not surface the OpenAI `usage` object. The numbers
are good enough for relative comparisons but not for paper figures or vendor
billing reconciliation.

**Fix sketch.** Switch the structured-output call to
`instructor.create_with_completion(...)`, which returns
`(parsed_model, raw_chat_completion)`. Read
`raw.usage.prompt_tokens` / `completion_tokens` directly. For the YAML
fallback path (`_call_yaml`), the raw completion is already in scope — just
read its `usage` field.

**Cost.** ~30 min. Two call sites. The estimator can stay as a fallback for
providers that don't return `usage`.

---

## 3. The eval matcher is substring-based and fragile

**Why it matters.** `_matches()` in `scripts/compute_eval_metrics.py` checks
`from_substring in extracted_from_variable.lower()`. This over-matches ("job"
matches *job satisfaction*, *job crafting*, *side job*) and under-matches
synonyms ("autonomy" vs "self-direction"). A reviewer will spot this in the
first round of peer review.

**Fix sketch.** Two options, in order of effort:

1. (cheap) Add an `aliases` field to each gold item in `evaluation_gold.json`
   and match against the alias set with normalized + lemmatized comparison.
2. (better) Embed gold items + extracted items with a small sentence model and
   match on cosine ≥ threshold; report the threshold + ablate.

**Cost.** Option 1 is ~1 hour. Option 2 is ~half a day plus a calibration sweep.

---

## 4. No retrieval-quality eval

**Why it matters.** The pipeline depends heavily on the survey store and the
literature RAG, but nothing measures whether retrieval returns relevant
context. RESEARCH_PLAN §3.3 calls for Recall@k / nDCG; none exists. Without
it, "did switching the embedder help?" or "did adding a reranker help?" is
unanswerable.

**Fix sketch.** Add a small relevance-judgment fixture (query → list of
chunk_ids that should be retrieved) and a `scripts/eval_retrieval.py` that
computes Recall@1/3/5 and nDCG@5. Wire it into the ablation matrix as a
separate metric column.

**Cost.** Half a day for the harness. Labeling the relevance fixture is the
slow part — 50 query/relevant-doc pairs is the minimum useful size.

---

## 5. The ablation harness re-runs the whole pipeline per variant

**Why it matters.** Five variants × full pipeline runtime = 5× LLM cost. On
the synthetic 20-row corpus this is fine. On a real 500-document corpus it is
a serious blocker — you'll burn $20–50 per ablation run.

**Fix sketch.** Restructure: run the full pipeline once, capture every phase's
intermediate output, and ablate by *replaying* with phases stubbed out. The
extraction phase (the expensive one) only changes when you ablate retrieval
context, not when you ablate gap detection or refinement. Cache extraction
outputs by `(model, prompt_sha256, chunk_id, retrieval_context_sha256)`.

**Cost.** 1–2 days. Worth doing before the multi-corpus phase.

---

## 6. `_record_call_estimate` is duplicated four times in `rag_pipeline.py`

**Why it matters.** The pattern `_t0 = perf_counter(); try: call; except: record;
record after success` is copy-pasted with subtle variations. The
structured-extraction parse-error branch records *inside* the except, the
success branch records *outside* the try/except — both work, but inconsistency
is how "I forgot to instrument this branch" bugs creep in.

**Fix sketch.** One small helper:

```python
def _timed_llm_call(self, *, phase, prompt_text, fn):
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception:
        self._record_call_estimate(prompt_text=prompt_text, completion_text="",
                                   wall_seconds=time.perf_counter() - t0, phase=phase)
        raise
    self._record_call_estimate(prompt_text=prompt_text,
                               completion_text=_extract_completion_text(result),
                               wall_seconds=time.perf_counter() - t0, phase=phase)
    return result
```

**Cost.** ~30 min. Pure refactor, no behavior change.

---

## 7. Tests don't exercise any of the new modules

**Why it matters.** No unit tests for `eval/stats.py`, `eval/cost.py`,
`eval/runlog.py`, `eval/ablation.py`, `prompts/registry.py`, `config.py`, or
`logging_config.py`. The CI eval-stability job catches *one* property
(byte-identical output across two runs) — that's determinism, not correctness.

**Fix sketch.** Add ~50 lines of tests:

- `test_stats.py`: property-based test (via `hypothesis`) that the bootstrap CI
  contains the point estimate; that paired-bootstrap on identical inputs
  returns 0 ± noise; that McNemar with `b=c` gives p=1.0.
- `test_registry.py`: round-trip — write a known prompt, load it, assert
  sha256 matches, assert frontmatter parses.
- `test_runlog.py`: dump → reload → equality.
- `test_cost.py`: phase context manager attributes wall-clock correctly;
  USD totals sum.

**Cost.** ~2 hours. High ROI — catches real bugs in the modules I just
shipped.

---

## 8. Prompts are now duplicated between two sources of truth

**Why it matters.** `src/llm_survey/prompts/model_extraction_prompts.py` still
defines `BASE_EXTRACTION_PROMPT`, `RAG_ENHANCED_PROMPT`, etc. as Python
constants. The same text now also lives in
`src/llm_survey/prompts/registry/v1.0/*.md`. They will drift the moment
someone edits one and not the other.

**Fix sketch.** Have `model_extraction_prompts.py` load from the registry on
import:

```python
from llm_survey.prompts.registry import default_registry
_r = default_registry()
EXTRACTION_SYSTEM_PROMPT = _r.text("extraction_system")
BASE_EXTRACTION_PROMPT  = _r.text("base_extraction")
# etc.
```

Old call sites keep working; the registry is the single source of truth.

**Cost.** ~20 min.

---

## 9. `runlog.json` records `git_dirty` but not the actual diff

**Why it matters.** If `git_dirty=True`, the recorded `git_commit` is a lie —
the actual code that ran is `commit + uncommitted changes`. Anyone trying to
reproduce will get different numbers and not know why.

**Fix sketch.** Two options:

1. (paranoid) Refuse to run unless either the tree is clean or `--allow-dirty`
   is passed.
2. (forgiving) When dirty, save `git diff HEAD > runlog.diff` next to
   `runlog.json` and reference it from the run log.

Option 2 is lower friction during dev, option 1 is what serious labs do for
paper figures. Could do both, gated by an env var.

**Cost.** ~30 min for option 2.

---

## 10. Structured logging is wired but unused

**Why it matters.** `logging_config.py` provides a structlog-backed
`get_logger()`, but nothing in the pipeline actually calls it — every phase
still uses `print(...)`. The JSON-log story is currently aspirational. When an
eval run fails at hour 3 of a 4-hour run, you'll wish there were grep-able
events instead of free-text prints.

**Fix sketch.** Replace `print(f"Processing chunk {i}/{n}")` style lines in
`rag_pipeline.py` with `log.info("chunk_processing", i=i, n=n, chunk_id=cid)`.
Don't try to convert everything at once — start with the per-phase boundary
events (extraction start/end, gap-detection start/end). Keep the
human-readable prints in `main.py` since the CLI is end-user-facing.

**Cost.** ~1 hour for the high-value events. Spread the rest over time.

---

## Suggested ordering

If working solo and time-boxed, do them in this order:

1. **#2 + #6** as a quick cleanup batch (~1 hour total). Both fix things I
   shipped half-baked in the last round.
2. **#8** to prevent prompt drift before it happens (~20 min).
3. **#7** so the new modules are actually tested (~2 hours).
4. **#1** — grow the gold set. Without this everything else is invisible.
5. **#3** — fix the matcher, which becomes the next bottleneck once #1 lands.
6. **#4 + #5** — retrieval eval and replay-based ablation; these unblock the
   multi-corpus / reranker work in RESEARCH_PLAN §1.1 and §3.3.
7. **#9, #10** — quality-of-life polish; do whenever convenient.

After #1–5 the project is in a meaningfully stronger position to start the
multi-corpus and human-eval work that the RESEARCH_PLAN actually cares about.
