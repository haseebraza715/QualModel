from __future__ import annotations

import json
import os
import re
import urllib.error
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

import instructor
import yaml
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, BadRequestError, OpenAI, RateLimitError
from pydantic import ValidationError

from llm_survey.agents import (
    ClarificationAgent,
    ConflictDetector,
    CrossChunkGapDetector,
    LiteratureValidator,
    ModelConsolidator,
)
from llm_survey.eval.cost import RunRecorder
from llm_survey.eval.runlog import RunLog
from llm_survey.prompts.model_extraction_prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    format_structured_extraction_prompt,
)
from llm_survey.prompts.registry import default_registry
from llm_survey.rag import CachedEmbedder, LiteratureStore, PubMedClient, SemanticScholarClient, SurveyStore
from llm_survey.schemas.consolidation import ConsolidatedModel, ScoredHypothesis
from llm_survey.schemas.extraction import ChunkExtractionResult
from llm_survey.utils.cost_estimate import count_tokens
from llm_survey.utils.export_reports import (
    build_causal_graph_html,
    build_evidence_report_markdown,
    build_final_model_spec_yaml,
    build_mermaid_diagram,
)
from llm_survey.utils.preprocess import (
    generate_run_id,
    process_survey_data,
    save_processed_data,
    save_processed_data_for_run,
)
from llm_survey.utils.prompt_safety import build_refinement_user_message, build_thematic_analysis_user_message

def _inject_provenance(model: Dict[str, Any], chunk_id: str) -> None:
    if not chunk_id or not isinstance(model, dict):
        return
    for key in ("variables", "relationships", "moderators"):
        for item in model.get(key) or []:
            if isinstance(item, dict) and not item.get("source_chunk_ids"):
                item["source_chunk_ids"] = [chunk_id]
    for hyp in model.get("hypotheses") or []:
        if isinstance(hyp, dict) and not hyp.get("source_chunk_ids"):
            hyp["source_chunk_ids"] = [chunk_id]


def _is_empty_extraction(model: ChunkExtractionResult) -> bool:
    data = model.model_dump()
    return not (
        data.get("variables")
        or data.get("relationships")
        or data.get("hypotheses")
        or data.get("moderators")
    )


def summarize_extraction_failures(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts for dashboard / gap-detection warnings."""
    total = len(results)
    by_kind: Dict[str, int] = {"api_error": 0, "parse_error": 0, "empty_extraction": 0}
    for row in results:
        if row.get("success"):
            continue
        kind = row.get("failure_kind") or "api_error"
        if kind in by_kind:
            by_kind[kind] += 1
    failed = sum(1 for r in results if not r.get("success"))
    return {
        "total_chunks": total,
        "failed_chunks": failed,
        "failure_rate": (failed / total) if total else 0.0,
        "by_kind": by_kind,
    }


class RAGModelExtractor:
    """Pipeline for ingestion, dual-RAG retrieval, and typed extraction."""

    def __init__(
        self,
        openai_api_key: str = "",
        llm_model: str = "google/gemma-4-31b-it",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        extra_headers: Dict[str, str] | None = None,
        survey_chroma_path: str = "data/chroma/survey",
        survey_collection: str = "survey",
        literature_chroma_path: str = "data/chroma/literature",
        literature_collection: str = "literature",
        max_retries: int = 2,
        enable_literature_retrieval: bool = True,
        literature_target_papers: int = 20,
    ):
        # Read directly from env vars (populated by `load_dotenv()` in entry
        # points). The typed Settings object is used elsewhere; here we keep
        # env-var semantics so tests can `monkeypatch.delenv` to assert this
        # constructor refuses to run without a key.
        api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required (OPENROUTER_API_KEY).")

        self.embedding_model_name = embedding_model
        self.embedder = CachedEmbedder(model_name=embedding_model)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers or {},
        )
        self.structured_client = instructor.from_openai(self.client, mode=instructor.Mode.JSON)

        self.llm_model = llm_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_literature_retrieval = enable_literature_retrieval
        self.literature_target_papers = min(20, max(1, int(literature_target_papers)))

        self.survey_store = SurveyStore(
            persist_dir=survey_chroma_path,
            collection_name=survey_collection,
            embedder=self.embedder,
        )
        self.literature_store = LiteratureStore(
            persist_dir=literature_chroma_path,
            collection_name=literature_collection,
            embedder=self.embedder,
        )

        self.semantic_scholar = SemanticScholarClient()
        self.pubmed = PubMedClient()
        self.gap_detector = CrossChunkGapDetector()
        self.clarification_agent = ClarificationAgent()
        self.consolidator = ModelConsolidator()
        self.conflict_detector = ConflictDetector()
        self.literature_validator = LiteratureValidator()

        self.processed_chunks: List[Dict[str, Any]] = []
        self.run_id: str = generate_run_id("pipeline")

        # Provenance and cost tracking — every LLM call is recorded against the
        # currently-active phase (set via `with self.recorder.phase(name): ...`).
        # See dump_run_artifacts() to persist the report alongside outputs.
        self.recorder = RunRecorder(model=self.llm_model)
        self.run_log = RunLog(
            run_id=self.run_id,
            model=self.llm_model,
            temperature=float(self.temperature),
            seed=20260101,
            embedding_model=self.embedding_model_name,
        )
        try:
            registry = default_registry()
            for version in registry.list_versions():
                for md in (registry.root / version).glob("*.md"):
                    rec = registry.get(md.stem, version=version)
                    self.run_log.attach_prompt(f"{version}/{md.stem}", rec.sha256)
        except Exception:  # pragma: no cover - registry is optional
            pass
        self.run_log.attach_lockfile_hash()

    @staticmethod
    def _usage_from_raw(raw: Any) -> tuple[int | None, int | None]:
        """Extract (prompt_tokens, completion_tokens) from a raw ChatCompletion.

        Returns (None, None) if the SDK didn't surface a `usage` field — caller
        should then fall back to a tiktoken estimate.
        """
        usage = getattr(raw, "usage", None) if raw is not None else None
        if usage is None:
            return None, None
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is None or completion is None:
            return None, None
        try:
            return int(prompt), int(completion)
        except (TypeError, ValueError):
            return None, None

    def _record_llm_call(
        self,
        *,
        phase: str | None,
        wall_seconds: float,
        prompt_text: str,
        completion_text: str = "",
        raw_completion: Any = None,
    ) -> None:
        """Record one LLM call. Uses real `usage` tokens when the SDK returns
        them; otherwise falls back to a tiktoken estimate from string lengths.

        Both LLM call sites (`extract_model_from_chunk` via instructor's
        structured-output mode, and `_call_yaml` via the raw OpenAI client)
        funnel through here so the timing/recording pattern lives in one place.
        """
        p_toks, c_toks = self._usage_from_raw(raw_completion)
        if p_toks is None or c_toks is None:
            try:
                p_toks = count_tokens(prompt_text or "", self.llm_model)
                c_toks = count_tokens(completion_text or "", self.llm_model)
            except Exception:
                p_toks, c_toks = 0, 0
        self.recorder.record_llm_call(
            prompt_tokens=int(p_toks or 0),
            completion_tokens=int(c_toks or 0),
            wall_seconds=wall_seconds,
            phase=phase,
        )

    def dump_run_artifacts(self, output_dir: str) -> Dict[str, str]:
        """Write `cost_report.json` + `runlog.json` for the current run.

        Safe to call multiple times; later calls overwrite earlier ones.
        """
        os.makedirs(output_dir, exist_ok=True)
        cost_path = os.path.join(output_dir, "cost_report.json")
        runlog_path = os.path.join(output_dir, "runlog.json")
        self.recorder.dump(cost_path)
        self.run_log.dump(runlog_path)
        return {"cost_report": cost_path, "runlog": runlog_path}

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_text(path: str, payload: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)

    def process_and_store_data(
        self,
        file_path: str,
        max_tokens: int = 500,
        save_processed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process source files and persist chunk embeddings into survey store."""
        self.run_id = generate_run_id("pipeline")
        print(f"Processing data from {file_path}...")
        self.processed_chunks = process_survey_data(file_path, max_tokens=max_tokens)

        store_stats = self.survey_store.add_chunks(self.processed_chunks)
        print(
            f"Survey store updated. Added={store_stats['added']} "
            f"Skipped(existing hash)={store_stats['skipped']}"
        )

        if save_processed:
            latest_output = "data/processed/processed_chunks.json"
            save_processed_data(self.processed_chunks, latest_output)
            run_output = save_processed_data_for_run(self.processed_chunks, run_id=self.run_id)
            print(f"Processed data saved to {latest_output}")
            print(f"Run-scoped processed data saved to {run_output}")

        if self.enable_literature_retrieval:
            self._populate_literature_store(self.processed_chunks)

        return self.processed_chunks

    def _extract_topic_queries(self, chunks: List[Dict[str, Any]], max_queries: int = 8) -> List[str]:
        """Generate literature search queries from survey chunk corpus."""
        stop_words = {
            "the",
            "and",
            "for",
            "that",
            "with",
            "this",
            "from",
            "have",
            "been",
            "were",
            "when",
            "what",
            "where",
            "there",
            "their",
            "about",
            "into",
            "them",
            "they",
            "your",
            "you",
            "our",
            "not",
            "but",
            "are",
            "too",
            "very",
            "just",
            "more",
        }

        tokens: List[str] = []
        for chunk in chunks:
            words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", chunk.get("text", "").lower())
            tokens.extend([w for w in words if w not in stop_words])

        if not tokens:
            return []

        top_terms = [word for word, _ in Counter(tokens).most_common(max_queries * 4)]
        queries: List[str] = []
        for i in range(0, len(top_terms), 3):
            phrase = " ".join(top_terms[i : i + 3]).strip()
            if len(phrase.split()) >= 2:
                queries.append(phrase)
            if len(queries) >= max_queries:
                break

        if not queries:
            queries = [" ".join(top_terms[:3])]

        return queries

    def _populate_literature_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Fetch and index literature abstracts from Semantic Scholar + PubMed."""
        queries = self._extract_topic_queries(chunks)
        if not queries:
            print("Literature enrichment skipped: no viable search queries.")
            return

        papers_per_query = max(5, self.literature_target_papers // max(1, len(queries) * 2))
        gathered: List[Dict[str, Any]] = []

        for query in queries:
            try:
                gathered.extend(self.semantic_scholar.search_papers(query, limit=papers_per_query))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError, ValueError) as err:
                print(f"Semantic Scholar lookup failed for '{query}': {err}")

            try:
                gathered.extend(self.pubmed.search_papers(query, limit=papers_per_query))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError, ValueError) as err:
                print(f"PubMed lookup failed for '{query}': {err}")

            if len(gathered) >= self.literature_target_papers:
                break

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for paper in gathered:
            key = f"{paper.get('source')}::{paper.get('paper_id')}"
            abstract = (paper.get("abstract") or "").strip()
            if not abstract or key in seen:
                continue
            seen.add(key)
            deduped.append(paper)

        if not deduped:
            print("Literature enrichment completed with 0 usable abstracts.")
            return

        store_stats = self.literature_store.add_papers(deduped[: self.literature_target_papers])
        print(
            f"Literature store updated. Added={store_stats['added']} "
            f"Skipped(existing/empty)={store_stats['skipped']}"
        )

    def extract_model_from_chunk(
        self,
        chunk_text: str,
        use_rag: bool = True,
        num_context_docs: int = 3,
        num_literature_docs: int = 3,
        enriched_context: str = "",
        chunk_id: str = "",
    ) -> Dict[str, Any]:
        """Extract one structured model from a chunk using survey + literature context."""
        survey_context = ""
        literature_context = ""

        if use_rag:
            survey_context = self.survey_store.format_context(text=chunk_text, k=num_context_docs)
            literature_context = self.literature_store.format_context(text=chunk_text, k=num_literature_docs)

        if enriched_context.strip():
            literature_context = (
                f"{literature_context}\n\nRefinement Context:\n{enriched_context}".strip()
                if literature_context.strip()
                else f"Refinement Context:\n{enriched_context}".strip()
            )

        prompt = format_structured_extraction_prompt(
            chunk_text=chunk_text,
            survey_context=survey_context,
            literature_context=literature_context,
        )
        if chunk_id:
            prompt = f"{prompt}\n\nExtraction metadata:\nchunk_id: {chunk_id}\n"

        base_out: Dict[str, Any] = {
            "survey_context": survey_context,
            "literature_context": literature_context,
            "failure_kind": None,
        }

        import time as _time
        _t0 = _time.perf_counter()
        _full_prompt = EXTRACTION_SYSTEM_PROMPT + "\n" + prompt
        _raw_completion: Any = None
        try:
            # Prefer `create_with_completion` so we can read real
            # `usage.prompt_tokens` / `completion_tokens` from the raw chat
            # completion. Fall back to `create(...)` for older `instructor`
            # versions and for test mocks that only stub `create`.
            _completions = self.structured_client.chat.completions
            _create_kwargs = dict(
                model=self.llm_model,
                temperature=self.temperature,
                response_model=ChunkExtractionResult,
                max_retries=self.max_retries + 1,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            if hasattr(_completions, "create_with_completion"):
                response_model, _raw_completion = _completions.create_with_completion(**_create_kwargs)
            else:
                response_model = _completions.create(**_create_kwargs)
                _raw_completion = None
        except (RateLimitError, APITimeoutError, APIConnectionError, AuthenticationError, BadRequestError, APIError) as err:
            self._record_llm_call(
                phase="extraction",
                wall_seconds=_time.perf_counter() - _t0,
                prompt_text=_full_prompt,
            )
            return {
                **base_out,
                "model": None,
                "raw_response": "",
                "success": False,
                "error": str(err),
                "failure_kind": "api_error",
            }
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as err:
            self._record_llm_call(
                phase="extraction",
                wall_seconds=_time.perf_counter() - _t0,
                prompt_text=_full_prompt,
            )
            return {
                **base_out,
                "model": None,
                "raw_response": "",
                "success": False,
                "error": str(err),
                "failure_kind": "parse_error",
            }

        _completion_text = response_model.model_dump_json()
        self._record_llm_call(
            phase="extraction",
            wall_seconds=_time.perf_counter() - _t0,
            prompt_text=_full_prompt,
            completion_text=_completion_text,
            raw_completion=_raw_completion,
        )

        model_dict = response_model.model_dump()
        _inject_provenance(model_dict, chunk_id)
        if _is_empty_extraction(response_model):
            return {
                **base_out,
                "model": model_dict,
                "raw_response": response_model.model_dump_json(),
                "success": False,
                "error": "Extraction returned no variables, relationships, hypotheses, or moderators.",
                "failure_kind": "empty_extraction",
            }

        return {
            **base_out,
            "model": model_dict,
            "raw_response": response_model.model_dump_json(),
            "success": True,
            "failure_kind": None,
        }

    def extract_models_from_all_chunks(
        self,
        use_rag: bool = True,
        num_context_docs: int = 3,
        num_literature_docs: int = 3,
        save_results: bool = True,
        enriched_context: str = "",
        output_suffix: str = "",
    ) -> List[Dict[str, Any]]:
        """Extract structured models from all available chunks."""
        if not self.processed_chunks:
            raise ValueError("No processed chunks available. Run process_and_store_data first.")

        results: List[Dict[str, Any]] = []
        print(f"Extracting models from {len(self.processed_chunks)} chunks...")
        with self.recorder.phase("extraction"):
            for i, chunk in enumerate(self.processed_chunks, start=1):
                print(f"Processing chunk {i}/{len(self.processed_chunks)}")
                result = self.extract_model_from_chunk(
                    chunk_text=chunk["text"],
                    use_rag=use_rag,
                    num_context_docs=num_context_docs,
                    num_literature_docs=num_literature_docs,
                    enriched_context=enriched_context,
                    chunk_id=str(chunk.get("id", "")),
                )
                result["chunk_id"] = chunk["id"]
                result["chunk_metadata"] = chunk["metadata"]
                result["chunk_text"] = chunk["text"]
                results.append(result)

        if save_results:
            suffix = f"_{output_suffix}" if output_suffix else ""
            latest_path = f"outputs/extracted_models{suffix}.json"
            run_path = f"outputs/extracted_models_{self.run_id}{suffix}.json"
            self._write_json(latest_path, results)
            self._write_json(run_path, results)
            summary_path = f"outputs/extraction_failure_summary{suffix}.json"
            self._write_json(summary_path, summarize_extraction_failures(results))
            print(f"Results saved to {latest_path}")
            print(f"Run-scoped extraction results saved to {run_path}")

        return results

    def detect_cross_chunk_gaps(
        self,
        extraction_results: List[Dict[str, Any]],
        save_results: bool = True,
        output_suffix: str = "",
    ) -> Dict[str, Any]:
        """Detect cross-chunk gaps and score completeness/testability."""
        with self.recorder.phase("gap_detection"):
            report_model = self.gap_detector.detect(extraction_results)
            report = report_model.model_dump()

        if save_results:
            suffix = f"_{output_suffix}" if output_suffix else ""
            latest_path = f"outputs/cross_chunk_gap_report{suffix}.json"
            run_path = f"outputs/cross_chunk_gap_report_{self.run_id}{suffix}.json"
            self._write_json(latest_path, report)
            self._write_json(run_path, report)
            print(f"Cross-chunk gap report saved to {latest_path}")
            print(f"Run-scoped gap report saved to {run_path}")

        return report

    def generate_clarification_plan(
        self,
        gap_report: Dict[str, Any],
        save_results: bool = True,
        auto_answer_top_k: int = 3,
        output_suffix: str = "",
    ) -> Dict[str, Any]:
        """Convert gap report into actionable clarification questions."""
        with self.recorder.phase("clarification"):
            plan_model = self.clarification_agent.build_plan(
                gap_report=gap_report,
                literature_store=self.literature_store,
                auto_answer_top_k=auto_answer_top_k,
            )
            plan = plan_model.model_dump()

        if save_results:
            suffix = f"_{output_suffix}" if output_suffix else ""
            latest_path = f"outputs/clarification_plan{suffix}.json"
            run_path = f"outputs/clarification_plan_{self.run_id}{suffix}.json"
            self._write_json(latest_path, plan)
            self._write_json(run_path, plan)
            print(f"Clarification plan saved to {latest_path}")
            print(f"Run-scoped clarification plan saved to {run_path}")

        return plan

    def consolidate_model(
        self,
        extraction_results: List[Dict[str, Any]],
        gap_report: Dict[str, Any] | None = None,
        clarification_plan: Dict[str, Any] | None = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Merge chunk-level extractions into one consolidated model."""
        model = self.consolidator.consolidate(
            extraction_results=extraction_results,
            gap_report=gap_report or {},
            clarification_plan=clarification_plan or {},
        )
        payload = model.model_dump()
        if save_results:
            self._write_json("outputs/consolidated_model.json", payload)
            self._write_json(f"outputs/consolidated_model_{self.run_id}.json", payload)
        return payload

    def detect_conflicts(
        self,
        consolidated_model: Dict[str, Any],
        extraction_results: List[Dict[str, Any]],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Flag contradictions in the consolidated model."""
        report = self.conflict_detector.detect(
            consolidated_model=ConsolidatedModel.model_validate(consolidated_model),
            extraction_results=extraction_results,
            literature_store=self.literature_store if self.enable_literature_retrieval else None,
        )
        payload = report.model_dump()
        if save_results:
            self._write_json("outputs/conflict_report.json", payload)
            self._write_json(f"outputs/conflict_report_{self.run_id}.json", payload)
        return payload

    def validate_hypotheses(
        self,
        consolidated_model: Dict[str, Any],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Validate consolidated hypotheses against the literature store."""
        report = self.literature_validator.validate(
            hypotheses=[ScoredHypothesis.model_validate(row) for row in consolidated_model.get("hypotheses", [])],
            literature_store=self.literature_store if self.enable_literature_retrieval else None,
        )
        payload = report.model_dump()
        if save_results:
            self._write_json("outputs/literature_validation_report.json", payload)
            self._write_json(f"outputs/literature_validation_report_{self.run_id}.json", payload)
        return payload

    @staticmethod
    def _merge_validation_into_model(
        consolidated_model: Dict[str, Any],
        conflict_report: Dict[str, Any] | None,
        validation_report: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        merged = json.loads(json.dumps(consolidated_model))
        merged["contradictions"] = list((conflict_report or {}).get("contradictions", []))
        validation_map = {
            str(row.get("hypothesis_id", "")): row
            for row in (validation_report or {}).get("validations", [])
            if isinstance(row, dict)
        }
        for hypothesis in merged.get("hypotheses", []):
            validation = validation_map.get(str(hypothesis.get("id", "")), {})
            if validation:
                hypothesis["literature_support_score"] = validation.get("literature_support_score", 0.0)
                hypothesis["consensus_strength"] = validation.get("consensus_strength", hypothesis.get("consensus_strength", "weak"))
                hypothesis["novelty_flag"] = validation.get("novelty_flag", False)
        return merged

    def export_final_outputs(
        self,
        consolidated_model: Dict[str, Any],
        conflict_report: Dict[str, Any] | None = None,
        validation_report: Dict[str, Any] | None = None,
        total_chunks: int = 0,
        iterations_completed: int = 0,
        output_dir: str = "outputs",
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Build the final model-spec artifacts for review and sharing."""
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "1.0.0",
            "total_chunks": total_chunks,
            "iterations_completed": iterations_completed,
        }
        yaml_text = build_final_model_spec_yaml(
            consolidated_model=consolidated_model,
            validations=validation_report,
            conflict_report=conflict_report,
            metadata=metadata,
        )
        mermaid_text = build_mermaid_diagram(consolidated_model)
        html_text = build_causal_graph_html(
            consolidated_model=consolidated_model,
            validations=validation_report,
            conflict_report=conflict_report,
        )
        evidence_md = build_evidence_report_markdown(
            consolidated_model=consolidated_model,
            validations=validation_report,
            conflict_report=conflict_report,
        )

        output = {
            "model_spec_yaml": yaml_text,
            "causal_graph_mermaid": mermaid_text,
            "causal_graph_html": html_text,
            "evidence_report_markdown": evidence_md,
            "paths": {},
        }
        if save_results:
            yaml_path = os.path.join(output_dir, "final_model_spec.yaml")
            mermaid_path = os.path.join(output_dir, "causal_graph.mmd")
            html_path = os.path.join(output_dir, "causal_graph.html")
            evidence_path = os.path.join(output_dir, "evidence_report.md")
            self._write_text(yaml_path, yaml_text)
            self._write_text(mermaid_path, mermaid_text)
            self._write_text(html_path, html_text)
            self._write_text(evidence_path, evidence_md)
            self._write_text(os.path.join(output_dir, f"final_model_spec_{self.run_id}.yaml"), yaml_text)
            self._write_text(os.path.join(output_dir, f"causal_graph_{self.run_id}.mmd"), mermaid_text)
            self._write_text(os.path.join(output_dir, f"causal_graph_{self.run_id}.html"), html_text)
            self._write_text(os.path.join(output_dir, f"evidence_report_{self.run_id}.md"), evidence_md)
            output["paths"] = {
                "yaml": yaml_path,
                "mermaid": mermaid_path,
                "html": html_path,
                "evidence_markdown": evidence_path,
            }
        return output

    def finalize_model_outputs(
        self,
        extraction_results: List[Dict[str, Any]],
        gap_report: Dict[str, Any],
        clarification_plan: Dict[str, Any],
        refinement_report: Dict[str, Any] | None = None,
        output_dir: str = "outputs",
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Run consolidation, conflict detection, literature validation, and final exports."""
        with self.recorder.phase("consolidation"):
            consolidated_model = self.consolidate_model(
                extraction_results=extraction_results,
                gap_report=gap_report,
                clarification_plan=clarification_plan,
                save_results=False,
            )
            conflict_report = self.detect_conflicts(
                consolidated_model=consolidated_model,
                extraction_results=extraction_results,
                save_results=False,
            )
            validation_report = self.validate_hypotheses(
                consolidated_model=consolidated_model,
                save_results=False,
            )
        merged_model = self._merge_validation_into_model(
            consolidated_model=consolidated_model,
            conflict_report=conflict_report,
            validation_report=validation_report,
        )
        if save_results:
            self._write_json("outputs/consolidated_model.json", merged_model)
            self._write_json(f"outputs/consolidated_model_{self.run_id}.json", merged_model)
            self._write_json("outputs/conflict_report.json", conflict_report)
            self._write_json(f"outputs/conflict_report_{self.run_id}.json", conflict_report)
            self._write_json("outputs/literature_validation_report.json", validation_report)
            self._write_json(f"outputs/literature_validation_report_{self.run_id}.json", validation_report)
        exports = self.export_final_outputs(
            consolidated_model=merged_model,
            conflict_report=conflict_report,
            validation_report=validation_report,
            total_chunks=len(self.processed_chunks),
            iterations_completed=int((refinement_report or {}).get("iterations_completed", 0)),
            output_dir=output_dir,
            save_results=save_results,
        )
        if save_results:
            try:
                self.dump_run_artifacts(output_dir)
            except Exception:  # pragma: no cover - artifact dump must never break the pipeline
                pass

        return {
            "consolidated_model": merged_model,
            "conflict_report": conflict_report,
            "literature_validation": validation_report,
            "final_exports": exports,
        }

    def _build_enriched_context(self, clarification_plan: Dict[str, Any], gap_report: Dict[str, Any]) -> str:
        """Build enriched context for refinement iterations."""
        priority_gaps = clarification_plan.get("questions", [])[:5]
        auto_answers = clarification_plan.get("auto_answers", [])
        top_gap_descriptions = gap_report.get("priority_gaps", [])[:3]

        segments: List[str] = []
        if top_gap_descriptions:
            segments.append("Priority unresolved gaps:\n- " + "\n- ".join(top_gap_descriptions))

        if priority_gaps:
            q_lines = []
            for q in priority_gaps:
                q_lines.append(f"{q.get('question_id', 'Q?')}: {q.get('question_text', '').strip()}")
            if q_lines:
                segments.append("Clarification questions:\n- " + "\n- ".join(q_lines))

        if auto_answers:
            a_lines = []
            for a in auto_answers[:8]:
                answer_text = str(a.get("answer_text", "")).strip()
                if not answer_text:
                    continue
                a_lines.append(f"{a.get('question_id', 'Q?')}: {answer_text}")
            if a_lines:
                segments.append("Literature-backed clarification answers:\n- " + "\n- ".join(a_lines))

        return "\n\n".join(seg for seg in segments if seg.strip())

    def run_refinement_loop(
        self,
        extraction_results: List[Dict[str, Any]],
        gap_report: Dict[str, Any],
        clarification_plan: Dict[str, Any],
        use_rag: bool = True,
        num_context_docs: int = 3,
        num_literature_docs: int = 3,
        max_iterations: int = 2,
        completeness_threshold: float = 0.75,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Iteratively re-extract with enriched clarification context until completeness threshold or max iterations.
        """
        max_iterations = max(0, max_iterations)
        completeness_threshold = max(0.0, min(1.0, completeness_threshold))

        current_results = extraction_results
        current_gap_report = gap_report
        current_clarification_plan = clarification_plan

        def _coverage(rep: Dict[str, Any]) -> float:
            return float(
                rep.get("structural_coverage_score", rep.get("overall_model_completeness", 0.0)) or 0.0
            )

        history: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "completeness": _coverage(current_gap_report),
                "testability": float(current_gap_report.get("model_testability_score", 0.0) or 0.0),
                "gap_count": len(current_gap_report.get("gaps", [])),
                "question_count": len(current_clarification_plan.get("questions", [])),
                "auto_answer_count": len(current_clarification_plan.get("auto_answers", [])),
            }
        ]

        iterations_completed = 0
        stop_reason = "max_iterations_reached"
        prior_coverage = _coverage(current_gap_report)

        for iteration in range(1, max_iterations + 1):
            completeness = _coverage(current_gap_report)
            if completeness >= completeness_threshold:
                stop_reason = "threshold_reached"
                break

            enriched_context = self._build_enriched_context(
                clarification_plan=current_clarification_plan,
                gap_report=current_gap_report,
            )
            if not enriched_context.strip():
                stop_reason = "no_enriched_context"
                break

            current_results = self.extract_models_from_all_chunks(
                use_rag=use_rag,
                num_context_docs=num_context_docs,
                num_literature_docs=num_literature_docs,
                save_results=save_results,
                enriched_context=enriched_context,
                output_suffix=f"iter_{iteration}",
            )
            current_gap_report = self.detect_cross_chunk_gaps(
                extraction_results=current_results,
                save_results=save_results,
                output_suffix=f"iter_{iteration}",
            )
            current_clarification_plan = self.generate_clarification_plan(
                gap_report=current_gap_report,
                save_results=save_results,
                output_suffix=f"iter_{iteration}",
            )

            new_coverage = _coverage(current_gap_report)
            if new_coverage - prior_coverage < 0.05:
                stop_reason = "convergence_no_coverage_gain"
                iterations_completed = iteration
                history.append(
                    {
                        "iteration": iteration,
                        "completeness": new_coverage,
                        "testability": float(current_gap_report.get("model_testability_score", 0.0) or 0.0),
                        "gap_count": len(current_gap_report.get("gaps", [])),
                        "question_count": len(current_clarification_plan.get("questions", [])),
                        "auto_answer_count": len(current_clarification_plan.get("auto_answers", [])),
                    }
                )
                prior_coverage = new_coverage
                break

            prior_coverage = new_coverage
            iterations_completed = iteration
            history.append(
                {
                    "iteration": iteration,
                    "completeness": new_coverage,
                    "testability": float(current_gap_report.get("model_testability_score", 0.0) or 0.0),
                    "gap_count": len(current_gap_report.get("gaps", [])),
                    "question_count": len(current_clarification_plan.get("questions", [])),
                    "auto_answer_count": len(current_clarification_plan.get("auto_answers", [])),
                }
            )

        if stop_reason == "max_iterations_reached":
            final_completeness = _coverage(current_gap_report)
            if final_completeness >= completeness_threshold:
                stop_reason = "threshold_reached"

        report = {
            "iterations_completed": iterations_completed,
            "max_iterations": max_iterations,
            "completeness_threshold": completeness_threshold,
            "stop_reason": stop_reason,
            "history": history,
            "final_completeness": _coverage(current_gap_report),
            "final_testability": float(current_gap_report.get("model_testability_score", 0.0) or 0.0),
            "final_gap_count": len(current_gap_report.get("gaps", [])),
        }

        if save_results:
            self._write_json("outputs/extracted_models.json", current_results)
            self._write_json(f"outputs/extracted_models_{self.run_id}.json", current_results)
            self._write_json("outputs/cross_chunk_gap_report.json", current_gap_report)
            self._write_json(f"outputs/cross_chunk_gap_report_{self.run_id}.json", current_gap_report)
            self._write_json("outputs/clarification_plan.json", current_clarification_plan)
            self._write_json(f"outputs/clarification_plan_{self.run_id}.json", current_clarification_plan)
            self._write_json("outputs/refinement_loop_report.json", report)
            self._write_json(f"outputs/refinement_loop_report_{self.run_id}.json", report)
            print("Refinement loop report saved to outputs/refinement_loop_report.json")
            print(f"Run-scoped refinement loop report saved to outputs/refinement_loop_report_{self.run_id}.json")

        return {
            "report": report,
            "final_extraction_results": current_results,
            "final_gap_report": current_gap_report,
            "final_clarification_plan": current_clarification_plan,
        }

    def _safe_completion_text(self, completion: Any) -> str:
        """Extract completion text defensively across provider edge cases."""
        choices = getattr(completion, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    def _call_yaml(self, prompt: str) -> Dict[str, Any]:
        import time as _time
        _t0 = _time.perf_counter()
        completion: Any = None
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "Return only valid YAML. No markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_response = self._safe_completion_text(completion)
        except (RateLimitError, APITimeoutError, APIConnectionError, AuthenticationError, BadRequestError, APIError) as err:
            self._record_llm_call(
                phase=None,
                wall_seconds=_time.perf_counter() - _t0,
                prompt_text=prompt,
            )
            return {"payload": None, "raw_response": "", "success": False, "error": str(err)}
        self._record_llm_call(
            phase=None,
            wall_seconds=_time.perf_counter() - _t0,
            prompt_text=prompt,
            completion_text=raw_response,
            raw_completion=completion,
        )
        try:
            return {"payload": yaml.safe_load(raw_response), "raw_response": raw_response, "success": True}
        except yaml.YAMLError as err:
            return {"payload": None, "raw_response": raw_response, "success": False, "error": str(err)}

    def perform_thematic_analysis(
        self,
        text_excerpts: List[str],
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Perform thematic analysis across multiple text excerpts."""
        combined_text = "\n\n---\n\n".join(text_excerpts)

        prompt = build_thematic_analysis_user_message(combined_text)

        with self.recorder.phase("thematic_analysis"):
            response = self._call_yaml(prompt)
        result = {
            "thematic_analysis": response.get("payload"),
            "raw_response": response.get("raw_response"),
            "success": response.get("success", False),
        }
        if not response.get("success", False):
            result["error"] = response.get("error", "Failed to parse YAML.")

        if save_results:
            output_path = "outputs/thematic_analysis.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Thematic analysis saved to {output_path}")

        return result

    def refine_model(
        self,
        original_model: Dict[str, Any],
        context: str = "",
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Refine and validate a model specification."""
        model_yaml = yaml.dump(original_model, default_flow_style=False)

        prompt = build_refinement_user_message(model_yaml, context)

        with self.recorder.phase("refinement"):
            response = self._call_yaml(prompt)
        result = {
            "refined_model": response.get("payload"),
            "raw_response": response.get("raw_response"),
            "success": response.get("success", False),
        }
        if not response.get("success", False):
            result["error"] = response.get("error", "Failed to parse YAML.")

        if save_results:
            output_path = "outputs/model_refinement.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Model refinement saved to {output_path}")

        return result
