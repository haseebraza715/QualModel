"""Unit tests for `llm_survey.eval.cost` and `llm_survey.eval.runlog`."""
from __future__ import annotations

import json
import time
from pathlib import Path

from llm_survey.eval.cost import RunRecorder
from llm_survey.eval.runlog import RunLog


# ----- RunRecorder ----------------------------------------------------------


def test_recorder_attributes_calls_to_active_phase() -> None:
    rec = RunRecorder(model="google/gemma-4-31b-it")
    with rec.phase("extraction"):
        rec.record_llm_call(prompt_tokens=100, completion_tokens=50, wall_seconds=1.0)
        rec.record_llm_call(prompt_tokens=200, completion_tokens=80, wall_seconds=1.5)
    with rec.phase("consolidation"):
        rec.record_llm_call(prompt_tokens=50, completion_tokens=20, wall_seconds=0.7)

    summary = rec.summary()
    assert summary["per_phase"]["extraction"]["calls"] == 2
    assert summary["per_phase"]["extraction"]["prompt_tokens"] == 300
    assert summary["per_phase"]["extraction"]["completion_tokens"] == 130
    assert summary["per_phase"]["consolidation"]["calls"] == 1


def test_recorder_totals_sum_per_phase_values() -> None:
    rec = RunRecorder(model="google/gemma-4-31b-it")
    with rec.phase("a"):
        rec.record_llm_call(prompt_tokens=10, completion_tokens=5, wall_seconds=0.1)
    with rec.phase("b"):
        rec.record_llm_call(prompt_tokens=20, completion_tokens=10, wall_seconds=0.2)

    summary = rec.summary()
    assert summary["totals"]["calls"] == 2
    assert summary["totals"]["prompt_tokens"] == 30
    assert summary["totals"]["completion_tokens"] == 15
    # Totals USD = sum of per-phase USD (rounding dependent but should agree).
    sum_per_phase = sum(d["estimated_usd"] for d in summary["per_phase"].values())
    assert abs(summary["totals"]["estimated_usd"] - round(sum_per_phase, 6)) < 1e-9


def test_recorder_calls_outside_phase_default_to_unknown() -> None:
    rec = RunRecorder(model="google/gemma-4-31b-it")
    rec.record_llm_call(prompt_tokens=1, completion_tokens=1, wall_seconds=0.1)
    summary = rec.summary()
    assert "unknown" in summary["per_phase"]


def test_recorder_dump_writes_valid_json(tmp_path: Path) -> None:
    rec = RunRecorder(model="google/gemma-4-31b-it")
    with rec.phase("extraction"):
        rec.record_llm_call(prompt_tokens=10, completion_tokens=5, wall_seconds=0.1)
    out = tmp_path / "cost.json"
    rec.dump(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["model"] == "google/gemma-4-31b-it"
    assert payload["totals"]["calls"] == 1


def test_recorder_unknown_model_zero_cost() -> None:
    rec = RunRecorder(model="totally/not-a-real-model")
    with rec.phase("x"):
        rec.record_llm_call(prompt_tokens=1000, completion_tokens=1000, wall_seconds=0.1)
    summary = rec.summary()
    # Unknown model -> price table returns 0 -> estimated_usd is 0.
    assert summary["totals"]["estimated_usd"] == 0.0


# ----- RunLog ---------------------------------------------------------------


def test_runlog_dump_and_reload(tmp_path: Path) -> None:
    log = RunLog(
        run_id="test-001",
        model="google/gemma-4-31b-it",
        temperature=0.0,
        seed=20260101,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    log.attach_prompt("v1.0/extraction_system", "deadbeef" * 8)
    out = tmp_path / "runlog.json"
    log.dump(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test-001"
    assert payload["temperature"] == 0.0
    assert payload["prompts"]["v1.0/extraction_system"] == "deadbeef" * 8
    assert payload["ended_at"] is not None


def test_runlog_finalize_records_end_time() -> None:
    log = RunLog(
        run_id="test-002",
        model="x",
        temperature=0.0,
        seed=1,
        embedding_model="y",
    )
    started = log.started_at
    time.sleep(0.01)
    log.finalize()
    assert log.ended_at is not None
    assert log.ended_at >= started
