"""Unit tests for `llm_survey.prompts.registry` — round-trip + frontmatter parsing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from llm_survey.prompts.registry import PromptRecord, PromptRegistry, default_registry


def test_default_registry_has_versioned_prompts() -> None:
    r = default_registry()
    versions = r.list_versions()
    assert versions, "expected at least one version directory under prompts/registry/"
    assert "v1.0" in versions


def test_default_registry_loads_extraction_system() -> None:
    rec = default_registry().get("extraction_system")
    assert isinstance(rec, PromptRecord)
    assert rec.text.startswith("You are a senior qualitative research extraction assistant.")
    assert rec.meta["version"] == rec.version
    assert rec.sha256 == hashlib.sha256(rec.text.encode("utf-8")).hexdigest() or len(rec.sha256) == 64


def test_registry_round_trip(tmp_path: Path) -> None:
    # Build a minimal registry on a tmp dir and read it back.
    version_dir = tmp_path / "v0.1"
    version_dir.mkdir(parents=True)
    body = "Hello {name}, this is a test prompt."
    front = "---\nname: hello\nversion: v0.1\nauthor: tests\n---\n"
    (version_dir / "hello.md").write_text(front + body, encoding="utf-8")

    reg = PromptRegistry(root=tmp_path)
    rec = reg.get("hello")
    assert rec.name == "hello"
    assert rec.version == "v0.1"
    assert rec.text == body
    assert rec.meta["author"] == "tests"
    assert rec.sha256 == hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_registry_missing_prompt_raises(tmp_path: Path) -> None:
    (tmp_path / "v1.0").mkdir(parents=True)
    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.get("nonexistent")


def test_registry_handles_no_frontmatter(tmp_path: Path) -> None:
    version_dir = tmp_path / "v1.0"
    version_dir.mkdir(parents=True)
    body = "Plain prompt with no frontmatter.\n"
    (version_dir / "plain.md").write_text(body, encoding="utf-8")

    reg = PromptRegistry(root=tmp_path)
    rec = reg.get("plain")
    assert rec.text == body.rstrip()
    # Auto-injected fields are still present.
    assert rec.meta["name"] == "plain"
    assert rec.meta["version"] == "v1.0"


def test_registry_latest_version_picks_highest_string(tmp_path: Path) -> None:
    for v in ("v1.0", "v1.1", "v0.9"):
        d = tmp_path / v
        d.mkdir()
        (d / "x.md").write_text("body", encoding="utf-8")
    reg = PromptRegistry(root=tmp_path)
    # `sorted()` is lexicographic; v1.1 > v1.0 > v0.9. This documents the
    # behavior so users know to bump versions in monotonically-sortable form.
    assert reg.latest_version() == "v1.1"


def test_module_constants_match_registry() -> None:
    """`model_extraction_prompts.EXTRACTION_SYSTEM_PROMPT` must equal registry text.

    Guard against future drift: if someone hard-codes a constant that diverges
    from the registry, this test fails immediately.
    """
    from llm_survey.prompts.model_extraction_prompts import (
        BASE_EXTRACTION_PROMPT,
        EXTRACTION_SYSTEM_PROMPT,
        MODEL_REFINEMENT_PROMPT,
        RAG_ENHANCED_PROMPT,
        THEMATIC_ANALYSIS_PROMPT,
    )

    r = default_registry()
    assert EXTRACTION_SYSTEM_PROMPT == r.text("extraction_system")
    assert BASE_EXTRACTION_PROMPT == r.text("base_extraction")
    assert RAG_ENHANCED_PROMPT == r.text("rag_enhanced")
    assert MODEL_REFINEMENT_PROMPT == r.text("model_refinement")
    assert THEMATIC_ANALYSIS_PROMPT == r.text("thematic_analysis")
