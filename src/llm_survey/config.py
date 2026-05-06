"""Typed configuration object for the LLM survey pipeline.

Centralises model names, API keys, paths, seeds and feature toggles. Replaces
ad-hoc `os.getenv(...)` calls scattered across the codebase. All fields can be
overridden via env vars (case-insensitive) or a `.env` file at the repo root.

Usage:

    from llm_survey.config import get_settings
    cfg = get_settings()
    extractor = RAGModelExtractor(
        openai_api_key=cfg.openrouter_api_key,
        llm_model=cfg.llm_model,
        ...
    )

Existing call sites that read env vars directly are NOT yet migrated — the goal
of this module is to make the migration mechanical and reversible. New code
should prefer `get_settings()`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Field names match accepted env var names (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- API access ---
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    http_referer: str = Field(default="", validation_alias="HTTP_REFERER")
    x_title: str = Field(default="", validation_alias="X_TITLE")

    # --- Models ---
    llm_model: str = Field(default="google/gemma-4-31b-it", validation_alias="LLM_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )

    # --- Determinism ---
    # Default temperature is 0 for determinism; set LLM_TEMPERATURE>0 to override.
    llm_temperature: float = Field(default=0.0, validation_alias="LLM_TEMPERATURE")
    seed: int = Field(default=20260101, validation_alias="LLM_SEED")

    # --- Paths ---
    data_dir: Path = Field(default=REPO_ROOT / "data", validation_alias="DATA_DIR")
    outputs_dir: Path = Field(default=REPO_ROOT / "outputs", validation_alias="OUTPUTS_DIR")
    survey_chroma_path: Path = Field(
        default=REPO_ROOT / "data" / "chroma" / "survey",
        validation_alias="SURVEY_CHROMA_PATH",
    )
    literature_chroma_path: Path = Field(
        default=REPO_ROOT / "data" / "chroma" / "literature",
        validation_alias="LITERATURE_CHROMA_PATH",
    )

    # --- Feature toggles ---
    enable_literature_retrieval: bool = Field(
        default=True, validation_alias="ENABLE_LITERATURE_RETRIEVAL"
    )
    enable_refinement_loop: bool = Field(
        default=True, validation_alias="ENABLE_REFINEMENT_LOOP"
    )
    max_refinement_iterations: int = Field(default=2, validation_alias="MAX_REFINEMENT_ITERATIONS")
    completeness_threshold: float = Field(default=0.75, validation_alias="COMPLETENESS_THRESHOLD")
    literature_target_papers: int = Field(default=20, validation_alias="LITERATURE_TARGET_PAPERS")

    def http_extra_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        return headers


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance. Call `get_settings.cache_clear()` to reset (mostly for tests)."""
    return Settings()


__all__ = ["Settings", "get_settings", "REPO_ROOT"]
