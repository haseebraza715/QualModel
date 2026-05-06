"""Smoke tests for `llm_survey.config.Settings`."""
from __future__ import annotations

import os

from llm_survey.config import Settings


def test_settings_defaults_are_deterministic() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_temperature == 0.0
    assert s.seed == 20260101
    assert s.llm_model == "google/gemma-4-31b-it"
    assert s.openrouter_base_url.startswith("https://")


def test_settings_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-haiku-4-5")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-from-env")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_model == "anthropic/claude-haiku-4-5"
    assert s.llm_temperature == 0.7
    assert s.openrouter_api_key == "sk-test-from-env"


def test_http_extra_headers_omits_empty(monkeypatch) -> None:
    monkeypatch.delenv("HTTP_REFERER", raising=False)
    monkeypatch.delenv("X_TITLE", raising=False)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    headers = s.http_extra_headers()
    # Empty values are NOT emitted as headers.
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


def test_http_extra_headers_emits_set_values(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("X_TITLE", "MyApp")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    headers = s.http_extra_headers()
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-Title"] == "MyApp"
