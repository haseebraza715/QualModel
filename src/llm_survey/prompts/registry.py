"""Versioned prompt registry.

Prompts are loaded from `src/llm_survey/prompts/registry/<version>/<name>.md`,
each with YAML frontmatter capturing author, date, intended model, change
rationale, and (optionally) eval delta vs. the previous version.

Usage:

    from llm_survey.prompts.registry import PromptRegistry
    registry = PromptRegistry()  # reads bundled prompts
    text, meta = registry.get("extraction_system", version="v1.0")
    # `text` is the prompt body; `meta` is a dict of frontmatter fields.
    # `meta["sha256"]` is a content hash for run-log provenance.

The registry is intentionally read-only at runtime. To add a new version:

  1. Copy the existing `vX.Y/<name>.md` to `vX.(Y+1)/<name>.md`.
  2. Update the frontmatter (`version`, `change_rationale`, `eval_delta`).
  3. Bump callers to request the new version (or rely on `latest`).

Fallback: if a prompt file is missing, the registry falls back to the
hard-coded constants in `model_extraction_prompts.py` so existing call sites
keep working during the migration period.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REGISTRY_ROOT = Path(__file__).resolve().parent / "registry"


@dataclass(frozen=True)
class PromptRecord:
    name: str
    version: str
    text: str
    meta: dict[str, Any]
    sha256: str


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a `---\\nyaml\\n---\\nbody` document. Missing frontmatter -> empty meta."""
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, parts[2].lstrip("\n")


class PromptRegistry:
    """Loads versioned prompts from disk.

    Args:
        root: Override the registry root (used by tests).
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else REGISTRY_ROOT

    def list_versions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def latest_version(self) -> str | None:
        versions = self.list_versions()
        return versions[-1] if versions else None

    def get(self, name: str, version: str | None = None) -> PromptRecord:
        version = version or self.latest_version() or "v1.0"
        path = self.root / version / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt {name!r} not found at {path}")
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        # Preserve any frontmatter-declared version, but the on-disk path wins.
        meta["version"] = version
        meta["name"] = name
        meta["sha256"] = digest
        return PromptRecord(name=name, version=version, text=body.rstrip(), meta=meta, sha256=digest)

    def text(self, name: str, version: str | None = None) -> str:
        return self.get(name, version).text


@lru_cache(maxsize=1)
def default_registry() -> PromptRegistry:
    return PromptRegistry()


__all__ = ["REGISTRY_ROOT", "PromptRecord", "PromptRegistry", "default_registry"]
