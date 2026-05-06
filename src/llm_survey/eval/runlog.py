"""Run log for reproducibility provenance.

A `RunLog` is a small JSON document attached to every pipeline output. It
captures everything needed to re-run a result: prompt versions + sha256s,
model + temperature + seed, embedder + tokenizer version, dependency-lock
hash, git commit, wall clock, and an optional pointer to a W&B / MLflow run.

The goal is that a paper figure can cite a `runlog.json` and a reader can
diff it against a fresh run to localise any drift.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        return False


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@dataclass
class RunLog:
    run_id: str
    model: str
    temperature: float
    seed: int
    embedding_model: str
    prompts: dict[str, str] = field(default_factory=dict)  # name -> sha256
    git_commit: str = field(default_factory=_git_commit)
    git_dirty: bool = field(default_factory=_git_dirty)
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def attach_lockfile_hash(self, lock_path: str | Path = "requirements.lock") -> None:
        h = _hash_file(Path(lock_path))
        if h:
            self.extras["requirements_lock_sha256"] = h

    def attach_prompt(self, name: str, sha256: str) -> None:
        self.prompts[name] = sha256

    def finalize(self) -> None:
        self.ended_at = time.time()

    def dump(self, path: str | Path) -> None:
        if self.ended_at is None:
            self.finalize()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2, default=str), encoding="utf-8")

    @classmethod
    def from_settings(cls, *, run_id: str, settings, prompt_registry=None) -> "RunLog":
        log = cls(
            run_id=run_id,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            seed=settings.seed,
            embedding_model=settings.embedding_model,
        )
        if prompt_registry is not None:
            for version in prompt_registry.list_versions():
                version_dir = prompt_registry.root / version
                for md in version_dir.glob("*.md"):
                    rec = prompt_registry.get(md.stem, version=version)
                    log.attach_prompt(f"{version}/{md.stem}", rec.sha256)
        log.attach_lockfile_hash()
        log.extras["env"] = {k: os.environ.get(k, "") for k in ("PYTHONHASHSEED",)}
        return log


__all__ = ["RunLog"]
