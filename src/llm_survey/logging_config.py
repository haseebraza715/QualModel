"""Structured logging setup using `structlog`.

Use this from any module that wants grep-able JSON logs. Default output is
human-readable on a TTY and JSON when piped (e.g. CI / log shipper). Calling
`configure_logging()` is idempotent.

    from llm_survey.logging_config import get_logger
    log = get_logger(__name__)
    log.info("extracted_chunks", n=42, chunk_id="r1c0")
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    import structlog
except ImportError:  # pragma: no cover - structlog is in pyproject; runtime guard
    structlog = None  # type: ignore[assignment]


_CONFIGURED = False


def configure_logging(level: str | int = "INFO", *, force_json: bool | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_level = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )
    if structlog is None:  # pragma: no cover
        _CONFIGURED = True
        return
    use_json = force_json if force_json is not None else not sys.stderr.isatty()
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if use_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


class _StdlibAdapter:
    """Minimal stdlib-logger shim that accepts kwargs like structlog."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _format(self, event: str, **kwargs: Any) -> str:
        if not kwargs:
            return event
        kv = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{event} {kv}"

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log.debug(self._format(event, **kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self._log.info(self._format(event, **kwargs))

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log.warning(self._format(event, **kwargs))

    def error(self, event: str, **kwargs: Any) -> None:
        self._log.error(self._format(event, **kwargs))

    def exception(self, event: str, **kwargs: Any) -> None:
        self._log.exception(self._format(event, **kwargs))


def get_logger(name: str | None = None) -> Any:
    if not _CONFIGURED:
        configure_logging(level=os.environ.get("LLM_SURVEY_LOG_LEVEL", "INFO"))
    if structlog is None:  # pragma: no cover - structlog absent fallback
        return _StdlibAdapter(name or "llm_survey")
    return structlog.get_logger(name or "llm_survey")


__all__ = ["configure_logging", "get_logger"]
