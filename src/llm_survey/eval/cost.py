"""Per-phase cost / latency / token accounting.

The pipeline already counts tokens in `llm_survey.utils.cost_estimate`; this
module adds a thin run-level recorder that any phase can append events to and
that emits aggregated per-phase stats with means, totals and (optional)
bootstrap CIs.

Usage:

    from llm_survey.eval.cost import RunRecorder, USD_PRICES
    rec = RunRecorder(model="google/gemma-4-31b-it")
    with rec.phase("extraction"):
        # ... LLM call ...
        rec.record_llm_call(prompt_tokens=512, completion_tokens=128)
    rec.dump("outputs/cost_report.json")

The cost numbers are estimates only — pricing changes constantly. Update
`USD_PRICES` (per 1M tokens) when you re-run the eval for a paper figure.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterator


# Indicative OpenRouter / vendor pricing in USD per 1M tokens. These are
# defaults; override per-run if pricing has shifted.
USD_PRICES: dict[str, dict[str, float]] = {
    "google/gemma-4-31b-it": {"in": 0.20, "out": 0.30},
    "anthropic/claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "anthropic/claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "anthropic/claude-opus-4-7": {"in": 15.00, "out": 75.00},
    "openai/gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "openai/gpt-4o": {"in": 2.50, "out": 10.00},
}


@dataclass
class CallEvent:
    phase: str
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    timestamp: float


@dataclass
class PhaseEvent:
    phase: str
    start: float
    end: float | None = None

    @property
    def duration(self) -> float:
        return (self.end or time.time()) - self.start


@dataclass
class RunRecorder:
    model: str
    prices_per_million: dict[str, dict[str, float]] = field(default_factory=lambda: dict(USD_PRICES))
    calls: list[CallEvent] = field(default_factory=list)
    phases: list[PhaseEvent] = field(default_factory=list)
    _current_phase: str | None = None
    _phase_start: float | None = None

    def _resolve_phase(self, override: str | None) -> str:
        return override or self._current_phase or "unknown"

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        prev_name, prev_start = self._current_phase, self._phase_start
        self._current_phase = name
        self._phase_start = time.perf_counter()
        ev = PhaseEvent(phase=name, start=time.time())
        self.phases.append(ev)
        try:
            yield
        finally:
            ev.end = time.time()
            self._current_phase = prev_name
            self._phase_start = prev_start

    def record_llm_call(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        wall_seconds: float | None = None,
        phase: str | None = None,
    ) -> None:
        if wall_seconds is None and self._phase_start is not None:
            wall_seconds = time.perf_counter() - self._phase_start
        self.calls.append(
            CallEvent(
                phase=self._resolve_phase(phase),
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                wall_seconds=float(wall_seconds or 0.0),
                timestamp=time.time(),
            )
        )

    def _phase_durations(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for ev in self.phases:
            out[ev.phase] = out.get(ev.phase, 0.0) + ev.duration
        return out

    def summary(self) -> dict[str, Any]:
        per_phase: dict[str, dict[str, Any]] = {}
        durations = self._phase_durations()
        prices = self.prices_per_million.get(self.model, {"in": 0.0, "out": 0.0})
        for ev in self.calls:
            d = per_phase.setdefault(
                ev.phase,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "wall_seconds_total": 0.0,
                    "per_call_wall_seconds": [],
                },
            )
            d["calls"] += 1
            d["prompt_tokens"] += ev.prompt_tokens
            d["completion_tokens"] += ev.completion_tokens
            d["wall_seconds_total"] += ev.wall_seconds
            d["per_call_wall_seconds"].append(ev.wall_seconds)
        for phase, d in per_phase.items():
            calls = d["calls"]
            samples = d.pop("per_call_wall_seconds")
            d["mean_wall_seconds"] = round(mean(samples), 4) if samples else 0.0
            d["std_wall_seconds"] = round(pstdev(samples), 4) if len(samples) > 1 else 0.0
            in_cost = (d["prompt_tokens"] / 1_000_000) * prices.get("in", 0.0)
            out_cost = (d["completion_tokens"] / 1_000_000) * prices.get("out", 0.0)
            d["estimated_usd"] = round(in_cost + out_cost, 6)
            d["phase_duration_seconds"] = round(durations.get(phase, 0.0), 4)
        totals = {
            "calls": sum(d["calls"] for d in per_phase.values()),
            "prompt_tokens": sum(d["prompt_tokens"] for d in per_phase.values()),
            "completion_tokens": sum(d["completion_tokens"] for d in per_phase.values()),
            "estimated_usd": round(sum(d["estimated_usd"] for d in per_phase.values()), 6),
            "wall_seconds_total": round(sum(durations.values()), 4),
        }
        return {"model": self.model, "totals": totals, "per_phase": per_phase}

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")


__all__ = ["RunRecorder", "USD_PRICES", "CallEvent", "PhaseEvent"]
