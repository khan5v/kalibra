"""Core data model — Span and Trace.

These are the canonical types that all of Kalibra operates on.
Traces are loaded from JSONL, metrics compute over them, results reference them.
No external dependencies — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Outcome constants ─────────────────────────────────────────────────────────

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"


# ── Span ──────────────────────────────────────────────────────────────────────

@dataclass
class Span:
    """One step in an agent's execution.

    A single LLM call, tool invocation, search query, code execution, etc.
    Every span belongs to a trace. Spans form a tree via parent_id.
    """

    span_id: str = ""
    name: str = ""
    parent_id: str | None = None

    # Timing (nanoseconds for precision, 0 if unavailable).
    start_ns: int = 0
    end_ns: int = 0

    # Telemetry — None means "not measured" (e.g. non-LLM span),
    # 0 means "measured as zero" (e.g. cached response, free tool call).
    cost: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    error: bool = False

    # Freeform extra data.
    attributes: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        if self.start_ns and self.end_ns:
            return (self.end_ns - self.start_ns) / 1e9
        return 0.0

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


# ── Trace ─────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    """A single agent run — a tree of spans plus outcome and metadata.

    The fundamental unit of comparison. Kalibra compares populations of traces.

    Telemetry can come from two sources:
    - Spans: when trace.spans is non-empty, properties aggregate from spans.
    - Trace-level fields: when trace.spans is empty, the loader sets
      _cost, _input_tokens, etc. directly from the JSONL row.

    None means "not measured." 0 means "measured as zero." Metrics must
    respect this distinction — None values should be excluded, not treated as 0.
    """

    trace_id: str = ""
    spans: list[Span] = field(default_factory=list)
    outcome: str | None = None  # OUTCOME_SUCCESS | OUTCOME_FAILURE | None
    metadata: dict = field(default_factory=dict)

    # Trace-level telemetry for span-less traces. None = not measured.
    _cost: float | None = None
    _input_tokens: int | None = None
    _output_tokens: int | None = None
    _duration_s: float | None = None

    @property
    def duration(self) -> float | None:
        """Wall-clock duration in seconds. None if not measured."""
        if self.spans:
            times = [
                (s.start_ns, s.end_ns)
                for s in self.spans
                if s.start_ns or s.end_ns
            ]
            if times:
                return (max(e for _, e in times) - min(s for s, _ in times)) / 1e9
            return None
        return self._duration_s

    @property
    def total_cost(self) -> float | None:
        """Sum of all span costs. None if no span has cost and no trace-level cost."""
        if self.spans:
            costs = [s.cost for s in self.spans if s.cost is not None]
            if not costs:
                return None
            return sum(costs)
        return self._cost

    @property
    def total_tokens(self) -> int | None:
        """Sum of all span tokens. None if no span has tokens and no trace-level tokens."""
        if self.spans:
            tokens = [s.total_tokens for s in self.spans if s.total_tokens is not None]
            if not tokens:
                return None
            return sum(tokens)
        if self._input_tokens is not None or self._output_tokens is not None:
            return (self._input_tokens or 0) + (self._output_tokens or 0)
        return None

    def root_spans(self) -> list[Span]:
        """Spans with no parent."""
        return [s for s in self.spans if s.parent_id is None]
