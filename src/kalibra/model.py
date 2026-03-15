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

    # Telemetry.
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
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
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ── Trace ─────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    """A single agent run — a tree of spans plus outcome and metadata.

    The fundamental unit of comparison. Kalibra compares populations of traces.
    """

    trace_id: str = ""
    spans: list[Span] = field(default_factory=list)
    outcome: str | None = None  # OUTCOME_SUCCESS | OUTCOME_FAILURE | None
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Wall-clock duration: max(end) - min(start) across all spans, in seconds."""
        if not self.spans:
            return 0.0
        times = [
            (s.start_ns, s.end_ns)
            for s in self.spans
            if s.start_ns or s.end_ns
        ]
        if not times:
            return 0.0
        return (max(e for _, e in times) - min(s for s, _ in times)) / 1e9

    @property
    def total_cost(self) -> float:
        """Sum of all span costs."""
        return sum(s.cost for s in self.spans)

    @property
    def total_tokens(self) -> int:
        """Sum of all span tokens (input + output)."""
        return sum(s.total_tokens for s in self.spans)

    def root_spans(self) -> list[Span]:
        """Spans with no parent."""
        return [s for s in self.spans if s.parent_id is None]
