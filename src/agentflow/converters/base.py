"""Base data structures for traces — the common representation all converters produce."""

from dataclasses import dataclass, field


@dataclass
class Span:
    """A single span (step) in an agent trace."""

    span_id: str
    parent_id: str | None
    name: str
    start_time: float  # unix timestamp
    end_time: float
    attributes: dict = field(default_factory=dict)
    # Commonly extracted attributes (optional, populated by converters):
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    status: str = "ok"  # "ok" | "error"


@dataclass
class Trace:
    """A single agent run — a collection of spans forming a trace tree."""

    trace_id: str
    spans: list[Span]
    outcome: str | None = None  # "success" | "failure" | None
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans)
        return end - start

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    def root_spans(self) -> list["Span"]:
        """Return spans with no parent (roots of the trace tree)."""
        return [s for s in self.spans if s.parent_id is None]

    def children_of(self, span_id: str) -> list["Span"]:
        """Return direct children of a span."""
        return [s for s in self.spans if s.parent_id == span_id]
