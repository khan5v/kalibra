"""TraceCollection — unified query interface over any trace source."""

from __future__ import annotations

from collections import defaultdict
from functools import cached_property

from opentelemetry.sdk.trace import ReadableSpan

from agentflow.converters.base import Trace


class TraceCollection:
    """A queryable, indexed collection of traces from any source.

    Factory methods handle loading and format conversion; the query
    interface is uniform regardless of origin (files, connectors, etc.).

    Example::

        baseline = TraceCollection.from_path("./baseline-traces/")
        current  = TraceCollection.from_path("./current-traces/")
        current  = TraceCollection.from_traces(connector.fetch(...), source="langfuse")
    """

    def __init__(self, traces: list[Trace], source: str = ""):
        self._traces = traces
        self.source = source

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_path(cls, path: str, progress: bool = True) -> "TraceCollection":
        """Load from a file or directory (SWE-bench, JSONL — auto-detected)."""
        from agentflow.converters import load_traces
        traces = load_traces(path, trace_format="auto", progress=progress)
        return cls(traces, source=path)

    @classmethod
    def from_traces(cls, traces: list[Trace], source: str = "") -> "TraceCollection":
        """Wrap an existing list of Trace objects."""
        return cls(traces, source=source)

    # ── Query interface ────────────────────────────────────────────────────────

    def all_traces(self) -> list[Trace]:
        """All traces in the collection."""
        return self._traces

    def get_trace(self, trace_id: str) -> Trace | None:
        """Look up a single trace by ID."""
        return self._by_id.get(trace_id)

    def spans_for_node(self, node: str) -> list[ReadableSpan]:
        """All spans with the given node name, across all traces."""
        return self._spans_by_node.get(node, [])

    def traces_with_outcome(self, outcome: str) -> list[Trace]:
        """All traces matching the given outcome ('success' or 'failure')."""
        return [t for t in self._traces if t.outcome == outcome]

    def __len__(self) -> int:
        return len(self._traces)

    def __repr__(self) -> str:
        return f"TraceCollection(n={len(self._traces)}, source={self.source!r})"

    # ── Precomputed indices (built once, on first access) ──────────────────────

    @cached_property
    def _by_id(self) -> dict[str, Trace]:
        return {t.trace_id: t for t in self._traces}

    @cached_property
    def _spans_by_node(self) -> dict[str, list[ReadableSpan]]:
        index: dict[str, list[Span]] = defaultdict(list)
        for t in self._traces:
            for s in t.spans:
                index[s.name].append(s)
        return dict(index)
