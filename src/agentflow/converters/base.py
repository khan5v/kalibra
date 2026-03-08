"""Base data structures for traces — the common representation all converters produce.

Spans are represented as OpenTelemetry ``ReadableSpan`` objects.
``Trace`` is a thin wrapper that holds a list of spans plus outcome and metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import Status, StatusCode
from opentelemetry.trace import SpanContext, TraceFlags

# ── OTel GenAI semantic convention attribute keys ─────────────────────────────

GEN_AI_MODEL         = "gen_ai.request.model"
GEN_AI_INPUT_TOKENS  = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
AF_COST              = "agentflow.cost"


# ── Factory ───────────────────────────────────────────────────────────────────

def make_span(
    name: str,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    start_ns: int,
    end_ns: int,
    attributes: dict | None = None,
    error: bool = False,
) -> ReadableSpan:
    """Create an OTel ``ReadableSpan``.

    Args:
        name:           Span name.
        trace_id:       Trace ID — UUID or arbitrary string (auto-hashed to 128 bits).
        span_id:        Span ID — hex string up to 16 chars, or arbitrary string (auto-hashed).
        parent_span_id: Parent span ID hex string, or ``None`` for a root span.
        start_ns:       Start time in nanoseconds.
        end_ns:         End time in nanoseconds.
        attributes:     Span attributes dict (OTel GenAI keys encouraged).
        error:          Set status to ERROR if ``True``.
    """
    trace_id_int = _to_trace_id_int(trace_id)
    span_id_int  = _to_span_id_int(span_id)

    ctx = SpanContext(
        trace_id=trace_id_int,
        span_id=span_id_int,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    parent_ctx: SpanContext | None = None
    if parent_span_id is not None:
        parent_ctx = SpanContext(
            trace_id=trace_id_int,
            span_id=_to_span_id_int(parent_span_id),
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

    return ReadableSpan(
        name=name,
        context=ctx,
        parent=parent_ctx,
        attributes=attributes or {},
        status=Status(StatusCode.ERROR if error else StatusCode.OK),
        start_time=start_ns,
        end_time=end_ns,
    )


# ── Span accessors ────────────────────────────────────────────────────────────

def span_is_error(s: ReadableSpan) -> bool:
    return s.status.status_code == StatusCode.ERROR


def span_cost(s: ReadableSpan) -> float:
    return float((s.attributes or {}).get(AF_COST, 0.0))


def span_duration_s(s: ReadableSpan) -> float:
    if s.start_time is None or s.end_time is None:
        return 0.0
    return (s.end_time - s.start_time) / 1e9


def span_model(s: ReadableSpan) -> str | None:
    return (s.attributes or {}).get(GEN_AI_MODEL)


def span_input_tokens(s: ReadableSpan) -> int:
    return int((s.attributes or {}).get(GEN_AI_INPUT_TOKENS, 0))


def span_output_tokens(s: ReadableSpan) -> int:
    return int((s.attributes or {}).get(GEN_AI_OUTPUT_TOKENS, 0))


# ── Trace ─────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    """A single agent run — a list of OTel spans plus outcome and metadata."""

    trace_id: str
    spans: list[ReadableSpan]
    outcome: str | None = None  # "success" | "failure" | None
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if not self.spans:
            return 0.0
        times = [(s.start_time, s.end_time) for s in self.spans
                 if s.start_time is not None and s.end_time is not None]
        if not times:
            return 0.0
        return (max(e for _, e in times) - min(st for st, _ in times)) / 1e9

    @property
    def total_cost(self) -> float:
        return sum(span_cost(s) for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(span_input_tokens(s) + span_output_tokens(s) for s in self.spans)

    def root_spans(self) -> list[ReadableSpan]:
        return [s for s in self.spans if s.parent is None]

    def children_of(self, span_id_int: int) -> list[ReadableSpan]:
        return [s for s in self.spans if s.parent and s.parent.span_id == span_id_int]


# ── ID conversion helpers ─────────────────────────────────────────────────────

def _to_trace_id_int(s: str) -> int:
    """Convert arbitrary string → 128-bit int for OTel trace ID."""
    clean = s.replace("-", "")
    if len(clean) == 32 and _is_hex(clean):
        return int(clean, 16)
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _to_span_id_int(s: str) -> int:
    """Convert hex string → 64-bit int for OTel span ID."""
    if len(s) <= 16 and _is_hex(s):
        return int(s, 16)
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)


def _is_hex(s: str) -> bool:
    try:
        int(s, 16)
        return True
    except ValueError:
        return False
