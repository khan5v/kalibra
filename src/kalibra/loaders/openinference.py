"""OpenInference format — converts Phoenix/Arize trace exports to Kalibra Traces.

OpenInference is a set of attribute conventions on top of OpenTelemetry,
maintained by Arize AI. Phoenix exports traces as flat span arrays
(JSONL, one span per line) with attributes like:

    llm.token_count.prompt, llm.cost.total, openinference.span.kind

Assumption: each span carries its own cost/token values, not aggregated
subtotals of children. This follows the OpenInference convention.
"""

from __future__ import annotations

import json
from pathlib import Path

from kalibra.loaders import TraceLoader
from kalibra.loaders._utils import (
    _find_root_span,
    _flatten_dict,
    _group_by_trace_id,
    _iso_to_ns,
    _normalize_status,
    _resolve_attr,
    _safe_float,
    _safe_int,
)
from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS, Span, Trace


class OpenInferenceLoader(TraceLoader):
    """OpenInference/Phoenix trace format (flat span arrays with context.trace_id)."""

    name = "openinference"

    def detect(self, item: dict) -> bool:
        return is_openinference(item)

    def load(self, path: Path) -> list[Trace]:
        return load_openinference_jsonl(path)


# ── Detection ────────────────────────────────────────────────────────────────

_OI_SPAN_KINDS = {"LLM", "TOOL", "CHAIN", "AGENT", "RETRIEVER",
                   "RERANKER", "EMBEDDING", "GUARDRAIL", "EVALUATOR"}


def is_openinference(item: dict) -> bool:
    """Check if a JSON object looks like an OpenInference span.

    Requires a recognized OpenInference span_kind — either as a top-level
    field, a dot-flattened attribute, or a nested attribute. The loader
    peeks at multiple items (not just the first), so spans with UNKNOWN
    kind are fine as long as at least one span in the sample has a
    recognized kind.
    """
    context = item.get("context")
    if not isinstance(context, dict):
        return False
    if "trace_id" not in context:
        return False

    # Top-level span_kind (Phoenix JSONL format).
    if item.get("span_kind") in _OI_SPAN_KINDS:
        return True

    attrs = item.get("attributes")
    if isinstance(attrs, dict):
        # Dot-flattened key.
        if attrs.get("openinference.span.kind") in _OI_SPAN_KINDS:
            return True
        # Nested dict: attributes.openinference.span.kind
        oi = attrs.get("openinference")
        if isinstance(oi, dict):
            span = oi.get("span")
            if isinstance(span, dict) and span.get("kind") in _OI_SPAN_KINDS:
                return True

    return False


# ── Loading ──────────────────────────────────────────────────────────────────

def load_openinference_jsonl(path: Path) -> list[Trace]:
    """Load OpenInference spans from JSONL (one span per line)."""
    raw_spans: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                raw_spans.append(row)
    return _group_spans(raw_spans)


# ── Internal ─────────────────────────────────────────────────────────────────


def _group_spans(raw_spans: list[dict]) -> list[Trace]:
    """Group flat OpenInference spans by trace_id and build Traces."""
    trace_groups = _group_by_trace_id(raw_spans)

    traces: list[Trace] = []
    for trace_id, spans_raw in trace_groups.items():
        root = _find_root_span(spans_raw)

        # Determine outcome from span statuses.
        outcome = None
        has_finish = False
        has_failure = False

        for s in spans_raw:
            s_attrs = s.get("attributes") or {}
            finish = _extract_finish_reason(s_attrs)
            if finish is not None:
                has_finish = True
                if finish in ("truncated", "filtered"):
                    has_failure = True

        if has_finish:
            outcome = OUTCOME_FAILURE if has_failure else OUTCOME_SUCCESS
        elif root is not None:
            code = _normalize_status(root.get("status_code"))
            if not code:
                status = root.get("status") or {}
                code = _normalize_status(status.get("status_code"))
            if code == "ERROR":
                outcome = OUTCOME_FAILURE
            elif code == "OK":
                outcome = OUTCOME_SUCCESS

        # Build Kalibra Spans.
        spans = [_to_span(s) for s in spans_raw]
        spans.sort(key=lambda sp: sp.start_ns)

        # Collect root span attributes as trace metadata.
        metadata: dict = {}
        if root is not None:
            attrs = root.get("attributes") or {}
            span_kind = (
                root.get("span_kind")
                or _resolve_attr(attrs, "openinference.span.kind")
                or ""
            )
            if span_kind:
                metadata["span_kind"] = span_kind
            flat: dict = {}
            _flatten_dict(attrs, "", flat)
            _SKIP_PREFIXES = ("llm.", "openinference.", "input.", "output.")
            for k, v in flat.items():
                if not any(k.startswith(p) for p in _SKIP_PREFIXES):
                    metadata[k] = v

        traces.append(Trace(
            trace_id=trace_id,
            spans=spans,
            outcome=outcome,
            metadata=metadata,
        ))

    return traces


# ── LLM finish reason extraction ──────────────────────────────────────────────

_ANTHROPIC_REASON_MAP = {
    "end_turn": "complete",
    "max_tokens": "truncated",
    "tool_use": "tool_call",
    "stop_sequence": "complete",
}

_OPENAI_REASON_MAP = {
    "stop": "complete",
    "length": "truncated",
    "tool_calls": "tool_call",
    "content_filter": "filtered",
    "function_call": "tool_call",
}

_GOOGLE_REASON_MAP = {
    "STOP": "complete",
    "MAX_TOKENS": "truncated",
    "SAFETY": "filtered",
    "RECITATION": "filtered",
    "OTHER": "complete",
}


def _extract_finish_reason(attrs: dict) -> str | None:
    """Best-effort extraction of LLM completion reason from output.value."""
    output_raw = _resolve_attr(attrs, "output.value")
    if not output_raw or not isinstance(output_raw, str):
        return None

    try:
        output = json.loads(output_raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(output, dict):
        return None

    # Anthropic: top-level stop_reason
    reason = output.get("stop_reason")
    if reason and reason in _ANTHROPIC_REASON_MAP:
        return _ANTHROPIC_REASON_MAP[reason]

    # OpenAI: choices[0].finish_reason
    choices = output.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            reason = first.get("finish_reason")
            if reason and reason in _OPENAI_REASON_MAP:
                return _OPENAI_REASON_MAP[reason]

    # Google: candidates[0].finishReason
    candidates = output.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            reason = first.get("finishReason")
            if reason and reason in _GOOGLE_REASON_MAP:
                return _GOOGLE_REASON_MAP[reason]

    return None


def _to_span(raw: dict) -> Span:
    """Convert a single OpenInference span dict to a Kalibra Span."""
    context = raw.get("context") or {}
    attrs = raw.get("attributes") or {}

    span_id = str(context.get("span_id") or "")
    parent_id = raw.get("parent_id")
    if not parent_id:
        parent_id = None
    else:
        parent_id = str(parent_id)

    span_kind = (
        raw.get("span_kind")
        or _resolve_attr(attrs, "openinference.span.kind")
        or ""
    )
    name = raw.get("name") or span_kind

    start_ns = _iso_to_ns(raw.get("start_time", ""))
    end_ns = _iso_to_ns(raw.get("end_time", ""))

    raw_in = _resolve_attr(attrs, "llm.token_count.prompt")
    raw_out = _resolve_attr(attrs, "llm.token_count.completion")
    if raw_in is None and raw_out is None:
        raw_total = _resolve_attr(attrs, "llm.token_count.total")
        if raw_total is not None:
            raw_out = raw_total

    raw_cost = _resolve_attr(attrs, "llm.cost.total")
    model = _resolve_attr(attrs, "llm.model_name")

    status_code = _normalize_status(raw.get("status_code"))
    if not status_code:
        status = raw.get("status") or {}
        status_code = _normalize_status(status.get("status_code"))
    is_error = status_code == "ERROR"

    flat_attrs: dict = {}
    _flatten_dict(attrs, "", flat_attrs)

    return Span(
        span_id=span_id,
        name=name,
        parent_id=parent_id,
        start_ns=start_ns,
        end_ns=end_ns,
        cost=_safe_float(raw_cost),
        input_tokens=_safe_int(raw_in),
        output_tokens=_safe_int(raw_out),
        model=str(model) if model is not None else None,
        error=is_error,
        attributes=flat_attrs,
    )


