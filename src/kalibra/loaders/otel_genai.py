"""OTel GenAI format — OpenTelemetry gen_ai.* semantic conventions.

Loads traces from platforms that emit the official OpenTelemetry GenAI
semantic conventions (Langfuse, Datadog, PydanticAI/Logfire, OpenLLMetry).

Attributes use the gen_ai.* namespace:
    gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
    gen_ai.request.model, gen_ai.response.finish_reasons,
    gen_ai.operation.name, gen_ai.system

Key difference from OpenInference:
- Finish reason is a direct attribute (gen_ai.response.finish_reasons),
  not buried in output.value JSON.
- No cost attribute in the standard — cost stays None unless mapped via
  fields config.
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


# ── Finish reason mapping ────────────────────────────────────────────────────
# gen_ai.response.finish_reasons is an array of strings.
# The OTel spec normalises provider values, but instrumentation libraries
# may pass through raw provider values. Handle both.

_SUCCESS_REASONS = {"stop", "end_turn", "stop_sequence"}
_FAILURE_REASONS = {"length", "max_tokens", "content_filter", "safety",
                    "recitation"}
_TOOL_REASONS = {"tool_calls", "tool_use", "function_call"}


class OTelGenAILoader(TraceLoader):
    """OTel GenAI semantic conventions (gen_ai.* attributes)."""

    name = "otel-genai"

    def detect(self, item: dict) -> bool:
        """Detect gen_ai.* attributes without openinference.* attributes."""
        attrs = item.get("attributes")
        if not isinstance(attrs, dict):
            return False

        has_genai = (
            _resolve_attr(attrs, "gen_ai.operation.name") is not None
            or _resolve_attr(attrs, "gen_ai.system") is not None
            or _resolve_attr(attrs, "gen_ai.request.model") is not None
        )
        if not has_genai:
            return False

        # Reject if OpenInference attributes are present — let that loader
        # handle it instead (it's earlier in the registry).
        has_oi = (
            _resolve_attr(attrs, "openinference.span.kind") is not None
            or item.get("span_kind") in _OI_SPAN_KINDS
        )
        return not has_oi

    def load(self, path: Path) -> list[Trace]:
        return _load_otel_genai_jsonl(path)


# OpenInference span kinds — used only for negative detection.
_OI_SPAN_KINDS = {"LLM", "TOOL", "CHAIN", "AGENT", "RETRIEVER",
                   "RERANKER", "EMBEDDING", "GUARDRAIL", "EVALUATOR"}


# ── Loading ──────────────────────────────────────────────────────────────────

def _load_otel_genai_jsonl(path: Path) -> list[Trace]:
    """Load OTel GenAI spans from JSONL (one span per line)."""
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
    return _build_traces(raw_spans)


def _build_traces(raw_spans: list[dict]) -> list[Trace]:
    """Group flat OTel GenAI spans by trace_id and build Traces."""
    trace_groups = _group_by_trace_id(raw_spans)

    traces: list[Trace] = []
    for trace_id, spans_raw in trace_groups.items():
        root = _find_root_span(spans_raw)

        # Determine outcome from finish reasons across all spans.
        outcome = _determine_outcome(spans_raw, root)

        # Build Kalibra Spans.
        spans = [_to_span(s) for s in spans_raw]
        spans.sort(key=lambda sp: sp.start_ns)

        # Collect root span attributes as trace metadata.
        metadata = _extract_metadata(root)

        traces.append(Trace(
            trace_id=trace_id,
            spans=spans,
            outcome=outcome,
            metadata=metadata,
        ))

    return traces


# ── Outcome detection ────────────────────────────────────────────────────────

def _determine_outcome(spans_raw: list[dict], root: dict | None) -> str | None:
    """Determine trace outcome from gen_ai.response.finish_reasons.

    If ANY span has a failure reason, the trace failed.
    If all spans with finish reasons completed successfully, the trace succeeded.
    Falls back to OTel status_code on root span.
    """
    has_finish = False
    has_failure = False

    for s in spans_raw:
        attrs = s.get("attributes") or {}
        reasons = _get_finish_reasons(attrs)
        if reasons is not None and len(reasons) > 0:
            has_finish = True
            for reason in reasons:
                r = str(reason).lower()
                if r in _FAILURE_REASONS:
                    has_failure = True

    if has_finish:
        return OUTCOME_FAILURE if has_failure else OUTCOME_SUCCESS

    # Fall back to OTel status_code on root span.
    if root is not None:
        code = _normalize_status(root.get("status_code"))
        if not code:
            status = root.get("status") or {}
            code = _normalize_status(status.get("status_code"))
        if code == "ERROR":
            return OUTCOME_FAILURE
        if code == "OK":
            return OUTCOME_SUCCESS

    return None


def _get_finish_reasons(attrs: dict) -> list | None:
    """Extract finish reasons from gen_ai.response.finish_reasons.

    The spec defines this as an array of strings. Handle:
    - Array: ["stop"] → ["stop"]
    - String (non-spec but defensive): "stop" → ["stop"]
    - JSON string: '["stop"]' → ["stop"]
    - Missing/None: → None
    """
    raw = _resolve_attr(attrs, "gen_ai.response.finish_reasons")
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        # Could be a JSON-encoded array.
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        # Or a bare string value.
        return [raw]
    return None


# ── Span conversion ──────────────────────────────────────────────────────────

def _to_span(raw: dict) -> Span:
    """Convert a single OTel GenAI span dict to a Kalibra Span."""
    context = raw.get("context") or {}
    attrs = raw.get("attributes") or {}

    span_id = str(context.get("span_id") or "")
    parent_id = raw.get("parent_id")
    if not parent_id:
        parent_id = None
    else:
        parent_id = str(parent_id)

    # Name: use span name, fall back to gen_ai.operation.name + model.
    operation = _resolve_attr(attrs, "gen_ai.operation.name") or ""
    model = _resolve_attr(attrs, "gen_ai.request.model") or ""
    name = raw.get("name") or f"{operation} {model}".strip() or "unknown"

    # Timing.
    start_ns = _iso_to_ns(raw.get("start_time", ""))
    end_ns = _iso_to_ns(raw.get("end_time", ""))

    # Tokens.
    raw_in = _resolve_attr(attrs, "gen_ai.usage.input_tokens")
    raw_out = _resolve_attr(attrs, "gen_ai.usage.output_tokens")

    # No standard cost attribute in OTel GenAI.
    # Users can map vendor-specific cost via fields config.
    raw_cost = None

    # Error status.
    status_code = _normalize_status(raw.get("status_code"))
    if not status_code:
        status = raw.get("status") or {}
        status_code = _normalize_status(status.get("status_code"))
    is_error = status_code == "ERROR"

    # Flatten attributes for metadata.
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
        model=str(model) if model else None,
        error=is_error,
        attributes=flat_attrs,
    )


# ── Metadata extraction ──────────────────────────────────────────────────────

def _extract_metadata(root: dict | None) -> dict:
    """Extract trace metadata from the root span's attributes."""
    metadata: dict = {}
    if root is None:
        return metadata

    attrs = root.get("attributes") or {}
    flat: dict = {}
    _flatten_dict(attrs, "", flat)

    # Skip attributes already captured in Span fields (tokens, model)
    # and large text blobs (input/output messages).
    _SKIP_PREFIXES = (
        "gen_ai.input.", "gen_ai.output.",  # large message blobs
        "gen_ai.usage.",                     # already in Span.input_tokens/output_tokens
        "gen_ai.request.",                   # already in Span.model
    )
    for k, v in flat.items():
        if not any(k.startswith(p) for p in _SKIP_PREFIXES):
            metadata[k] = v

    return metadata


