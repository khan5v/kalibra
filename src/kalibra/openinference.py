"""OpenInference loader — converts Phoenix/Arize trace exports to Kalibra Traces.

OpenInference is a set of attribute conventions on top of OpenTelemetry,
maintained by Arize AI. Phoenix exports traces as flat span arrays
(JSON array or JSONL, one span per line) with attributes like:

    llm.token_count.prompt, llm.cost.total, openinference.span.kind

This module detects and parses both export formats:
- JSON array of spans
- JSONL with one span per line

Assumption: each span carries its own cost/token values, not aggregated
subtotals of children. This follows the OpenInference convention.
"""

from __future__ import annotations

import json
from pathlib import Path

from kalibra.loader import _iso_to_ns
from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS, Span, Trace


def is_openinference(item: dict) -> bool:
    """Check if a JSON object looks like an OpenInference span.

    Detection uses two signals, in order:
    1. Strong: recognized span_kind (top-level, dot-flattened, or nested).
    2. Structural: context.trace_id + context.span_id + parent_id key.
       This catches spans with UNKNOWN or None span_kind, which appear
       in real Phoenix exports. Regular Kalibra JSONL never has a nested
       context dict, so this is a safe discriminator.
    """
    context = item.get("context")
    if not isinstance(context, dict):
        return False
    if "trace_id" not in context:
        return False

    _OI_SPAN_KINDS = {"LLM", "TOOL", "CHAIN", "AGENT", "RETRIEVER",
                       "RERANKER", "EMBEDDING", "GUARDRAIL", "EVALUATOR"}

    # Strong signal: recognized OpenInference span kind.
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

    # Structural signal: context has both trace_id and span_id,
    # and parent_id is a known key (even if None). This triple is
    # the fingerprint of an OpenInference/OTel span export.
    if "span_id" in context and "parent_id" in item:
        return True

    return False


def load_openinference_json(data: list[dict]) -> list[Trace]:
    """Load traces from a JSON array of OpenInference spans."""
    return _group_spans(data)


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
    # Group spans by trace_id.
    trace_groups: dict[str, list[dict]] = {}
    for item in raw_spans:
        if not isinstance(item, dict):
            continue
        context = item.get("context")
        if not isinstance(context, dict):
            continue
        trace_id = str(context.get("trace_id") or "")
        if not trace_id:
            continue
        trace_groups.setdefault(trace_id, []).append(item)

    traces: list[Trace] = []
    for trace_id, spans_raw in trace_groups.items():
        # Find root span (no parent_id).
        root = None
        for s in spans_raw:
            pid = s.get("parent_id")
            if pid is None or pid == "":
                if root is None:
                    root = s
                else:
                    # Multiple roots — pick earliest start time.
                    if (s.get("start_time") or "") < (root.get("start_time") or ""):
                        root = s

        # Determine outcome from span statuses.
        #
        # Priority:
        # 1. LLM finish reason from any LLM span's output.value.
        #    The root may be a CHAIN/AGENT with no output.value —
        #    the actual LLM completions live on child spans.
        #    If ANY LLM span was truncated/filtered, the trace failed.
        # 2. OTel status_code on root span (fallback).
        #
        # OTel status_code "OK" just means the API call succeeded, NOT
        # that the model completed its answer. A truncated response is
        # status_code "OK" but finish_reason "max_tokens" — that's a
        # failure for quality purposes.
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
            # Fall back to OTel status code on root.
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
            # Preserve non-standard attributes for field mapping.
            # Exclude llm.*, openinference.*, input.*, output.* — these
            # are either already parsed or large text blobs.
            flat: dict = {}
            _flatten_attrs(attrs, "", flat)
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
#
# OpenInference doesn't standardize the LLM completion reason. It lives
# inside output.value as provider-specific JSON. We parse it for the
# three major providers.
#
# Returns: "complete", "truncated", "tool_call", "filtered", or None.
# Provider response formats are subject to change — update the maps
# and extraction logic when they do.

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
    """Extract LLM completion reason from output.value.

    Parses the provider-specific JSON inside the output.value attribute
    to determine whether the model completed naturally, was truncated,
    or made a tool call.

    Returns: "complete", "truncated", "tool_call", "filtered", or None.
    """
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


# OTel status codes: 0=UNSET, 1=OK, 2=ERROR.
_OTEL_STATUS_MAP = {0: "", 1: "OK", 2: "ERROR"}


def _normalize_status(raw) -> str:
    """Normalize OTel status code to uppercase string.

    Handles: "OK", "ERROR", "Ok", 1, 2, etc.
    """
    if raw is None:
        return ""
    if isinstance(raw, int):
        return _OTEL_STATUS_MAP.get(raw, "")
    return str(raw).upper()


def _safe_float(val) -> float | None:
    """Convert to float, returning None if absent or unconvertible."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    """Convert to int, returning None if absent or unconvertible."""
    if val is None:
        return None
    try:
        return int(float(val))  # handles "500" and 500.0
    except (ValueError, TypeError):
        return None


def _to_span(raw: dict) -> Span:
    """Convert a single OpenInference span dict to a Kalibra Span.

    Handles two common export layouts:
    - Phoenix JSONL: top-level span_kind, status_code, nested attrs
    - JSON array: status.status_code, dot-flattened attrs
    """
    context = raw.get("context") or {}
    attrs = raw.get("attributes") or {}

    # Span identity.
    span_id = str(context.get("span_id") or "")
    parent_id = raw.get("parent_id")
    if not parent_id:  # Normalize both None and "" to None
        parent_id = None
    else:
        parent_id = str(parent_id)

    # Name: use span name, fall back to span kind.
    span_kind = (
        raw.get("span_kind")
        or _resolve_attr(attrs, "openinference.span.kind")
        or ""
    )
    name = raw.get("name") or span_kind

    # Timing: ISO 8601 timestamps.
    start_ns = _iso_to_ns(raw.get("start_time", ""))
    end_ns = _iso_to_ns(raw.get("end_time", ""))

    # Tokens (OpenInference convention).
    # Prefer prompt+completion breakdown. Fall back to total if neither exists.
    raw_in = _resolve_attr(attrs, "llm.token_count.prompt")
    raw_out = _resolve_attr(attrs, "llm.token_count.completion")
    if raw_in is None and raw_out is None:
        raw_total = _resolve_attr(attrs, "llm.token_count.total")
        if raw_total is not None:
            # Only total available — assign to output (conservative:
            # total without breakdown is better than losing the data).
            raw_out = raw_total

    # Cost (OpenInference convention).
    raw_cost = _resolve_attr(attrs, "llm.cost.total")

    # Model.
    model = _resolve_attr(attrs, "llm.model_name")

    # Error status — top-level or nested.
    status_code = _normalize_status(raw.get("status_code"))
    if not status_code:
        status = raw.get("status") or {}
        status_code = _normalize_status(status.get("status_code"))
    is_error = status_code == "ERROR"

    # Flatten attributes for metadata.
    flat_attrs: dict = {}
    _flatten_attrs(attrs, "", flat_attrs)

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


def _resolve_attr(attrs: dict, dot_path: str):
    """Resolve an OpenInference attribute by dot-path.

    Handles both layouts:
    - Dot-flattened: attrs["llm.token_count.prompt"]
    - Nested dicts:  attrs["llm"]["token_count"]["prompt"]
    """
    # Try flat key first.
    if dot_path in attrs:
        return attrs[dot_path]
    # Try nested traversal.
    parts = dot_path.split(".")
    current = attrs
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _flatten_attrs(obj: dict, prefix: str, out: dict) -> None:
    """Flatten nested OpenInference attributes to dot-notation."""
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_attrs(v, key, out)
        elif v is not None and not isinstance(v, list):
            out[key] = v


