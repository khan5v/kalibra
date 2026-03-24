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
from kalibra.loaders._utils import _iso_to_ns, _safe_float, _safe_int
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

    # Strong signal: recognized OpenInference span kind.
    if item.get("span_kind") in _OI_SPAN_KINDS:
        return True

    attrs = item.get("attributes")
    if isinstance(attrs, dict):
        if attrs.get("openinference.span.kind") in _OI_SPAN_KINDS:
            return True
        oi = attrs.get("openinference")
        if isinstance(oi, dict):
            span = oi.get("span")
            if isinstance(span, dict) and span.get("kind") in _OI_SPAN_KINDS:
                return True

    # Structural signal: context has both trace_id and span_id,
    # and parent_id is a known key (even if None).
    if "span_id" in context and "parent_id" in item:
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
                    if (s.get("start_time") or "") < (root.get("start_time") or ""):
                        root = s

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


# OTel status codes: 0=UNSET, 1=OK, 2=ERROR.
_OTEL_STATUS_MAP = {0: "", 1: "OK", 2: "ERROR"}


def _normalize_status(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, int):
        return _OTEL_STATUS_MAP.get(raw, "")
    return str(raw).upper()


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
    """Resolve an OpenInference attribute by dot-path."""
    if dot_path in attrs:
        return attrs[dot_path]
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
