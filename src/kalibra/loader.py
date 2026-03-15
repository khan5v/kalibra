"""Trace loader — reads JSONL into Trace objects.

Supports one format: JSONL with one trace per line.
Traces contain nested spans. Field names are configurable via FieldsConfig.
JSON strings embedded in field values are auto-parsed recursively.

Minimal trace (no spans):
    {"trace_id": "t1", "outcome": "success", "cost": 0.05}

Full trace with spans:
    {"trace_id": "t1", "outcome": "success", "spans": [
        {"span_id": "s1", "name": "plan", "cost": 0.03, "input_tokens": 500},
        {"span_id": "s2", "name": "search", "parent_id": "s1", "error": true}
    ]}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS, Span, Trace

# Avoid circular import — FieldsConfig is only needed for type hints
# and the actual resolution logic lives here.
_SUCCESS_VALUES = {"success", "true", "1", "pass", "passed", "resolved"}
_FAILURE_VALUES = {"failure", "false", "0", "fail", "failed", "error"}


def load_traces(
    path: str | Path,
    trace_id_field: str | None = None,
    fields: object | None = None,
) -> list[Trace]:
    """Load traces from a JSONL file.

    Args:
        path: Path to the JSONL file.
        trace_id_field: If set, use this field instead of ``trace_id``.
            Deprecated — prefer ``fields.trace_id``.
        fields: Optional FieldsConfig with field mappings for outcome,
            cost, input_tokens, output_tokens, trace_id, task_id.

    Returns:
        List of Trace objects.
    """
    # Resolve trace_id_field from fields config or explicit arg.
    if fields and hasattr(fields, "trace_id") and fields.trace_id:
        trace_id_field = fields.trace_id

    p = Path(path)

    # Try nested JSON array first (e.g. TRAIL dataset).
    traces = _try_load_json_array(p, trace_id_field)
    if traces is not None:
        if fields:
            _apply_fields(traces, fields)
        return traces

    # Standard JSONL: one line per trace.
    traces = _load_jsonl(p, trace_id_field)
    if fields:
        _apply_fields(traces, fields)
    return traces


# ── JSONL loader ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path, trace_id_field: str | None) -> list[Trace]:
    """Load traces from JSONL — one line per trace, spans nested inside."""
    id_field = trace_id_field or "trace_id"
    traces = []

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                _error(path, line_no, f"invalid JSON — {exc}")

            if not isinstance(row, dict):
                _error(path, line_no, f"expected JSON object, got {type(row).__name__}")

            row = _auto_parse_json_strings(row)
            trace_id = _resolve_trace_id(row, id_field, path, line_no)
            traces.append(_row_to_trace(row, trace_id))

    return traces


# ── JSON array loader (nested OTel format) ────────────────────────────────────

def _try_load_json_array(path: Path, trace_id_field: str | None) -> list[Trace] | None:
    """Try to parse as a JSON array of trace objects. Returns None if not this format."""
    try:
        with open(path) as f:
            first = f.read(1).strip()
            if first != "[":
                return None
            f.seek(0)
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(data, list) or not data:
        return None

    first_item = data[0]
    if not isinstance(first_item, dict):
        return None

    id_field = trace_id_field or "trace_id"
    has_id = id_field in first_item or "trace_id" in first_item

    # Check if it's nested OTel (has "spans" with "child_spans").
    if has_id and "spans" in first_item:
        spans_val = first_item["spans"]
        if isinstance(spans_val, list) and spans_val:
            first_span = spans_val[0]
            if isinstance(first_span, dict) and "child_spans" in first_span:
                return _load_nested_otel(data, id_field)

    if not has_id:
        return None

    # Regular JSON array of trace dicts.
    traces = []
    for item in data:
        if not isinstance(item, dict):
            continue
        item = _auto_parse_json_strings(item)
        trace_id = item.get(id_field) or item.get("trace_id", "")
        traces.append(_row_to_trace(item, str(trace_id)))
    return traces


# ── Nested OTel loader ────────────────────────────────────────────────────────

def _load_nested_otel(data: list[dict], id_field: str) -> list[Trace]:
    """Load traces from nested OTel format (child_spans trees)."""
    traces = []
    for trace_data in data:
        if not isinstance(trace_data, dict):
            continue
        trace_id = str(trace_data.get(id_field) or trace_data.get("trace_id", ""))
        root_spans = trace_data.get("spans") or []

        spans: list[Span] = []
        for root in root_spans:
            _flatten_otel_span(root, trace_id, parent_id=None, out=spans)

        spans.sort(key=lambda s: s.start_ns)

        # Detect outcome from root span status.
        outcome = None
        for root in root_spans:
            if str(root.get("status_code", "")).lower() == "error":
                outcome = "failure"
                break

        metadata = _extract_otel_metadata(trace_data, root_spans)

        traces.append(Trace(
            trace_id=trace_id,
            spans=spans,
            outcome=outcome,
            metadata=metadata,
        ))
    return traces


def _flatten_otel_span(
    span: dict, trace_id: str, parent_id: str | None, out: list[Span],
) -> None:
    """Recursively flatten a nested OTel span tree."""
    span_id = span.get("span_id", "")
    name = span.get("span_name") or span.get("name") or "unknown"
    attrs = span.get("span_attributes") or {}

    # Extract tokens from OpenInference or OTel GenAI attribute names.
    input_tokens = int(
        attrs.get("llm.token_count.prompt")
        or attrs.get("gen_ai.usage.input_tokens")
        or 0
    )
    output_tokens = int(
        attrs.get("llm.token_count.completion")
        or attrs.get("gen_ai.usage.output_tokens")
        or 0
    )
    model = (
        attrs.get("llm.model_name")
        or attrs.get("gen_ai.request.model")
    )
    cost = float(
        attrs.get("llm.cost.total")
        or attrs.get("kalibra.cost")
        or 0.0
    )

    start_ns, end_ns = _parse_otel_timing(span)
    is_error = str(span.get("status_code", "")).lower() == "error"

    out.append(Span(
        span_id=span_id,
        name=name,
        parent_id=parent_id,
        start_ns=start_ns,
        end_ns=end_ns,
        cost=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        error=is_error,
        attributes={k: v for k, v in attrs.items() if v is not None},
    ))

    for child in span.get("child_spans") or []:
        _flatten_otel_span(child, trace_id, parent_id=span_id, out=out)


def _parse_otel_timing(span: dict) -> tuple[int, int]:
    """Parse timing from nested OTel span."""
    ts = span.get("timestamp", "")
    dur = span.get("duration", "")

    start_ns = _iso_to_ns(ts) if ts else 0
    end_ns = start_ns

    if dur:
        end_ns = start_ns + _parse_iso_duration(dur)
    elif span.get("end_time"):
        end_ns = _iso_to_ns(span["end_time"])

    return start_ns, end_ns


def _extract_otel_metadata(trace_data: dict, root_spans: list[dict]) -> dict:
    """Extract metadata from nested OTel trace."""
    meta: dict = {}
    if root_spans:
        root = root_spans[0]
        svc = root.get("service_name", "")
        if svc:
            meta["service"] = svc
        res = root.get("resource_attributes") or {}
        for k, v in res.items():
            if v is not None:
                meta[f"resource.{k}"] = v
    return meta


# ── Row → Trace conversion ────────────────────────────────────────────────────

def _row_to_trace(row: dict, trace_id: str) -> Trace:
    """Convert a parsed JSONL row to a Trace object."""
    outcome = row.get("outcome")

    # Parse spans if present.
    raw_spans = row.get("spans")
    if isinstance(raw_spans, list) and raw_spans:
        spans = [_dict_to_span(s) for s in raw_spans if isinstance(s, dict)]
    else:
        # No spans array — create a synthetic span from trace-level fields.
        spans = [_trace_level_to_span(row, trace_id)]

    spans.sort(key=lambda s: s.start_ns)

    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    return Trace(
        trace_id=trace_id,
        spans=spans,
        outcome=outcome,
        metadata=metadata,
    )


def _dict_to_span(d: dict) -> Span:
    """Convert a span dict to a Span object."""
    start_ns, end_ns = _parse_timing(d)
    return Span(
        span_id=d.get("span_id", ""),
        name=d.get("name", ""),
        parent_id=d.get("parent_id"),
        start_ns=start_ns,
        end_ns=end_ns,
        cost=float(d.get("cost", 0.0)),
        input_tokens=int(d.get("input_tokens", 0)),
        output_tokens=int(d.get("output_tokens", 0)),
        model=d.get("model"),
        error=bool(d.get("error", False)),
        attributes=d.get("attributes") or {},
    )


def _trace_level_to_span(row: dict, trace_id: str) -> Span:
    """Create a synthetic span from trace-level fields (no spans array)."""
    start_ns, end_ns = _parse_timing(row)

    # Collect any extra fields as attributes.
    known = {
        "trace_id", "outcome", "metadata", "spans",
        "cost", "input_tokens", "output_tokens", "model",
        "start_time", "end_time", "duration_s", "start_ns", "end_ns",
        "error", "name", "span_id", "parent_id", "attributes",
    }
    attrs = {}
    for k, v in row.items():
        if k not in known and v is not None:
            if isinstance(v, dict):
                _flatten_dict(v, prefix=k, out=attrs)
            elif not isinstance(v, list):
                attrs[k] = v

    # Also include explicit attributes dict.
    extra = row.get("attributes")
    if isinstance(extra, dict):
        attrs.update(extra)

    return Span(
        span_id=row.get("span_id", f"{hash(trace_id) & 0xFFFFFFFF:08x}"),
        name=row.get("name", "eval"),
        parent_id=row.get("parent_id"),
        start_ns=start_ns,
        end_ns=end_ns,
        cost=float(row.get("cost", 0.0)),
        input_tokens=int(row.get("input_tokens", 0)),
        output_tokens=int(row.get("output_tokens", 0)),
        model=row.get("model"),
        error=bool(row.get("error", False)),
        attributes=attrs,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

_LIKELY_ID_FIELDS = [
    "uuid", "id", "instance_id", "task_name", "request_id",
    "trace_id", "traj_id", "run_id", "session_id",
]


def _resolve_trace_id(
    row: dict, id_field: str, path: Path, line_no: int,
) -> str:
    """Resolve the trace ID from a row, with helpful error on failure."""
    # Configured field or default trace_id.
    if id_field != "trace_id" and id_field in row:
        return str(row[id_field])
    if "trace_id" in row:
        return str(row["trace_id"])

    # Not found — build helpful error.
    row_keys = [k for k in row.keys()]
    candidates = [k for k in row_keys if k in _LIKELY_ID_FIELDS]
    for k in row_keys:
        if (k.endswith("_id") or k.endswith("_name")) and k not in candidates:
            candidates.append(k)

    msg = "no trace ID field found"
    if id_field != "trace_id":
        msg += f" (looked for '{id_field}' from config)"

    hints = [f"Available fields: {row_keys}"]
    if candidates:
        hints.append(f"These might be the trace ID: {candidates}")
    hints.append(
        "Set in kalibra.yml:\n"
        "  fields:\n"
        f"    trace_id: {candidates[0] if candidates else '<field_name>'}"
    )

    _error(path, line_no, msg, hint="\n".join(hints))
    return ""  # unreachable — _error raises


def _parse_timing(row: dict) -> tuple[int, int]:
    """Extract start/end nanoseconds from a row."""
    # Explicit nanoseconds.
    if "start_ns" in row:
        return int(row["start_ns"]), int(row.get("end_ns", row["start_ns"]))

    # ISO or unix timestamps.
    if "start_time" in row:
        start = _parse_ts_to_ns(row["start_time"])
        end = _parse_ts_to_ns(row.get("end_time", row["start_time"]))
        return start, end

    # Duration shorthand.
    if "duration_s" in row:
        dur_ns = int(float(row["duration_s"]) * 1e9)
        return 0, dur_ns

    return 0, 0


def _parse_ts_to_ns(val) -> int:
    """Parse a timestamp to nanoseconds."""
    if isinstance(val, (int, float)):
        if val < 1e12:
            return int(val * 1e9)
        return int(val)
    if isinstance(val, str):
        return _iso_to_ns(val)
    return 0


def _iso_to_ns(ts: str) -> int:
    """Parse ISO 8601 timestamp to nanoseconds."""
    try:
        ts = ts.rstrip("Z")
        if "." in ts:
            dt = datetime.strptime(ts[:26], "%Y-%m-%dT%H:%M:%S.%f")
        else:
            dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1e9)
    except (ValueError, TypeError):
        return 0


def _parse_iso_duration(dur: str) -> int:
    """Parse ISO 8601 duration (e.g. PT1M24.6S) to nanoseconds."""
    if not dur.startswith("PT"):
        return 0
    dur = dur[2:]
    total_seconds = 0.0
    for unit, multiplier in [("H", 3600), ("M", 60), ("S", 1)]:
        if unit in dur:
            val_str, dur = dur.split(unit, 1)
            total_seconds += float(val_str) * multiplier
    return int(total_seconds * 1e9)


def _auto_parse_json_strings(obj):
    """Recursively parse JSON strings embedded in field values."""
    if isinstance(obj, dict):
        return {k: _auto_parse_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_auto_parse_json_strings(v) for v in obj]
    if isinstance(obj, str) and obj and obj[0] in ("{", "["):
        try:
            return _auto_parse_json_strings(json.loads(obj))
        except (json.JSONDecodeError, ValueError):
            pass
    return obj


def _flatten_dict(d: dict, prefix: str, out: dict) -> None:
    """Flatten a nested dict into dot-notation keys."""
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            _flatten_dict(v, prefix=key, out=out)
        elif v is not None and not isinstance(v, list):
            out[key] = v


def _error(
    path: Path, line_no: int, msg: str, *, hint: str | None = None,
) -> None:
    """Raise a formatted parse error."""
    parts = [f"\n  {path}:{line_no} — {msg}"]
    if hint:
        for line in hint.split("\n"):
            parts.append(f"  {line}")
    raise ValueError("\n".join(parts))


# ── Field mapping (post-load) ────────────────────────────────────────────────

def _resolve_dot_path(obj: dict, path: str):
    """Resolve a dot-notation path against a dict.

    Tries two strategies:
    1. Direct flat key: obj["agent_cost.total_cost"] (flattened attributes)
    2. Nested traversal: obj["agent_cost"]["total_cost"] (nested dicts)
    """
    # Strategy 1: the full path is already a flat key (from _flatten_dict).
    if path in obj:
        return obj[path]

    # Strategy 2: walk nested dicts.
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _apply_fields(traces: list[Trace], fields: object) -> None:
    """Apply FieldsConfig mappings to loaded traces.

    Resolves outcome, cost, input_tokens, output_tokens from custom
    field paths. Modifies traces in place.
    """
    outcome_field = getattr(fields, "outcome", None)
    cost_field = getattr(fields, "cost", None)
    input_tokens_field = getattr(fields, "input_tokens", None)
    output_tokens_field = getattr(fields, "output_tokens", None)
    task_id_field = getattr(fields, "task_id", None)

    if not any([outcome_field, cost_field, input_tokens_field,
                output_tokens_field, task_id_field]):
        return

    for trace in traces:
        # ── Outcome mapping ──────────────────────────────────────────
        if outcome_field and trace.outcome is None:
            # Try metadata first, then span attributes.
            val = _resolve_dot_path(trace.metadata, outcome_field)
            if val is None:
                for s in trace.spans:
                    val = _resolve_dot_path(s.attributes, outcome_field)
                    if val is not None:
                        break
            if val is not None:
                trace.outcome = _classify_outcome(val)

        # ── Task ID mapping ──────────────────────────────────────────
        if task_id_field:
            val = _resolve_dot_path(trace.metadata, task_id_field)
            if val is not None:
                trace.metadata["task_id"] = val

        # ── Span-level field mappings ────────────────────────────────
        if cost_field or input_tokens_field or output_tokens_field:
            for span in trace.spans:
                # Build a combined lookup: span dict fields + attributes.
                lookup = span.attributes

                if cost_field and span.cost == 0:
                    val = _resolve_dot_path(lookup, cost_field)
                    if val is not None:
                        span.cost = float(val)

                if input_tokens_field and span.input_tokens == 0:
                    val = _resolve_dot_path(lookup, input_tokens_field)
                    if val is not None:
                        span.input_tokens = int(val)

                if output_tokens_field and span.output_tokens == 0:
                    val = _resolve_dot_path(lookup, output_tokens_field)
                    if val is not None:
                        span.output_tokens = int(val)


def _classify_outcome(val) -> str | None:
    """Classify a raw value as success/failure outcome."""
    if isinstance(val, bool):
        return OUTCOME_SUCCESS if val else OUTCOME_FAILURE
    val_str = str(val).lower().strip()
    if val_str in _SUCCESS_VALUES:
        return OUTCOME_SUCCESS
    if val_str in _FAILURE_VALUES:
        return OUTCOME_FAILURE
    return None
