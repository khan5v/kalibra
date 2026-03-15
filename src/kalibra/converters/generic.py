"""JSONL trace format — kalibra's portable interchange format.

Two input formats, auto-detected per file:

**Flat eval** — one row per trace, no span detail::

    {"trace_id": "task-1", "outcome": "success", "cost": 0.012, "duration_s": 5.2}

**Flat spans** — one row per span, grouped by trace_id::

    {"trace_id": "t1", "span_id": "s1", "name": "planner", "outcome": "success", ...}
    {"trace_id": "t1", "span_id": "s2", "name": "tool-call", "parent_id": "s1", ...}

Save always writes flat spans (one row per span, trace-level fields on root span).

.. note:: Round-trip format change

   Loading a flat-eval file and saving it produces flat-span JSONL (one row
   per span instead of one row per trace).  All data is preserved — only the
   layout changes.  This happens when ``pull`` caches a ``source: jsonl``
   file or when the programmatic API round-trips through ``save_jsonl``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from opentelemetry.sdk.trace import StatusCode

from kalibra.converters.base import (
    AF_COST,
    GEN_AI_INPUT_TOKENS,
    GEN_AI_MODEL,
    GEN_AI_OUTPUT_TOKENS,
    Trace,
    make_span,
)

# ── Friendly field names ↔ OTel attribute keys ───────────────────────────────

_FRIENDLY_TO_ATTR = {
    "model": GEN_AI_MODEL,
    "cost": AF_COST,
    "input_tokens": GEN_AI_INPUT_TOKENS,
    "output_tokens": GEN_AI_OUTPUT_TOKENS,
}

_ATTR_TO_FRIENDLY = {v: k for k, v in _FRIENDLY_TO_ATTR.items()}

# Fields consumed by the parser — everything else goes into attributes.
_TRACE_FIELDS = {"trace_id", "outcome", "metadata"}
_SPAN_FIELDS = {"span_id", "parent_id", "name", "start_time", "end_time", "error"}
_TIMING_FIELDS = {"duration_s", "start_ns", "end_ns"}
_INTERNAL = {"attributes", "_line"}
_RESERVED = _TRACE_FIELDS | _SPAN_FIELDS | _TIMING_FIELDS | set(_FRIENDLY_TO_ATTR) | _INTERNAL


# ── Load ─────────────────────────────────────────────────────────────────────

def load_json_traces(
    path: Path,
    trace_id_field: str | None = None,
) -> list[Trace]:
    """Load traces from JSONL or JSON. Auto-detects format:

    - Nested OTel traces (JSON array with ``spans`` containing ``child_spans``)
    - Flat spans (JSONL, one row per span with ``span_id``)
    - Flat evals (JSONL, one row per trace without ``span_id``)

    Args:
        trace_id_field: If set, use this field instead of ``trace_id``.
    """
    # Try nested OTel format first (JSON array of trace objects).
    nested = _try_load_nested(path, trace_id_field=trace_id_field)
    if nested is not None:
        return nested

    # Fall back to JSONL (flat eval or flat span).
    rows = _read_rows(path, trace_id_field=trace_id_field)
    if not rows:
        return []

    fmt = _detect_format(rows, path)

    if fmt == "flat_span":
        return _load_flat_spans(rows, path)
    return _load_flat_evals(rows, path)


def _read_rows(
    path: Path,
    trace_id_field: str | None = None,
) -> list[dict]:
    """Parse JSONL into a list of dicts with clear error messages."""
    # Which field to use as trace_id.
    id_field = trace_id_field or "trace_id"

    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                _error(path, line_no, f"invalid JSON — {exc}", hint=(
                    "Each line must be a valid JSON object. Example:\n"
                    '  {"trace_id": "task-1", "outcome": "success"}'
                ))

            if not isinstance(d, dict):
                _error(
                    path, line_no,
                    f"expected a JSON object, got {type(d).__name__}",
                )

            # Resolve trace ID: use configured field, remap to trace_id.
            if id_field != "trace_id" and id_field in d:
                d["trace_id"] = d[id_field]
            elif "trace_id" not in d:
                _trace_id_error(path, line_no, d, id_field)

            d["_line"] = line_no
            rows.append(d)
    return rows


# Common ID-like field names to suggest when trace_id is missing.
_LIKELY_ID_FIELDS = [
    "uuid", "id", "instance_id", "task_name", "request_id",
    "trace_id", "traj_id", "run_id", "session_id",
]


def _trace_id_error(
    path: Path, line_no: int, row: dict, configured_field: str,
) -> None:
    """Raise a helpful error when trace_id can't be found."""
    row_keys = [k for k in row.keys() if k != "_line"]

    # Find fields that look like IDs.
    candidates = [k for k in row_keys if k in _LIKELY_ID_FIELDS]
    # Also check for fields ending in _id or _name.
    for k in row_keys:
        if (k.endswith("_id") or k.endswith("_name")) and k not in candidates:
            candidates.append(k)

    msg = f"no trace ID field found"
    if configured_field != "trace_id":
        msg += f" (looked for '{configured_field}' from config)"

    hint_lines = [f"Available fields: {row_keys}"]
    if candidates:
        hint_lines.append(
            f"These might be the trace ID: {candidates}"
        )
    hint_lines.append(
        "Set in kalibra.yml:\n"
        "  fields:\n"
        f"    trace_id: {candidates[0] if candidates else '<field_name>'}"
    )

    _error(path, line_no, msg, hint="\n".join(hint_lines))


def _detect_format(rows: list[dict], path: Path) -> str:
    """Detect format by checking all rows for span_id consistency.

    If any row has ``span_id``, the file is treated as flat span format.
    Warns if some rows have ``span_id`` and others don't — this usually
    means the file mixes formats, which can produce unexpected results.
    """
    has_span_id = [("span_id" in row) for row in rows]
    n_with = sum(has_span_id)

    if n_with == 0:
        return "flat_eval"
    if n_with == len(rows):
        return "flat_span"

    # Mixed — some rows have span_id, some don't. Treat as flat span but warn.
    first_without = next(r for r, has in zip(rows, has_span_id) if not has)
    _error(path, first_without.get("_line", 0),
           f"mixed format — {n_with} of {len(rows)} rows have 'span_id'",
           trace_id=first_without.get("trace_id"),
           hint=(
               "All rows in a file must use the same format.\n"
               "  Flat eval:  no span_id (one row per trace)\n"
               "  Flat span:  every row has span_id (one row per span)\n\n"
               "  If this is a flat-span file, add span_id to every row."
           ))


# ── Nested OTel loader ───────────────────────────────────────────────────────

# OpenInference attribute keys (used by TRAIL / Phoenix / Arize).
_OI_TOKEN_PROMPT = "llm.token_count.prompt"
_OI_TOKEN_COMPLETION = "llm.token_count.completion"
_OI_MODEL = "llm.model_name"
_OI_COST = "llm.cost.total"


def _try_load_nested(
    path: Path,
    trace_id_field: str | None = None,
) -> list[Trace] | None:
    """Try to parse as a JSON array of nested OTel traces.

    Returns None if the file isn't in nested format (falls through to JSONL).
    """
    try:
        with open(path) as f:
            first_char = f.read(1).strip()
            if first_char != "[":
                return None
            f.seek(0)
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(data, list) or not data:
        return None

    # Check if it looks like nested OTel: objects with trace_id + "spans".
    first = data[0]
    if not isinstance(first, dict):
        return None
    id_field = trace_id_field or "trace_id"
    has_id = id_field in first or "trace_id" in first
    if not has_id or "spans" not in first:
        return None
    # Remap if using a custom field.
    if id_field != "trace_id":
        for item in data:
            if isinstance(item, dict) and id_field in item:
                item["trace_id"] = item[id_field]

    return [_convert_nested_trace(t) for t in data if isinstance(t, dict)]


def _convert_nested_trace(trace_data: dict) -> Trace:
    """Convert a nested OTel trace (with child_spans) to a Kalibra Trace."""
    trace_id = trace_data["trace_id"]
    root_spans = trace_data.get("spans") or []

    spans = []
    for root in root_spans:
        _flatten_span(root, trace_id, parent_id=None, out=spans)

    spans.sort(key=lambda s: s.start_time)

    # Detect outcome from status or errors.
    outcome = None
    for root in root_spans:
        status = root.get("status_code", "").lower()
        if status == "error":
            outcome = "failure"
            break

    return Trace(
        trace_id=trace_id,
        spans=spans,
        outcome=outcome,
        metadata=_extract_nested_metadata(trace_data, root_spans),
    )


def _flatten_span(
    span: dict, trace_id: str, parent_id: str | None, out: list,
) -> None:
    """Recursively flatten a nested span tree into a flat list."""
    span_id = span.get("span_id", "")
    name = span.get("span_name") or span.get("name") or "unknown"
    attrs = dict(span.get("span_attributes") or {})

    # Map OpenInference token/cost attributes to Kalibra's OTel keys.
    prompt_tokens = attrs.pop(_OI_TOKEN_PROMPT, 0)
    completion_tokens = attrs.pop(_OI_TOKEN_COMPLETION, 0)
    attrs.pop("llm.token_count.total", None)  # derived, don't double-count
    model = attrs.pop(_OI_MODEL, None)
    cost = attrs.pop(_OI_COST, 0.0)

    # Also check for standard OTel GenAI keys (some exporters use these).
    if not prompt_tokens:
        prompt_tokens = attrs.pop(GEN_AI_INPUT_TOKENS, 0)
    if not completion_tokens:
        completion_tokens = attrs.pop(GEN_AI_OUTPUT_TOKENS, 0)
    if not model:
        model = attrs.pop(GEN_AI_MODEL, None)
    if not cost:
        cost = attrs.pop(AF_COST, 0.0)

    otel_attrs = {
        GEN_AI_INPUT_TOKENS: int(prompt_tokens or 0),
        GEN_AI_OUTPUT_TOKENS: int(completion_tokens or 0),
        AF_COST: float(cost or 0.0),
    }
    if model:
        otel_attrs[GEN_AI_MODEL] = model

    # Forward remaining span_attributes.
    for k, v in attrs.items():
        if v is not None:
            otel_attrs[k] = v

    # Parse timing.
    start_ns, end_ns = _parse_nested_timing(span)

    is_error = span.get("status_code", "").lower() == "error"

    out.append(make_span(
        name=name,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_id,
        start_ns=start_ns,
        end_ns=end_ns,
        attributes=otel_attrs,
        error=is_error,
    ))

    for child in span.get("child_spans") or []:
        _flatten_span(child, trace_id, parent_id=span_id, out=out)


def _parse_nested_timing(span: dict) -> tuple[int, int]:
    """Parse start/end from nested span. Handles ISO timestamps and durations."""
    ts = span.get("timestamp", "")
    duration_str = span.get("duration", "")

    start_ns = 0
    if ts:
        start_ns = _iso_to_ns(ts)

    end_ns = start_ns
    if duration_str:
        dur_ns = _parse_iso_duration(duration_str)
        end_ns = start_ns + dur_ns
    elif span.get("end_time"):
        end_ns = _iso_to_ns(span["end_time"])

    return start_ns, end_ns


def _iso_to_ns(ts: str) -> int:
    """Parse ISO 8601 timestamp to nanoseconds."""
    try:
        # Handle both 'Z' and '+00:00' suffixes.
        ts = ts.rstrip("Z")
        if "." in ts:
            dt = datetime.strptime(ts[:26], "%Y-%m-%dT%H:%M:%S.%f")
        else:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1e9)
    except (ValueError, TypeError):
        return 0


def _parse_iso_duration(dur: str) -> int:
    """Parse ISO 8601 duration (e.g. PT1M24.635189S) to nanoseconds."""
    if not dur.startswith("PT"):
        return 0
    dur = dur[2:]  # strip "PT"
    total_seconds = 0.0
    # Parse hours, minutes, seconds.
    for unit, multiplier in [("H", 3600), ("M", 60), ("S", 1)]:
        if unit in dur:
            val_str, dur = dur.split(unit, 1)
            total_seconds += float(val_str) * multiplier
    return int(total_seconds * 1e9)


def _extract_nested_metadata(
    trace_data: dict, root_spans: list[dict],
) -> dict:
    """Extract trace metadata from nested OTel format."""
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


# ── Flat eval loader ─────────────────────────────────────────────────────────

def _load_flat_evals(rows: list[dict], path: Path) -> list[Trace]:
    """Each row is one trace. Synthesize a single span from top-level fields."""
    traces = []
    for row in rows:
        line_no = row.pop("_line", 0)
        trace_id = row["trace_id"]
        outcome = _parse_outcome(row.get("outcome"), path, line_no, trace_id)
        start_ns, end_ns = _parse_timing(row)
        attrs = _extract_attrs(row)

        span = make_span(
            name=row.get("name", "eval"),
            trace_id=trace_id,
            span_id=f"{hash(trace_id) & 0xFFFFFFFF:08x}",
            parent_span_id=None,
            start_ns=start_ns,
            end_ns=end_ns,
            attributes=attrs,
            error=(row.get("error") is True),
        )
        traces.append(Trace(
            trace_id=trace_id,
            spans=[span],
            outcome=outcome,
            metadata=row.get("metadata", {}),
        ))
    return traces


# ── Flat span loader ─────────────────────────────────────────────────────────

def _load_flat_spans(rows: list[dict], path: Path) -> list[Trace]:
    """Rows are spans, grouped by trace_id. Trace fields from root span row."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["trace_id"]].append(row)

    traces = []
    for trace_id, span_rows in grouped.items():
        outcome = None
        metadata: dict = {}
        spans = []

        for row in span_rows:
            line_no = row.pop("_line", 0)

            # Trace-level fields: take from first row that has them
            if outcome is None and "outcome" in row:
                outcome = _parse_outcome(row["outcome"], path, line_no, trace_id)
            if not metadata and "metadata" in row:
                metadata = row["metadata"]

            start_ns, end_ns = _parse_timing(row)
            attrs = _extract_attrs(row)

            span = make_span(
                name=row.get("name", "span"),
                trace_id=trace_id,
                span_id=row["span_id"],
                parent_span_id=row.get("parent_id"),
                start_ns=start_ns,
                end_ns=end_ns,
                attributes=attrs,
                error=(row.get("error") is True),
            )
            spans.append(span)

        spans.sort(key=lambda s: s.start_time)
        traces.append(Trace(
            trace_id=trace_id, spans=spans, outcome=outcome, metadata=metadata,
        ))
    return traces


# ── Save ─────────────────────────────────────────────────────────────────────

def save_jsonl(traces: list[Trace], path: str) -> None:
    """Write traces as flat-span JSONL (one row per span)."""
    with open(path, "w") as f:
        for t in traces:
            for i, s in enumerate(t.spans):
                row = _span_to_row(t, s, is_root=(i == 0))
                f.write(json.dumps(row) + "\n")


def _span_to_row(trace: Trace, s, is_root: bool) -> dict:
    """Serialize a span as a flat row. Root span carries trace-level fields."""
    attrs = dict(s.attributes or {})

    row: dict = {"trace_id": trace.trace_id}

    # Root span carries outcome and metadata
    if is_root:
        row["outcome"] = trace.outcome
        if trace.metadata:
            row["metadata"] = trace.metadata

    row["span_id"] = format(s.context.span_id, "016x") if s.context else ""
    row["parent_id"] = format(s.parent.span_id, "016x") if s.parent else None
    row["name"] = s.name
    row["start_time"] = _ns_to_iso(s.start_time)
    row["end_time"] = _ns_to_iso(s.end_time)

    # Extract known attributes as top-level friendly fields
    for otel_key, friendly in _ATTR_TO_FRIENDLY.items():
        val = attrs.pop(otel_key, None)
        if val is not None:
            row[friendly] = val

    if s.status.status_code == StatusCode.ERROR:
        row["error"] = True

    # Remaining attributes
    if attrs:
        row["attributes"] = attrs

    return row


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_attrs(row: dict) -> dict:
    """Extract OTel attributes from a flat row. Friendly names → OTel keys."""
    attrs = {}
    for friendly, otel_key in _FRIENDLY_TO_ATTR.items():
        val = row.get(friendly)
        if val is not None:
            attrs[otel_key] = val

    # Pass through any explicit attributes dict
    extra = row.get("attributes")
    if isinstance(extra, dict):
        attrs.update(extra)

    # Any unknown fields → attributes
    for k, v in row.items():
        if k not in _RESERVED and v is not None:
            attrs[k] = v

    return attrs


def _parse_timing(row: dict) -> tuple[int, int]:
    """Extract start/end nanoseconds from a row, accepting multiple formats."""
    # Nanoseconds (internal/precise)
    if "start_ns" in row:
        return int(row["start_ns"]), int(row["end_ns"])

    # ISO or unix timestamps
    if "start_time" in row:
        start = _parse_ts_to_ns(row["start_time"])
        end = _parse_ts_to_ns(row.get("end_time", row["start_time"]))
        return start, end

    # Duration shorthand
    if "duration_s" in row:
        dur_ns = int(float(row["duration_s"]) * 1e9)
        return 0, dur_ns

    return 0, 0


def _parse_ts_to_ns(val) -> int:
    """Parse a timestamp value to nanoseconds. Accepts ISO string or numeric."""
    if isinstance(val, (int, float)):
        # If it looks like seconds (< 1e12), convert; otherwise assume nanos
        if val < 1e12:
            return int(val * 1e9)
        return int(val)
    if isinstance(val, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(val.rstrip("Z"), fmt.rstrip("Z"))
                return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e9)
            except ValueError:
                continue
        # Try ISO format with timezone
        try:
            dt = datetime.fromisoformat(val)
            return int(dt.timestamp() * 1e9)
        except ValueError:
            pass
    return 0


def _ns_to_iso(ns: int | None) -> str:
    """Convert nanoseconds to ISO 8601 string."""
    if not ns:
        return "1970-01-01T00:00:00Z"
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_outcome(val, path: Path, line_no: int, trace_id: str) -> str | None:
    """Validate outcome value."""
    if val is None:
        return None
    if val in ("success", "failure"):
        return val
    _error(path, line_no,
           f"'outcome' must be \"success\", \"failure\", or null — got {val!r}",
           trace_id=trace_id)


def _error(path: Path, line_no: int, msg: str, *,
           trace_id: str | None = None, hint: str | None = None) -> None:
    """Raise a clear, formatted parse error."""
    parts = [f"\n  {path}:{line_no} — {msg}"]
    if trace_id:
        parts.append(f"  trace_id: {trace_id}")
    if hint:
        for hint_line in hint.split("\n"):
            parts.append(f"  {hint_line}")
    raise ValueError("\n".join(parts))
