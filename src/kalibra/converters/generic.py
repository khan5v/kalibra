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

def load_json_traces(path: Path) -> list[Trace]:
    """Load traces from a JSONL file. Auto-detects flat eval or flat span format."""
    rows = _read_rows(path)
    if not rows:
        return []

    fmt = _detect_format(rows, path)

    if fmt == "flat_span":
        return _load_flat_spans(rows, path)
    return _load_flat_evals(rows, path)


def _read_rows(path: Path) -> list[dict]:
    """Parse JSONL into a list of dicts with clear error messages."""
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
                    '  {"trace_id": "task-1", "outcome": "success", "cost": 0.012}'
                ))

            if not isinstance(d, dict):
                _error(path, line_no, f"expected a JSON object, got {type(d).__name__}")

            if "trace_id" not in d:
                _error(path, line_no, "missing required field 'trace_id'", hint=(
                    "Every row needs a trace_id. Minimal example:\n"
                    '  {"trace_id": "task-1", "outcome": "success"}'
                ))

            d["_line"] = line_no
            rows.append(d)
    return rows


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
