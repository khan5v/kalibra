"""Flat JSONL format — Kalibra's built-in trace format.

One JSON object per line, one trace per line. Spans nested inside.
This is the fallback format when no other format matches.

Minimal trace:
    {"trace_id": "t1", "outcome": "success", "cost": 0.05}

Full trace with spans:
    {"trace_id": "t1", "outcome": "success", "spans": [
        {"span_id": "s1", "name": "plan", "cost": 0.03, "input_tokens": 500},
        {"span_id": "s2", "name": "search", "parent_id": "s1", "error": true}
    ]}
"""

from __future__ import annotations

import json
from pathlib import Path

from kalibra.loaders import TraceFormat
from kalibra.loaders._utils import (
    _auto_parse_json_strings,
    _classify_outcome,
    _flatten_dict,
    _parse_ts_to_ns,
    _safe_float,
    _safe_int,
)
from kalibra.model import Span, Trace


class FlatFormat(TraceFormat):
    """Kalibra's built-in flat JSONL format (one trace per line)."""

    name = "flat"

    def detect(self, item: dict) -> bool:
        # Flat is the fallback — never claims a match during auto-detection.
        return False

    def load(self, path: Path) -> list[Trace]:
        return _load_flat_jsonl(path)


# ── Known field names (excluded from metadata) ──────────────────────────────

_TRACE_KNOWN_FIELDS = {
    "trace_id", "outcome", "metadata", "spans",
    "cost", "input_tokens", "output_tokens", "model",
    "start_time", "end_time", "duration_s", "start_ns", "end_ns",
    "error", "name", "span_id", "parent_id", "attributes",
}


# ── Loading ──────────────────────────────────────────────────────────────────

def _load_flat_jsonl(path: Path, trace_id_field: str | None = None) -> list[Trace]:
    """Load flat JSONL — one trace per line, spans nested inside."""
    id_field = trace_id_field or "trace_id"
    traces = []

    bad_lines: list[int] = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines.append(line_no)
                continue

            if not isinstance(row, dict):
                bad_lines.append(line_no)
                continue

            row = _auto_parse_json_strings(row)
            trace_id = _resolve_trace_id(row, id_field)
            traces.append(_row_to_trace(row, trace_id))

    if bad_lines:
        n = len(bad_lines)
        sample = ", ".join(str(ln) for ln in bad_lines[:5])
        if n > 5:
            sample += f", ... ({n} total)"
        raise ValueError(
            f"\n  {path} — {n} malformed line(s): {sample}\n"
            f"  Successfully parsed {len(traces)} trace(s), "
            f"but cannot proceed with corrupt input.\n"
            f"  Fix or remove the invalid lines and retry."
        )

    return traces


# ── Row → Trace conversion ───────────────────────────────────────────────────

def _row_to_trace(row: dict, trace_id: str) -> Trace:
    """Convert a parsed JSONL row to a Trace object."""
    outcome = _classify_outcome(row.get("outcome")) if row.get("outcome") is not None else None

    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    # Collect extra fields as metadata — always, regardless of spans.
    for k, v in row.items():
        if k not in _TRACE_KNOWN_FIELDS and v is not None:
            if isinstance(v, dict):
                _flatten_dict(v, prefix=k, out=metadata)
            elif not isinstance(v, list):
                metadata[k] = v
    extra = row.get("attributes")
    if isinstance(extra, dict):
        metadata.update(extra)

    # Parse spans if present.
    raw_spans = row.get("spans")
    if isinstance(raw_spans, list) and raw_spans:
        spans = [_dict_to_span(s) for s in raw_spans if isinstance(s, dict)]
        spans.sort(key=lambda s: s.start_ns)
        return Trace(
            trace_id=trace_id,
            spans=spans,
            outcome=outcome,
            metadata=metadata,
        )

    # No spans — set trace-level fields directly.
    start_ns, end_ns = _parse_timing(row)

    raw_cost = row.get("cost")
    raw_in = row.get("input_tokens")
    raw_out = row.get("output_tokens")
    duration = (end_ns - start_ns) / 1e9 if (start_ns or end_ns) else None
    if duration is None and "duration_s" in row:
        duration = _safe_float(row["duration_s"])

    return Trace(
        trace_id=trace_id,
        spans=[],
        outcome=outcome,
        metadata=metadata,
        _cost=_safe_float(raw_cost),
        _input_tokens=_safe_int(raw_in),
        _output_tokens=_safe_int(raw_out),
        _duration_s=duration,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dict_to_span(d: dict) -> Span:
    """Convert a span dict to a Span object."""
    start_ns, end_ns = _parse_timing(d)
    raw_cost = d.get("cost")
    raw_in = d.get("input_tokens")
    raw_out = d.get("output_tokens")
    return Span(
        span_id=d.get("span_id", ""),
        name=d.get("name", ""),
        parent_id=d.get("parent_id"),
        start_ns=start_ns,
        end_ns=end_ns,
        cost=_safe_float(raw_cost),
        input_tokens=_safe_int(raw_in),
        output_tokens=_safe_int(raw_out),
        model=d.get("model"),
        error=bool(d.get("error", False)),
        attributes=d.get("attributes") or {},
    )


def _resolve_trace_id(row: dict, id_field: str) -> str:
    """Resolve the trace ID from a row."""
    if id_field != "trace_id" and id_field in row:
        return str(row[id_field])
    if "trace_id" in row:
        return str(row["trace_id"])
    return ""


def _parse_timing(row: dict) -> tuple[int, int]:
    """Extract start/end nanoseconds from a row."""
    if "start_ns" in row:
        return int(row["start_ns"]), int(row.get("end_ns", row["start_ns"]))
    if "start_time" in row:
        start = _parse_ts_to_ns(row["start_time"])
        end = _parse_ts_to_ns(row.get("end_time", row["start_time"]))
        return start, end
    if "duration_s" in row:
        dur_ns = int(float(row["duration_s"]) * 1e9)
        return 0, dur_ns
    return 0, 0
