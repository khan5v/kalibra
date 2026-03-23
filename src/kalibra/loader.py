"""Trace loader — thin orchestrator that detects formats and delegates.

The actual loading logic lives in kalibra.loaders.* (one module per format).
This module handles: format detection, field mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

from kalibra.loaders._utils import (
    _classify_outcome,
    _resolve_dot_path,
    _safe_float,
    _safe_int,
)
from kalibra.loaders.flat import _load_flat_jsonl
from kalibra.model import Trace


# ── Format registry ──────────────────────────────────────────────────────────

_ALL_FORMATS = None
_FORMAT_MAP = None


def _get_formats():
    """Lazy-initialize the format registry (avoids circular imports at module level).

    Detection order matters: first match wins. Formats must check for
    mutually exclusive signals (e.g. openinference.span.kind vs gen_ai.*).
    FlatFormat is the fallback — it never participates in auto-detection.
    """
    global _ALL_FORMATS, _FORMAT_MAP
    if _ALL_FORMATS is None:
        from kalibra.loaders.openinference import OpenInferenceFormat
        from kalibra.loaders.flat import FlatFormat
        _ALL_FORMATS = [OpenInferenceFormat()]
        _FORMAT_MAP = {f.name: f for f in _ALL_FORMATS}
        _FORMAT_MAP["flat"] = FlatFormat()
    return _ALL_FORMATS, _FORMAT_MAP


# ── Public API ───────────────────────────────────────────────────────────────

def load_traces(
    path: str | Path,
    trace_id_field: str | None = None,
    fields: object | None = None,
    format: str = "auto",
) -> list[Trace]:
    """Load traces from a JSONL file.

    Args:
        path: Path to the JSONL file.
        trace_id_field: If set, use this field instead of ``trace_id``.
            Deprecated — prefer ``fields.trace_id``.
        fields: Optional FieldsConfig with field mappings for outcome,
            cost, input_tokens, output_tokens, trace_id, task_id.
        format: Trace format — ``"auto"`` to detect, or an explicit name
            like ``"openinference"``, ``"otel-genai"``, ``"flat"``.

    Returns:
        List of Trace objects.
    """
    all_formats, format_map = _get_formats()

    # Resolve trace_id_field from fields config or explicit arg.
    if fields and hasattr(fields, "trace_id") and fields.trace_id:
        trace_id_field = fields.trace_id

    p = Path(path)

    # Explicit format — skip detection.
    if format != "auto":
        if format not in format_map:
            known = ", ".join(sorted(format_map))
            raise ValueError(
                f"Unknown format {format!r}. Available: {known}"
            )
        fmt = format_map[format]
        traces = fmt.load(p)
        if fields:
            _apply_fields(traces, fields)
        return traces

    # Auto-detect: peek at first JSONL line, try each format, fall back to flat.
    first_item = _peek_first_item(p)
    if first_item is not None:
        for fmt in all_formats:
            if fmt.detect(first_item):
                traces = fmt.load(p)
                if fields:
                    _apply_fields(traces, fields)
                return traces

    # Fallback: flat JSONL.
    traces = _load_flat_jsonl(p, trace_id_field)
    if fields:
        _apply_fields(traces, fields)
    return traces


def _peek_first_item(path: Path) -> dict | None:
    """Read the first valid JSON object from a JSONL file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    return row
            except json.JSONDecodeError:
                pass
    return None


# ── Field mapping (post-load, format-agnostic) ──────────────────────────────

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
    duration_field = getattr(fields, "duration", None)

    if not any([outcome_field, cost_field, input_tokens_field,
                output_tokens_field, task_id_field, duration_field]):
        return

    for trace in traces:
        # ── Outcome mapping ──────────────────────────────────────────
        if outcome_field and trace.outcome is None:
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

        # ── Cost/token/duration field mappings ───────────────────────
        if cost_field or input_tokens_field or output_tokens_field:
            if trace.spans:
                for span in trace.spans:
                    lookup = span.attributes
                    if cost_field and span.cost is None:
                        val = _resolve_dot_path(lookup, cost_field)
                        if val is not None:
                            span.cost = _safe_float(val)
                    if input_tokens_field and span.input_tokens is None:
                        val = _resolve_dot_path(lookup, input_tokens_field)
                        if val is not None:
                            span.input_tokens = _safe_int(val)
                    if output_tokens_field and span.output_tokens is None:
                        val = _resolve_dot_path(lookup, output_tokens_field)
                        if val is not None:
                            span.output_tokens = _safe_int(val)
            else:
                if cost_field and trace._cost is None:
                    val = _resolve_dot_path(trace.metadata, cost_field)
                    if val is not None:
                        trace._cost = _safe_float(val)
                if input_tokens_field and trace._input_tokens is None:
                    val = _resolve_dot_path(trace.metadata, input_tokens_field)
                    if val is not None:
                        trace._input_tokens = _safe_int(val)
                if output_tokens_field and trace._output_tokens is None:
                    val = _resolve_dot_path(trace.metadata, output_tokens_field)
                    if val is not None:
                        trace._output_tokens = _safe_int(val)

        # Duration mapping — independent of cost/token fields.
        if duration_field and not trace.spans and trace._duration_s is None:
            val = _resolve_dot_path(trace.metadata, duration_field)
            if val is not None:
                trace._duration_s = _safe_float(val)
