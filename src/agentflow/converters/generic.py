"""JSONL trace format — agentflow's portable interchange format.

Written by ``agentflow pull``, readable by ``agentflow compare``.
One JSON object per line: {trace_id, outcome, metadata, spans}.

Span fields:
  span_id, parent_span_id, name, start_ns, end_ns, attributes, error
"""

from __future__ import annotations

import json
from pathlib import Path

from agentflow.converters.base import Trace, make_span
from opentelemetry.sdk.trace import StatusCode


def load_json_traces(path: Path) -> list[Trace]:
    """Load traces from a JSONL file (supports both current and legacy formats)."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            trace_id = d["trace_id"]
            spans = [_load_span(s, trace_id) for s in d.get("spans", [])]
            traces.append(Trace(
                trace_id=trace_id,
                spans=spans,
                outcome=d.get("outcome"),
                metadata=d.get("metadata", {}),
            ))
    return traces


def _load_span(s: dict, trace_id: str):
    """Load a span from dict, handling both current and legacy serialization."""
    from agentflow.converters.base import AF_COST, GEN_AI_INPUT_TOKENS, GEN_AI_MODEL, GEN_AI_OUTPUT_TOKENS

    # Current format: start_ns / end_ns
    if "start_ns" in s:
        return make_span(
            name=s["name"],
            trace_id=trace_id,
            span_id=s["span_id"],
            parent_span_id=s.get("parent_span_id"),
            start_ns=s["start_ns"],
            end_ns=s["end_ns"],
            attributes=s.get("attributes", {}),
            error=s.get("error", False),
        )

    # Legacy format: start_time (float seconds), model/cost/status as top-level fields
    attrs = dict(s.get("attributes", {}))
    if s.get("model"):
        attrs[GEN_AI_MODEL] = s["model"]
    if s.get("input_tokens"):
        attrs[GEN_AI_INPUT_TOKENS] = s["input_tokens"]
    if s.get("output_tokens"):
        attrs[GEN_AI_OUTPUT_TOKENS] = s["output_tokens"]
    if s.get("cost"):
        attrs[AF_COST] = s["cost"]
    return make_span(
        name=s["name"],
        trace_id=trace_id,
        span_id=s["span_id"],
        parent_span_id=s.get("parent_id"),
        start_ns=int(float(s.get("start_time", 0)) * 1e9),
        end_ns=int(float(s.get("end_time", 0)) * 1e9),
        attributes=attrs,
        error=(s.get("status") == "error"),
    )


def save_jsonl(traces: list[Trace], path: str) -> None:
    """Write traces to a JSONL file."""
    with open(path, "w") as f:
        for t in traces:
            row = {
                "trace_id": t.trace_id,
                "outcome": t.outcome,
                "metadata": t.metadata,
                "spans": [_span_to_dict(s) for s in t.spans],
            }
            f.write(json.dumps(row) + "\n")


def _span_to_dict(s) -> dict:
    return {
        "span_id":        format(s.context.span_id, "016x") if s.context else "",
        "parent_span_id": format(s.parent.span_id, "016x") if s.parent else None,
        "name":           s.name,
        "start_ns":       s.start_time,
        "end_ns":         s.end_time,
        "attributes":     dict(s.attributes or {}),
        "error":          s.status.status_code == StatusCode.ERROR,
    }
