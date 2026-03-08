"""JSONL trace format — agentflow's portable interchange format.

Written by ``agentflow pull``, readable by ``agentflow compare``.
One JSON object per line: {trace_id, outcome, metadata, spans}.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentflow.converters.base import Span, Trace


def load_json_traces(path: Path) -> list[Trace]:
    """Load traces from a JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            spans = [Span(**s) for s in d.get("spans", [])]
            traces.append(Trace(
                trace_id=d["trace_id"],
                spans=spans,
                outcome=d.get("outcome"),
                metadata=d.get("metadata", {}),
            ))
    return traces


def save_jsonl(traces: list[Trace], path: str) -> None:
    """Write traces to a JSONL file."""
    with open(path, "w") as f:
        for t in traces:
            row = {
                "trace_id": t.trace_id,
                "outcome": t.outcome,
                "metadata": t.metadata,
                "spans": [
                    {
                        "span_id": s.span_id,
                        "parent_id": s.parent_id,
                        "name": s.name,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "attributes": s.attributes,
                        "model": s.model,
                        "input_tokens": s.input_tokens,
                        "output_tokens": s.output_tokens,
                        "cost": s.cost,
                        "status": s.status,
                    }
                    for s in t.spans
                ],
            }
            f.write(json.dumps(row) + "\n")
