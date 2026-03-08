"""LangSmith connector — pull traces via langsmith SDK."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone

from agentflow.converters.base import (
    AF_COST, GEN_AI_INPUT_TOKENS, GEN_AI_MODEL, GEN_AI_OUTPUT_TOKENS,
    Trace, make_span,
)


class LangSmithConnector:
    """Pull runs from LangSmith and convert to agentflow Trace objects.

    Env vars:
        LANGSMITH_API_KEY
        LANGSMITH_API_URL   default: https://api.smith.langchain.com
    """

    def __init__(self, api_key: str, api_url: str | None = None):
        self.api_key = api_key
        self.api_url = api_url or "https://api.smith.langchain.com"

    def fetch(
        self,
        project_name: str,
        since: datetime | None = None,
        limit: int = 5000,
        progress: bool = True,
    ) -> list[Trace]:
        """Fetch root runs from LangSmith and convert to agentflow Traces."""
        try:
            from langsmith import Client
        except ImportError:
            raise ImportError(
                "langsmith package is required: pip install 'agentflow[langsmith]'"
            )

        client = Client(api_key=self.api_key, api_url=self.api_url)
        since = since or (datetime.now(timezone.utc) - timedelta(days=7))

        traces = []
        fetched = 0

        # List root runs only (no parent) — each is a trace
        runs = client.list_runs(
            project_name=project_name,
            start_time=since,
            is_root=True,
            limit=limit,
        )

        for run in runs:
            child_runs = list(client.list_runs(
                project_name=project_name,
                parent_run_id=str(run.id),
            ))
            trace = self._convert(run, child_runs)
            if trace:
                traces.append(trace)
            fetched += 1
            if progress and fetched % 100 == 0:
                print(f"  Fetched {fetched} traces from LangSmith...")
            if fetched >= limit:
                break

        if progress:
            print(f"  Done. {len(traces)} traces fetched from LangSmith.")
        return traces

    def _convert(self, run, child_runs: list) -> Trace | None:
        """Convert a LangSmith root run + children → agentflow Trace."""
        trace_id = str(run.id)

        # Outcome from run error / feedback
        outcome = None
        if run.error:
            outcome = "failure"
        elif hasattr(run, "feedback_stats") and run.feedback_stats:
            # LangSmith stores feedback like {"score": {"avg": 1.0}}
            score = run.feedback_stats.get("score", {}).get("avg")
            if score is not None:
                outcome = "success" if score >= 0.5 else "failure"

        # Build spans from child runs
        spans = [self._run_to_span(trace_id, run, is_root=True)]
        for child in child_runs:
            span = self._run_to_span(trace_id, child, is_root=False)
            if span:
                spans.append(span)

        spans = [s for s in spans if s is not None]
        spans.sort(key=lambda s: s.start_time)

        return Trace(
            trace_id=trace_id,
            spans=spans,
            outcome=outcome,
            metadata={
                "source": "langsmith",
                "name": run.name or "",
                "project": getattr(run, "session_name", "") or "",
                "tags": list(run.tags or []),
            },
        )

    def _run_to_span(self, trace_id: str, run, is_root: bool = False):
        run_id = str(run.id)
        name = run.name or run.run_type or "unknown"

        t0 = _to_ts(run.start_time)
        t1 = _to_ts(run.end_time) if run.end_time else t0

        usage = getattr(run, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0

        model = None
        extra = getattr(run, "extra", None) or {}
        invocation_params = extra.get("invocation_params", {})
        if invocation_params:
            model = invocation_params.get("model_name") or invocation_params.get("model")

        # Parent: compute span ID the same way (md5 of trace_id:parent_run_id)
        parent_run_id = str(run.parent_run_id) if run.parent_run_id else None
        parent_span_id = (
            hashlib.md5(f"{trace_id}:{parent_run_id}".encode()).hexdigest()[:16]
            if parent_run_id else None
        )

        attrs: dict = {"run_type": run.run_type or ""}
        if model:
            attrs[GEN_AI_MODEL] = model
        attrs[GEN_AI_INPUT_TOKENS]  = input_tokens
        attrs[GEN_AI_OUTPUT_TOKENS] = output_tokens
        attrs[AF_COST]              = 0.0

        return make_span(
            name=name,
            trace_id=trace_id,
            span_id=hashlib.md5(f"{trace_id}:{run_id}".encode()).hexdigest()[:16],
            parent_span_id=parent_span_id,
            start_ns=int(t0 * 1e9),
            end_ns=int(t1 * 1e9),
            attributes=attrs,
            error=bool(run.error),
        )


def _to_ts(dt) -> float:
    """Convert datetime or None to unix timestamp."""
    if dt is None:
        return time.time()
    if isinstance(dt, (int, float)):
        return float(dt)
    if hasattr(dt, "timestamp"):
        return dt.timestamp()
    return time.time()
