"""LangSmith connector — pull traces via langsmith SDK."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone

from agentflow.converters.base import Span, Trace


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

    def _run_to_span(self, trace_id: str, run, is_root: bool = False) -> Span | None:
        run_id = str(run.id)
        name = run.name or run.run_type or "unknown"

        t0 = _to_ts(run.start_time)
        t1 = _to_ts(run.end_time) if run.end_time else t0

        # Tokens from usage_metadata
        usage = getattr(run, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0

        # Cost — LangSmith doesn't always expose this
        cost = 0.0

        model = None
        extra = getattr(run, "extra", None) or {}
        invocation_params = extra.get("invocation_params", {})
        if invocation_params:
            model = invocation_params.get("model_name") or invocation_params.get("model")

        status = "error" if run.error else "ok"
        parent_id = str(run.parent_run_id) if run.parent_run_id else None

        return Span(
            span_id=hashlib.md5(f"{trace_id}:{run_id}".encode()).hexdigest()[:16],
            parent_id=parent_id,
            name=name,
            start_time=t0,
            end_time=t1,
            attributes={"run_type": run.run_type or ""},
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            status=status,
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
