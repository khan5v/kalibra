"""LangSmith connector — pull traces via langsmith SDK."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone

from langsmith import Client
from langsmith.utils import LangSmithError, LangSmithRateLimitError

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

    MAX_RETRIES = 5

    def __init__(self, api_key: str, api_url: str | None = None):
        self.api_key = api_key
        self.api_url = api_url or "https://api.smith.langchain.com"

    def fetch(
        self,
        project_name: str,
        since: datetime | None = None,
        limit: int = 5000,
        progress: bool = True,
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[Trace]:
        """Fetch root runs from LangSmith and convert to agentflow Traces."""
        client = self._make_client()
        since = since or (datetime.now(timezone.utc) - timedelta(days=7))

        traces: list[Trace] = []
        fetched = 0

        # Build filter string for tag/session filtering
        filter_str = self._build_filter(tags=tags, session_id=session_id)

        # List root runs only (no parent) — each is a trace
        runs = self._retry(
            lambda: client.list_runs(
                project_name=project_name,
                start_time=since,
                is_root=True,
                limit=limit,
                filter=filter_str or None,
            ),
            "list root runs",
        )

        if progress:
            print(f"  Fetching up to {limit:,} traces from LangSmith...")

        for run in runs:
            child_runs = list(self._retry(
                lambda r=run: client.list_runs(
                    project_name=project_name,
                    parent_run_id=str(r.id),
                ),
                f"list children of {run.id}",
            ))
            trace = self._convert(run, child_runs)
            if trace:
                traces.append(trace)
            fetched += 1
            if progress and (fetched % 5 == 0):
                print(f"  Fetched {fetched} traces...")
            if fetched >= limit:
                break

        if progress:
            print(f"  Done. {len(traces)} traces fetched from LangSmith.")
        return traces

    def _make_client(self) -> Client:
        return Client(api_key=self.api_key, api_url=self.api_url)

    def _retry(self, fn, description: str = "request"):
        """Call *fn* with exponential backoff on rate limits and server errors."""
        delay = 1.0
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn()
            except LangSmithRateLimitError:
                if attempt == self.MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"LangSmith rate limit exceeded after {self.MAX_RETRIES} retries ({description})"
                    )
                print(f"  Rate limited ({description}) — waiting {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except LangSmithError as exc:
                if attempt == self.MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"LangSmith request failed after {self.MAX_RETRIES} retries ({description}): {exc}"
                    )
                print(f"  LangSmith error ({description}, attempt {attempt + 1}/{self.MAX_RETRIES}) — retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except (ConnectionError, TimeoutError, OSError) as exc:
                if attempt == self.MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"LangSmith connection failed after {self.MAX_RETRIES} retries ({description}): {exc}"
                    )
                print(f"  Connection error ({description}, attempt {attempt + 1}/{self.MAX_RETRIES}) — retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    @staticmethod
    def _build_filter(
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Build a LangSmith filter string for list_runs.

        LangSmith filter syntax:
          has(tags, "foo")          — run has tag "foo"
          eq(session_id, "bar")    — run belongs to session "bar"
        Multiple clauses are ANDed with 'and'.
        """
        clauses: list[str] = []
        if tags:
            for tag in tags:
                clauses.append(f'has(tags, "{tag}")')
        if session_id:
            clauses.append(f'eq(session_id, "{session_id}")')
        if len(clauses) <= 1:
            return clauses[0] if clauses else ""
        return f"and({', '.join(clauses)})"

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
        else:
            # Heuristic: check output for outcome keywords
            output = run.outputs or {}
            if isinstance(output, dict):
                out_str = str(output).lower()
            else:
                out_str = str(output).lower()
            if "success" in out_str:
                outcome = "success"
            elif any(kw in out_str for kw in ("failure", "error", "failed", "exception")):
                outcome = "failure"

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
                "session_id": str(getattr(run, "session_id", "") or ""),
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
        # Fallback: check serialized kwargs
        if not model:
            serialized = extra.get("metadata", {})
            if isinstance(serialized, dict):
                model = serialized.get("ls_model_name")

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
        # LangSmith doesn't expose cost natively — read from extra.metadata.agentflow_cost
        metadata = extra.get("metadata", {}) if isinstance(extra, dict) else {}
        attrs[AF_COST] = float(metadata.get("agentflow_cost", 0.0))

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
