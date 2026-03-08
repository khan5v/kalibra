"""Langfuse connector — pull traces via REST API."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Generator

from agentflow.converters.base import Span, Trace


class LangfuseConnector:
    """Pull traces + observations from Langfuse REST API.

    Env vars:
        LANGFUSE_HOST        default: https://cloud.langfuse.com
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_SECRET_KEY
    """

    def __init__(self, host: str, public_key: str, secret_key: str):
        self.host = host.rstrip("/")
        self.auth = (public_key, secret_key)

    def fetch(
        self,
        project_id: str | None = None,
        since: datetime | None = None,
        limit: int = 5000,
        progress: bool = True,
    ) -> list[Trace]:
        """Fetch traces from Langfuse and convert to agentflow Trace objects."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for the Langfuse connector: pip install httpx")

        since = since or (datetime.now(timezone.utc) - timedelta(days=7))
        since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        traces = []
        fetched = 0

        for raw_trace in self._iter_traces(httpx, since_str, limit):
            observations = self._fetch_trace_observations(httpx, raw_trace["id"])
            trace = self._convert(raw_trace, observations)
            if trace:
                traces.append(trace)
            fetched += 1
            if progress and fetched % 100 == 0:
                print(f"  Fetched {fetched} traces from Langfuse...")
            time.sleep(0.05)  # 50ms between observation fetches to stay under rate limit
            if fetched >= limit:
                break

        if progress:
            print(f"  Done. {len(traces)} traces fetched from Langfuse.")
        return traces

    def _get(self, httpx, url: str, params: dict) -> dict:
        """GET with exponential backoff on 429."""
        delay = 1.0
        for attempt in range(5):
            resp = httpx.get(url, auth=self.auth, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", delay))
                wait = max(retry_after, delay)
                print(f"  Rate limited — waiting {wait:.0f}s...")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Langfuse rate limit exceeded after 5 retries")

    def _iter_traces(self, httpx, since_str: str, limit: int) -> Generator[dict, None, None]:
        page = 1
        total_yielded = 0
        while True:
            data = self._get(
                httpx,
                f"{self.host}/api/public/traces",
                {"page": page, "limit": 50, "fromTimestamp": since_str},
            )
            items = data.get("data", [])
            if not items:
                break
            for item in items:
                yield item
                total_yielded += 1
                if total_yielded >= limit:
                    return
            meta = data.get("meta", {})
            if page >= meta.get("totalPages", 1):
                break
            page += 1

    def _fetch_trace_observations(self, httpx, trace_id: str) -> list[dict]:
        """Fetch observations for a trace via the individual trace detail endpoint.

        Uses GET /api/public/traces/{traceId} which returns observations inline —
        avoids the /api/public/observations?traceId= endpoint that returns 400 on
        Langfuse's free tier.
        """
        data = self._get(
            httpx,
            f"{self.host}/api/public/traces/{trace_id}",
            {},
        )
        return data.get("observations", [])

    def _convert(self, raw: dict, observations: list[dict]) -> Trace | None:
        """Convert Langfuse trace + observations → agentflow Trace."""
        trace_id = raw["id"]
        outcome = None
        if raw.get("output"):
            # Heuristic: output field contains outcome keywords
            out = str(raw["output"]).lower()
            if "success" in out:
                outcome = "success"
            elif "failure" in out or "error" in out or "failed" in out or "exception" in out:
                outcome = "failure"

        spans = []
        for obs in observations:
            span = self._obs_to_span(trace_id, obs)
            if span:
                spans.append(span)

        # Sort by start time
        spans.sort(key=lambda s: s.start_time)

        if not spans:
            # Create a single synthetic span from trace metadata
            t0 = _parse_ts(raw.get("timestamp"))
            spans = [Span(
                span_id=hashlib.md5(trace_id.encode()).hexdigest()[:16],
                parent_id=None,
                name=raw.get("name", "unknown"),
                start_time=t0,
                end_time=t0,
                attributes={},
                model=None,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                status="ok",
            )]

        return Trace(
            trace_id=trace_id,
            spans=spans,
            outcome=outcome,
            metadata={
                "source": "langfuse",
                "name": raw.get("name", ""),
                "session_id": raw.get("sessionId", ""),
                "user_id": raw.get("userId", ""),
            },
        )

    def _obs_to_span(self, trace_id: str, obs: dict) -> Span | None:
        obs_id = obs.get("id", "")
        name = obs.get("name") or obs.get("type", "unknown")

        t0 = _parse_ts(obs.get("startTime"))
        t1 = _parse_ts(obs.get("endTime")) or t0

        usage = obs.get("usage") or {}
        input_tokens = usage.get("input", 0) or 0
        output_tokens = usage.get("output", 0) or 0
        cost = obs.get("calculatedTotalCost") or 0.0

        model = obs.get("model") or None

        # Error detection
        status = "ok"
        level = obs.get("level", "DEFAULT")
        if level in ("ERROR", "WARNING"):
            status = "error"
        if obs.get("statusMessage") and "error" in str(obs["statusMessage"]).lower():
            status = "error"

        return Span(
            span_id=hashlib.md5(f"{trace_id}:{obs_id}".encode()).hexdigest()[:16],
            parent_id=obs.get("parentObservationId"),
            name=name,
            start_time=t0,
            end_time=t1,
            attributes={"langfuse_type": obs.get("type", "")},
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            status=status,
        )


def _parse_ts(ts_str: str | None) -> float:
    """Parse ISO timestamp to unix float. Returns current time if missing."""
    if not ts_str:
        return time.time()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str[:26].rstrip("Z"), fmt.rstrip("Z")).replace(
                tzinfo=timezone.utc
            ).timestamp()
        except ValueError:
            continue
    return time.time()


def parse_since(since_str: str) -> datetime:
    """Parse a duration string (7d, 24h, 2h) or ISO date to a datetime."""
    since_str = since_str.strip()
    now = datetime.now(timezone.utc)
    if since_str.endswith("d"):
        return now - timedelta(days=int(since_str[:-1]))
    if since_str.endswith("h"):
        return now - timedelta(hours=int(since_str[:-1]))
    if since_str.endswith("m"):
        return now - timedelta(minutes=int(since_str[:-1]))
    # Try ISO date
    try:
        return datetime.fromisoformat(since_str).replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Cannot parse --since value: {since_str!r}. Use e.g. 7d, 24h, 2026-01-01")
