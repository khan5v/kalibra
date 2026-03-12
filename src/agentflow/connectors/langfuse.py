"""Langfuse connector — pull traces via REST API."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Generator

import httpx

from agentflow.converters.base import (
    AF_COST, GEN_AI_INPUT_TOKENS, GEN_AI_MODEL, GEN_AI_OUTPUT_TOKENS,
    Trace, make_span,
)


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
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[Trace]:
        """Fetch traces from Langfuse and convert to agentflow Trace objects."""
        since = since or (datetime.now(timezone.utc) - timedelta(days=7))
        since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        traces = []
        fetched = 0
        total_available: int | None = None

        for raw_trace, total in self._iter_traces(since_str, limit, tags=tags, session_id=session_id):
            if total_available is None:
                total_available = total
                effective_total = min(total, limit)
                if progress:
                    print(f"  Found {total:,} traces in Langfuse (fetching up to {effective_total:,})...")
            observations = self._fetch_trace_observations(raw_trace["id"])
            trace = self._convert(raw_trace, observations)
            if trace:
                traces.append(trace)
            fetched += 1
            if progress and (fetched % 5 == 0 or fetched == effective_total):
                print(f"  Fetched {fetched}/{effective_total} traces...")
            time.sleep(0.05)  # 50ms between observation fetches to stay under rate limit
            if fetched >= limit:
                break

        if progress:
            print(f"  Done. {len(traces)} traces fetched from Langfuse.")
        return traces

    def _get(self, url: str, params: dict) -> dict:
        """GET with exponential backoff on 429, 5xx, and connection errors."""
        delay = 1.0
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = httpx.get(url, auth=self.auth, params=params, timeout=30)
            except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Langfuse request failed after {max_retries} retries: {exc}")
                print(f"  Connection error (attempt {attempt + 1}/{max_retries}) — retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", delay))
                wait = max(retry_after, delay)
                print(f"  Rate limited — waiting {wait:.0f}s...")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code >= 500:
                if attempt == max_retries - 1:
                    resp.raise_for_status()
                print(f"  Server error {resp.status_code} (attempt {attempt + 1}/{max_retries}) — retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Langfuse rate limit exceeded after {max_retries} retries")

    def _iter_traces(
        self,
        since_str: str,
        limit: int,
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> Generator[tuple[dict, int], None, None]:
        """Yield (trace_dict, total_items) tuples."""
        page = 1
        total_yielded = 0
        while True:
            params: dict = {"page": page, "limit": 50, "fromTimestamp": since_str}
            if tags:
                params["tags"] = tags
            if session_id:
                params["sessionId"] = session_id
            data = self._get(
                f"{self.host}/api/public/traces",
                params,
            )
            items = data.get("data", [])
            meta = data.get("meta", {})
            total_items = meta.get("totalItems", 0)
            if not items:
                break
            for item in items:
                yield item, total_items
                total_yielded += 1
                if total_yielded >= limit:
                    return
            if page >= meta.get("totalPages", 1):
                break
            page += 1

    def _fetch_trace_observations(self, trace_id: str) -> list[dict]:
        """Fetch observations for a trace via the individual trace detail endpoint.

        Uses GET /api/public/traces/{traceId} which returns observations inline —
        avoids the /api/public/observations?traceId= endpoint that returns 400 on
        Langfuse's free tier.
        """
        data = self._get(
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
            t0_ns = int(_parse_ts(raw.get("timestamp")) * 1e9)
            spans = [make_span(
                name=raw.get("name", "unknown"),
                trace_id=trace_id,
                span_id=hashlib.md5(trace_id.encode()).hexdigest()[:16],
                parent_span_id=None,
                start_ns=t0_ns,
                end_ns=t0_ns,
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

    def _obs_to_span(self, trace_id: str, obs: dict):
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
        is_error = False
        level = obs.get("level", "DEFAULT")
        if level in ("ERROR", "WARNING"):
            is_error = True
        if obs.get("statusMessage") and "error" in str(obs["statusMessage"]).lower():
            is_error = True

        # Parent span ID: compute the same way as span_id (md5 of trace_id:obs_id)
        parent_obs_id = obs.get("parentObservationId")
        parent_span_id = (
            hashlib.md5(f"{trace_id}:{parent_obs_id}".encode()).hexdigest()[:16]
            if parent_obs_id else None
        )

        attrs: dict = {"langfuse_type": obs.get("type", "")}
        if model:
            attrs[GEN_AI_MODEL] = model
        attrs[GEN_AI_INPUT_TOKENS]  = input_tokens
        attrs[GEN_AI_OUTPUT_TOKENS] = output_tokens
        attrs[AF_COST]              = cost

        return make_span(
            name=name,
            trace_id=trace_id,
            span_id=hashlib.md5(f"{trace_id}:{obs_id}".encode()).hexdigest()[:16],
            parent_span_id=parent_span_id,
            start_ns=int(t0 * 1e9),
            end_ns=int(t1 * 1e9),
            attributes=attrs,
            error=is_error,
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
