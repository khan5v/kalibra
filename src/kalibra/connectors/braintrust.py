"""Braintrust connector — pull traces via BTQL REST API."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone

import httpx

from kalibra.converters.base import (
    AF_COST, GEN_AI_INPUT_TOKENS, GEN_AI_MODEL, GEN_AI_OUTPUT_TOKENS,
    Trace, make_span,
)


class BraintrustConnector:
    """Pull traces from Braintrust via the BTQL query API.

    Braintrust stores data in two tables:
      - **project_logs**: production traces from instrumented apps
      - **experiments**: eval runs from ``braintrust.init()`` + ``start_span()``

    The connector queries both by default and merges the results.

    Env vars:
        BRAINTRUST_API_KEY
        BRAINTRUST_API_URL   default: https://api.braintrust.dev
    """

    MAX_RETRIES = 5

    def __init__(self, api_key: str, api_url: str = "https://api.braintrust.dev"):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")

    def fetch(
        self,
        project_name: str,
        since: datetime | None = None,
        limit: int = 5000,
        progress: bool = True,
        tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[Trace]:
        """Fetch traces from Braintrust and convert to kalibra Trace objects."""
        since = since or (datetime.now(timezone.utc) - timedelta(days=7))
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Resolve project name → UUID (BTQL requires UUID).
        project_id = self._resolve_project_id(project_name)
        if progress:
            print(f"  Querying Braintrust project {project_name!r} ({project_id[:8]}...)...")

        # Query both experiments and project_logs.
        raw_traces: dict[str, list[dict]] = {}
        for source_fn in ("experiment", "project_logs"):
            source_traces = self._fetch_from_source(
                source_fn, project_id, since_iso, limit, tags,
            )
            for trace_id, spans in source_traces.items():
                raw_traces.setdefault(trace_id, []).extend(spans)

        if progress:
            print(f"  Found {len(raw_traces)} traces.")

        traces = []
        for trace_id, spans in raw_traces.items():
            trace = self._convert(trace_id, spans)
            if trace:
                traces.append(trace)

        if progress:
            print(f"  Done. {len(traces)} traces converted.")
        return traces

    # ── Project resolution ────────────────────────────────────────────────────

    def _resolve_project_id(self, name_or_id: str) -> str:
        """Resolve a project name to its UUID. Passes through UUIDs unchanged."""
        # If it looks like a UUID already, use it directly.
        if len(name_or_id) == 36 and name_or_id.count("-") == 4:
            return name_or_id

        data = self._rest_get(f"/v1/project?project_name={name_or_id}")
        projects = data.get("objects", [])
        if not projects:
            raise RuntimeError(
                f"Braintrust project {name_or_id!r} not found.\n"
                f"  Check the project name in the Braintrust UI."
            )
        return projects[0]["id"]

    # ── Experiment listing ────────────────────────────────────────────────────

    def _list_experiment_ids(self, project_id: str) -> list[str]:
        """List all experiment IDs for a project."""
        data = self._rest_get(f"/v1/experiment?project_id={project_id}")
        return [exp["id"] for exp in data.get("objects", [])]

    # ── BTQL query ────────────────────────────────────────────────────────────

    _BTQL_FIELDS = (
        "id, span_id, root_span_id, span_parents, is_root, "
        "span_attributes, output, error, scores, metadata, "
        "metrics, tags, created"
    )

    def _fetch_from_source(
        self,
        source_fn: str,
        project_id: str,
        since_iso: str,
        limit: int,
        tags: list[str] | None,
    ) -> dict[str, list[dict]]:
        """Fetch spans from one BTQL source (experiment or project_logs)."""
        if source_fn == "experiment":
            exp_ids = self._list_experiment_ids(project_id)
            if not exp_ids:
                return {}
            # Query each experiment and merge.
            all_traces: dict[str, list[dict]] = {}
            for exp_id in exp_ids:
                traces = self._btql_fetch(f"experiment('{exp_id}')", since_iso, limit, tags)
                for trace_id, spans in traces.items():
                    all_traces.setdefault(trace_id, []).extend(spans)
            return all_traces
        else:
            return self._btql_fetch(f"project_logs('{project_id}')", since_iso, limit, tags)

    def _btql_fetch(
        self,
        from_clause: str,
        since_iso: str,
        limit: int,
        tags: list[str] | None,
    ) -> dict[str, list[dict]]:
        """Run a BTQL query and group results by trace ID.

        Tags only exist on root spans in Braintrust, so we use a two-step approach:
        1. Find root_span_ids matching the tag + time filter
        2. Fetch ALL spans (including children) for those traces
        """
        time_filter = f"created > '{since_iso}'"

        if tags:
            # Step 1: find matching trace IDs from root spans.
            tag_filters = " AND ".join(
                f"tags LIKE '%{tag.replace(chr(39), '')}%'" for tag in tags
            )
            id_query = (
                f"SELECT root_span_id "
                f"FROM {from_clause} "
                f"WHERE {time_filter} AND is_root = true AND {tag_filters} "
                f"LIMIT {min(limit, 1000)}"
            )
            id_rows, _ = self._btql_request(id_query)
            trace_ids = [r["root_span_id"] for r in id_rows if r.get("root_span_id")]
            if not trace_ids:
                return {}

            # Step 2: fetch all spans for those traces.
            # Build an IN clause with the trace IDs.
            id_list = ", ".join(f"'{tid}'" for tid in trace_ids)
            query = (
                f"SELECT {self._BTQL_FIELDS} "
                f"FROM {from_clause} "
                f"WHERE root_span_id IN ({id_list}) "
                f"LIMIT 1000"
            )
        else:
            query = (
                f"SELECT {self._BTQL_FIELDS} "
                f"FROM {from_clause} "
                f"WHERE {time_filter} "
                f"LIMIT {min(limit, 1000)}"
            )

        all_rows: list[dict] = []
        cursor: str | None = None

        while True:
            paginated_query = query
            if cursor:
                paginated_query += f" OFFSET '{cursor}'"

            rows, next_cursor = self._btql_request(paginated_query)
            if not rows:
                break

            all_rows.extend(rows)
            if not next_cursor or len(all_rows) >= limit * 10:  # spans >> traces
                break
            cursor = next_cursor

        # Group by trace ID.
        traces: dict[str, list[dict]] = {}
        for row in all_rows:
            trace_id = row.get("root_span_id") or row.get("span_id", "")
            traces.setdefault(trace_id, []).append(row)
        return traces

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _rest_get(self, path: str) -> dict:
        """GET a REST endpoint with retry."""
        return self._raw_request("GET", path).json()

    def _raw_request(
        self, method: str, path: str,
        json_body: dict | None = None,
        timeout: int = 30,
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff. Returns the response."""
        delay = 1.0
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = httpx.request(
                    method,
                    f"{self.api_url}{path}",
                    headers=self._auth_headers(),
                    json=json_body,
                    timeout=timeout,
                )
            except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
                if attempt == self.MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Braintrust request failed after {self.MAX_RETRIES} retries: {exc}"
                    ) from exc
                print(f"  Connection error (attempt {attempt + 1}/{self.MAX_RETRIES}) "
                      f"— retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code == 429:
                wait = max(float(resp.headers.get("Retry-After", delay)), delay)
                print(f"  Rate limited — waiting {wait:.0f}s...")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code >= 500:
                if attempt == self.MAX_RETRIES - 1:
                    resp.raise_for_status()
                print(f"  Server error {resp.status_code} (attempt {attempt + 1}/{self.MAX_RETRIES}) "
                      f"— retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"Braintrust authentication failed (HTTP {resp.status_code}).\n"
                    "  Check that BRAINTRUST_API_KEY is valid.\n"
                    "  Create one at: Settings → API Keys in the Braintrust UI."
                )

            if resp.status_code == 400:
                body = resp.text[:500]
                raise RuntimeError(f"Braintrust request error (HTTP 400): {body}")

            resp.raise_for_status()
            return resp

        raise RuntimeError(f"Braintrust request failed after {self.MAX_RETRIES} retries")

    def _btql_request(self, query: str) -> tuple[list[dict], str | None]:
        """Execute a BTQL query. Returns (rows, next_cursor)."""
        resp = self._raw_request(
            "POST", "/btql",
            json_body={"query": query, "fmt": "json"},
            timeout=60,
        )
        next_cursor = (
            resp.headers.get("x-bt-cursor")
            or resp.headers.get("x-amz-meta-bt_cursor")
        )
        body = resp.json()
        rows = body.get("data", []) if isinstance(body, dict) else body
        return rows, next_cursor

    # ── Conversion ────────────────────────────────────────────────────────────

    def _convert(self, trace_id: str, raw_spans: list[dict]) -> Trace | None:
        """Convert a group of Braintrust spans into a kalibra Trace."""
        if not raw_spans:
            return None

        spans = []
        root_span: dict | None = None

        for raw in raw_spans:
            span = self._span_to_otel(trace_id, raw)
            if span:
                spans.append(span)
            if raw.get("is_root"):
                root_span = raw

        if not spans:
            return None

        spans.sort(key=lambda s: s.start_time)

        if root_span is None:
            root_span = raw_spans[0]

        outcome = self._detect_outcome(root_span)

        # Build trace metadata from root span.
        trace_meta: dict = {"source": "braintrust"}
        root_meta = root_span.get("metadata") or {}
        if isinstance(root_meta, dict):
            for k, v in root_meta.items():
                if v is not None:
                    trace_meta[f"braintrust.{k}"] = v
        root_tags = root_span.get("tags")
        if root_tags:
            trace_meta["tags"] = root_tags

        span_attrs = root_span.get("span_attributes") or {}
        name = span_attrs.get("name", "")
        if name:
            trace_meta["name"] = name

        return Trace(
            trace_id=trace_id,
            spans=spans,
            outcome=outcome,
            metadata=trace_meta,
        )

    def _span_to_otel(self, trace_id: str, raw: dict) -> "ReadableSpan | None":
        """Convert a single Braintrust span to an OTel ReadableSpan."""
        span_id = raw.get("span_id", raw.get("id", ""))
        if not span_id:
            return None

        span_attrs = raw.get("span_attributes") or {}
        name = span_attrs.get("name") or span_attrs.get("type") or "unknown"

        metrics = raw.get("metrics") or {}
        start_ts = metrics.get("start")
        end_ts = metrics.get("end")

        # Fall back to created timestamp if metrics.start/end missing.
        if start_ts is None:
            created = raw.get("created", "")
            start_ts = _parse_iso(created) if created else time.time()
        if end_ts is None:
            end_ts = start_ts

        start_ns = int(float(start_ts) * 1e9)
        end_ns = int(float(end_ts) * 1e9)

        # Parent span: first entry in span_parents array (null for root spans).
        parents = raw.get("span_parents") or []
        parent_span_id = parents[0] if parents else None

        # Attributes.
        input_tokens = int(metrics.get("prompt_tokens") or 0)
        output_tokens = int(metrics.get("completion_tokens") or 0)
        cost = float(metrics.get("estimated_cost") or 0.0)

        metadata = raw.get("metadata") or {}
        model = metadata.get("model") or None

        attrs: dict = {
            GEN_AI_INPUT_TOKENS: input_tokens,
            GEN_AI_OUTPUT_TOKENS: output_tokens,
            AF_COST: cost,
            "braintrust.span_type": span_attrs.get("type", ""),
        }
        if model:
            attrs[GEN_AI_MODEL] = model

        # Forward scores as attributes.
        scores = raw.get("scores") or {}
        for score_name, score_val in scores.items():
            if score_val is not None:
                attrs[f"braintrust.score.{score_name}"] = float(score_val)

        is_error = raw.get("error") is not None

        return make_span(
            name=name,
            trace_id=trace_id,
            span_id=_stable_id(trace_id, span_id),
            parent_span_id=_stable_id(trace_id, parent_span_id) if parent_span_id else None,
            start_ns=start_ns,
            end_ns=end_ns,
            attributes=attrs,
            error=is_error,
        )

    @staticmethod
    def _detect_outcome(root: dict) -> str | None:
        """Detect success/failure from a root span.

        Priority:
          1. error field present → failure
          2. scores with success/failure keywords → mapped
          3. output field with keyword heuristic → mapped
        """
        if root.get("error") is not None:
            return "failure"

        scores = root.get("scores") or {}
        for score_name, score_val in scores.items():
            if score_val is None:
                continue
            if score_name.lower() in ("correctness", "accuracy", "pass", "success"):
                return "success" if float(score_val) >= 0.5 else "failure"

        output = root.get("output")
        if output is not None:
            out_str = str(output).lower()
            if "success" in out_str:
                return "success"
            if any(kw in out_str for kw in ("failure", "error", "failed", "exception")):
                return "failure"

        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stable_id(trace_id: str, span_id: str) -> str:
    """Create a deterministic hex ID from trace + span IDs."""
    return hashlib.md5(f"{trace_id}:{span_id}".encode()).hexdigest()[:16]


def _parse_iso(ts: str) -> float:
    """Parse ISO 8601 timestamp to Unix float."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts[:26].rstrip("Z"), fmt.rstrip("Z")).replace(
                tzinfo=timezone.utc
            ).timestamp()
        except ValueError:
            continue
    return time.time()
