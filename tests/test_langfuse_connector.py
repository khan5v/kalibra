"""Tests for the Langfuse connector — all HTTP calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kalibra.connectors.langfuse import LangfuseConnector, _parse_ts, parse_since
from kalibra.converters.base import span_cost, span_input_tokens, span_is_error, span_model, span_output_tokens


# ── fixtures ──────────────────────────────────────────────────────────────────

HOST = "https://langfuse.example.com"
CONNECTOR = LangfuseConnector(HOST, public_key="pk-test", secret_key="sk-test")

RAW_TRACE = {
    "id": "trace-001",
    "name": "solve-task",
    "timestamp": "2026-01-01T10:00:00Z",
    "output": {"outcome": "success"},
    "sessionId": "sess-1",
    "userId": "user-1",
}

RAW_TRACE_FAILURE = {
    "id": "trace-002",
    "name": "solve-task",
    "timestamp": "2026-01-01T10:01:00Z",
    "output": {"outcome": "failure"},
    "sessionId": "sess-1",
    "userId": "user-1",
}

RAW_TRACE_NO_OUTPUT = {
    "id": "trace-003",
    "name": "solve-task",
    "timestamp": "2026-01-01T10:02:00Z",
    "output": None,
    "sessionId": "",
    "userId": "",
}

OBSERVATION = {
    "id": "obs-001",
    "name": "bash_tool",
    "type": "SPAN",
    "startTime": "2026-01-01T10:00:01Z",
    "endTime": "2026-01-01T10:00:05Z",
    "level": "DEFAULT",
    "usage": {"input": 100, "output": 50},
    "calculatedTotalCost": 0.002,
    "model": "gpt-4o",
    "parentObservationId": None,
    "statusMessage": None,
}

OBSERVATION_ERROR = {
    **OBSERVATION,
    "id": "obs-002",
    "level": "ERROR",
    "statusMessage": "Tool call failed",
}

OBSERVATION_RICH = {
    "id": "obs-rich",
    "name": "llm-call",
    "type": "GENERATION",
    "startTime": "2026-01-01T10:00:01Z",
    "endTime": "2026-01-01T10:00:05Z",
    "level": "DEFAULT",
    "usage": {"input": 100, "output": 50, "total": 150, "unit": "TOKENS"},
    "calculatedTotalCost": 0.005,
    "model": "gpt-4o",
    "parentObservationId": None,
    "statusMessage": None,
    "modelParameters": {"temperature": 0.7, "max_tokens": 4096},
    "completionStartTime": "2026-01-01T10:00:01.500Z",
    "version": "v2.1",
    "environment": "staging",
    "metadata": {
        "custom_field": "custom_value",
        "rag_chunk_count": 5,
        "attributes": {
            "gen_ai.agent.name": "my-agent",
            "custom.pipeline.step": "retrieval",
        },
        "resourceAttributes": {
            "service.name": "my-service",
            "service.version": "1.0.0",
        },
    },
    "input": {"prompt": "What is the weather?"},
    "output": {"completion": "It is sunny."},
}


def _make_httpx_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    return resp


# ── outcome detection ──────────────────────────────────────────────────────────

def test_convert_outcome_success():
    trace = CONNECTOR._convert(RAW_TRACE, [])
    assert trace.outcome == "success"


def test_convert_outcome_failure():
    trace = CONNECTOR._convert(RAW_TRACE_FAILURE, [])
    assert trace.outcome == "failure"


def test_convert_outcome_none_when_no_output():
    trace = CONNECTOR._convert(RAW_TRACE_NO_OUTPUT, [])
    assert trace.outcome is None


@pytest.mark.parametrize("output_text,expected", [
    ("task failed with error", "failure"),
    ("exception raised", "failure"),
    ("execution failed", "failure"),
    ("outcome: success", "success"),
    ("unknown result", None),
])
def test_convert_outcome_heuristics(output_text, expected):
    raw = {**RAW_TRACE, "output": output_text}
    trace = CONNECTOR._convert(raw, [])
    assert trace.outcome == expected


# ── span conversion ────────────────────────────────────────────────────────────

def test_obs_to_span_basic():
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION)
    assert span.name == "bash_tool"
    assert span_input_tokens(span) == 100
    assert span_output_tokens(span) == 50
    assert span_cost(span) == 0.002
    assert span_model(span) == "gpt-4o"
    assert not span_is_error(span)


def test_obs_to_span_error_level():
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_ERROR)
    assert span_is_error(span)


def test_obs_to_span_error_in_status_message():
    obs = {**OBSERVATION, "level": "DEFAULT", "statusMessage": "connection error"}
    span = CONNECTOR._obs_to_span("trace-001", obs)
    assert span_is_error(span)


def test_obs_to_span_missing_usage():
    obs = {**OBSERVATION, "usage": None}
    span = CONNECTOR._obs_to_span("trace-001", obs)
    assert span_input_tokens(span) == 0
    assert span_output_tokens(span) == 0


def test_convert_creates_synthetic_span_when_no_observations():
    trace = CONNECTOR._convert(RAW_TRACE, [])
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "solve-task"


def test_convert_uses_observations_when_present():
    trace = CONNECTOR._convert(RAW_TRACE, [OBSERVATION])
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "bash_tool"


def test_convert_sorts_spans_by_start_time():
    obs_late = {**OBSERVATION, "id": "obs-late", "startTime": "2026-01-01T10:00:10Z"}
    obs_early = {**OBSERVATION, "id": "obs-early", "startTime": "2026-01-01T10:00:00Z"}
    trace = CONNECTOR._convert(RAW_TRACE, [obs_late, obs_early])
    assert trace.spans[0].start_time < trace.spans[1].start_time


# ── attribute pass-through ─────────────────────────────────────────────────────

def test_obs_to_span_forwards_model_parameters():
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    attrs = dict(span.attributes)
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["gen_ai.request.max_tokens"] == 4096


def test_obs_to_span_forwards_completion_start_time():
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    assert span.attributes["langfuse.completion_start_time"] == "2026-01-01T10:00:01.500Z"


def test_obs_to_span_forwards_version_and_environment():
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    assert span.attributes["langfuse.version"] == "v2.1"
    assert span.attributes["langfuse.environment"] == "staging"


def test_obs_to_span_forwards_custom_metadata():
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    attrs = dict(span.attributes)
    assert attrs["langfuse.metadata.custom_field"] == "custom_value"
    assert attrs["langfuse.metadata.rag_chunk_count"] == 5


def test_obs_to_span_hoists_otel_attributes_from_metadata():
    """OTel attributes stored in metadata.attributes should be top-level span attrs."""
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    attrs = dict(span.attributes)
    assert attrs["gen_ai.agent.name"] == "my-agent"
    assert attrs["custom.pipeline.step"] == "retrieval"


def test_obs_to_span_hoists_resource_attributes():
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    attrs = dict(span.attributes)
    assert attrs["resource.service.name"] == "my-service"
    assert attrs["resource.service.version"] == "1.0.0"


def test_obs_to_span_forwards_extra_usage_fields():
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    attrs = dict(span.attributes)
    assert attrs["langfuse.usage.total"] == 150
    assert attrs["langfuse.usage.unit"] == "TOKENS"


def test_obs_to_span_forwards_input_output():
    """Input and output dicts should be forwarded as langfuse.* attributes."""
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    attrs = dict(span.attributes)
    # Dicts get JSON-serialized by _coerce_attr_value
    assert "langfuse.input" in attrs
    assert "langfuse.output" in attrs


def test_obs_to_span_does_not_duplicate_known_fields():
    """Fields already mapped (model, tokens, cost) should not appear twice."""
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION_RICH)
    attrs = dict(span.attributes)
    # gen_ai.request.model is set from obs["model"], not duplicated
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    # Should not have langfuse.model or langfuse.usage.input
    assert "langfuse.model" not in attrs
    assert "langfuse.usage.input" not in attrs


def test_convert_forwards_trace_metadata():
    raw = {
        **RAW_TRACE,
        "tags": ["v2", "experiment"],
        "release": "2026-03-01",
        "version": "1.2.3",
        "environment": "production",
        "metadata": {"team": "ml-platform", "experiment_id": "exp-42"},
    }
    trace = CONNECTOR._convert(raw, [])
    assert trace.metadata["tags"] == ["v2", "experiment"]
    assert trace.metadata["release"] == "2026-03-01"
    assert trace.metadata["version"] == "1.2.3"
    assert trace.metadata["environment"] == "production"
    assert trace.metadata["langfuse.team"] == "ml-platform"
    assert trace.metadata["langfuse.experiment_id"] == "exp-42"


def test_obs_minimal_has_no_extra_attrs():
    """Basic observation without extra fields should not gain spurious attributes."""
    span = CONNECTOR._obs_to_span("trace-001", OBSERVATION)
    attrs = dict(span.attributes)
    # Should have the core set and nothing more
    assert attrs["langfuse.type"] == "SPAN"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert "langfuse.metadata" not in str(list(attrs.keys()))


# ── HTTP layer: pagination and rate limiting ───────────────────────────────────

def _mock_get_side_effect(responses: list):
    """Returns a side_effect function that pops from a response list."""
    it = iter(responses)
    def side_effect(url, *, auth, params, timeout):
        return next(it)
    return side_effect


def test_iter_traces_single_page():
    page1 = _make_httpx_response({
        "data": [RAW_TRACE],
        "meta": {"totalPages": 1, "totalItems": 1},
    })

    with patch("kalibra.connectors.langfuse.httpx") as mock_httpx:
        mock_httpx.get.side_effect = [page1]
        items = list(CONNECTOR._iter_traces("2026-01-01T00:00:00Z", limit=100))

    assert len(items) == 1
    assert items[0][0]["id"] == "trace-001"


def test_iter_traces_respects_limit():
    items_page = [{"id": f"trace-{i:03d}", "name": "t", "timestamp": "2026-01-01T10:00:00Z"}
                  for i in range(50)]
    page1 = _make_httpx_response({"data": items_page, "meta": {"totalPages": 3}})

    with patch("kalibra.connectors.langfuse.httpx") as mock_httpx:
        mock_httpx.get.return_value = page1
        result = list(CONNECTOR._iter_traces("2026-01-01T00:00:00Z", limit=10))

    assert len(result) == 10


def test_get_retries_on_429():
    rate_limited = _make_httpx_response({}, status_code=429)
    rate_limited.headers = {"Retry-After": "0"}
    rate_limited.raise_for_status = MagicMock(side_effect=Exception("429"))

    success = _make_httpx_response({"data": [], "meta": {"totalPages": 1}})

    with patch("kalibra.connectors.langfuse.httpx") as mock_httpx, \
         patch("time.sleep"):
        mock_httpx.get.side_effect = [rate_limited, success]
        result = CONNECTOR._get(f"{HOST}/api/public/traces", {})

    assert result == {"data": [], "meta": {"totalPages": 1}}
    assert mock_httpx.get.call_count == 2


def test_get_raises_after_five_429s():
    rate_limited = _make_httpx_response({}, status_code=429)
    rate_limited.headers = {}
    rate_limited.raise_for_status = MagicMock()

    with patch("kalibra.connectors.langfuse.httpx") as mock_httpx, \
         patch("time.sleep"):
        mock_httpx.get.return_value = rate_limited
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            CONNECTOR._get(f"{HOST}/api/public/traces", {})


def test_fetch_trace_observations_uses_detail_endpoint():
    detail_response = _make_httpx_response({
        **RAW_TRACE,
        "observations": [OBSERVATION],
    })

    with patch("kalibra.connectors.langfuse.httpx") as mock_httpx:
        mock_httpx.get.return_value = detail_response
        obs = CONNECTOR._fetch_trace_observations("trace-001")

    assert len(obs) == 1
    assert obs[0]["id"] == "obs-001"
    called_url = mock_httpx.get.call_args[0][0]
    assert "/api/public/traces/trace-001" in called_url
    assert "observations?" not in called_url


# ── full fetch integration (mocked HTTP) ──────────────────────────────────────

def test_fetch_returns_traces_with_correct_outcomes():
    traces_page = _make_httpx_response({
        "data": [RAW_TRACE, RAW_TRACE_FAILURE],
        "meta": {"totalPages": 1},
    })
    detail_success = _make_httpx_response({**RAW_TRACE, "observations": [OBSERVATION]})
    detail_failure = _make_httpx_response({**RAW_TRACE_FAILURE, "observations": []})

    with patch("kalibra.connectors.langfuse.httpx") as mock_httpx, \
         patch("time.sleep"):
        mock_httpx.get.side_effect = [traces_page, detail_success, detail_failure]
        traces = CONNECTOR.fetch(since=None, limit=10, progress=False)

    assert len(traces) == 2
    outcomes = {t.trace_id: t.outcome for t in traces}
    assert outcomes["trace-001"] == "success"
    assert outcomes["trace-002"] == "failure"


# ── parse_since ────────────────────────────────────────────────────────────────

def test_parse_since_days():
    result = parse_since("7d")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    delta = now - result
    assert 6.9 < delta.total_seconds() / 86400 < 7.1


def test_parse_since_hours():
    result = parse_since("24h")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    delta = now - result
    assert 23.9 < delta.total_seconds() / 3600 < 24.1


def test_parse_since_minutes():
    result = parse_since("30m")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    delta = now - result
    assert 29 < delta.total_seconds() / 60 < 31


def test_parse_since_iso_date():
    result = parse_since("2026-01-01")
    assert result.year == 2026
    assert result.month == 1
    assert result.day == 1


def test_parse_since_invalid():
    with pytest.raises(ValueError):
        parse_since("not-a-valid-date")


# ── _parse_ts ──────────────────────────────────────────────────────────────────

def test_parse_ts_with_microseconds():
    ts = _parse_ts("2026-01-01T10:00:00.123456Z")
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.hour == 10
    assert dt.microsecond == 123456


def test_parse_ts_without_microseconds():
    ts = _parse_ts("2026-01-01T10:00:00Z")
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.hour == 10


def test_parse_ts_none_returns_float():
    with patch("time.time", return_value=1234567890.0):
        ts = _parse_ts(None)
    assert ts == 1234567890.0
