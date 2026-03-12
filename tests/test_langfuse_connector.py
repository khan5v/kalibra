"""Tests for the Langfuse connector — all HTTP calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentflow.connectors.langfuse import LangfuseConnector, _parse_ts, parse_since
from agentflow.converters.base import span_cost, span_input_tokens, span_is_error, span_model, span_output_tokens


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
    trace_detail = _make_httpx_response({**RAW_TRACE, "observations": []})

    httpx = MagicMock()
    httpx.get.side_effect = [page1, trace_detail]

    with patch("time.sleep"):
        traces = CONNECTOR.fetch.__wrapped__ if hasattr(CONNECTOR.fetch, "__wrapped__") else None
        # Call fetch directly via _iter_traces + _convert
        items = list(CONNECTOR._iter_traces(httpx, "2026-01-01T00:00:00Z", limit=100))

    assert len(items) == 1
    assert items[0][0]["id"] == "trace-001"


def test_iter_traces_respects_limit():
    items_page = [{"id": f"trace-{i:03d}", "name": "t", "timestamp": "2026-01-01T10:00:00Z"}
                  for i in range(50)]
    page1 = _make_httpx_response({"data": items_page, "meta": {"totalPages": 3}})

    httpx = MagicMock()
    httpx.get.return_value = page1

    result = list(CONNECTOR._iter_traces(httpx, "2026-01-01T00:00:00Z", limit=10))
    assert len(result) == 10


def test_get_retries_on_429():
    rate_limited = _make_httpx_response({}, status_code=429)
    rate_limited.headers = {"Retry-After": "0"}
    rate_limited.raise_for_status = MagicMock(side_effect=Exception("429"))

    success = _make_httpx_response({"data": [], "meta": {"totalPages": 1}})

    httpx = MagicMock()
    httpx.get.side_effect = [rate_limited, success]

    with patch("time.sleep"):
        result = CONNECTOR._get(httpx, f"{HOST}/api/public/traces", {})

    assert result == {"data": [], "meta": {"totalPages": 1}}
    assert httpx.get.call_count == 2


def test_get_raises_after_five_429s():
    rate_limited = _make_httpx_response({}, status_code=429)
    rate_limited.headers = {}
    rate_limited.raise_for_status = MagicMock()

    httpx = MagicMock()
    httpx.get.return_value = rate_limited

    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            CONNECTOR._get(httpx, f"{HOST}/api/public/traces", {})


def test_fetch_trace_observations_uses_detail_endpoint():
    detail_response = _make_httpx_response({
        **RAW_TRACE,
        "observations": [OBSERVATION],
    })
    httpx = MagicMock()
    httpx.get.return_value = detail_response

    obs = CONNECTOR._fetch_trace_observations(httpx, "trace-001")

    assert len(obs) == 1
    assert obs[0]["id"] == "obs-001"
    # Verify it called the detail endpoint, NOT the /observations?traceId= endpoint
    called_url = httpx.get.call_args[0][0]
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

    with patch("httpx.get", side_effect=[traces_page, detail_success, detail_failure]), \
         patch("time.sleep"):
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
