"""Tests for the Braintrust connector — all HTTP calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kalibra.connectors.braintrust import BraintrustConnector, _parse_iso, _stable_id
from kalibra.converters.base import (
    span_cost, span_input_tokens, span_is_error, span_model, span_output_tokens,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

CONNECTOR = BraintrustConnector(api_key="sk-test")

ROOT_SPAN = {
    "id": "evt-001",
    "span_id": "span-root",
    "root_span_id": "span-root",
    "span_parents": [],
    "is_root": True,
    "span_attributes": {"name": "agent-run", "type": "task"},
    "input": {"task": "summarize-invoice"},
    "output": {"outcome": "success"},
    "error": None,
    "scores": {"Correctness": 1.0},
    "metadata": {"model": "claude-sonnet-4-20250514", "user_id": "u-1"},
    "metrics": {
        "start": 1700000000.0,
        "end": 1700000060.0,
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "estimated_cost": 0.003,
    },
    "tags": ["production", "v1.2"],
    "created": "2026-01-01T10:00:00Z",
}

CHILD_SPAN = {
    "id": "evt-002",
    "span_id": "span-child-1",
    "root_span_id": "span-root",
    "span_parents": ["span-root"],
    "is_root": False,
    "span_attributes": {"name": "search_tool", "type": "tool"},
    "input": {"query": "find invoices"},
    "output": {"result": "found 3"},
    "error": None,
    "scores": {},
    "metadata": {},
    "metrics": {
        "start": 1700000010.0,
        "end": 1700000020.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost": 0.0,
    },
    "tags": [],
    "created": "2026-01-01T10:00:10Z",
}

ERROR_SPAN = {
    **CHILD_SPAN,
    "id": "evt-003",
    "span_id": "span-err",
    "span_attributes": {"name": "failing_tool", "type": "tool"},
    "error": "Tool call failed: timeout",
    "metrics": {
        "start": 1700000020.0,
        "end": 1700000025.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost": 0.0,
    },
}


# ── conversion tests ─────────────────────────────────────────────────────────

class TestConvert:
    def test_basic_trace_conversion(self):
        trace = CONNECTOR._convert("span-root", [ROOT_SPAN, CHILD_SPAN])
        assert trace is not None
        assert trace.trace_id == "span-root"
        assert len(trace.spans) == 2
        assert trace.outcome == "success"

    def test_span_attributes(self):
        trace = CONNECTOR._convert("span-root", [ROOT_SPAN])
        span = trace.spans[0]
        assert span.name == "agent-run"
        assert span_input_tokens(span) == 500
        assert span_output_tokens(span) == 200
        assert span_cost(span) == 0.003
        assert span_model(span) == "claude-sonnet-4-20250514"

    def test_child_span_has_parent(self):
        trace = CONNECTOR._convert("span-root", [ROOT_SPAN, CHILD_SPAN])
        child = [s for s in trace.spans if s.name == "search_tool"][0]
        assert child.parent is not None

    def test_error_span(self):
        trace = CONNECTOR._convert("span-root", [ROOT_SPAN, ERROR_SPAN])
        err_span = [s for s in trace.spans if s.name == "failing_tool"][0]
        assert span_is_error(err_span)

    def test_empty_spans_returns_none(self):
        assert CONNECTOR._convert("t1", []) is None

    def test_metadata_forwarded(self):
        trace = CONNECTOR._convert("span-root", [ROOT_SPAN])
        assert trace.metadata["source"] == "braintrust"
        assert trace.metadata["braintrust.user_id"] == "u-1"
        assert trace.metadata["tags"] == ["production", "v1.2"]
        assert trace.metadata["name"] == "agent-run"

    def test_scores_as_attributes(self):
        trace = CONNECTOR._convert("span-root", [ROOT_SPAN])
        span = trace.spans[0]
        assert span.attributes["braintrust.score.Correctness"] == 1.0

    def test_spans_sorted_by_start_time(self):
        trace = CONNECTOR._convert("span-root", [CHILD_SPAN, ROOT_SPAN])
        assert trace.spans[0].start_time < trace.spans[1].start_time

    def test_missing_metrics_uses_created_timestamp(self):
        no_metrics = {**ROOT_SPAN, "metrics": {}, "created": "2026-01-01T12:00:00Z"}
        trace = CONNECTOR._convert("span-root", [no_metrics])
        assert trace is not None
        assert trace.spans[0].start_time > 0


# ── outcome detection ─────────────────────────────────────────────────────────

class TestOutcomeDetection:
    def test_error_field_means_failure(self):
        root = {**ROOT_SPAN, "error": "something broke", "scores": {}}
        trace = CONNECTOR._convert("t1", [root])
        assert trace.outcome == "failure"

    def test_score_based_success(self):
        root = {**ROOT_SPAN, "error": None, "output": {}, "scores": {"Correctness": 0.8}}
        trace = CONNECTOR._convert("t1", [root])
        assert trace.outcome == "success"

    def test_score_based_failure(self):
        root = {**ROOT_SPAN, "error": None, "output": {}, "scores": {"accuracy": 0.2}}
        trace = CONNECTOR._convert("t1", [root])
        assert trace.outcome == "failure"

    def test_output_keyword_success(self):
        root = {**ROOT_SPAN, "error": None, "scores": {}, "output": "task completed successfully"}
        trace = CONNECTOR._convert("t1", [root])
        assert trace.outcome == "success"

    def test_output_keyword_failure(self):
        root = {**ROOT_SPAN, "error": None, "scores": {}, "output": "agent failed to complete"}
        trace = CONNECTOR._convert("t1", [root])
        assert trace.outcome == "failure"

    def test_no_signal_returns_none(self):
        root = {**ROOT_SPAN, "error": None, "scores": {}, "output": {"data": 42}}
        trace = CONNECTOR._convert("t1", [root])
        assert trace.outcome is None


# ── BTQL query building ──────────────────────────────────────────────────────

class TestBtqlFetch:
    def test_groups_spans_by_root_span_id(self):
        """Verify that _btql_fetch groups spans by root_span_id."""
        rows = [
            {"span_id": "s1", "root_span_id": "trace-A", "is_root": True},
            {"span_id": "s2", "root_span_id": "trace-A", "is_root": False},
            {"span_id": "s3", "root_span_id": "trace-B", "is_root": True},
        ]
        with patch.object(CONNECTOR, "_btql_request", return_value=(rows, None)):
            result = CONNECTOR._btql_fetch("experiment('x')", "2026-01-01T00:00:00Z", 100, None)
        assert len(result) == 2
        assert len(result["trace-A"]) == 2
        assert len(result["trace-B"]) == 1


# ── fetch with mock HTTP ─────────────────────────────────────────────────────

class TestFetchEndToEnd:
    @patch.object(BraintrustConnector, "_list_experiment_ids", return_value=["exp-1"])
    @patch.object(BraintrustConnector, "_resolve_project_id", return_value="proj-uuid")
    @patch.object(BraintrustConnector, "_btql_request")
    def test_fetch_returns_traces(self, mock_btql, mock_resolve, mock_list_exp):
        # First call: experiment query returns spans. Second call: project_logs returns empty.
        mock_btql.side_effect = [
            ([ROOT_SPAN, CHILD_SPAN], None),  # experiment
            ([], None),                         # project_logs
        ]
        traces = CONNECTOR.fetch(project_name="test-proj", progress=False)
        assert len(traces) == 1
        assert traces[0].trace_id == "span-root"
        assert len(traces[0].spans) == 2

    @patch.object(BraintrustConnector, "_list_experiment_ids", return_value=[])
    @patch.object(BraintrustConnector, "_resolve_project_id", return_value="proj-uuid")
    @patch.object(BraintrustConnector, "_btql_request")
    def test_fetch_empty_response(self, mock_btql, mock_resolve, mock_list_exp):
        mock_btql.return_value = ([], None)  # project_logs empty
        traces = CONNECTOR.fetch(project_name="test-proj", progress=False)
        assert traces == []

    @patch.object(BraintrustConnector, "_list_experiment_ids", return_value=["exp-1"])
    @patch.object(BraintrustConnector, "_resolve_project_id", return_value="proj-uuid")
    def test_fetch_400_raises_with_message(self, mock_resolve, mock_list_exp):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Invalid BTQL query: unknown column 'foo'"
        mock_resp.raise_for_status = MagicMock()
        with patch("kalibra.connectors.braintrust.httpx.request", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="request error"):
                CONNECTOR.fetch(project_name="test-proj", progress=False)


# ── helpers ───────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_parse_iso_full(self):
        ts = _parse_iso("2026-01-01T12:00:00Z")
        assert abs(ts - 1767268800.0) < 2  # within 2 seconds

    def test_parse_iso_with_millis(self):
        ts = _parse_iso("2026-01-01T12:00:00.123Z")
        assert abs(ts - 1767268800.123) < 2

    def test_parse_iso_garbage_returns_now(self):
        ts = _parse_iso("not-a-date")
        assert ts > 0  # falls back to time.time()

    def test_stable_id_deterministic(self):
        a = _stable_id("trace-1", "span-1")
        b = _stable_id("trace-1", "span-1")
        assert a == b
        assert len(a) == 16

    def test_stable_id_different_inputs(self):
        a = _stable_id("trace-1", "span-1")
        b = _stable_id("trace-1", "span-2")
        assert a != b


# ── connector registry ────────────────────────────────────────────────────────

class TestRegistry:
    def test_get_connector_braintrust(self):
        with patch.dict("os.environ", {"BRAINTRUST_API_KEY": "sk-test"}):
            from kalibra.connectors import get_connector
            conn = get_connector("braintrust")
            assert isinstance(conn, BraintrustConnector)

    def test_get_connector_missing_key(self):
        with patch.dict("os.environ", {"BRAINTRUST_API_KEY": ""}, clear=False):
            from kalibra.connectors import get_connector
            with pytest.raises(RuntimeError, match="Missing Braintrust"):
                get_connector("braintrust")
