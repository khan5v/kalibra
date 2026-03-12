"""Tests for the LangSmith connector — all SDK calls are mocked."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentflow.connectors.langsmith import LangSmithConnector, _to_ts
from agentflow.converters.base import span_cost, span_input_tokens, span_is_error, span_model, span_output_tokens


# ── fixtures ──────────────────────────────────────────────────────────────────

CONNECTOR = LangSmithConnector(api_key="test-key", api_url="https://api.test.com")

_NOW = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
_LATER = datetime(2026, 1, 15, 10, 0, 30, tzinfo=timezone.utc)


def _make_run(
    run_id: str = "run-001",
    name: str = "agent-run",
    run_type: str = "chain",
    start_time: datetime = _NOW,
    end_time: datetime = _LATER,
    error: str | None = None,
    parent_run_id: str | None = None,
    tags: list | None = None,
    feedback_stats: dict | None = None,
    outputs: dict | None = None,
    usage_metadata: dict | None = None,
    extra: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=run_id,
        name=name,
        run_type=run_type,
        start_time=start_time,
        end_time=end_time,
        error=error,
        parent_run_id=parent_run_id,
        tags=tags or [],
        feedback_stats=feedback_stats,
        outputs=outputs or {},
        usage_metadata=usage_metadata or {},
        extra=extra or {},
        session_name="test-project",
        session_id="session-1",
    )


ROOT_RUN = _make_run(
    run_id="root-001",
    outputs={"outcome": "success"},
)

ROOT_RUN_FAILURE = _make_run(
    run_id="root-002",
    error="Agent crashed",
    outputs={},
)

CHILD_RUN_LLM = _make_run(
    run_id="child-001",
    name="plan",
    run_type="llm",
    parent_run_id="root-001",
    usage_metadata={"input_tokens": 500, "output_tokens": 200},
    extra={"invocation_params": {"model_name": "gpt-4o"}},
)

CHILD_RUN_TOOL = _make_run(
    run_id="child-002",
    name="run_tests",
    run_type="tool",
    parent_run_id="root-001",
    usage_metadata={},
    extra={},
)

CHILD_RUN_ERROR = _make_run(
    run_id="child-003",
    name="edit_file",
    run_type="llm",
    parent_run_id="root-001",
    error="Tool call failed",
    usage_metadata={"input_tokens": 100, "output_tokens": 50},
    extra={"invocation_params": {"model_name": "claude-sonnet-4-20250514"}},
)


# ── outcome detection ──────────────────────────────────────────────────────────

def test_convert_outcome_success_from_output():
    trace = CONNECTOR._convert(ROOT_RUN, [])
    assert trace.outcome == "success"


def test_convert_outcome_failure_from_error():
    trace = CONNECTOR._convert(ROOT_RUN_FAILURE, [])
    assert trace.outcome == "failure"


def test_convert_outcome_from_feedback_stats():
    run = _make_run(
        run_id="fb-001",
        feedback_stats={"score": {"avg": 0.8}},
        outputs={},
    )
    trace = CONNECTOR._convert(run, [])
    assert trace.outcome == "success"


def test_convert_outcome_failure_from_feedback_stats():
    run = _make_run(
        run_id="fb-002",
        feedback_stats={"score": {"avg": 0.2}},
        outputs={},
    )
    trace = CONNECTOR._convert(run, [])
    assert trace.outcome == "failure"


def test_convert_outcome_none_when_no_signals():
    run = _make_run(run_id="no-signal", outputs={}, feedback_stats=None, error=None)
    trace = CONNECTOR._convert(run, [])
    assert trace.outcome is None


@pytest.mark.parametrize("output_val,expected", [
    ({"outcome": "failure"}, "failure"),
    ({"result": "exception thrown"}, "failure"),
    ({"outcome": "success"}, "success"),
    ({"info": "something neutral"}, None),
])
def test_convert_outcome_heuristics(output_val, expected):
    run = _make_run(run_id="heur", outputs=output_val)
    trace = CONNECTOR._convert(run, [])
    assert trace.outcome == expected


# ── span conversion ────────────────────────────────────────────────────────────

def test_run_to_span_llm():
    span = CONNECTOR._run_to_span("root-001", CHILD_RUN_LLM)
    assert span.name == "plan"
    assert span_input_tokens(span) == 500
    assert span_output_tokens(span) == 200
    assert span_model(span) == "gpt-4o"
    assert span_cost(span) == 0.0  # no agentflow_cost in metadata
    assert not span_is_error(span)


def test_run_to_span_tool():
    span = CONNECTOR._run_to_span("root-001", CHILD_RUN_TOOL)
    assert span.name == "run_tests"
    assert span_input_tokens(span) == 0
    assert span_output_tokens(span) == 0
    assert span_model(span) is None


def test_run_to_span_error():
    span = CONNECTOR._run_to_span("root-001", CHILD_RUN_ERROR)
    assert span_is_error(span)
    assert span_model(span) == "claude-sonnet-4-20250514"


def test_run_to_span_model_from_ls_model_name():
    """Fallback model detection from extra.metadata.ls_model_name."""
    run = _make_run(
        run_id="model-fb",
        run_type="llm",
        extra={"metadata": {"ls_model_name": "claude-haiku-4-5-20251001"}},
    )
    span = CONNECTOR._run_to_span("root-001", run)
    assert span_model(span) == "claude-haiku-4-5-20251001"


def test_run_to_span_cost_from_metadata():
    """Cost is read from extra.metadata.agentflow_cost when present."""
    run = _make_run(
        run_id="cost-run",
        run_type="llm",
        extra={"metadata": {"agentflow_cost": 0.0042}},
    )
    span = CONNECTOR._run_to_span("root-001", run)
    assert span_cost(span) == pytest.approx(0.0042)


def test_run_to_span_parent_id():
    span = CONNECTOR._run_to_span("root-001", CHILD_RUN_LLM)
    assert span.parent is not None  # has a parent context


def test_run_to_span_root_no_parent():
    span = CONNECTOR._run_to_span("root-001", ROOT_RUN, is_root=True)
    assert span.parent is None


# ── full convert ──────────────────────────────────────────────────────────────

def test_convert_builds_all_spans():
    trace = CONNECTOR._convert(ROOT_RUN, [CHILD_RUN_LLM, CHILD_RUN_TOOL])
    # 1 root + 2 children
    assert len(trace.spans) == 3


def test_convert_sorts_by_start_time():
    late = _make_run(run_id="late", start_time=_LATER, end_time=_LATER, parent_run_id="root-001")
    early = _make_run(run_id="early", start_time=_NOW, end_time=_NOW, parent_run_id="root-001")
    trace = CONNECTOR._convert(ROOT_RUN, [late, early])
    times = [s.start_time for s in trace.spans]
    assert times == sorted(times)


def test_convert_metadata_includes_tags():
    run = _make_run(run_id="tagged", tags=["agentflow", "baseline"], outputs={"outcome": "success"})
    trace = CONNECTOR._convert(run, [])
    assert trace.metadata["tags"] == ["agentflow", "baseline"]
    assert trace.metadata["source"] == "langsmith"


# ── filter building ──────────────────────────────────────────────────────────

def test_build_filter_empty():
    assert CONNECTOR._build_filter() == ""


def test_build_filter_tags_only():
    f = CONNECTOR._build_filter(tags=["agentflow", "baseline"])
    assert 'has(tags, "agentflow")' in f
    assert 'has(tags, "baseline")' in f
    assert f.startswith("and(")


def test_build_filter_session_only():
    f = CONNECTOR._build_filter(session_id="sess-42")
    assert f == 'eq(session_id, "sess-42")'


def test_build_filter_both():
    f = CONNECTOR._build_filter(tags=["v2"], session_id="sess-1")
    assert 'has(tags, "v2")' in f
    assert 'eq(session_id, "sess-1")' in f
    assert f.startswith("and(")


# ── retry logic ──────────────────────────────────────────────────────────────

def test_retry_succeeds_first_try():
    fn = MagicMock(return_value="ok")
    result = CONNECTOR._retry(fn, "test")
    assert result == "ok"
    assert fn.call_count == 1


def test_retry_succeeds_after_failures():
    fn = MagicMock(side_effect=[ConnectionError("fail"), ConnectionError("fail"), "ok"])
    with patch("time.sleep"):
        result = CONNECTOR._retry(fn, "test")
    assert result == "ok"
    assert fn.call_count == 3


def test_retry_raises_after_max():
    fn = MagicMock(side_effect=ConnectionError("fail"))
    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="connection failed after 5 retries"):
            CONNECTOR._retry(fn, "test")
    assert fn.call_count == 5


# ── fetch integration (mocked SDK) ──────────────────────────────────────────

def test_fetch_converts_runs_to_traces():
    mock_client = MagicMock()
    mock_client.list_runs.side_effect = [
        # First call: root runs
        iter([ROOT_RUN]),
        # Second call: children of root-001
        iter([CHILD_RUN_LLM, CHILD_RUN_TOOL]),
    ]

    with patch.object(CONNECTOR, "_make_client", return_value=mock_client):
        traces = CONNECTOR.fetch(project_name="test", limit=10, progress=False)

    assert len(traces) == 1
    assert traces[0].trace_id == "root-001"
    assert traces[0].outcome == "success"
    assert len(traces[0].spans) == 3  # root + 2 children


def test_fetch_passes_filter_for_tags():
    mock_client = MagicMock()
    mock_client.list_runs.return_value = iter([])

    with patch.object(CONNECTOR, "_make_client", return_value=mock_client):
        CONNECTOR.fetch(
            project_name="test",
            tags=["agentflow", "baseline"],
            limit=10,
            progress=False,
        )

    call_kwargs = mock_client.list_runs.call_args[1]
    assert 'has(tags, "agentflow")' in call_kwargs["filter"]
    assert 'has(tags, "baseline")' in call_kwargs["filter"]


def test_fetch_passes_filter_for_session():
    mock_client = MagicMock()
    mock_client.list_runs.return_value = iter([])

    with patch.object(CONNECTOR, "_make_client", return_value=mock_client):
        CONNECTOR.fetch(
            project_name="test",
            session_id="sess-42",
            limit=10,
            progress=False,
        )

    call_kwargs = mock_client.list_runs.call_args[1]
    assert 'eq(session_id, "sess-42")' in call_kwargs["filter"]


def test_fetch_respects_limit():
    runs = [_make_run(run_id=f"run-{i}", outputs={"outcome": "success"}) for i in range(10)]
    mock_client = MagicMock()
    # Root runs returns all 10, children returns empty for each
    call_count = [0]
    def list_runs_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return iter(runs)
        return iter([])
    mock_client.list_runs.side_effect = list_runs_side_effect

    with patch.object(CONNECTOR, "_make_client", return_value=mock_client):
        traces = CONNECTOR.fetch(project_name="test", limit=3, progress=False)

    assert len(traces) == 3


def test_fetch_no_filter_when_no_tags_or_session():
    mock_client = MagicMock()
    mock_client.list_runs.return_value = iter([])

    with patch.object(CONNECTOR, "_make_client", return_value=mock_client):
        CONNECTOR.fetch(project_name="test", limit=10, progress=False)

    call_kwargs = mock_client.list_runs.call_args[1]
    assert call_kwargs["filter"] is None


# ── _to_ts ───────────────────────────────────────────────────────────────────

def test_to_ts_datetime():
    dt = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _to_ts(dt) == dt.timestamp()


def test_to_ts_float():
    assert _to_ts(1234567890.5) == 1234567890.5


def test_to_ts_none():
    with patch("time.time", return_value=9999.0):
        assert _to_ts(None) == 9999.0
