"""Tests — core data structures, collection, metrics, comparison."""

from agentflow.collection import TraceCollection
from agentflow.compare import compare, CompareResult
from agentflow.converters.base import Span, Trace
from agentflow.metrics import (
    CostMetric,
    DEFAULT_METRICS,
    DurationMetric,
    PathDistributionMetric,
    PerTaskMetric,
    StepsMetric,
    SuccessRateMetric,
    ToolErrorRateMetric,
    _extract_task_id,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _span(name: str, i: int = 0, status: str = "ok", cost: float = 0.0) -> Span:
    return Span(
        span_id=f"{name}_{i}", parent_id=None, name=name,
        start_time=float(i), end_time=float(i + 1),
        cost=cost, status=status,
    )


def _trace(trace_id: str, steps: list[str], outcome: str = "success",
           costs: list[float] | None = None, errors: list[str] | None = None) -> Trace:
    spans = []
    for i, name in enumerate(steps):
        cost = costs[i] if costs else 0.0
        status = "error" if errors and name in errors else "ok"
        spans.append(_span(name, i, status=status, cost=cost))
    return Trace(trace_id=trace_id, spans=spans, outcome=outcome)


def _col(*traces: Trace) -> TraceCollection:
    return TraceCollection.from_traces(list(traces))


# ── Trace properties ───────────────────────────────────────────────────────────

def test_trace_duration():
    t = _trace("t1", ["a", "b", "c"])
    assert t.duration == 3.0


def test_trace_total_cost():
    t = _trace("t1", ["a", "b"], costs=[1.0, 2.5])
    assert t.total_cost == 3.5


def test_trace_total_tokens():
    spans = [Span("s1", None, "a", 0.0, 1.0, input_tokens=10, output_tokens=5)]
    t = Trace("t", spans)
    assert t.total_tokens == 15


# ── TraceCollection ────────────────────────────────────────────────────────────

def test_collection_len():
    col = _col(_trace("t1", ["a"]), _trace("t2", ["b"]))
    assert len(col) == 2


def test_collection_all_traces():
    t1 = _trace("t1", ["a"])
    col = _col(t1)
    assert col.all_traces() == [t1]


def test_collection_get_trace():
    t = _trace("abc", ["a"])
    col = _col(t)
    assert col.get_trace("abc") is t
    assert col.get_trace("missing") is None


def test_collection_spans_for_node():
    col = _col(
        _trace("t1", ["bash", "edit"]),
        _trace("t2", ["bash", "submit"]),
    )
    assert len(col.spans_for_node("bash")) == 2
    assert len(col.spans_for_node("edit")) == 1
    assert col.spans_for_node("unknown") == []


def test_collection_traces_with_outcome():
    col = _col(
        _trace("t1", ["a"], outcome="success"),
        _trace("t2", ["a"], outcome="failure"),
        _trace("t3", ["a"], outcome="success"),
    )
    assert len(col.traces_with_outcome("success")) == 2
    assert len(col.traces_with_outcome("failure")) == 1


# ── SuccessRateMetric ──────────────────────────────────────────────────────────

def test_success_rate_zero_delta():
    col = _col(_trace("t1", ["a"], "success"), _trace("t2", ["a"], "failure"))
    m = SuccessRateMetric()
    s = m.summarize(col)
    assert s["total"] == 2
    assert s["successes"] == 1
    assert s["rate"] == 0.5
    result = m.compare(s, s)
    assert result.delta == 0.0


def test_success_rate_positive_delta():
    m = SuccessRateMetric()
    b = m.summarize(_col(_trace("t1", ["a"], "failure"), _trace("t2", ["a"], "failure")))
    c = m.summarize(_col(_trace("t1", ["a"], "success"), _trace("t2", ["a"], "success")))
    result = m.compare(b, c)
    assert result.delta == 100.0


def test_success_rate_threshold_fields():
    m = SuccessRateMetric()
    s = m.summarize(_col(_trace("t1", ["a"], "success")))
    result = m.compare(s, s)
    fields = m.threshold_fields(result)
    assert "success_rate_delta" in fields
    assert "success_rate" in fields


# ── PerTaskMetric ──────────────────────────────────────────────────────────────

def test_per_task_regression():
    m = PerTaskMetric()
    b = m.summarize(_col(_trace("task__model__0", ["a"], "success")))
    c = m.summarize(_col(_trace("task__model__0", ["a"], "failure")))
    result = m.compare(b, c)
    assert len(result.metadata["regressions"]) == 1
    assert len(result.metadata["improvements"]) == 0


def test_per_task_improvement():
    m = PerTaskMetric()
    b = m.summarize(_col(_trace("task__model__0", ["a"], "failure")))
    c = m.summarize(_col(_trace("task__model__0", ["a"], "success")))
    result = m.compare(b, c)
    assert len(result.metadata["improvements"]) == 1
    assert len(result.metadata["regressions"]) == 0


def test_extract_task_id_swebench():
    assert _extract_task_id("django__django-12345__gpt4__0") == "django__django-12345"
    assert _extract_task_id("simple") == "simple"


# ── CostMetric ────────────────────────────────────────────────────────────────

def test_cost_metric_delta():
    m = CostMetric()
    b = m.summarize(_col(_trace("t1", ["a"], costs=[1.0])))
    c = m.summarize(_col(_trace("t1", ["a"], costs=[2.0])))
    result = m.compare(b, c)
    assert result.delta == 100.0


def test_cost_metric_threshold():
    m = CostMetric()
    s = m.summarize(_col(_trace("t1", ["a"], costs=[0.05])))
    result = m.compare(s, s)
    assert "cost_delta_pct" in m.threshold_fields(result)


def test_cost_metric_no_data_warning():
    m = CostMetric()
    s = m.summarize(_col(_trace("t1", ["a"])))  # cost=0 by default
    result = m.compare(s, s)
    assert result.warnings
    assert result.delta is None


# ── StepsMetric ───────────────────────────────────────────────────────────────

def test_steps_metric():
    m = StepsMetric()
    b = m.summarize(_col(_trace("t1", ["a", "b"])))
    c = m.summarize(_col(_trace("t1", ["a", "b", "c"])))
    result = m.compare(b, c)
    assert result.delta == 50.0


# ── DurationMetric ────────────────────────────────────────────────────────────

def test_duration_metric_p95():
    m = DurationMetric()
    s = m.summarize(_col(_trace("t1", ["a", "b", "c"])))
    assert s["avg"] == 3.0
    assert s["p95"] == 3.0


def test_duration_metric_threshold_fields():
    m = DurationMetric()
    s = m.summarize(_col(_trace("t1", ["a"])))
    result = m.compare(s, s)
    fields = m.threshold_fields(result)
    assert "duration_delta_pct" in fields
    assert "duration_p95_delta_pct" in fields


# ── ToolErrorRateMetric ───────────────────────────────────────────────────────

def test_tool_error_rate():
    m = ToolErrorRateMetric()
    col = _col(_trace("t1", ["bash", "edit"], errors=["bash"]))
    s = m.summarize(col)
    assert s["errors"] == 1
    assert s["total"] == 2
    assert s["rate"] == 0.5


def test_tool_error_rate_delta():
    m = ToolErrorRateMetric()
    b = m.summarize(_col(_trace("t1", ["a", "b"])))
    c = m.summarize(_col(_trace("t1", ["a", "b"], errors=["a", "b"])))
    result = m.compare(b, c)
    assert result.delta == 100.0


# ── PathDistributionMetric ────────────────────────────────────────────────────

def test_path_jaccard_identical():
    m = PathDistributionMetric()
    col = _col(_trace("t1", ["a", "b", "c"]))
    s = m.summarize(col)
    result = m.compare(s, s)
    assert result.metadata["jaccard"] == 1.0


def test_path_jaccard_disjoint():
    m = PathDistributionMetric()
    b = m.summarize(_col(_trace("t1", ["a", "b"])))
    c = m.summarize(_col(_trace("t1", ["x", "y"])))
    result = m.compare(b, c)
    assert result.metadata["jaccard"] == 0.0
    assert len(result.metadata["new_paths"]) == 1
    assert len(result.metadata["dropped_paths"]) == 1


# ── Full compare() integration ────────────────────────────────────────────────

def test_compare_same_data(tmp_path):
    from agentflow.converters.generic import save_jsonl
    traces = [
        _trace("t1__m__0", ["bash", "edit", "submit"], "success"),
        _trace("t2__m__1", ["bash", "edit", "submit"], "failure"),
    ]
    f = str(tmp_path / "traces.jsonl")
    save_jsonl(traces, f)
    result = compare(f, f)
    assert isinstance(result, CompareResult)
    assert result.thresholds_passed is True
    assert result["success_rate"].delta == 0.0
    assert result["per_task"].metadata["regressions"] == []


def test_compare_threshold_pass(tmp_path):
    from agentflow.converters.generic import save_jsonl
    traces = [_trace("t1__m__0", ["a"], "success")]
    f = str(tmp_path / "traces.jsonl")
    save_jsonl(traces, f)
    result = compare(f, f, require=["success_rate_delta >= -5"])
    assert result.thresholds_passed is True


def test_compare_threshold_fail(tmp_path):
    from agentflow.converters.generic import save_jsonl
    traces = [_trace("t1__m__0", ["a"], "success")]
    f = str(tmp_path / "traces.jsonl")
    save_jsonl(traces, f)
    result = compare(f, f, require=["success_rate_delta >= 99"])
    assert result.thresholds_passed is False


def test_all_default_metrics_present(tmp_path):
    from agentflow.converters.generic import save_jsonl
    traces = [_trace("t1__m__0", ["a"], "success")]
    f = str(tmp_path / "traces.jsonl")
    save_jsonl(traces, f)
    result = compare(f, f)
    for m in DEFAULT_METRICS:
        assert m.name in result.metrics, f"Missing metric: {m.name}"
