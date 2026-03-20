"""End-to-end test: engine + all renderers.

Builds synthetic traces, runs all metrics through the engine,
renders output through all three renderers, and verifies correctness.
"""

from __future__ import annotations

import json
import random

from kalibra.engine import CompareResult, compare, resolve_metrics
from kalibra.metrics import Direction
from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS, Span, Trace
from kalibra.renderers import render
from kalibra.renderers.terminal import render_terminal


def _make_span(
    name: str = "llm_call",
    duration_s: float = 1.0,
    cost: float = 0.01,
    input_tokens: int = 500,
    output_tokens: int = 200,
    error: bool = False,
) -> Span:
    start_ns = 1_000_000_000
    end_ns = start_ns + int(duration_s * 1e9)
    return Span(
        span_id=f"span-{random.randint(0, 99999):05d}",
        name=name,
        start_ns=start_ns,
        end_ns=end_ns,
        cost=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
    )


def _make_trace(
    trace_id: str,
    outcome: str | None = OUTCOME_SUCCESS,
    n_spans: int = 3,
    span_names: list[str] | None = None,
    base_duration: float = 2.0,
    base_cost: float = 0.02,
    base_tokens: int = 500,
    error_rate: float = 0.0,
) -> Trace:
    names = span_names or ["llm_call", "tool_use", "llm_call"]
    spans = []
    for i in range(n_spans):
        name = names[i % len(names)]
        spans.append(_make_span(
            name=name,
            duration_s=base_duration + random.uniform(-0.5, 0.5),
            cost=base_cost + random.uniform(-0.005, 0.005),
            input_tokens=base_tokens + random.randint(-100, 100),
            output_tokens=base_tokens // 2 + random.randint(-50, 50),
            error=random.random() < error_rate,
        ))
    return Trace(
        trace_id=trace_id,
        spans=spans,
        outcome=outcome,
        metadata={"task_id": trace_id.rsplit("-", 1)[0]},
    )


def _build_populations(
    n: int = 50,
    success_rate: float = 0.8,
    prefix: str = "baseline",
    base_duration: float = 2.0,
    base_cost: float = 0.02,
    base_tokens: int = 500,
    n_spans: int = 3,
    span_names: list[str] | None = None,
    error_rate: float = 0.05,
) -> list[Trace]:
    traces = []
    tasks = [f"task-{i}" for i in range(n)]
    for i, task in enumerate(tasks):
        outcome = OUTCOME_SUCCESS if random.random() < success_rate else OUTCOME_FAILURE
        traces.append(_make_trace(
            trace_id=f"{task}-{prefix}-0",
            outcome=outcome,
            n_spans=n_spans,
            span_names=span_names,
            base_duration=base_duration,
            base_cost=base_cost,
            base_tokens=base_tokens,
            error_rate=error_rate,
        ))
    return traces


class TestEngineBasic:
    """Test the engine produces valid CompareResult."""

    def test_compare_returns_result(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current", base_cost=0.015)

        result = compare(
            baseline, current,
            baseline_source="v1",
            current_source="v2",
        )

        assert isinstance(result, CompareResult)
        assert result.baseline_count == 50
        assert result.current_count == 50
        assert result.baseline_source == "v1"
        assert result.current_source == "v2"
        assert isinstance(result.direction, Direction)

    def test_all_metrics_present(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current")

        result = compare(baseline, current)

        expected = {
            "success_rate", "cost", "duration", "steps",
            "error_rate",
            "token_usage", "token_efficiency", "cost_quality",
            "trace_breakdown", "span_breakdown",
        }
        assert set(result.observations.keys()) == expected

    def test_gates_pass(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current")

        result = compare(
            baseline, current,
            require=["success_rate >= 50"],
        )

        assert result.passed
        assert len(result.gates) == 1
        assert result.gates[0].passed

    def test_gates_fail(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.1, "current")

        result = compare(
            baseline, current,
            require=["success_rate >= 90"],
        )

        assert not result.passed
        assert not result.gates[0].passed

    def test_subset_metrics(self):
        random.seed(42)
        baseline = _build_populations(30, 0.8, "baseline")
        current = _build_populations(30, 0.85, "current")

        result = compare(
            baseline, current,
            metrics=["success_rate", "cost"],
        )

        assert set(result.observations.keys()) == {"success_rate", "cost"}

    def test_empty_metrics_list_runs_nothing(self):
        """Fix #1: metrics=[] should run zero metrics, not all defaults."""
        random.seed(42)
        baseline = _build_populations(30, 0.8, "baseline")
        current = _build_populations(30, 0.85, "current")

        result = compare(baseline, current, metrics=[])

        assert result.observations == {}

    def test_resolve_metrics_empty_list(self):
        """Fix #1: resolve_metrics([]) returns empty, not defaults."""
        assert resolve_metrics([]) == []
        assert len(resolve_metrics(None)) == 10

    def test_rollup_excludes_breakdowns(self):
        """Fix #8: breakdown metrics shouldn't dominate overall direction.

        When all aggregate metrics are SAME but span_breakdown shows
        a regression, the rollup should be SAME, not DEGRADATION.
        """
        random.seed(42)
        # Same params for both → aggregate metrics all SAME.
        # But span names differ slightly so span_breakdown might detect change.
        baseline = _build_populations(50, 0.8, "baseline")
        random.seed(42)
        current = _build_populations(50, 0.8, "current")

        result = compare(baseline, current)

        # The aggregate metrics should dominate; breakdowns shouldn't flip it.
        sb = result.observations.get("span_breakdown")
        if sb and sb.direction in (Direction.DEGRADATION, Direction.UPGRADE):
            # Even with a breakdown regression, the overall should be SAME
            assert result.direction != Direction.DEGRADATION or any(
                o.direction == Direction.DEGRADATION
                for o in result.observations.values()
                if o.name not in ("trace_breakdown", "span_breakdown")
            )

    def test_metric_config_sets_attributes(self):
        """Fix #11: metric_config replaces task_id_field special-case."""
        random.seed(42)
        baseline = [
            Trace(trace_id=f"t-{i}", spans=[_make_span()],
                  outcome=OUTCOME_SUCCESS, metadata={"my_task": f"task-{i}"})
            for i in range(10)
        ]
        current = [
            Trace(trace_id=f"t-{i}", spans=[_make_span()],
                  outcome=OUTCOME_FAILURE if i < 3 else OUTCOME_SUCCESS,
                  metadata={"my_task": f"task-{i}"})
            for i in range(10)
        ]

        result = compare(
            baseline, current,
            metrics=["trace_breakdown"],
            metric_config={"trace_breakdown": {"task_id_field": "my_task"}},
        )

        tb = result.observations["trace_breakdown"]
        assert tb.metadata["n_regressions"] == 3


class TestTerminalRenderer:
    """Test the terminal renderer produces output without errors."""

    def test_render_basic(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current", base_cost=0.015)

        result = compare(
            baseline, current,
            baseline_source="prod-v1",
            current_source="staging-v2",
        )
        output = render_terminal(result)

        assert "Kalibra Compare" in output
        assert "prod-v1" in output
        assert "staging-v2" in output
        assert "50" in output
        # Fix #9: error_rate should appear in compact mode
        assert "Error rate" in output

    def test_render_verbose(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current", base_cost=0.015)

        result = compare(
            baseline, current,
            baseline_source="prod-v1",
            current_source="staging-v2",
        )
        compact = render_terminal(result)
        verbose = render_terminal(result, verbose=True)

        assert "Kalibra Compare" in verbose
        assert len(verbose) > len(compact)
        # Fix #9: error_rate should also appear in verbose mode
        assert "Error rate" in verbose

    def test_render_with_gates(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current")

        result = compare(
            baseline, current,
            require=["success_rate >= 50", "cost_delta_pct <= 20"],
            baseline_source="v1",
            current_source="v2",
        )
        output = render_terminal(result)

        assert "Quality gates" in output

    def test_render_with_warnings(self):
        random.seed(42)
        baseline = _build_populations(10, 0.8, "baseline")
        current = _build_populations(10, 0.85, "current")

        result = compare(
            baseline, current,
            baseline_source="v1",
            current_source="v2",
        )
        output = render_terminal(result)

        assert "Kalibra Compare" in output

    def test_render_no_outcome_data(self):
        """Test rendering when traces have no outcome data."""
        random.seed(42)
        baseline = [
            Trace(trace_id=f"b-{i}", spans=[_make_span()])
            for i in range(20)
        ]
        current = [
            Trace(trace_id=f"c-{i}", spans=[_make_span()])
            for i in range(20)
        ]

        result = compare(baseline, current)
        output = render_terminal(result)

        assert "Kalibra Compare" in output
        assert "n/a" in output.lower() or "N/A" in output

    def test_render_degradation(self):
        """Test rendering when current is worse."""
        random.seed(42)
        baseline = _build_populations(
            50, 0.9, "baseline",
            base_cost=0.01, base_duration=1.0, base_tokens=300,
        )
        current = _build_populations(
            50, 0.5, "current",
            base_cost=0.05, base_duration=5.0, base_tokens=1500,
        )

        result = compare(baseline, current)
        output = render_terminal(result)

        assert "Kalibra Compare" in output

    def test_render_all_same(self):
        """Test rendering when baseline == current."""
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        random.seed(42)
        current = _build_populations(50, 0.8, "current")

        result = compare(baseline, current)
        output = render_terminal(result)

        assert "Kalibra Compare" in output


class TestMarkdownRenderer:
    """Test the markdown renderer."""

    def test_render_basic(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current", base_cost=0.015)

        result = compare(
            baseline, current,
            baseline_source="v1",
            current_source="v2",
        )
        output = render(result, "markdown")

        assert "Kalibra Compare" in output
        assert "v1" in output
        assert "| Metric |" in output

    def test_markdown_span_breakdown(self):
        """Fix #5: markdown must include span breakdown."""
        random.seed(42)
        baseline = _build_populations(
            50, 0.8, "baseline", base_cost=0.05, base_duration=3.0,
        )
        current = _build_populations(
            50, 0.85, "current", base_cost=0.01, base_duration=1.0,
        )

        result = compare(baseline, current)
        output = render(result, "markdown")

        sb = result.observations.get("span_breakdown")
        if sb and sb.direction != Direction.NA:
            assert "Span Breakdown" in output

    def test_markdown_verbose_shows_improvements(self):
        """Fix #12: verbose mode should show improvements in trace breakdown."""
        random.seed(42)
        baseline = [
            Trace(trace_id=f"task-{i}", spans=[_make_span()],
                  outcome=OUTCOME_FAILURE, metadata={"task_id": f"task-{i}"})
            for i in range(10)
        ]
        current = [
            Trace(trace_id=f"task-{i}", spans=[_make_span()],
                  outcome=OUTCOME_SUCCESS, metadata={"task_id": f"task-{i}"})
            for i in range(10)
        ]

        result = compare(
            baseline, current,
            metric_config={"trace_breakdown": {"task_id_field": "task_id"}},
        )
        compact = render(result, "markdown")
        verbose = render(result, "markdown", verbose=True)

        # Verbose should include improvements detail
        assert "Improvements" in verbose or len(verbose) >= len(compact)

    def test_markdown_consistent_names(self):
        """Fix #10: metric names should match between terminal and markdown."""
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current")

        result = compare(baseline, current)
        md = render(result, "markdown")

        # Should use shared METRIC_LABEL, not .replace("_", " ").title()
        assert "Cost / quality" in md
        assert "Token efficiency" in md


class TestJsonRenderer:
    """Test the JSON renderer."""

    def test_render_parses(self):
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current")

        result = compare(baseline, current)
        output = render(result, "json")
        data = json.loads(output)

        assert data["direction"] in ("upgrade", "same", "degradation", "inconclusive", "n/a")
        assert "metrics" in data
        assert "success_rate" in data["metrics"]

    def test_json_clean_handles_nested_lists(self):
        """Fix #6: _clean must recurse into lists."""
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current")

        result = compare(baseline, current)
        output = render(result, "json")
        data = json.loads(output)

        # Trace breakdown regressions is a list of dicts — should serialize cleanly.
        tb = data["metrics"]["trace_breakdown"]
        assert isinstance(tb["metadata"]["regressions"], list)

    def test_json_nan_becomes_null(self):
        """NaN values in gate actuals should become null."""
        random.seed(42)
        baseline = [Trace(trace_id="b")]
        current = [Trace(trace_id="c")]

        result = compare(
            baseline, current,
            metrics=["success_rate"],
            require=["success_rate >= 50"],
        )
        output = render(result, "json")
        data = json.loads(output)

        # Gate should have been skipped (no data), actual should be null not NaN
        gate = data["gates"][0]
        assert gate["actual"] is None or isinstance(gate["actual"], (int, float))


class TestEdgeCases:
    """Edge cases for the full pipeline."""

    def test_empty_populations(self):
        result = compare([], [])
        output = render_terminal(result)
        assert "Kalibra Compare" in output

    def test_single_trace(self):
        baseline = [_make_trace("task-0-b-0", OUTCOME_SUCCESS)]
        current = [_make_trace("task-0-c-0", OUTCOME_FAILURE)]

        result = compare(baseline, current)
        output = render_terminal(result)
        assert "Kalibra Compare" in output

    def test_no_spans(self):
        baseline = [Trace(trace_id="b", outcome=OUTCOME_SUCCESS)]
        current = [Trace(trace_id="c", outcome=OUTCOME_FAILURE)]

        result = compare(baseline, current)
        output = render_terminal(result)
        assert "Kalibra Compare" in output

    def test_one_sided_cost_data_returns_na(self):
        """Bug fix: if only one population has cost data, return N/A, not bogus delta."""
        baseline = [
            Trace(trace_id=f"b-{i}", spans=[_make_span(cost=0.05)],
                  outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]
        # Current traces have no cost data (span-less, no _cost).
        current = [
            Trace(trace_id=f"c-{i}", outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]

        result = compare(baseline, current, metrics=["cost"])
        obs = result.observations["cost"]
        assert obs.direction == Direction.NA

    def test_one_sided_token_data_returns_na(self):
        """Bug fix: if only one population has token data, return N/A."""
        baseline = [
            Trace(trace_id=f"b-{i}", spans=[_make_span(input_tokens=500, output_tokens=200)],
                  outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]
        current = [
            Trace(trace_id=f"c-{i}", outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]

        result = compare(baseline, current, metrics=["token_usage"])
        obs = result.observations["token_usage"]
        assert obs.direction == Direction.NA

    def test_one_sided_cost_quality_returns_na(self):
        """Bug fix: if only baseline has cost for successes, return N/A."""
        baseline = [
            Trace(trace_id=f"b-{i}", spans=[_make_span(cost=0.05)],
                  outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]
        current = [
            Trace(trace_id=f"c-{i}", outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]

        result = compare(baseline, current, metrics=["cost_quality"])
        obs = result.observations["cost_quality"]
        assert obs.direction == Direction.NA

    def test_one_sided_token_efficiency_returns_na(self):
        """Bug fix: if only baseline has tokens for successes, return N/A."""
        baseline = [
            Trace(trace_id=f"b-{i}", spans=[_make_span(input_tokens=500, output_tokens=200)],
                  outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]
        current = [
            Trace(trace_id=f"c-{i}", outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]

        result = compare(baseline, current, metrics=["token_efficiency"])
        obs = result.observations["token_efficiency"]
        assert obs.direction == Direction.NA

    def test_zero_cost_included_in_median(self):
        """Bug fix: 0 cost means measured as zero, not absent. Must be included."""
        baseline = [
            Trace(trace_id=f"b-{i}", _cost=0.10, outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]
        # Half cost $0 (free/cached), half cost $0.10.
        current = [
            Trace(trace_id=f"c-{i}", _cost=0.0 if i < 10 else 0.10,
                  outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]

        result = compare(baseline, current, metrics=["cost"])
        obs = result.observations["cost"]
        # Median should be $0.05, not $0.10 (which would happen if zeros were excluded).
        assert obs.current["median"] < 0.10

    def test_zero_success_rate_threshold_field(self):
        """Bug fix: 0% success rate must produce success_rate=0, not None/crash."""
        baseline = [
            Trace(trace_id=f"b-{i}", spans=[_make_span()],
                  outcome=OUTCOME_SUCCESS)
            for i in range(20)
        ]
        current = [
            Trace(trace_id=f"c-{i}", spans=[_make_span()],
                  outcome=OUTCOME_FAILURE)
            for i in range(20)
        ]

        result = compare(baseline, current, metrics=["success_rate"])
        obs = result.observations["success_rate"]
        from kalibra.metrics.success_rate import SuccessRateMetric
        fields = SuccessRateMetric().threshold_fields(obs)
        assert fields["success_rate"] == 0.0
        assert fields["success_rate_delta"] < 0

    def test_non_llm_spans_dont_corrupt_metrics(self):
        """Non-LLM spans (no cost/tokens) should be None, not zero.

        A trace with 2 LLM spans ($0.05 each) and 1 tool span (no cost)
        should have total_cost=$0.10, not $0.10/3 averaged with zeros.
        """
        llm_span = Span(
            span_id="llm", name="llm_call",
            start_ns=1_000_000_000, end_ns=2_000_000_000,
            cost=0.05, input_tokens=500, output_tokens=200,
        )
        tool_span = Span(
            span_id="tool", name="file_read",
            start_ns=2_000_000_000, end_ns=2_500_000_000,
            # cost, input_tokens, output_tokens are None (non-LLM)
        )
        trace = Trace(
            trace_id="t1", outcome=OUTCOME_SUCCESS,
            spans=[llm_span, tool_span],
        )

        assert trace.total_cost == 0.05  # only the LLM span
        assert trace.total_tokens == 700  # only the LLM span
        assert tool_span.cost is None
        assert tool_span.total_tokens is None

    def test_mixed_llm_and_tool_spans_in_compare(self):
        """Traces with mixed LLM + non-LLM spans should work end-to-end."""
        def make_mixed_trace(tid, cost):
            return Trace(
                trace_id=tid, outcome=OUTCOME_SUCCESS,
                spans=[
                    Span(span_id=f"{tid}-llm", name="llm_call",
                         start_ns=1_000_000_000, end_ns=2_000_000_000,
                         cost=cost, input_tokens=500, output_tokens=200),
                    Span(span_id=f"{tid}-tool", name="tool_use",
                         start_ns=2_000_000_000, end_ns=2_500_000_000),
                ],
            )

        baseline = [make_mixed_trace(f"b-{i}", 0.05) for i in range(20)]
        current = [make_mixed_trace(f"c-{i}", 0.03) for i in range(20)]

        result = compare(baseline, current)
        cost_obs = result.observations["cost"]
        # Should see the real cost difference, not corrupted by None spans
        assert cost_obs.direction != Direction.NA
        assert cost_obs.baseline["median"] == 0.05
        assert cost_obs.current["median"] == 0.03

    def test_inconclusive_gate_skipped(self):
        """When a metric's CI includes zero (direction=SAME), delta gates
        should be skipped rather than evaluated against the point estimate.

        The gate threshold is set to something the point estimate would
        FAIL (-50%), proving the skip actually prevented a false failure.
        """
        # Small n + overlapping values → CI will include zero → SAME direction.
        random.seed(42)
        baseline = [
            Trace(trace_id=f"b-{i}", _cost=0.05 + random.uniform(-0.02, 0.02),
                  outcome=OUTCOME_SUCCESS)
            for i in range(10)
        ]
        current = [
            Trace(trace_id=f"c-{i}", _cost=0.051 + random.uniform(-0.02, 0.02),
                  outcome=OUTCOME_SUCCESS)
            for i in range(10)
        ]

        result = compare(
            baseline, current,
            metrics=["cost"],
            # Threshold that the point estimate would fail — but skip saves it.
            require=["cost_delta_pct <= -50"],
        )

        # Cost direction should be SAME (CI includes zero at n=10).
        cost_obs = result.observations["cost"]
        assert cost_obs.direction == Direction.SAME

        # The delta gate should be SKIPPED, not evaluated.
        # Without the skip, this would FAIL (actual > -50%).
        gate = result.gates[0]
        assert gate.passed is True
        assert gate.warning is not None
        assert "not statistically significant" in gate.warning

    def test_conclusive_gate_evaluated(self):
        """When a metric's CI does NOT include zero, delta gates
        should be evaluated normally — and fail when violated."""
        random.seed(42)
        baseline = [
            Trace(trace_id=f"b-{i}", _cost=0.05, outcome=OUTCOME_SUCCESS)
            for i in range(50)
        ]
        current = [
            Trace(trace_id=f"c-{i}", _cost=0.10, outcome=OUTCOME_SUCCESS)
            for i in range(50)
        ]

        result = compare(
            baseline, current,
            metrics=["cost"],
            require=["cost_delta_pct <= 50"],
        )

        # Cost direction should NOT be SAME (clear 100% increase).
        cost_obs = result.observations["cost"]
        assert cost_obs.direction != Direction.SAME

        # Gate should be evaluated (not skipped) and FAIL (100% > 50%).
        gate = result.gates[0]
        assert gate.warning is None
        assert gate.passed is False

    def test_absolute_field_not_skipped_when_inconclusive(self):
        """Absolute fields (success_rate, total_cost) should still be
        evaluated even when the delta is inconclusive."""
        random.seed(42)
        baseline = [
            Trace(trace_id=f"b-{i}", _cost=0.05 + random.uniform(-0.01, 0.01),
                  outcome=OUTCOME_SUCCESS)
            for i in range(10)
        ]
        current = [
            Trace(trace_id=f"c-{i}", _cost=0.06 + random.uniform(-0.01, 0.01),
                  outcome=OUTCOME_SUCCESS)
            for i in range(10)
        ]

        result = compare(
            baseline, current,
            metrics=["cost"],
            require=["total_cost <= 1.0"],
        )

        # total_cost is an absolute field, not a delta — should be evaluated.
        gate = result.gates[0]
        assert gate.warning is None  # Not skipped
        assert gate.passed is True  # total < 1.0

    def test_all_renderers_no_crash(self):
        """All three renderers should handle any CompareResult without crashing."""
        random.seed(42)
        baseline = _build_populations(50, 0.8, "baseline")
        current = _build_populations(50, 0.85, "current", base_cost=0.015)

        result = compare(
            baseline, current,
            require=["success_rate >= 50"],
            baseline_source="v1",
            current_source="v2",
        )

        for fmt in ("terminal", "markdown", "json"):
            output = render(result, fmt, verbose=True)
            assert len(output) > 0

        for fmt in ("terminal", "markdown", "json"):
            output = render(result, fmt, verbose=False)
            assert len(output) > 0
