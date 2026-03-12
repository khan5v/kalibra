"""Tests for report rendering — terminal, markdown, JSON formats."""

import click

from agentflow.collection import TraceCollection
from agentflow.compare import compare_collections
from agentflow.config import CompareConfig
from agentflow.converters.base import AF_COST, Trace, make_span
from agentflow.report import render


# ── Helpers ────────────────────────────────────────────────────────────────────

def _span(name, i=0, cost=0.01):
    return make_span(
        name=name, trace_id="t", span_id=f"{name}_{i}",
        parent_span_id=None, start_ns=int(i * 1e9), end_ns=int((i + 1) * 1e9),
        attributes={AF_COST: cost},
    )


def _trace(tid, steps, outcome="success", cost=0.01):
    return Trace(tid, [_span(s, i, cost) for i, s in enumerate(steps)], outcome=outcome)


def _make_result(require=None):
    """Build a CompareResult with known baseline/current."""
    baseline = TraceCollection.from_traces([
        _trace("task1__m__0", ["plan", "edit", "test"], "success", cost=0.02),
        _trace("task2__m__1", ["plan", "edit", "test"], "failure", cost=0.03),
        _trace("task3__m__2", ["plan", "edit"], "success", cost=0.01),
    ], source="baseline.jsonl")
    current = TraceCollection.from_traces([
        _trace("task1__m__0", ["plan", "edit", "test"], "success", cost=0.01),
        _trace("task2__m__1", ["plan", "edit", "test"], "success", cost=0.02),
        _trace("task3__m__2", ["plan", "edit"], "success", cost=0.01),
    ], source="current.jsonl")
    config = CompareConfig(require=require or [])
    return compare_collections(baseline, current, config=config)


# ── Terminal format ────────────────────────────────────────────────────────────

class TestTerminalReport:
    def test_header_present(self):
        text = render(_make_result(), "terminal")
        assert "AgentFlow Compare" in text
        assert "baseline.jsonl" in text
        assert "current.jsonl" in text

    def test_direction_line(self):
        text = render(_make_result(), "terminal")
        assert "Direction" in text

    def test_metrics_have_detail_lines(self):
        text = render(_make_result(), "terminal")
        # Cost should show median on headline and avg/total as sub-lines
        assert "median" in text
        assert "avg" in text
        assert "total" in text

    def test_no_gates_footer(self):
        text = render(_make_result(), "terminal")
        assert "no quality gates configured" in text

    def test_gates_passed(self):
        text = render(_make_result(require=["success_rate_delta >= -50"]), "terminal")
        plain = click.unstyle(text)
        assert "PASSED" in plain
        assert "[ OK ]" in plain

    def test_gates_failed(self):
        text = render(_make_result(require=["success_rate_delta >= 99"]), "terminal")
        plain = click.unstyle(text)
        assert "FAILED" in plain
        assert "[FAIL]" in plain

    def test_per_task_capped(self):
        """Per-task output should cap task IDs, not flood the terminal."""
        text = render(_make_result(), "terminal")
        # Should not contain more than 3 task IDs per line
        lines = text.split("\n")
        for line in lines:
            if "regressed:" in line or "improved:" in line:
                # Count commas — at most 2 (= 3 items)
                assert line.count(",") <= 2


# ── Markdown format ────────────────────────────────────────────────────────────

class TestMarkdownReport:
    def test_header(self):
        text = render(_make_result(), "markdown")
        assert "## AgentFlow" in text
        assert "**Baseline:**" in text

    def test_metrics_as_sections(self):
        text = render(_make_result(), "markdown")
        # Should have bold metric labels, not table rows
        assert "**" in text

    def test_per_task_capped_in_markdown(self):
        text = render(_make_result(), "markdown")
        # Should have at most 5 task bullets
        bullets = [l for l in text.split("\n") if l.startswith("- `")]
        assert len(bullets) <= 10  # 5 improved + 5 regressed max

    def test_gates_in_markdown(self):
        text = render(_make_result(require=["success_rate_delta >= -50"]), "markdown")
        assert "### Thresholds" in text
        assert "PASSED" in text


# ── JSON format ────────────────────────────────────────────────────────────────

class TestJsonReport:
    def test_valid_json(self):
        import json
        text = render(_make_result(), "json")
        data = json.loads(text)
        assert "comparison" in data
        assert "validation" in data

    def test_detail_lines_in_json(self):
        import json
        text = render(_make_result(), "json")
        data = json.loads(text)
        cost_obs = data["comparison"]["observations"]["cost"]
        assert "detail_lines" in cost_obs
        assert isinstance(cost_obs["detail_lines"], list)
