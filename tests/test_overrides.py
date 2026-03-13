"""Tests for outcome and cost overrides — config parsing and post-processing."""

from __future__ import annotations

import pytest

from agentflow.config import CostConfig, OutcomeConfig, SourceConfig
from agentflow.converters.base import (
    AF_COST,
    Trace,
    apply_overrides,
    make_span,
    span_cost,
    _resolve_field,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _span(name: str, cost: float = 0.0, extra_attrs: dict | None = None) -> object:
    attrs = {AF_COST: cost}
    if extra_attrs:
        attrs.update(extra_attrs)
    return make_span(
        name=name,
        trace_id="trace-001",
        span_id=f"{hash(name) & 0xFFFF:016x}",
        parent_span_id=None,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
        attributes=attrs,
    )


def _trace(
    trace_id: str = "t1",
    outcome: str | None = None,
    metadata: dict | None = None,
    cost: float = 0.01,
) -> Trace:
    return Trace(
        trace_id=trace_id,
        spans=[_span("llm-call", cost=cost)],
        outcome=outcome,
        metadata=metadata or {},
    )


def _source_config(outcome: dict | None = None, cost: dict | None = None) -> SourceConfig:
    return SourceConfig(
        source="langfuse",
        project="test",
        outcome=OutcomeConfig.from_dict(outcome),
        cost=CostConfig.from_dict(cost),
    )


# ── OutcomeConfig parsing ────────────────────────────────────────────────────

def test_outcome_config_from_dict():
    cfg = OutcomeConfig.from_dict({
        "field": "metadata.result",
        "success": ["pass", "resolved"],
        "failure": ["fail"],
    })
    assert cfg.field == "metadata.result"
    assert cfg.success == ["pass", "resolved"]
    assert cfg.failure == ["fail"]


def test_outcome_config_from_none():
    assert OutcomeConfig.from_dict(None) is None


def test_outcome_config_defaults():
    cfg = OutcomeConfig.from_dict({"field": "metadata.status"})
    assert cfg.success == ["success"]
    assert cfg.failure == ["failure", "error", "failed"]


# ── CostConfig parsing ──────────────────────────────────────────────────────

def test_cost_config_from_dict():
    cfg = CostConfig.from_dict({"attr": "custom.cost_usd"})
    assert cfg.attr == "custom.cost_usd"


def test_cost_config_from_none():
    assert CostConfig.from_dict(None) is None


# ── SourceConfig parsing with overrides ──────────────────────────────────────

def test_source_config_parses_outcome():
    src = SourceConfig.from_dict({
        "source": "langfuse",
        "project": "test",
        "outcome": {"field": "metadata.eval", "success": ["pass"]},
    })
    assert src.outcome is not None
    assert src.outcome.field == "metadata.eval"


def test_source_config_parses_cost():
    src = SourceConfig.from_dict({
        "source": "langfuse",
        "project": "test",
        "cost": {"attr": "my.cost"},
    })
    assert src.cost is not None
    assert src.cost.attr == "my.cost"


def test_source_config_no_overrides():
    src = SourceConfig.from_dict({"source": "langfuse", "project": "test"})
    assert src.outcome is None
    assert src.cost is None


# ── _resolve_field ───────────────────────────────────────────────────────────

def test_resolve_field_bare_key():
    t = _trace(metadata={"status": "pass"})
    assert _resolve_field(t, "status") == "pass"


def test_resolve_field_metadata_prefix():
    t = _trace(metadata={"result": "fail"})
    assert _resolve_field(t, "metadata.result") == "fail"


def test_resolve_field_dotted_key():
    """Keys with dots (e.g., langfuse.evaluation) are tried as exact keys first."""
    t = _trace(metadata={"langfuse.evaluation": "pass"})
    assert _resolve_field(t, "metadata.langfuse.evaluation") == "pass"


def test_resolve_field_nested_dict():
    t = _trace(metadata={"eval": {"result": "success"}})
    assert _resolve_field(t, "metadata.eval.result") == "success"


def test_resolve_field_missing():
    t = _trace(metadata={})
    assert _resolve_field(t, "metadata.nonexistent") is None


def test_resolve_field_outcome():
    t = _trace(outcome="failure")
    assert _resolve_field(t, "outcome") == "failure"


def test_resolve_field_trace_id():
    t = _trace(trace_id="abc-123")
    assert _resolve_field(t, "trace_id") == "abc-123"


# ── Outcome override ────────────────────────────────────────────────────────

def test_outcome_override_success_match():
    t = _trace(outcome=None, metadata={"result": "pass"})
    src = _source_config(outcome={"field": "result", "success": ["pass"]})
    apply_overrides([t], src)
    assert t.outcome == "success"


def test_outcome_override_failure_match():
    t = _trace(outcome=None, metadata={"result": "timeout"})
    src = _source_config(outcome={
        "field": "result",
        "success": ["pass"],
        "failure": ["timeout", "crash"],
    })
    apply_overrides([t], src)
    assert t.outcome == "failure"


def test_outcome_override_replaces_connector_heuristic():
    """Override should replace the connector's default outcome."""
    t = _trace(outcome="success", metadata={"eval": "fail"})
    src = _source_config(outcome={"field": "eval", "failure": ["fail"]})
    apply_overrides([t], src)
    assert t.outcome == "failure"


def test_outcome_override_no_match_keeps_original():
    t = _trace(outcome="success", metadata={"result": "unknown_value"})
    src = _source_config(outcome={"field": "result", "success": ["pass"]})
    apply_overrides([t], src)
    assert t.outcome == "success"  # unchanged


def test_outcome_override_missing_field_keeps_original():
    t = _trace(outcome="failure", metadata={})
    src = _source_config(outcome={"field": "nonexistent"})
    apply_overrides([t], src)
    assert t.outcome == "failure"  # unchanged


def test_outcome_override_case_insensitive():
    t = _trace(outcome=None, metadata={"status": "PASS"})
    src = _source_config(outcome={"field": "status", "success": ["pass"]})
    apply_overrides([t], src)
    assert t.outcome == "success"


def test_outcome_override_with_metadata_prefix():
    t = _trace(outcome=None, metadata={"langfuse.eval_result": "resolved"})
    src = _source_config(outcome={
        "field": "metadata.langfuse.eval_result",
        "success": ["resolved"],
    })
    apply_overrides([t], src)
    assert t.outcome == "success"


def test_outcome_override_no_field_configured():
    """OutcomeConfig with no field should be a no-op."""
    t = _trace(outcome="failure", metadata={"result": "pass"})
    src = _source_config(outcome={"success": ["pass"]})  # no field
    apply_overrides([t], src)
    assert t.outcome == "failure"  # unchanged


# ── Cost override ────────────────────────────────────────────────────────────

def test_cost_override_reads_alternate_attr():
    span = _span("llm", cost=0.01, extra_attrs={"custom.cost": 0.05})
    t = Trace(trace_id="t1", spans=[span], outcome="success")
    src = _source_config(cost={"attr": "custom.cost"})
    apply_overrides([t], src)
    assert span_cost(t.spans[0]) == pytest.approx(0.05)


def test_cost_override_missing_attr_defaults_to_zero():
    span = _span("llm", cost=0.01)
    t = Trace(trace_id="t1", spans=[span], outcome="success")
    src = _source_config(cost={"attr": "nonexistent.cost"})
    apply_overrides([t], src)
    assert span_cost(t.spans[0]) == 0.0


def test_cost_override_no_attr_configured():
    """CostConfig with no attr should be a no-op."""
    span = _span("llm", cost=0.01)
    t = Trace(trace_id="t1", spans=[span], outcome="success")
    src = _source_config(cost={})  # no attr
    apply_overrides([t], src)
    assert span_cost(t.spans[0]) == pytest.approx(0.01)  # unchanged


# ── apply_overrides edge cases ───────────────────────────────────────────────

def test_apply_overrides_none_config():
    t = _trace(outcome="success")
    apply_overrides([t], None)
    assert t.outcome == "success"


def test_apply_overrides_no_overrides_configured():
    t = _trace(outcome="success")
    src = _source_config()  # no outcome or cost overrides
    apply_overrides([t], src)
    assert t.outcome == "success"


def test_apply_overrides_both_outcome_and_cost():
    span = _span("llm", cost=0.01, extra_attrs={"my.cost": 0.99})
    t = Trace(
        trace_id="t1",
        spans=[span],
        outcome=None,
        metadata={"status": "pass"},
    )
    src = _source_config(
        outcome={"field": "status", "success": ["pass"]},
        cost={"attr": "my.cost"},
    )
    apply_overrides([t], src)
    assert t.outcome == "success"
    assert span_cost(t.spans[0]) == pytest.approx(0.99)
