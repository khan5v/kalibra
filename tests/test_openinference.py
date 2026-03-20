"""Tests for OpenInference loader — detection, grouping, span conversion, finish reason."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS
from kalibra.openinference import (
    _extract_finish_reason,
    _flatten_attrs,
    _group_spans,
    _normalize_status,
    _resolve_attr,
    _safe_float,
    _safe_int,
    _to_span,
    is_openinference,
    load_openinference_json,
    load_openinference_jsonl,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _span(
    trace_id="t1",
    span_id="s1",
    parent_id=None,
    name="llm_call",
    span_kind="LLM",
    start_time="2026-01-01T10:00:00Z",
    end_time="2026-01-01T10:00:01Z",
    status_code="OK",
    attrs=None,
    **extra,
):
    """Build a minimal OpenInference span dict."""
    item = {
        "context": {"trace_id": trace_id, "span_id": span_id},
        "name": name,
        "span_kind": span_kind,
        "start_time": start_time,
        "end_time": end_time,
        "status_code": status_code,
        "attributes": attrs or {},
        **extra,
    }
    if parent_id is not None:
        item["parent_id"] = parent_id
    return item


def _write_jsonl(tmp_path, spans, filename="traces.jsonl"):
    p = tmp_path / filename
    p.write_text("\n".join(json.dumps(s) for s in spans) + "\n")
    return p


# ── is_openinference detection ───────────────────────────────────────────────

class TestIsOpenInference:
    def test_valid_top_level_span_kind(self):
        item = _span(span_kind="LLM")
        assert is_openinference(item) is True

    def test_valid_dot_flattened_attr(self):
        item = {
            "context": {"trace_id": "t1"},
            "attributes": {"openinference.span.kind": "AGENT"},
        }
        assert is_openinference(item) is True

    def test_valid_nested_attr(self):
        item = {
            "context": {"trace_id": "t1"},
            "attributes": {"openinference": {"span": {"kind": "TOOL"}}},
        }
        assert is_openinference(item) is True

    def test_rejects_missing_context(self):
        assert is_openinference({"span_kind": "LLM"}) is False

    def test_rejects_missing_trace_id(self):
        assert is_openinference({"context": {}, "span_kind": "LLM"}) is False

    def test_rejects_non_dict_context(self):
        assert is_openinference({"context": "string", "span_kind": "LLM"}) is False

    def test_rejects_unknown_span_kind(self):
        item = _span(span_kind="UNKNOWN_KIND")
        # Remove attributes to avoid dot-flattened match
        item["attributes"] = {}
        assert is_openinference(item) is False

    def test_structural_detection_unknown_kind(self):
        """UNKNOWN span_kind still detected via structural signal."""
        item = {
            "context": {"trace_id": "t1", "span_id": "s1"},
            "span_kind": "UNKNOWN",
            "parent_id": None,
            "attributes": {},
        }
        assert is_openinference(item) is True

    def test_structural_detection_null_kind(self):
        """None span_kind still detected via structural signal."""
        item = {
            "context": {"trace_id": "t1", "span_id": "s1"},
            "span_kind": None,
            "parent_id": None,
            "attributes": {},
        }
        assert is_openinference(item) is True

    def test_rejects_no_structural_signals(self):
        """No span_kind, no span_id in context, no parent_id → not OI."""
        item = {"context": {"trace_id": "t1"}, "attributes": {}}
        assert is_openinference(item) is False

    def test_all_known_span_kinds(self):
        for kind in ["LLM", "TOOL", "CHAIN", "AGENT", "RETRIEVER",
                      "RERANKER", "EMBEDDING", "GUARDRAIL", "EVALUATOR"]:
            item = _span(span_kind=kind)
            assert is_openinference(item) is True, f"Failed for {kind}"


# ── _group_spans — trace assembly ────────────────────────────────────────────

class TestGroupSpans:
    def test_single_trace_single_span(self):
        spans = [_span(trace_id="t1", span_id="s1")]
        traces = _group_spans(spans)
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert len(traces[0].spans) == 1

    def test_two_traces(self):
        spans = [
            _span(trace_id="t1", span_id="s1"),
            _span(trace_id="t2", span_id="s2"),
        ]
        traces = _group_spans(spans)
        assert len(traces) == 2
        trace_ids = {t.trace_id for t in traces}
        assert trace_ids == {"t1", "t2"}

    def test_multiple_spans_same_trace(self):
        spans = [
            _span(trace_id="t1", span_id="root", parent_id=None),
            _span(trace_id="t1", span_id="child", parent_id="root"),
        ]
        traces = _group_spans(spans)
        assert len(traces) == 1
        assert len(traces[0].spans) == 2

    def test_root_span_detected(self):
        """Root span is the one with no parent_id."""
        spans = [
            _span(trace_id="t1", span_id="child", parent_id="root",
                   start_time="2026-01-01T10:00:01Z"),
            _span(trace_id="t1", span_id="root", parent_id=None,
                   start_time="2026-01-01T10:00:00Z"),
        ]
        traces = _group_spans(spans)
        # Root's span_kind should be in metadata.
        assert traces[0].metadata.get("span_kind") is not None

    def test_multiple_roots_picks_earliest(self):
        """When two spans have no parent_id, pick the one with earlier start_time."""
        spans = [
            _span(trace_id="t1", span_id="late_root", parent_id=None,
                   start_time="2026-01-01T10:00:05Z", span_kind="CHAIN"),
            _span(trace_id="t1", span_id="early_root", parent_id=None,
                   start_time="2026-01-01T10:00:00Z", span_kind="AGENT"),
        ]
        traces = _group_spans(spans)
        assert traces[0].metadata["span_kind"] == "AGENT"

    def test_skips_non_dict_items(self):
        raw = [_span(), "not a dict", 42, None]
        traces = _group_spans(raw)
        assert len(traces) == 1

    def test_skips_items_without_context(self):
        raw = [_span(), {"no_context": True}]
        traces = _group_spans(raw)
        assert len(traces) == 1

    def test_skips_items_with_empty_trace_id(self):
        raw = [
            _span(trace_id="t1"),
            {
                "context": {"trace_id": "", "span_id": "s2"},
                "span_kind": "LLM",
            },
        ]
        traces = _group_spans(raw)
        assert len(traces) == 1

    def test_spans_sorted_by_start_time(self):
        spans = [
            _span(trace_id="t1", span_id="late",
                   start_time="2026-01-01T10:00:05Z"),
            _span(trace_id="t1", span_id="early",
                   start_time="2026-01-01T10:00:00Z"),
        ]
        traces = _group_spans(spans)
        assert traces[0].spans[0].span_id == "early"
        assert traces[0].spans[1].span_id == "late"

    def test_metadata_excludes_llm_and_oi_attrs(self):
        """Root span metadata should exclude llm.*, openinference.*, input.*, output.*."""
        spans = [_span(
            attrs={
                "llm.model_name": "claude-3",
                "openinference.span.kind": "LLM",
                "input.value": "long prompt text",
                "output.value": "long response text",
                "custom.field": "keep_this",
            },
        )]
        traces = _group_spans(spans)
        meta = traces[0].metadata
        assert "custom.field" in meta
        assert "llm.model_name" not in meta
        assert "openinference.span.kind" not in meta
        assert "input.value" not in meta
        assert "output.value" not in meta


# ── Outcome detection ────────────────────────────────────────────────────────

class TestOutcome:
    def test_success_from_otel_status(self):
        spans = [_span(status_code="OK")]
        traces = _group_spans(spans)
        assert traces[0].outcome == OUTCOME_SUCCESS

    def test_failure_from_otel_status(self):
        spans = [_span(status_code="ERROR")]
        traces = _group_spans(spans)
        assert traces[0].outcome == OUTCOME_FAILURE

    def test_failure_from_nested_status(self):
        """Phoenix JSON format: status.status_code instead of top-level."""
        s = _span()
        del s["status_code"]
        s["status"] = {"status_code": "ERROR"}
        traces = _group_spans([s])
        assert traces[0].outcome == OUTCOME_FAILURE

    def test_integer_status_codes(self):
        """OTel uses integers: 1=OK, 2=ERROR."""
        s_ok = _span(status_code=1)
        s_err = _span(trace_id="t2", span_id="s2", status_code=2)
        traces_ok = _group_spans([s_ok])
        traces_err = _group_spans([s_err])
        assert traces_ok[0].outcome == OUTCOME_SUCCESS
        assert traces_err[0].outcome == OUTCOME_FAILURE

    def test_finish_reason_overrides_status(self):
        """Finish reason from output.value takes priority over OTel status."""
        anthropic_response = json.dumps({"stop_reason": "max_tokens"})
        s = _span(
            status_code="OK",
            attrs={"output.value": anthropic_response},
        )
        traces = _group_spans([s])
        assert traces[0].outcome == OUTCOME_FAILURE

    def test_finish_reason_scans_all_spans(self):
        """Root may be CHAIN — finish_reason lives on child LLM spans."""
        truncated = json.dumps({"stop_reason": "max_tokens"})
        spans = [
            _span(trace_id="t1", span_id="root", span_kind="CHAIN",
                   parent_id=None, attrs={}),
            _span(trace_id="t1", span_id="child", span_kind="LLM",
                   parent_id="root", attrs={"output.value": truncated}),
        ]
        traces = _group_spans(spans)
        assert traces[0].outcome == OUTCOME_FAILURE

    def test_all_complete_means_success(self):
        complete = json.dumps({"stop_reason": "end_turn"})
        spans = [
            _span(trace_id="t1", span_id="s1", attrs={"output.value": complete}),
            _span(trace_id="t1", span_id="s2", parent_id="s1",
                   attrs={"output.value": complete}),
        ]
        traces = _group_spans(spans)
        assert traces[0].outcome == OUTCOME_SUCCESS

    def test_none_outcome_when_no_signals(self):
        s = _span(status_code="UNSET")
        s.pop("status_code")
        traces = _group_spans([s])
        assert traces[0].outcome is None


# ── _extract_finish_reason ───────────────────────────────────────────────────

class TestFinishReason:
    def test_anthropic_end_turn(self):
        attrs = {"output.value": json.dumps({"stop_reason": "end_turn"})}
        assert _extract_finish_reason(attrs) == "complete"

    def test_anthropic_max_tokens(self):
        attrs = {"output.value": json.dumps({"stop_reason": "max_tokens"})}
        assert _extract_finish_reason(attrs) == "truncated"

    def test_anthropic_tool_use(self):
        attrs = {"output.value": json.dumps({"stop_reason": "tool_use"})}
        assert _extract_finish_reason(attrs) == "tool_call"

    def test_openai_stop(self):
        output = {"choices": [{"finish_reason": "stop"}]}
        attrs = {"output.value": json.dumps(output)}
        assert _extract_finish_reason(attrs) == "complete"

    def test_openai_length(self):
        output = {"choices": [{"finish_reason": "length"}]}
        attrs = {"output.value": json.dumps(output)}
        assert _extract_finish_reason(attrs) == "truncated"

    def test_openai_content_filter(self):
        output = {"choices": [{"finish_reason": "content_filter"}]}
        attrs = {"output.value": json.dumps(output)}
        assert _extract_finish_reason(attrs) == "filtered"

    def test_google_stop(self):
        output = {"candidates": [{"finishReason": "STOP"}]}
        attrs = {"output.value": json.dumps(output)}
        assert _extract_finish_reason(attrs) == "complete"

    def test_google_max_tokens(self):
        output = {"candidates": [{"finishReason": "MAX_TOKENS"}]}
        attrs = {"output.value": json.dumps(output)}
        assert _extract_finish_reason(attrs) == "truncated"

    def test_google_safety(self):
        output = {"candidates": [{"finishReason": "SAFETY"}]}
        attrs = {"output.value": json.dumps(output)}
        assert _extract_finish_reason(attrs) == "filtered"

    def test_no_output_value(self):
        assert _extract_finish_reason({}) is None

    def test_non_json_output(self):
        attrs = {"output.value": "plain text response"}
        assert _extract_finish_reason(attrs) is None

    def test_output_is_list_not_dict(self):
        attrs = {"output.value": json.dumps([1, 2, 3])}
        assert _extract_finish_reason(attrs) is None

    def test_unknown_stop_reason(self):
        attrs = {"output.value": json.dumps({"stop_reason": "unknown_reason"})}
        assert _extract_finish_reason(attrs) is None

    def test_nested_output_value(self):
        """output.value resolved via nested dict."""
        attrs = {"output": {"value": json.dumps({"stop_reason": "end_turn"})}}
        assert _extract_finish_reason(attrs) == "complete"


# ── _to_span ─────────────────────────────────────────────────────────────────

class TestToSpan:
    def test_basic_conversion(self):
        raw = _span(
            span_id="s1", name="generate",
            start_time="2026-01-01T10:00:00Z",
            end_time="2026-01-01T10:00:02Z",
            attrs={
                "llm.token_count.prompt": 500,
                "llm.token_count.completion": 200,
                "llm.cost.total": 0.05,
                "llm.model_name": "claude-3.5-sonnet",
            },
        )
        span = _to_span(raw)
        assert span.span_id == "s1"
        assert span.name == "generate"
        assert span.input_tokens == 500
        assert span.output_tokens == 200
        assert span.cost == pytest.approx(0.05)
        assert span.model == "claude-3.5-sonnet"
        assert span.start_ns > 0
        assert span.end_ns > span.start_ns

    def test_parent_id_normalized(self):
        # Empty parent_id should become None.
        raw = _span(parent_id="")
        raw["parent_id"] = ""
        span = _to_span(raw)
        assert span.parent_id is None

    def test_none_parent_id(self):
        raw = _span()
        span = _to_span(raw)
        assert span.parent_id is None

    def test_real_parent_id(self):
        raw = _span(parent_id="parent_span")
        span = _to_span(raw)
        assert span.parent_id == "parent_span"

    def test_error_from_status(self):
        raw = _span(status_code="ERROR")
        span = _to_span(raw)
        assert span.error is True

    def test_no_error(self):
        raw = _span(status_code="OK")
        span = _to_span(raw)
        assert span.error is False

    def test_fallback_total_tokens_to_output(self):
        """When only total token count exists, assign to output_tokens."""
        raw = _span(attrs={"llm.token_count.total": 700})
        span = _to_span(raw)
        assert span.input_tokens is None
        assert span.output_tokens == 700

    def test_name_falls_back_to_span_kind(self):
        raw = _span(name=None, span_kind="LLM")
        raw.pop("name")
        span = _to_span(raw)
        assert span.name == "LLM"

    def test_nested_attrs_resolved(self):
        raw = _span(attrs={
            "llm": {
                "token_count": {"prompt": 100, "completion": 50},
                "cost": {"total": 0.01},
                "model_name": "gpt-4o",
            },
        })
        span = _to_span(raw)
        assert span.input_tokens == 100
        assert span.output_tokens == 50
        assert span.cost == pytest.approx(0.01)
        assert span.model == "gpt-4o"


# ── _resolve_attr ────────────────────────────────────────────────────────────

class TestResolveAttr:
    def test_flat_key(self):
        attrs = {"llm.cost.total": 0.05}
        assert _resolve_attr(attrs, "llm.cost.total") == 0.05

    def test_nested_key(self):
        attrs = {"llm": {"cost": {"total": 0.05}}}
        assert _resolve_attr(attrs, "llm.cost.total") == 0.05

    def test_flat_takes_priority(self):
        attrs = {
            "llm.cost.total": 0.05,
            "llm": {"cost": {"total": 0.99}},
        }
        assert _resolve_attr(attrs, "llm.cost.total") == 0.05

    def test_missing_returns_none(self):
        assert _resolve_attr({}, "llm.cost.total") is None

    def test_partial_path_returns_none(self):
        attrs = {"llm": {"cost": {}}}
        assert _resolve_attr(attrs, "llm.cost.total") is None


# ── _normalize_status ────────────────────────────────────────────────────────

class TestNormalizeStatus:
    def test_string_ok(self):
        assert _normalize_status("OK") == "OK"

    def test_string_error(self):
        assert _normalize_status("ERROR") == "ERROR"

    def test_mixed_case(self):
        assert _normalize_status("Ok") == "OK"

    def test_int_ok(self):
        assert _normalize_status(1) == "OK"

    def test_int_error(self):
        assert _normalize_status(2) == "ERROR"

    def test_int_unset(self):
        assert _normalize_status(0) == ""

    def test_none(self):
        assert _normalize_status(None) == ""

    def test_unknown_int(self):
        assert _normalize_status(99) == ""


# ── _safe_float / _safe_int ──────────────────────────────────────────────────

class TestSafeConversions:
    def test_float_normal(self):
        assert _safe_float(0.05) == 0.05

    def test_float_string(self):
        assert _safe_float("0.05") == 0.05

    def test_float_none(self):
        assert _safe_float(None) is None

    def test_float_invalid(self):
        assert _safe_float("not_a_number") is None

    def test_int_normal(self):
        assert _safe_int(500) == 500

    def test_int_from_float(self):
        assert _safe_int(500.0) == 500

    def test_int_from_string(self):
        assert _safe_int("500") == 500

    def test_int_none(self):
        assert _safe_int(None) is None

    def test_int_invalid(self):
        assert _safe_int("nope") is None


# ── _flatten_attrs ───────────────────────────────────────────────────────────

class TestFlattenAttrs:
    def test_flat_passthrough(self):
        out = {}
        _flatten_attrs({"a": 1, "b": "x"}, "", out)
        assert out == {"a": 1, "b": "x"}

    def test_nested(self):
        out = {}
        _flatten_attrs({"a": {"b": {"c": 42}}}, "", out)
        assert out == {"a.b.c": 42}

    def test_skips_none_and_lists(self):
        out = {}
        _flatten_attrs({"a": None, "b": [1, 2], "c": 3}, "", out)
        assert out == {"c": 3}


# ── JSONL loader integration ─────────────────────────────────────────────────

class TestLoadJsonl:
    def test_load_single_trace(self, tmp_path):
        spans = [
            _span(trace_id="t1", span_id="root"),
            _span(trace_id="t1", span_id="child", parent_id="root"),
        ]
        path = _write_jsonl(tmp_path, spans)
        traces = load_openinference_jsonl(path)
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert len(traces[0].spans) == 2

    def test_load_multiple_traces(self, tmp_path):
        spans = [
            _span(trace_id="t1", span_id="s1"),
            _span(trace_id="t2", span_id="s2"),
            _span(trace_id="t1", span_id="s3", parent_id="s1"),
        ]
        path = _write_jsonl(tmp_path, spans)
        traces = load_openinference_jsonl(path)
        assert len(traces) == 2

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "traces.jsonl"
        lines = [
            "",
            json.dumps(_span(trace_id="t1")),
            "",
            json.dumps(_span(trace_id="t2")),
            "",
        ]
        p.write_text("\n".join(lines))
        traces = load_openinference_jsonl(p)
        assert len(traces) == 2

    def test_skips_bad_json(self, tmp_path):
        p = tmp_path / "traces.jsonl"
        lines = [
            json.dumps(_span(trace_id="t1")),
            "not json",
            json.dumps(_span(trace_id="t2")),
        ]
        p.write_text("\n".join(lines))
        traces = load_openinference_jsonl(p)
        assert len(traces) == 2


# ── JSON array loader integration ────────────────────────────────────────────

class TestLoadJsonArray:
    def test_load_from_array(self):
        data = [
            _span(trace_id="t1", span_id="s1"),
            _span(trace_id="t1", span_id="s2", parent_id="s1"),
        ]
        traces = load_openinference_json(data)
        assert len(traces) == 1
        assert len(traces[0].spans) == 2

    def test_empty_array(self):
        traces = load_openinference_json([])
        assert traces == []


# ── Integration via load_traces (auto-detection) ────────────────────────────

class TestAutoDetection:
    def test_jsonl_autodetect(self, tmp_path):
        """OpenInference JSONL is auto-detected by load_traces."""
        from kalibra.loader import load_traces
        spans = [
            _span(trace_id="t1", span_id="root", span_kind="CHAIN"),
            _span(trace_id="t1", span_id="child", span_kind="LLM",
                   parent_id="root",
                   attrs={
                       "llm.token_count.prompt": 100,
                       "llm.token_count.completion": 50,
                       "llm.cost.total": 0.01,
                   }),
        ]
        path = _write_jsonl(tmp_path, spans)
        traces = load_traces(str(path))
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert len(traces[0].spans) == 2
        # Cost/tokens come from the LLM span.
        assert traces[0].total_cost == pytest.approx(0.01)
        assert traces[0].total_tokens == 150

    def test_json_array_autodetect(self, tmp_path):
        """OpenInference JSON array is auto-detected by load_traces."""
        from kalibra.loader import load_traces
        data = [
            _span(trace_id="t1", span_id="root"),
            _span(trace_id="t1", span_id="child", parent_id="root"),
        ]
        p = tmp_path / "traces.json"
        p.write_text(json.dumps(data))
        traces = load_traces(str(p))
        assert len(traces) == 1
        assert len(traces[0].spans) == 2


# ── Leaf spans and tree aggregation ──────────────────────────────────────────

class TestLeafSpans:
    """Verify leaf_spans() and that metrics don't double-count in tree traces."""

    def test_leaf_spans_flat_trace(self):
        """All spans are leaves when none have children."""
        spans = [
            _span(trace_id="t1", span_id="s1"),
            _span(trace_id="t1", span_id="s2"),
        ]
        traces = _group_spans(spans)
        leaves = traces[0].leaf_spans()
        assert len(leaves) == 2

    def test_leaf_spans_tree_trace(self):
        """CHAIN root with 3 LLM children → 3 leaves, not 4."""
        spans = [
            _span(trace_id="t1", span_id="root", span_kind="CHAIN"),
            _span(trace_id="t1", span_id="plan", span_kind="LLM", parent_id="root"),
            _span(trace_id="t1", span_id="tool", span_kind="TOOL", parent_id="root"),
            _span(trace_id="t1", span_id="respond", span_kind="LLM", parent_id="root"),
        ]
        traces = _group_spans(spans)
        assert len(traces[0].spans) == 4
        leaves = traces[0].leaf_spans()
        assert len(leaves) == 3
        leaf_ids = {s.span_id for s in leaves}
        assert "root" not in leaf_ids

    def test_leaf_spans_deep_tree(self):
        """root → step1 → (step1.1, step1.2) — only step1.1, step1.2 are leaves."""
        spans = [
            _span(trace_id="t1", span_id="root", span_kind="AGENT"),
            _span(trace_id="t1", span_id="step1", span_kind="CHAIN", parent_id="root"),
            _span(trace_id="t1", span_id="step1.1", span_kind="LLM", parent_id="step1"),
            _span(trace_id="t1", span_id="step1.2", span_kind="TOOL", parent_id="step1"),
        ]
        traces = _group_spans(spans)
        leaves = traces[0].leaf_spans()
        assert len(leaves) == 2
        leaf_ids = {s.span_id for s in leaves}
        assert leaf_ids == {"step1.1", "step1.2"}

    def test_cost_not_double_counted(self):
        """CHAIN root (no cost) + 2 LLM children (with cost) → sum of children only."""
        spans = [
            _span(trace_id="t1", span_id="root", span_kind="CHAIN",
                   attrs={}),  # No cost on orchestration span
            _span(trace_id="t1", span_id="llm1", span_kind="LLM", parent_id="root",
                   attrs={"llm.cost.total": 0.05}),
            _span(trace_id="t1", span_id="llm2", span_kind="LLM", parent_id="root",
                   attrs={"llm.cost.total": 0.03}),
        ]
        traces = _group_spans(spans)
        assert traces[0].total_cost == pytest.approx(0.08)

    def test_tokens_not_double_counted(self):
        """Same as cost — orchestration span has None tokens, only LLM spans counted."""
        spans = [
            _span(trace_id="t1", span_id="root", span_kind="CHAIN", attrs={}),
            _span(trace_id="t1", span_id="llm1", span_kind="LLM", parent_id="root",
                   attrs={"llm.token_count.prompt": 100, "llm.token_count.completion": 50}),
            _span(trace_id="t1", span_id="llm2", span_kind="LLM", parent_id="root",
                   attrs={"llm.token_count.prompt": 200, "llm.token_count.completion": 100}),
        ]
        traces = _group_spans(spans)
        assert traces[0].total_tokens == 450

    def test_steps_metric_counts_leaves(self):
        """Steps metric should count leaf spans, not total spans."""
        from kalibra.metrics.steps import StepsMetric

        # Build two traces: CHAIN root + 3 LLM children each.
        all_spans = []
        for tid in ["t1", "t2"]:
            all_spans.append(
                _span(trace_id=tid, span_id=f"{tid}_root", span_kind="CHAIN"))
            for i in range(3):
                all_spans.append(
                    _span(trace_id=tid, span_id=f"{tid}_s{i}", span_kind="LLM",
                           parent_id=f"{tid}_root"))

        traces = _group_spans(all_spans)
        assert len(traces) == 2

        metric = StepsMetric()
        obs = metric.compare(traces, traces)
        # Each trace has 4 spans but 3 leaves. Median should be 3, not 4.
        assert obs.baseline["median"] == 3.0
        assert obs.current["median"] == 3.0


# ── Real Phoenix fixture files ───────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"


class TestPhoenixFixtures:
    """Validate against real Phoenix trace exports."""

    def test_context_retrieval_nested_attrs(self):
        """Real LlamaIndex RAG traces — nested dict attributes."""
        path = FIXTURES / "phoenix_context_retrieval.jsonl"
        if not path.exists():
            pytest.skip("fixture not downloaded")
        from kalibra.loader import load_traces
        traces = load_traces(str(path))
        assert len(traces) == 4
        for t in traces:
            assert len(t.spans) == 5
            assert len(t.leaf_spans()) == 2
            assert t.total_tokens is not None
            assert t.total_tokens > 0
            assert t.duration is not None
            assert t.duration > 0
            assert t.outcome == "success"

    def test_random_traces_dot_flattened(self):
        """Synthetic Phoenix traces — dot-flattened attributes, mixed span kinds."""
        path = FIXTURES / "phoenix_random.jsonl"
        if not path.exists():
            pytest.skip("fixture not downloaded")
        from kalibra.loader import load_traces
        traces = load_traces(str(path))
        assert len(traces) == 50
        # Should have grouped 198 spans into 50 traces.
        total_spans = sum(len(t.spans) for t in traces)
        assert total_spans == 198
        # Some traces should have token data (LLM spans).
        has_tokens = sum(1 for t in traces if t.total_tokens is not None)
        assert has_tokens > 0
        # All traces should have duration (all have timestamps).
        has_dur = sum(1 for t in traces if t.duration is not None)
        assert has_dur == 50


# ── Adversarial / poison payloads ────────────────────────────────────────────

class TestPoisonPayloads:
    """Every field null, empty, zero, or missing. Must not crash."""

    def test_all_nulls(self):
        """Span with every attribute null."""
        raw = {
            "context": {"trace_id": "t1", "span_id": None},
            "name": None,
            "span_kind": "LLM",
            "parent_id": None,
            "start_time": None,
            "end_time": None,
            "status_code": None,
            "attributes": {
                "llm.token_count.prompt": None,
                "llm.token_count.completion": None,
                "llm.cost.total": None,
                "llm.model_name": None,
                "output.value": None,
            },
        }
        traces = _group_spans([raw])
        assert len(traces) == 1
        span = traces[0].spans[0]
        assert span.cost is None
        assert span.input_tokens is None
        assert span.output_tokens is None
        assert span.model is None

    def test_all_empty_strings(self):
        raw = {
            "context": {"trace_id": "t1", "span_id": ""},
            "name": "",
            "span_kind": "LLM",
            "parent_id": "",
            "start_time": "",
            "end_time": "",
            "status_code": "",
            "attributes": {
                "llm.token_count.prompt": "",
                "llm.token_count.completion": "",
                "llm.cost.total": "",
                "llm.model_name": "",
            },
        }
        traces = _group_spans([raw])
        assert len(traces) == 1
        span = traces[0].spans[0]
        # Empty strings should not become 0 — they fail conversion → None.
        assert span.cost is None
        assert span.input_tokens is None
        assert span.output_tokens is None
        # parent_id "" should normalize to None (root span).
        assert span.parent_id is None

    def test_all_zeros(self):
        raw = {
            "context": {"trace_id": "t1", "span_id": "s1"},
            "name": "zero_span",
            "span_kind": "LLM",
            "start_time": "2026-01-01T10:00:00Z",
            "end_time": "2026-01-01T10:00:00Z",
            "status_code": 0,
            "attributes": {
                "llm.token_count.prompt": 0,
                "llm.token_count.completion": 0,
                "llm.cost.total": 0.0,
            },
        }
        traces = _group_spans([raw])
        span = traces[0].spans[0]
        # Zero is valid — 0 means "measured as zero", not None.
        assert span.cost == 0.0
        assert span.input_tokens == 0
        assert span.output_tokens == 0

    def test_minimal_fields_only(self):
        """Absolute minimum: just context with trace_id."""
        raw = {
            "context": {"trace_id": "t1"},
            "span_kind": "LLM",
        }
        traces = _group_spans([raw])
        assert len(traces) == 1
        span = traces[0].spans[0]
        assert span.span_id == ""
        assert span.cost is None
        assert span.start_ns == 0

    def test_mixed_poison(self):
        """Some fields null, some empty, some missing, some valid."""
        spans = [
            {
                "context": {"trace_id": "t1", "span_id": "s1"},
                "span_kind": "CHAIN",
                "name": "root",
                "start_time": "2026-01-01T10:00:00Z",
                "end_time": None,
                "parent_id": "",
                "status_code": "OK",
                "attributes": {},
            },
            {
                "context": {"trace_id": "t1", "span_id": "s2"},
                "span_kind": "LLM",
                "parent_id": "s1",
                "attributes": {
                    "llm.token_count.prompt": 500,
                    "llm.token_count.completion": None,
                    "llm.cost.total": 0.05,
                    "output.value": json.dumps({"stop_reason": "end_turn"}),
                },
            },
        ]
        traces = _group_spans(spans)
        assert len(traces) == 1
        assert traces[0].outcome == OUTCOME_SUCCESS

        llm_span = [s for s in traces[0].spans if s.name != "root"][0]
        assert llm_span.input_tokens == 500
        assert llm_span.output_tokens is None
        assert llm_span.cost == pytest.approx(0.05)
