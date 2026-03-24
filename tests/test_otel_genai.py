"""Tests for OTel GenAI loader — detection, span conversion, finish reasons."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalibra.loader import load_traces
from kalibra.loaders.otel_genai import (
    OTelGenAILoader,
    _build_traces,
    _determine_outcome,
    _get_finish_reasons,
    _to_span,
)
from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _span(
    trace_id="t1", span_id="s1", parent_id=None, name="chat gpt-4o",
    attrs=None, start_time="2026-01-01T00:00:00Z",
    end_time="2026-01-01T00:00:01Z", status_code="OK",
):
    """Build a minimal OTel GenAI span dict."""
    base_attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o",
    }
    if attrs:
        base_attrs.update(attrs)
    return {
        "context": {"trace_id": trace_id, "span_id": span_id},
        "parent_id": parent_id,
        "name": name,
        "start_time": start_time,
        "end_time": end_time,
        "status_code": status_code,
        "attributes": base_attrs,
    }


def _write_jsonl(tmp_path: Path, spans: list[dict]) -> Path:
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in spans) + "\n")
    return path


# ── Detection ────────────────────────────────────────────────────────────────

class TestDetection:
    def test_detects_genai_span(self):
        fmt = OTelGenAILoader()
        item = _span()
        assert fmt.detect(item) is True

    def test_detects_with_system_only(self):
        fmt = OTelGenAILoader()
        item = _span(attrs={"gen_ai.system": "anthropic"})
        assert fmt.detect(item) is True

    def test_detects_with_model_only(self):
        fmt = OTelGenAILoader()
        item = _span(attrs={"gen_ai.request.model": "claude-4.6"})
        assert fmt.detect(item) is True

    def test_rejects_no_genai_attrs(self):
        fmt = OTelGenAILoader()
        item = {"context": {"trace_id": "t1", "span_id": "s1"},
                "attributes": {"custom.key": "value"}}
        assert fmt.detect(item) is False

    def test_rejects_no_attributes(self):
        fmt = OTelGenAILoader()
        item = {"context": {"trace_id": "t1", "span_id": "s1"}}
        assert fmt.detect(item) is False

    def test_rejects_openinference_span(self):
        """Spans with openinference.span.kind should be handled by OI loader."""
        fmt = OTelGenAILoader()
        item = _span(attrs={
            "gen_ai.system": "openai",
            "openinference.span.kind": "LLM",
        })
        assert fmt.detect(item) is False

    def test_rejects_openinference_top_level_kind(self):
        fmt = OTelGenAILoader()
        item = _span()
        item["span_kind"] = "LLM"
        assert fmt.detect(item) is False

    def test_detects_nested_attrs(self):
        """Nested dict attributes (gen_ai: {system: openai})."""
        fmt = OTelGenAILoader()
        item = {
            "context": {"trace_id": "t1", "span_id": "s1"},
            "attributes": {"gen_ai": {"system": "openai"}},
        }
        assert fmt.detect(item) is True


# ── Finish reasons ───────────────────────────────────────────────────────────

class TestFinishReasons:
    def test_array_stop(self):
        attrs = {"gen_ai.response.finish_reasons": ["stop"]}
        assert _get_finish_reasons(attrs) == ["stop"]

    def test_array_length(self):
        attrs = {"gen_ai.response.finish_reasons": ["length"]}
        assert _get_finish_reasons(attrs) == ["length"]

    def test_string_value(self):
        """Non-spec but defensive — handle bare string."""
        attrs = {"gen_ai.response.finish_reasons": "stop"}
        assert _get_finish_reasons(attrs) == ["stop"]

    def test_json_string(self):
        """JSON-encoded array."""
        attrs = {"gen_ai.response.finish_reasons": '["stop"]'}
        assert _get_finish_reasons(attrs) == ["stop"]

    def test_missing(self):
        assert _get_finish_reasons({}) is None

    def test_none(self):
        attrs = {"gen_ai.response.finish_reasons": None}
        assert _get_finish_reasons(attrs) is None


# ── Outcome detection ────────────────────────────────────────────────────────

class TestOutcome:
    def test_success_from_finish_reason(self):
        spans = [_span(attrs={"gen_ai.response.finish_reasons": ["stop"]})]
        assert _determine_outcome(spans, spans[0]) == OUTCOME_SUCCESS

    def test_failure_from_length(self):
        spans = [_span(attrs={"gen_ai.response.finish_reasons": ["length"]})]
        assert _determine_outcome(spans, spans[0]) == OUTCOME_FAILURE

    def test_failure_from_content_filter(self):
        spans = [_span(attrs={"gen_ai.response.finish_reasons": ["content_filter"]})]
        assert _determine_outcome(spans, spans[0]) == OUTCOME_FAILURE

    def test_any_failure_means_trace_failed(self):
        """If ANY span has a failure reason, the whole trace fails."""
        spans = [
            _span(span_id="s1", attrs={"gen_ai.response.finish_reasons": ["stop"]}),
            _span(span_id="s2", attrs={"gen_ai.response.finish_reasons": ["length"]}),
        ]
        assert _determine_outcome(spans, spans[0]) == OUTCOME_FAILURE

    def test_fallback_to_status_code(self):
        """No finish reasons — fall back to OTel status_code."""
        spans = [_span(status_code="ERROR")]
        assert _determine_outcome(spans, spans[0]) == OUTCOME_FAILURE

    def test_no_signals_returns_none(self):
        spans = [_span(attrs={"gen_ai.system": "openai"}, status_code="UNSET")]
        assert _determine_outcome(spans, spans[0]) is None


# ── Span conversion ──────────────────────────────────────────────────────────

class TestToSpan:
    def test_basic_conversion(self):
        raw = _span(attrs={
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
            "gen_ai.request.model": "gpt-4o",
        })
        span = _to_span(raw)
        assert span.input_tokens == 100
        assert span.output_tokens == 50
        assert span.model == "gpt-4o"
        assert span.cost is None  # no standard cost attr

    def test_parent_id(self):
        raw = _span(parent_id="parent1")
        span = _to_span(raw)
        assert span.parent_id == "parent1"

    def test_none_parent_id(self):
        raw = _span(parent_id=None)
        span = _to_span(raw)
        assert span.parent_id is None

    def test_empty_parent_id_normalized(self):
        raw = _span(parent_id="")
        span = _to_span(raw)
        assert span.parent_id is None

    def test_error_from_status(self):
        raw = _span(status_code="ERROR")
        span = _to_span(raw)
        assert span.error is True

    def test_name_from_span(self):
        raw = _span(name="chat gpt-4o")
        span = _to_span(raw)
        assert span.name == "chat gpt-4o"

    def test_name_fallback_to_operation(self):
        raw = _span(name="")
        span = _to_span(raw)
        assert "chat" in span.name


# ── Full loading ─────────────────────────────────────────────────────────────

class TestBuildTraces:
    def test_single_trace(self):
        spans = [
            _span(trace_id="t1", span_id="root", parent_id=None),
            _span(trace_id="t1", span_id="child", parent_id="root",
                  attrs={"gen_ai.usage.input_tokens": 200,
                         "gen_ai.usage.output_tokens": 100}),
        ]
        traces = _build_traces(spans)
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert len(traces[0].spans) == 2
        assert traces[0].total_tokens == 300  # child: 200 in + 100 out

    def test_two_traces(self):
        spans = [
            _span(trace_id="t1", span_id="s1"),
            _span(trace_id="t2", span_id="s2"),
        ]
        traces = _build_traces(spans)
        assert len(traces) == 2

    def test_metadata_from_root(self):
        spans = [_span(attrs={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-4.6",
            "custom.tag": "production",
        })]
        traces = _build_traces(spans)
        meta = traces[0].metadata
        assert meta.get("gen_ai.system") == "anthropic"
        assert meta.get("custom.tag") == "production"

    def test_metadata_skips_input_output_messages(self):
        """Large message blobs should not be in metadata."""
        spans = [_span(attrs={
            "gen_ai.input.messages": "[{huge blob}]",
            "gen_ai.output.messages": "[{huge blob}]",
            "gen_ai.system": "openai",
        })]
        traces = _build_traces(spans)
        meta = traces[0].metadata
        assert "gen_ai.input.messages" not in meta
        assert "gen_ai.output.messages" not in meta
        assert meta.get("gen_ai.system") == "openai"


# ── Integration via load_traces ──────────────────────────────────────────────

class TestAutoDetection:
    def test_auto_detects_otel_genai(self, tmp_path):
        spans = [
            _span(trace_id="t1", span_id="root", parent_id=None),
            _span(trace_id="t1", span_id="child", parent_id="root",
                  attrs={"gen_ai.usage.input_tokens": 100,
                         "gen_ai.usage.output_tokens": 50}),
        ]
        path = _write_jsonl(tmp_path, spans)
        traces = load_traces(str(path))
        assert len(traces) == 1
        assert len(traces[0].spans) == 2

    def test_explicit_format(self, tmp_path):
        spans = [_span()]
        path = _write_jsonl(tmp_path, spans)
        traces = load_traces(str(path), format="otel-genai")
        assert len(traces) == 1

    def test_openinference_not_detected_as_genai(self, tmp_path):
        """An OpenInference span should NOT be detected as OTel GenAI."""
        oi_span = {
            "context": {"trace_id": "t1", "span_id": "s1"},
            "parent_id": None,
            "span_kind": "LLM",
            "name": "messages.create",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T00:00:01Z",
            "status_code": "OK",
            "attributes": {
                "openinference.span.kind": "LLM",
                "llm.token_count.prompt": 100,
                "llm.token_count.completion": 50,
            },
        }
        path = _write_jsonl(tmp_path, [oi_span])
        # Auto-detect should pick OpenInference, not OTel GenAI.
        traces = load_traces(str(path))
        assert len(traces) == 1
        # Verify it was loaded via OpenInference (tokens from llm.* attrs).
        assert traces[0].total_tokens == 150


# ── Poison payloads ──────────────────────────────────────────────────────────

class TestPoisonPayloads:
    def test_all_nulls(self):
        """All gen_ai fields null — tokens None, outcome from status_code fallback."""
        spans = [_span(attrs={
            "gen_ai.usage.input_tokens": None,
            "gen_ai.usage.output_tokens": None,
            "gen_ai.request.model": None,
            "gen_ai.response.finish_reasons": None,
        })]
        traces = _build_traces(spans)
        assert len(traces) == 1
        assert traces[0].total_tokens is None
        # status_code="OK" in _span default → success via fallback
        assert traces[0].outcome == OUTCOME_SUCCESS

    def test_all_zeros(self):
        spans = [_span(attrs={
            "gen_ai.usage.input_tokens": 0,
            "gen_ai.usage.output_tokens": 0,
        })]
        traces = _build_traces(spans)
        assert traces[0].total_tokens == 0  # 0 is measured, not None

    def test_empty_finish_reasons(self):
        """Empty array = no data, not explicit success. Falls through to status_code."""
        spans = [_span(attrs={"gen_ai.response.finish_reasons": []})]
        traces = _build_traces(spans)
        # Default _span has status_code="OK" → success via fallback, not via empty array
        assert traces[0].outcome == OUTCOME_SUCCESS

    def test_empty_finish_reasons_no_status(self):
        """Empty array + no status_code → None (no signals)."""
        spans = [_span(attrs={"gen_ai.response.finish_reasons": []}, status_code="UNSET")]
        traces = _build_traces(spans)
        assert traces[0].outcome is None
