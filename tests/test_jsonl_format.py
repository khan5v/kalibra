"""Tests for JSONL loading — nested traces, minimal traces, field mapping, errors."""

from __future__ import annotations

import json

import pytest

from kalibra.loader import load_traces

# ── helpers ──────────────────────────────────────────────────────────────────

def _write(tmp_path, lines: list[dict], filename: str = "traces.jsonl") -> str:
    p = tmp_path / filename
    p.write_text("\n".join(json.dumps(row) for row in lines) + "\n")
    return str(p)


# ── minimal traces (no spans) ───────────────────────────────────────────────

class TestMinimalTraces:
    def test_minimal(self, tmp_path):
        f = _write(tmp_path, [{"trace_id": "t1", "outcome": "success"}])
        traces = load_traces(f)
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert traces[0].outcome == "success"
        assert traces[0].spans == []  # no synthetic span

    def test_with_cost_and_tokens(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "cost": 0.012, "input_tokens": 500, "output_tokens": 200,
        }])
        traces = load_traces(f)
        assert traces[0].total_cost == pytest.approx(0.012)
        assert traces[0].total_tokens == 700
        assert traces[0].spans == []

    def test_with_duration(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success", "duration_s": 5.0,
        }])
        traces = load_traces(f)
        assert traces[0].duration == pytest.approx(5.0)

    def test_with_metadata(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "metadata": {"source": "custom", "team": "ml"},
        }])
        traces = load_traces(f)
        assert traces[0].metadata["source"] == "custom"
        assert traces[0].metadata["team"] == "ml"

    def test_null_outcome(self, tmp_path):
        f = _write(tmp_path, [{"trace_id": "t1", "outcome": None}])
        traces = load_traces(f)
        assert traces[0].outcome is None

    def test_no_outcome(self, tmp_path):
        f = _write(tmp_path, [{"trace_id": "t1"}])
        traces = load_traces(f)
        assert traces[0].outcome is None

    def test_multiple_rows(self, tmp_path):
        f = _write(tmp_path, [
            {"trace_id": "t1", "outcome": "success", "cost": 0.01},
            {"trace_id": "t2", "outcome": "failure", "cost": 0.02},
            {"trace_id": "t3", "outcome": "success", "cost": 0.005},
        ])
        traces = load_traces(f)
        assert len(traces) == 3
        assert [t.outcome for t in traces] == ["success", "failure", "success"]

    def test_unknown_fields_go_to_metadata(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "custom_score": 0.95, "evaluator": "gpt-4o",
        }])
        traces = load_traces(f)
        assert traces[0].metadata["custom_score"] == 0.95
        assert traces[0].metadata["evaluator"] == "gpt-4o"

    def test_iso_timestamps(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "start_time": "2026-01-01T10:00:00Z",
            "end_time": "2026-01-01T10:00:05Z",
        }])
        traces = load_traces(f)
        assert traces[0].duration == pytest.approx(5.0)


# ── nested span format ──────────────────────────────────────────────────────

class TestNestedSpans:
    def test_trace_with_spans(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "outcome": "success",
            "spans": [
                {"span_id": "s1", "name": "plan", "cost": 0.01,
                 "start_ns": 0, "end_ns": 3_000_000_000},
                {"span_id": "s2", "name": "tool", "parent_id": "s1",
                 "start_ns": 3_000_000_000, "end_ns": 5_000_000_000},
            ],
        }])
        traces = load_traces(f)
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert traces[0].outcome == "success"
        assert len(traces[0].spans) == 2
        assert traces[0].spans[0].name == "plan"
        assert traces[0].spans[0].cost == pytest.approx(0.01)
        assert traces[0].spans[1].parent_id == "s1"

    def test_multiple_traces_with_spans(self, tmp_path):
        f = _write(tmp_path, [
            {"trace_id": "t1", "outcome": "success",
             "spans": [{"span_id": "s1", "name": "a"}]},
            {"trace_id": "t2", "outcome": "failure",
             "spans": [{"span_id": "s2", "name": "b"}]},
        ])
        traces = load_traces(f)
        assert len(traces) == 2
        assert traces[0].trace_id == "t1"
        assert traces[1].outcome == "failure"

    def test_spans_sorted_by_start_time(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "spans": [
                {"span_id": "late", "name": "b", "start_ns": 10_000_000_000},
                {"span_id": "early", "name": "a", "start_ns": 1_000_000_000},
            ],
        }])
        traces = load_traces(f)
        assert traces[0].spans[0].name == "a"
        assert traces[0].spans[1].name == "b"

    def test_span_tokens_and_error(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "spans": [
                {"span_id": "s1", "name": "llm",
                 "input_tokens": 500, "output_tokens": 200, "error": True},
            ],
        }])
        traces = load_traces(f)
        s = traces[0].spans[0]
        assert s.input_tokens == 500
        assert s.output_tokens == 200
        assert s.error is True
        assert s.total_tokens == 700


# ── field mapping ───────────────────────────────────────────────────────────

class TestFieldMapping:
    def test_custom_trace_id(self, tmp_path):
        f = _write(tmp_path, [{"uuid": "abc", "outcome": "success"}])
        traces = load_traces(f, trace_id_field="uuid")
        assert traces[0].trace_id == "abc"

    def test_outcome_from_metadata(self, tmp_path):
        from kalibra.config import FieldsConfig
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "metadata": {"result": "success"},
        }])
        traces = load_traces(f, fields=FieldsConfig(outcome="result"))
        assert traces[0].outcome == "success"

    def test_outcome_boolean(self, tmp_path):
        from kalibra.config import FieldsConfig
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "metadata": {"passed": True},
        }])
        traces = load_traces(f, fields=FieldsConfig(outcome="passed"))
        assert traces[0].outcome == "success"

    def test_cost_from_attributes(self, tmp_path):
        from kalibra.config import FieldsConfig
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "attributes": {"custom.cost": 0.99},
        }])
        traces = load_traces(f, fields=FieldsConfig(cost="custom.cost"))
        assert traces[0].total_cost == pytest.approx(0.99)

    def test_tokens_from_nested_json(self, tmp_path):
        """JSON strings are auto-parsed, then field mapping resolves dot-paths."""
        from kalibra.config import FieldsConfig
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "key_stats": '{"input_tokens": 500, "output_tokens": 200}',
        }])
        traces = load_traces(f, fields=FieldsConfig(
            input_tokens="key_stats.input_tokens",
            output_tokens="key_stats.output_tokens",
        ))
        assert traces[0].total_tokens == 700

    def test_task_id_mapping(self, tmp_path):
        from kalibra.config import FieldsConfig
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "metadata": {"instance_id": "django-123"},
        }])
        traces = load_traces(f, fields=FieldsConfig(task_id="instance_id"))
        assert traces[0].metadata["task_id"] == "django-123"


# ── error messages ──────────────────────────────────────────────────────────

class TestErrors:
    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="malformed"):
            load_traces(str(p))

    def test_missing_trace_id_is_empty(self, tmp_path):
        """When no trace_id field exists, trace_id is empty string."""
        p = tmp_path / "no_id.jsonl"
        p.write_text('{"outcome": "success"}\n{"outcome": "failure"}\n')
        traces = load_traces(str(p))
        assert len(traces) == 2
        assert traces[0].trace_id == ""
        assert traces[1].trace_id == ""

    def test_missing_trace_id_does_not_guess(self, tmp_path):
        """Loader should NOT guess uuid as trace_id — only use configured field."""
        p = tmp_path / "has_uuid.jsonl"
        p.write_text('{"uuid": "abc", "outcome": "success"}\n')
        traces = load_traces(str(p))
        assert traces[0].trace_id == ""  # empty, not "abc"

    def test_custom_trace_id_field(self, tmp_path):
        f = _write(tmp_path, [{"uuid": "t1", "outcome": "success"}])
        traces = load_traces(f, trace_id_field="uuid")
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"

    def test_error_includes_line_number(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"trace_id": "t1"}\nnot json\n')
        with pytest.raises(ValueError, match="2"):
            load_traces(str(p))

    def test_error_includes_file_path(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="bad.jsonl"):
            load_traces(str(p))

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        traces = load_traces(str(p))
        assert traces == []

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "sparse.jsonl"
        p.write_text('\n{"trace_id": "t1"}\n\n{"trace_id": "t2"}\n\n')
        traces = load_traces(str(p))
        assert len(traces) == 2


# ── attributes handling ─────────────────────────────────────────────────────

class TestAttributes:
    def test_attributes_dict_merged_into_metadata(self, tmp_path):
        """For span-less traces, attributes go to metadata."""
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "attributes": {"custom.field": "hello"},
        }])
        traces = load_traces(f)
        assert traces[0].metadata["custom.field"] == "hello"

    def test_nested_dict_flattened_to_metadata(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "agent_cost": {"total_cost": 0.5, "total_tokens": 1000},
        }])
        traces = load_traces(f)
        assert traces[0].metadata["agent_cost.total_cost"] == 0.5
        assert traces[0].metadata["agent_cost.total_tokens"] == 1000

    def test_json_string_auto_parsed(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "stats": '{"tokens": 500, "cost": 0.01}',
        }])
        traces = load_traces(f)
        assert traces[0].metadata["stats.tokens"] == 500
        assert traces[0].metadata["stats.cost"] == 0.01

    def test_span_attributes_preserved(self, tmp_path):
        """For traces WITH spans, span attributes stay on spans."""
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "spans": [{"span_id": "s1", "name": "llm",
                        "attributes": {"model.name": "gpt-4o"}}],
        }])
        traces = load_traces(f)
        assert traces[0].spans[0].attributes["model.name"] == "gpt-4o"
