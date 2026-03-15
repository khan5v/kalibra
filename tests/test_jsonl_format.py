"""Tests for JSONL parsing — flat eval, flat spans, error messages."""

from __future__ import annotations

import json

import pytest

from kalibra.converters.base import span_cost, span_input_tokens, span_model, span_output_tokens
from kalibra.converters.generic import load_json_traces, save_jsonl


# ── helpers ──────────────────────────────────────────────────────────────────

def _write(tmp_path, lines: list[dict], filename: str = "traces.jsonl") -> str:
    p = tmp_path / filename
    p.write_text("\n".join(json.dumps(row) for row in lines) + "\n")
    return str(p)


# ── flat eval format ─────────────────────────────────────────────────────────

class TestFlatEval:
    def test_minimal(self, tmp_path):
        f = _write(tmp_path, [{"trace_id": "t1", "outcome": "success"}])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert traces[0].outcome == "success"
        assert len(traces[0].spans) == 1

    def test_with_cost_and_model(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "cost": 0.012, "model": "gpt-4o",
            "input_tokens": 500, "output_tokens": 200,
        }])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        s = traces[0].spans[0]
        assert span_cost(s) == pytest.approx(0.012)
        assert span_model(s) == "gpt-4o"
        assert span_input_tokens(s) == 500
        assert span_output_tokens(s) == 200

    def test_with_duration(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success", "duration_s": 5.0,
        }])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].duration == pytest.approx(5.0)

    def test_with_metadata(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "metadata": {"source": "custom", "team": "ml"},
        }])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].metadata["source"] == "custom"
        assert traces[0].metadata["team"] == "ml"

    def test_null_outcome(self, tmp_path):
        f = _write(tmp_path, [{"trace_id": "t1", "outcome": None}])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].outcome is None

    def test_no_outcome(self, tmp_path):
        f = _write(tmp_path, [{"trace_id": "t1"}])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].outcome is None

    def test_multiple_rows(self, tmp_path):
        f = _write(tmp_path, [
            {"trace_id": "t1", "outcome": "success", "cost": 0.01},
            {"trace_id": "t2", "outcome": "failure", "cost": 0.02},
            {"trace_id": "t3", "outcome": "success", "cost": 0.005},
        ])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert len(traces) == 3
        assert [t.outcome for t in traces] == ["success", "failure", "success"]

    def test_unknown_fields_become_attributes(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "custom_score": 0.95, "evaluator": "gpt-4o",
        }])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        attrs = dict(traces[0].spans[0].attributes)
        assert attrs["custom_score"] == 0.95
        assert attrs["evaluator"] == "gpt-4o"

    def test_span_name_defaults_to_eval(self, tmp_path):
        f = _write(tmp_path, [{"trace_id": "t1"}])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].spans[0].name == "eval"

    def test_custom_span_name(self, tmp_path):
        f = _write(tmp_path, [{"trace_id": "t1", "name": "my-agent"}])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].spans[0].name == "my-agent"

    def test_iso_timestamps(self, tmp_path):
        f = _write(tmp_path, [{
            "trace_id": "t1",
            "start_time": "2026-01-01T10:00:00Z",
            "end_time": "2026-01-01T10:00:05Z",
        }])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].duration == pytest.approx(5.0)


# ── flat span format ─────────────────────────────────────────────────────────

class TestFlatSpans:
    def test_single_trace_multiple_spans(self, tmp_path):
        f = _write(tmp_path, [
            {"trace_id": "t1", "span_id": "s1", "name": "planner",
             "parent_id": None, "outcome": "success",
             "start_time": "2026-01-01T10:00:00Z",
             "end_time": "2026-01-01T10:00:03Z",
             "model": "gpt-4o", "cost": 0.01},
            {"trace_id": "t1", "span_id": "s2", "name": "tool-call",
             "parent_id": "s1",
             "start_time": "2026-01-01T10:00:03Z",
             "end_time": "2026-01-01T10:00:05Z"},
        ])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert traces[0].outcome == "success"
        assert len(traces[0].spans) == 2
        assert traces[0].spans[0].name == "planner"
        assert traces[0].spans[1].name == "tool-call"

    def test_multiple_traces(self, tmp_path):
        f = _write(tmp_path, [
            {"trace_id": "t1", "span_id": "s1", "name": "agent", "outcome": "success"},
            {"trace_id": "t2", "span_id": "s2", "name": "agent", "outcome": "failure"},
        ])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert len(traces) == 2

    def test_parent_child_relationship(self, tmp_path):
        f = _write(tmp_path, [
            {"trace_id": "t1", "span_id": "root", "name": "agent", "parent_id": None},
            {"trace_id": "t1", "span_id": "child1", "name": "llm", "parent_id": "root"},
        ])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        root = traces[0].spans[0]
        child = traces[0].spans[1]
        assert root.parent is None
        assert child.parent is not None

    def test_outcome_from_first_row_with_it(self, tmp_path):
        """Outcome is taken from the first span row that has it."""
        f = _write(tmp_path, [
            {"trace_id": "t1", "span_id": "s1", "name": "a"},
            {"trace_id": "t1", "span_id": "s2", "name": "b", "outcome": "failure"},
        ])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].outcome == "failure"

    def test_spans_sorted_by_start_time(self, tmp_path):
        f = _write(tmp_path, [
            {"trace_id": "t1", "span_id": "late", "name": "b",
             "start_time": "2026-01-01T10:00:10Z", "end_time": "2026-01-01T10:00:15Z"},
            {"trace_id": "t1", "span_id": "early", "name": "a",
             "start_time": "2026-01-01T10:00:00Z", "end_time": "2026-01-01T10:00:05Z"},
        ])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        assert traces[0].spans[0].name == "a"
        assert traces[0].spans[1].name == "b"


# ── save + load round-trip ───────────────────────────────────────────────────

class TestRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path):
        from kalibra.converters.base import AF_COST, GEN_AI_MODEL, Trace, make_span

        spans = [
            make_span("planner", "t1", "0000000000000001", None,
                      1_000_000_000, 3_000_000_000,
                      {GEN_AI_MODEL: "gpt-4o", AF_COST: 0.01}),
            make_span("tool", "t1", "0000000000000002", "0000000000000001",
                      3_000_000_000, 5_000_000_000,
                      {AF_COST: 0.005}),
        ]
        original = [Trace("t1", spans, outcome="success",
                          metadata={"source": "test"})]

        out = str(tmp_path / "out.jsonl")
        save_jsonl(original, out)
        loaded = load_json_traces(tmp_path / "out.jsonl")

        assert len(loaded) == 1
        assert loaded[0].trace_id == "t1"
        assert loaded[0].outcome == "success"
        assert loaded[0].metadata["source"] == "test"
        assert len(loaded[0].spans) == 2
        assert span_model(loaded[0].spans[0]) == "gpt-4o"
        assert span_cost(loaded[0].spans[0]) == pytest.approx(0.01)
        assert span_cost(loaded[0].spans[1]) == pytest.approx(0.005)

    def test_save_writes_flat_spans(self, tmp_path):
        """Save should write one line per span, not one line per trace."""
        from kalibra.converters.base import Trace, make_span

        spans = [
            make_span("a", "t1", "0000000000000001", None, 0, int(1e9)),
            make_span("b", "t1", "0000000000000002", "0000000000000001", int(1e9), int(2e9)),
        ]
        traces = [Trace("t1", spans, outcome="success")]

        out = str(tmp_path / "out.jsonl")
        save_jsonl(traces, out)

        lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2  # one per span

        row1 = json.loads(lines[0])
        row2 = json.loads(lines[1])
        assert row1["trace_id"] == "t1"
        assert row1["outcome"] == "success"  # root span has outcome
        assert "outcome" not in row2  # non-root span doesn't

    def test_save_uses_friendly_names(self, tmp_path):
        from kalibra.converters.base import AF_COST, GEN_AI_MODEL, Trace, make_span

        spans = [make_span("a", "t1", "0000000000000001", None, 0, int(1e9),
                           {GEN_AI_MODEL: "gpt-4o", AF_COST: 0.01})]
        traces = [Trace("t1", spans)]

        out = str(tmp_path / "out.jsonl")
        save_jsonl(traces, out)

        row = json.loads((tmp_path / "out.jsonl").read_text().strip())
        assert row["model"] == "gpt-4o"
        assert row["cost"] == 0.01
        # OTel keys should NOT appear at top level
        assert "gen_ai.request.model" not in row
        assert "kalibra.cost" not in row


# ── error messages ───────────────────────────────────────────────────────────

class TestErrors:
    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_json_traces(p)

    def test_missing_trace_id(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"outcome": "success"}\n')
        with pytest.raises(ValueError, match="no trace ID field found"):
            load_json_traces(p)

    def test_missing_trace_id_shows_fields(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"outcome": "success"}\n')
        with pytest.raises(ValueError, match="Available fields.*outcome"):
            load_json_traces(p)

    def test_missing_trace_id_suggests_candidates(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"uuid": "abc", "outcome": "success"}\n')
        with pytest.raises(ValueError, match="might be the trace ID.*uuid"):
            load_json_traces(p)

    def test_custom_trace_id_field(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"uuid": "t1", "outcome": "success"}\n')
        traces = load_json_traces(p, trace_id_field="uuid")
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"

    def test_invalid_outcome(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"trace_id": "t1", "outcome": "maybe"}\n')
        with pytest.raises(ValueError, match="outcome.*must be"):
            load_json_traces(p)

    def test_error_includes_line_number(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"trace_id": "t1"}\nnot json\n')
        with pytest.raises(ValueError, match=":2"):
            load_json_traces(p)

    def test_error_includes_file_path(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n")
        with pytest.raises(ValueError, match="bad.jsonl"):
            load_json_traces(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        traces = load_json_traces(p)
        assert traces == []

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "sparse.jsonl"
        p.write_text('\n{"trace_id": "t1"}\n\n{"trace_id": "t2"}\n\n')
        traces = load_json_traces(p)
        assert len(traces) == 2

    def test_mixed_format_errors(self, tmp_path):
        """File with some rows having span_id and others not → error."""
        p = tmp_path / "mixed.jsonl"
        p.write_text(
            '{"trace_id": "t1", "span_id": "s1", "name": "root"}\n'
            '{"trace_id": "t1", "outcome": "success"}\n'
        )
        with pytest.raises(ValueError, match="mixed format"):
            load_json_traces(p)

    def test_mixed_format_error_shows_hint(self, tmp_path):
        p = tmp_path / "mixed.jsonl"
        p.write_text(
            '{"trace_id": "t1", "span_id": "s1", "name": "a"}\n'
            '{"trace_id": "t1", "outcome": "success"}\n'
        )
        with pytest.raises(ValueError, match="add span_id to every row"):
            load_json_traces(p)


# ── attributes handling ─────────────────────────────────────────────────────

class TestAttributes:
    def test_attributes_dict_not_leaked_as_attribute(self, tmp_path):
        """The 'attributes' dict should be merged, not stored as a raw attribute."""
        f = _write(tmp_path, [{
            "trace_id": "t1", "outcome": "success",
            "attributes": {"custom.field": "hello"},
        }])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        attrs = dict(traces[0].spans[0].attributes)
        assert "custom.field" in attrs
        assert "attributes" not in attrs

    def test_internal_line_field_not_in_attributes(self, tmp_path):
        """The internal _line field should never appear in span attributes."""
        f = _write(tmp_path, [{"trace_id": "t1", "outcome": "success"}])
        traces = load_json_traces(tmp_path / "traces.jsonl")
        attrs = dict(traces[0].spans[0].attributes)
        assert "_line" not in attrs
