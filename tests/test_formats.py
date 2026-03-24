"""Tests for the trace format registry and --trace-format flag."""

from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

from kalibra.cli import main
from kalibra.loaders import TraceLoader
from kalibra.loaders.openinference import OpenInferenceLoader
from kalibra.loaders.flat import FlatLoader
from kalibra.loader import load_traces


class TestFormatClasses:
    def test_openinference_detects_phoenix_span(self):
        fmt = OpenInferenceLoader()
        item = {
            "context": {"trace_id": "abc", "span_id": "def"},
            "parent_id": None,
            "span_kind": "LLM",
        }
        assert fmt.detect(item) is True

    def test_openinference_rejects_flat_trace(self):
        fmt = OpenInferenceLoader()
        item = {"trace_id": "abc", "outcome": "success", "cost": 0.01}
        assert fmt.detect(item) is False

    def test_flat_never_detects(self):
        fmt = FlatLoader()
        assert fmt.detect({"anything": "at all"}) is False
        assert fmt.detect({}) is False

    def test_format_names(self):
        assert OpenInferenceLoader().name == "openinference"
        assert FlatLoader().name == "flat"


class TestExplicitFormat:
    @pytest.fixture()
    def flat_traces(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        traces = [
            {"trace_id": "t1", "outcome": "success",
             "spans": [{"span_id": "s1", "name": "step",
                        "cost": 0.01, "input_tokens": 100, "output_tokens": 50}]},
            {"trace_id": "t2", "outcome": "failure",
             "spans": [{"span_id": "s2", "name": "step",
                        "cost": 0.02, "input_tokens": 200, "output_tokens": 80}]},
        ]
        path.write_text("\n".join(json.dumps(t) for t in traces) + "\n")
        return str(path)

    def test_explicit_flat_format(self, flat_traces):
        traces = load_traces(flat_traces, format="flat")
        assert len(traces) == 2
        assert traces[0].trace_id == "t1"

    def test_explicit_auto_format(self, flat_traces):
        traces = load_traces(flat_traces, format="auto")
        assert len(traces) == 2

    def test_unknown_format_raises(self, flat_traces):
        with pytest.raises(ValueError, match="Unknown format"):
            load_traces(flat_traces, format="nonexistent")

    @pytest.fixture()
    def oi_traces(self, tmp_path):
        """OpenInference-style spans."""
        path = tmp_path / "oi.jsonl"
        spans = [
            {"context": {"trace_id": "t1", "span_id": "s1"},
             "parent_id": None, "name": "agent", "span_kind": "CHAIN",
             "start_time": "2026-01-01T00:00:00Z",
             "end_time": "2026-01-01T00:00:01Z",
             "status_code": "OK", "attributes": {}},
            {"context": {"trace_id": "t1", "span_id": "s2"},
             "parent_id": "s1", "name": "llm", "span_kind": "LLM",
             "start_time": "2026-01-01T00:00:00Z",
             "end_time": "2026-01-01T00:00:01Z",
             "status_code": "OK",
             "attributes": {"llm.token_count.prompt": 100,
                            "llm.token_count.completion": 50}},
        ]
        path.write_text("\n".join(json.dumps(s) for s in spans) + "\n")
        return str(path)

    def test_explicit_openinference_format(self, oi_traces):
        traces = load_traces(oi_traces, format="openinference")
        assert len(traces) == 1
        assert traces[0].trace_id == "t1"
        assert len(traces[0].spans) == 2

    def test_auto_detects_openinference(self, oi_traces):
        traces = load_traces(oi_traces, format="auto")
        assert len(traces) == 1
        assert len(traces[0].spans) == 2


class TestFormatPrecedence:
    """CLI flag > per-population config > auto."""

    @pytest.fixture()
    def oi_traces(self, tmp_path):
        path = tmp_path / "oi.jsonl"
        spans = [
            {"context": {"trace_id": "t1", "span_id": "s1"},
             "parent_id": None, "name": "agent", "span_kind": "CHAIN",
             "start_time": "2026-01-01T00:00:00Z",
             "end_time": "2026-01-01T00:00:01Z",
             "status_code": "OK", "attributes": {}},
        ]
        path.write_text("\n".join(json.dumps(s) for s in spans) + "\n")
        return str(path)

    def test_cli_flag_overrides_config_format(self, oi_traces, tmp_path):
        """--trace-format overrides format: in config."""
        config = {
            "baseline": {"path": oi_traces, "format": "flat"},
            "current": {"path": oi_traces, "format": "flat"},
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Config says flat (would fail on OI data), but CLI says openinference
            result = runner.invoke(main, [
                "compare", "--trace-format", "openinference",
            ])
        assert result.exit_code in (0, 1), result.output
        assert "Kalibra Compare" in result.output

    def test_config_format_used_when_no_flag(self, oi_traces, tmp_path):
        """format: in config is used when --trace-format is not passed."""
        config = {
            "baseline": {"path": oi_traces, "format": "openinference"},
            "current": {"path": oi_traces, "format": "openinference"},
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        assert result.exit_code in (0, 1), result.output
        assert "1 traces" in result.output

    def test_mixed_formats_per_population(self, oi_traces, tmp_path):
        """Baseline and current can have different formats."""
        flat_path = tmp_path / "flat.jsonl"
        flat_path.write_text(json.dumps({
            "trace_id": "t1", "outcome": "success",
            "spans": [{"span_id": "s1", "name": "step",
                        "cost": 0.01, "input_tokens": 100, "output_tokens": 50,
                        "start_ns": 0, "end_ns": 1_000_000_000}],
        }) + "\n")

        config = {
            "baseline": {"path": oi_traces, "format": "openinference"},
            "current": {"path": str(flat_path), "format": "flat"},
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        assert result.exit_code in (0, 1), result.output
        assert "Kalibra Compare" in result.output


class TestWrongFormatErrors:
    """Explicit format on wrong data should fail gracefully, not crash."""

    def test_flat_format_on_openinference_data(self, tmp_path):
        """Forcing flat format on OI spans — should load but produce bad traces."""
        path = tmp_path / "oi.jsonl"
        path.write_text(json.dumps({
            "context": {"trace_id": "t1", "span_id": "s1"},
            "parent_id": None, "span_kind": "LLM",
            "attributes": {"llm.token_count.prompt": 100},
        }) + "\n")
        # Flat loader treats each line as a trace — won't crash but
        # the trace won't have meaningful data (no trace_id at top level).
        traces = load_traces(str(path), format="flat")
        assert len(traces) >= 1  # loads something, doesn't crash


class TestTraceLoaderViaCLI:
    @pytest.fixture()
    def sample_data(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        traces = [
            {"trace_id": "t1", "outcome": "success",
             "spans": [{"span_id": "s1", "name": "step",
                        "cost": 0.01, "input_tokens": 100, "output_tokens": 50,
                        "start_ns": 0, "end_ns": 1_000_000_000}]},
        ]
        path.write_text("\n".join(json.dumps(t) for t in traces) + "\n")
        return str(path)

    def test_trace_format_flag(self, sample_data):
        runner = CliRunner()
        result = runner.invoke(main, [
            "compare", "--baseline", sample_data, "--current", sample_data,
            "--trace-format", "flat",
        ])
        assert result.exit_code == 0, result.output
        assert "Kalibra Compare" in result.output

    def test_trace_format_in_config(self, sample_data, tmp_path):
        config = {
            "baseline": {"path": sample_data, "format": "flat"},
            "current": {"path": sample_data, "format": "flat"},
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        assert result.exit_code == 0, result.output
        assert "Kalibra Compare" in result.output
