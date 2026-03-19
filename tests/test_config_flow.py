"""Tests for unified kalibra.yml config — init, discovery, and CLI overrides."""

from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

from kalibra.cli import main
from kalibra.config import CompareConfig, FieldsConfig, PopulationConfig


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def sample_traces(tmp_path):
    """Create two small JSONL trace files for testing."""
    baseline = tmp_path / "baseline.jsonl"
    current = tmp_path / "current.jsonl"

    b_trace = {
        "trace_id": "t1",
        "outcome": "success",
        "spans": [{
            "span_id": "s1", "name": "step",
            "cost": 0.01, "input_tokens": 100, "output_tokens": 50,
            "start_ns": 0, "end_ns": 1_000_000_000,
        }],
    }
    c_trace = {
        "trace_id": "t1",
        "outcome": "success",
        "spans": [{
            "span_id": "s2", "name": "step",
            "cost": 0.02, "input_tokens": 200, "output_tokens": 80,
            "start_ns": 0, "end_ns": 2_000_000_000,
        }],
    }
    baseline.write_text(json.dumps(b_trace) + "\n")
    current.write_text(json.dumps(c_trace) + "\n")
    return str(baseline), str(current)


# ── CompareConfig parsing ─────────────────────────────────────────────────────

class TestCompareConfigParsing:
    def test_from_dict_full(self):
        data = {
            "baseline": {"path": "./baseline.jsonl"},
            "current": {"path": "./current.jsonl"},
            "metrics": ["success_rate", "cost"],
            "require": ["success_rate_delta >= -2"],
            "fields": {"task_id": "braintrust.task_id"},
        }
        cfg = CompareConfig.from_dict(data)
        assert cfg.baseline.path == "./baseline.jsonl"
        assert cfg.current.path == "./current.jsonl"
        assert cfg.metrics == ["success_rate", "cost"]
        assert cfg.require == ["success_rate_delta >= -2"]
        assert cfg.task_id == "braintrust.task_id"

    def test_from_dict_minimal(self):
        cfg = CompareConfig.from_dict({})
        assert cfg.baseline is None
        assert cfg.current is None
        assert cfg.metrics is None
        assert cfg.require == []
        assert cfg.task_id is None

    def test_from_dict_local_paths(self):
        data = {
            "baseline": {"path": "./a.jsonl"},
            "current": {"path": "./b.jsonl"},
        }
        cfg = CompareConfig.from_dict(data)
        assert cfg.baseline.path == "./a.jsonl"
        assert cfg.current.path == "./b.jsonl"

    def test_task_id_via_fields(self):
        cfg = CompareConfig.from_dict({"fields": {"task_id": "meta.tid"}})
        assert cfg.task_id == "meta.tid"

    def test_task_id_backward_compat(self):
        """Top-level task_id still works for old configs."""
        cfg = CompareConfig.from_dict({"task_id": "old.field"})
        assert cfg.task_id == "old.field"

    def test_population_empty_dict(self):
        pop = PopulationConfig.from_dict({})
        assert pop.path is None


# ── Config discovery ──────────────────────────────────────────────────────────

class TestConfigDiscovery:
    def test_kalibra_yml_discovered(self, runner, sample_traces, tmp_path):
        baseline, current = sample_traces
        config = {
            "baseline": {"path": baseline},
            "current": {"path": current},
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        assert result.exit_code == 0, result.output
        assert "Using" in result.output
        assert "Kalibra Compare" in result.output

    def test_no_config_no_flags_shows_welcome(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        assert result.exit_code == 0
        assert "kalibra demo" in result.output
        assert "kalibra init" in result.output

    def test_explicit_config_flag(self, runner, sample_traces, tmp_path):
        baseline, current = sample_traces
        config = {"baseline": {"path": baseline}, "current": {"path": current}}
        cfg_file = tmp_path / "custom.yml"
        cfg_file.write_text(yaml.dump(config))

        result = runner.invoke(main, [
            "compare", "--config", str(cfg_file),
        ])
        assert result.exit_code == 0, result.output
        assert "custom.yml" in result.output


# ── CLI overrides ─────────────────────────────────────────────────────────────

class TestCLIOverrides:
    def test_require_flag_replaces_config(self, runner, sample_traces, tmp_path):
        baseline, current = sample_traces
        config = {
            "baseline": {"path": baseline},
            "current": {"path": current},
            "require": ["success_rate_delta >= -2", "regressions <= 5"],
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, [
                "compare", "--require", "success_rate_delta >= -99",
            ])
        assert result.exit_code == 0, result.output
        assert "success_rate_delta >= -99" in result.output
        assert "regressions <= 5" not in result.output

    def test_baseline_flag_overrides_config(self, runner, sample_traces, tmp_path):
        baseline, current = sample_traces
        config = {
            "baseline": {"path": "/nonexistent.jsonl"},
            "current": {"path": current},
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, [
                "compare", "--baseline", baseline,
            ])
        assert result.exit_code == 0, result.output
        assert "Kalibra Compare" in result.output


# ── Named sources ────────────────────────────────────────────────────────────

class TestNamedSources:
    def test_sources_parsed(self):
        data = {
            "sources": {
                "data-a": {"path": "./a.jsonl"},
                "data-b": {"path": "./b.jsonl"},
            },
            "baseline": "data-a",
            "current": "data-b",
        }
        cfg = CompareConfig.from_dict(data)
        assert len(cfg.sources) == 2
        assert cfg.baseline.path == "./a.jsonl"
        assert cfg.current.path == "./b.jsonl"

    def test_unknown_source_reference_errors(self):
        data = {
            "sources": {"a": {"path": "./a.jsonl"}},
            "baseline": "nonexistent",
        }
        with pytest.raises(ValueError, match="Unknown source"):
            CompareConfig.from_dict(data)

    def test_inline_and_reference_mixed(self):
        data = {
            "sources": {
                "prod": {"path": "./prod.jsonl"},
            },
            "baseline": "prod",
            "current": {"path": "./local.jsonl"},
        }
        cfg = CompareConfig.from_dict(data)
        assert cfg.baseline.path == "./prod.jsonl"
        assert cfg.current.path == "./local.jsonl"

    def test_named_source_via_flag(self, runner, sample_traces, tmp_path):
        baseline, current = sample_traces
        config = {
            "sources": {
                "data-a": {"path": baseline},
                "data-b": {"path": current},
            },
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, [
                "compare", "--baseline", "data-a", "--current", "data-b",
            ])
        assert result.exit_code in (0, 1)
        assert "Kalibra Compare" in result.output


class TestInitCompareFlow:
    def test_init_then_compare(self, runner, sample_traces, tmp_path):
        baseline, current = sample_traces
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main, ["init", "--force"],
                input=f"{baseline}\n{current}\n",
            )
            assert result.exit_code == 0

            result = runner.invoke(main, ["compare"])
            assert "Kalibra Compare" in result.output
            assert "Quality gates" in result.output


# ── Fields config ─────────────────────────────────────────────────────────────

class TestFieldsConfig:
    def test_merge_override_wins(self):
        base = FieldsConfig(trace_id="id", outcome="result")
        override = FieldsConfig(outcome="status")
        merged = base.merge(override)
        assert merged.trace_id == "id"
        assert merged.outcome == "status"

    def test_merge_none_keeps_base(self):
        base = FieldsConfig(cost="my_cost")
        merged = base.merge(None)
        assert merged.cost == "my_cost"

    def test_per_source_fields(self):
        data = {
            "sources": {
                "a": {"path": "./a.jsonl", "fields": {"outcome": "result"}},
                "b": {"path": "./b.jsonl", "fields": {"outcome": "status"}},
            },
            "baseline": "a",
            "current": "b",
            "fields": {"cost": "total_cost"},
        }
        cfg = CompareConfig.from_dict(data)
        assert cfg.baseline.fields.outcome == "result"
        assert cfg.current.fields.outcome == "status"
        assert cfg.fields.cost == "total_cost"
