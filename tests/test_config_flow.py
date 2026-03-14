"""Tests for unified kalibra.yml config — init, discovery, and CLI overrides."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from kalibra.cli import main
from kalibra.config import CompareConfig, PopulationConfig, FieldsConfig


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def sample_traces(tmp_path):
    """Create two small JSONL trace files for testing."""
    from kalibra.converters.base import Trace, make_span
    from kalibra.converters.generic import save_jsonl

    spans_b = [make_span("step", "t1__m__0", "s1", None, 0, int(1e9),
                         {"kalibra.cost": 0.01, "gen_ai.usage.input_tokens": 100,
                          "gen_ai.usage.output_tokens": 50})]
    spans_c = [make_span("step", "t1__m__0", "s2", None, 0, int(2e9),
                         {"kalibra.cost": 0.02, "gen_ai.usage.input_tokens": 200,
                          "gen_ai.usage.output_tokens": 80})]

    baseline = tmp_path / "baseline.jsonl"
    current = tmp_path / "current.jsonl"
    save_jsonl([Trace("t1__m__0", spans_b, outcome="success")], str(baseline))
    save_jsonl([Trace("t1__m__0", spans_c, outcome="success")], str(current))
    return str(baseline), str(current)


# ── CompareConfig parsing ─────────────────────────────────────────────────────

class TestCompareConfigParsing:
    def test_from_dict_full(self):
        data = {
            "baseline": {"source": "langfuse", "project": "p1", "tags": ["v1"]},
            "current": {"path": "./current.jsonl"},
            "metrics": ["success_rate", "cost"],
            "require": ["success_rate_delta >= -2"],
            "fields": {"task_id": "braintrust.task_id"},
        }
        cfg = CompareConfig.from_dict(data)
        assert cfg.baseline.source == "langfuse"
        assert cfg.baseline.project == "p1"
        assert cfg.baseline.tags == ["v1"]
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
        assert cfg.baseline.source is None

    def test_task_id_via_fields(self):
        cfg = CompareConfig.from_dict({"fields": {"task_id": "meta.tid"}})
        assert cfg.task_id == "meta.tid"

    def test_task_id_backward_compat(self):
        """Top-level task_id still works for old configs."""
        cfg = CompareConfig.from_dict({"task_id": "old.field"})
        assert cfg.task_id == "old.field"

    def test_population_tags_as_string(self):
        pop = PopulationConfig.from_dict({"source": "langfuse", "project": "p", "tags": "v1"})
        assert pop.tags == ["v1"]

    def test_population_empty_dict(self):
        pop = PopulationConfig.from_dict({})
        assert pop.path is None
        assert pop.source is None
        assert pop.tags == []


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
        assert result.exit_code == 0
        assert "Using" in result.output
        assert "Kalibra Compare" in result.output

    def test_no_config_no_flags_errors(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        assert result.exit_code != 0
        assert "kalibra init" in result.output

    def test_explicit_config_flag(self, runner, sample_traces, tmp_path):
        baseline, current = sample_traces
        config = {"baseline": {"path": baseline}, "current": {"path": current}}
        cfg_file = tmp_path / "custom.yml"
        cfg_file.write_text(yaml.dump(config))

        result = runner.invoke(main, [
            "compare", "--config", str(cfg_file),
        ])
        assert result.exit_code == 0
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
        assert result.exit_code == 0
        # Only the CLI gate should appear, not the config ones.
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
        # Should succeed because CLI --baseline overrides the bad config path.
        assert result.exit_code == 0
        assert "Kalibra Compare" in result.output


# ── Init command ──────────────────────────────────────────────────────────────

class TestInit:
    def test_init_creates_file(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init", "--force"], input="4\n./a.jsonl\n./b.jsonl\n")
            assert result.exit_code == 0
            assert "Created kalibra.yml" in result.output

            config = yaml.safe_load(Path("kalibra.yml").read_text())
            assert config["baseline"]["path"] == "./a.jsonl"
            assert config["current"]["path"] == "./b.jsonl"
            assert "success_rate" in config["metrics"]
            assert len(config["require"]) > 0

    def test_init_remote_source(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main, ["init", "--force"],
                input="1\nmy-project\nprod,v1\nprod,v2\n",
            )
            assert result.exit_code == 0
            config = yaml.safe_load(Path("kalibra.yml").read_text())
            assert config["baseline"]["source"] == "langfuse"
            assert config["baseline"]["project"] == "my-project"
            assert config["current"]["tags"] == ["prod", "v2"]

    def test_init_empty_tags(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main, ["init", "--force"],
                input="2\nmy-project\n\n\n",
            )
            assert result.exit_code == 0
            config = yaml.safe_load(Path("kalibra.yml").read_text())
            assert config["baseline"]["source"] == "langsmith"
            assert "tags" not in config["baseline"]

    def test_init_no_overwrite_without_force(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("kalibra.yml").write_text("existing: true")
            result = runner.invoke(main, ["init"], input="n\n")
            assert "Aborted" in result.output
            assert yaml.safe_load(Path("kalibra.yml").read_text()) == {"existing": True}


# ── Init → Compare flow ──────────────────────────────────────────────────────

class TestNamedSources:
    def test_sources_parsed(self):
        data = {
            "sources": {
                "prod-v1": {"source": "langfuse", "project": "p", "tags": ["v1"]},
                "prod-v2": {"source": "langfuse", "project": "p", "tags": ["v2"]},
            },
            "baseline": "prod-v1",
            "current": "prod-v2",
        }
        cfg = CompareConfig.from_dict(data)
        assert len(cfg.sources) == 2
        assert cfg.baseline.tags == ["v1"]
        assert cfg.current.tags == ["v2"]
        assert cfg.get_source("prod-v1").project == "p"

    def test_unknown_source_reference_errors(self):
        data = {
            "sources": {"a": {"source": "langfuse", "project": "p"}},
            "baseline": "nonexistent",
        }
        with pytest.raises(ValueError, match="Unknown source"):
            CompareConfig.from_dict(data)

    def test_inline_and_reference_mixed(self):
        data = {
            "sources": {
                "prod": {"source": "langfuse", "project": "p", "tags": ["v1"]},
            },
            "baseline": "prod",
            "current": {"path": "./local.jsonl"},
        }
        cfg = CompareConfig.from_dict(data)
        assert cfg.baseline.source == "langfuse"
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
        assert result.exit_code in (0, 1)  # may fail gates
        assert "Kalibra Compare" in result.output


class TestInitCompareFlow:
    def test_init_then_compare(self, runner, sample_traces, tmp_path):
        baseline, current = sample_traces
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Init with local files pointing to our sample data.
            result = runner.invoke(
                main, ["init", "--force"],
                input=f"4\n{baseline}\n{current}\n",
            )
            assert result.exit_code == 0

            # Compare runs end-to-end (may pass or fail gates — we just
            # verify it ran the comparison, not the gate outcome).
            result = runner.invoke(main, ["compare"])
            assert "Kalibra Compare" in result.output
            assert "Thresholds" in result.output
