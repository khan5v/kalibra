"""Tests — CLI flag validation, error messages, and help output."""

import json

import pytest
from click.testing import CliRunner

from kalibra.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def trace_file(tmp_path):
    """Create a small JSONL trace file for testing."""
    f = tmp_path / "traces.jsonl"
    traces = [
        {"trace_id": "t1", "outcome": "success",
         "spans": [{"span_id": "s1", "name": "step",
                     "cost": 0.01, "input_tokens": 100, "output_tokens": 50,
                     "start_ns": 0, "end_ns": 1_000_000_000}]},
    ]
    f.write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    return str(f)


# ── --config ──────────────────────────────────────────────────────────────────

def test_config_missing_file_errors(runner, trace_file, tmp_path):
    result = runner.invoke(main, [
        "compare",
        "--baseline", trace_file,
        "--current", trace_file,
        "--config", str(tmp_path / "nonexistent.yml"),
    ])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_config_directory_errors(runner, trace_file, tmp_path):
    result = runner.invoke(main, [
        "compare",
        "--baseline", trace_file,
        "--current", trace_file,
        "--config", str(tmp_path),
    ])
    assert result.exit_code != 0
    assert "directory" in result.output


def test_config_valid_file(runner, trace_file, tmp_path):
    cfg = tmp_path / "compare.yml"
    cfg.write_text("metrics:\n  - success_rate\n")
    result = runner.invoke(main, [
        "compare",
        "--baseline", trace_file,
        "--current", trace_file,
        "--config", str(cfg),
    ])
    assert result.exit_code == 0


# ── help output ──────────────────────────────────────────────────────────────

def test_compare_help_shows_examples(runner):
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert "kalibra compare --baseline" in result.output
    assert "--current" in result.output



# ── early file-exists check ──────────────────────────────────────────────────

def test_compare_nonexistent_baseline_errors(runner, trace_file):
    result = runner.invoke(main, [
        "compare",
        "--baseline", "/no/such/file.jsonl",
        "--current", trace_file,
    ])
    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert "Baseline" in result.output


def test_compare_nonexistent_current_errors(runner, trace_file):
    result = runner.invoke(main, [
        "compare",
        "--baseline", trace_file,
        "--current", "/no/such/file.jsonl",
    ])
    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert "Current" in result.output


# ── CLI field overrides ──────────────────────────────────────────────────────

def test_compare_outcome_override(runner, tmp_path):
    """--outcome applies outcome field mapping."""
    f = tmp_path / "traces.jsonl"
    f.write_text("\n".join(json.dumps(row) for row in [
        {"trace_id": "t1", "metadata": {"result": "success"}},
        {"trace_id": "t2", "metadata": {"result": "failure"}},
    ]) + "\n")
    path = str(f)

    result = runner.invoke(main, [
        "compare",
        "--baseline", path,
        "--current", path,
        "--outcome", "result",
    ])
    assert result.exit_code == 0
    assert "50.0%" in result.output


def test_compare_cost_override(runner, tmp_path):
    """--cost applies cost field mapping."""
    f = tmp_path / "traces.jsonl"
    f.write_text("\n".join(json.dumps(row) for row in [
        {"trace_id": "t1", "outcome": "success", "cost": 0.01,
         "attributes": {"custom.cost": 0.99}},
        {"trace_id": "t2", "outcome": "success", "cost": 0.02,
         "attributes": {"custom.cost": 0.88}},
    ]) + "\n")
    path = str(f)

    result = runner.invoke(main, [
        "compare",
        "--baseline", path,
        "--current", path,
        "--cost", "custom.cost",
    ])
    assert result.exit_code == 0


# ── --metrics list ───────────────────────────────────────────────────────────

def test_compare_metrics_flag(runner):
    """--metrics lists all metric names and their threshold fields."""
    result = runner.invoke(main, ["compare", "--metrics"])
    assert result.exit_code == 0
    assert "success_rate" in result.output
    assert "cost_delta_pct" in result.output
    assert "regressions" in result.output
