"""Tests — CLI flag validation for --config and --sources."""

import pytest
from click.testing import CliRunner

from agentflow.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def trace_file(tmp_path):
    from agentflow.converters.generic import save_jsonl
    from agentflow.converters.base import Trace, make_span

    spans = [make_span(name="a", trace_id="t1", span_id="s1", parent_span_id=None,
                       start_ns=0, end_ns=int(1e9), attributes={})]
    traces = [Trace(trace_id="t1__m__0", spans=spans, outcome="success")]
    f = tmp_path / "traces.jsonl"
    save_jsonl(traces, str(f))
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
    assert "not a directory" in result.output or "directory" in result.output


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


# ── --sources ─────────────────────────────────────────────────────────────────

def test_sources_missing_dir_errors(runner, trace_file, tmp_path):
    result = runner.invoke(main, [
        "compare",
        "--baseline", trace_file,
        "--current", trace_file,
        "--sources", str(tmp_path / "no_such_dir"),
    ])
    assert result.exit_code != 0
    assert "not a directory" in result.output


def test_sources_file_errors(runner, trace_file, tmp_path):
    f = tmp_path / "not_a_dir.yml"
    f.write_text("")
    result = runner.invoke(main, [
        "compare",
        "--baseline", trace_file,
        "--current", trace_file,
        "--sources", str(f),
    ])
    assert result.exit_code != 0
    assert "not a directory" in result.output


def test_sources_valid_dir(runner, trace_file, tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    result = runner.invoke(main, [
        "compare",
        "--baseline", trace_file,
        "--current", trace_file,
        "--sources", str(sources_dir),
    ])
    assert result.exit_code == 0
