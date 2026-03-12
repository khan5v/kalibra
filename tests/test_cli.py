"""Tests — CLI flag validation, error messages, and help output."""

import pytest
from click.testing import CliRunner

from agentflow.cli import main, DEFAULT_CACHE_DIR


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


# ── helpful error messages ───────────────────────────────────────────────────

def test_compare_unknown_source_shows_hint(runner, trace_file):
    """@name not found → error suggests using a file path directly."""
    result = runner.invoke(main, [
        "compare",
        "--baseline", "@nonexistent",
        "--current", trace_file,
    ])
    assert result.exit_code != 0
    assert "not found" in result.output
    assert "agentflow compare --baseline ./traces.jsonl" in result.output


def test_compare_unknown_source_shows_available(runner, trace_file, tmp_path):
    """@name not found → error lists available sources."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    src_file = sources_dir / "my.yml"
    src_file.write_text(
        "prod:\n  source: langfuse\n  project: p\n  since: 7d\n  limit: 10\n"
    )
    result = runner.invoke(main, [
        "compare",
        "--baseline", "@missing",
        "--current", trace_file,
        "--sources", str(sources_dir),
    ])
    assert result.exit_code != 0
    assert "prod" in result.output
    assert "not found" in result.output


def test_pull_no_args_shows_examples(runner):
    """pull with no @name and no --source/--project shows usage examples."""
    result = runner.invoke(main, ["pull"])
    assert result.exit_code != 0
    assert "agentflow pull @my-baseline" in result.output
    assert "--source langfuse --project" in result.output


def test_pull_unknown_source_shows_hint(runner, tmp_path):
    """pull @unknown → error suggests explicit flags."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    result = runner.invoke(main, [
        "pull", "@ghost",
        "--sources-dir", str(sources_dir),
    ])
    assert result.exit_code != 0
    assert "not found" in result.output
    assert "agentflow pull --source langfuse --project" in result.output


def test_pull_unknown_source_lists_available(runner, tmp_path):
    """pull @unknown → error lists defined sources."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    src_file = sources_dir / "defs.yml"
    src_file.write_text(
        "staging:\n  source: langsmith\n  project: s\n  since: 3d\n  limit: 50\n"
    )
    result = runner.invoke(main, [
        "pull", "@nope",
        "--sources-dir", str(sources_dir),
    ])
    assert result.exit_code != 0
    assert "staging" in result.output


# ── help output ──────────────────────────────────────────────────────────────

def test_compare_help_shows_examples(runner):
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert "agentflow compare --baseline" in result.output
    assert "--current" in result.output


def test_pull_help_shows_both_modes(runner):
    result = runner.invoke(main, ["pull", "--help"])
    assert result.exit_code == 0
    assert "@current" in result.output
    assert "--source langfuse" in result.output


def test_compare_help_shows_cache_dir_default(runner):
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert DEFAULT_CACHE_DIR in result.output


def test_pull_help_shows_cache_dir_default(runner):
    result = runner.invoke(main, ["pull", "--help"])
    assert result.exit_code == 0
    assert DEFAULT_CACHE_DIR in result.output


# ── --cache-dir ──────────────────────────────────────────────────────────────

def test_compare_cache_dir_used(runner, trace_file, tmp_path):
    """--cache-dir is threaded through to compare (no crash with custom dir)."""
    custom_cache = tmp_path / "my_cache"
    result = runner.invoke(main, [
        "compare",
        "--baseline", trace_file,
        "--current", trace_file,
        "--cache-dir", str(custom_cache),
    ])
    assert result.exit_code == 0
