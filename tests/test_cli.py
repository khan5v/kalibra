"""Tests — CLI flag validation, error messages, and help output."""

import pytest
from click.testing import CliRunner

from kalibra.cli import main, DEFAULT_CACHE_DIR


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def trace_file(tmp_path):
    from kalibra.converters.generic import save_jsonl
    from kalibra.converters.base import Trace, make_span

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
    assert "kalibra compare --baseline ./traces.jsonl" in result.output


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
    assert "kalibra pull @my-baseline" in result.output
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
    assert "kalibra pull --source langfuse --project" in result.output


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
    assert "kalibra compare --baseline" in result.output
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


# ── early file-exists check (#10) ────────────────────────────────────────────

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


# ── CLI overrides (#1) ───────────────────────────────────────────────────────

def test_compare_outcome_field_override(runner, tmp_path):
    """--outcome-field applies outcome override to direct file paths."""
    import json

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
        "--outcome-field", "metadata.result",
    ])
    assert result.exit_code == 0
    assert "50.0%" in result.output


def test_compare_cost_attr_override(runner, tmp_path):
    """--cost-attr applies cost override to direct file paths."""
    import json

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
        "--cost-attr", "custom.cost",
    ])
    assert result.exit_code == 0


# ── JSONL source type (#12) ──────────────────────────────────────────────────

def test_pull_jsonl_source(runner, tmp_path):
    """pull @name with source: jsonl loads and caches the file."""
    import json

    # Create source JSONL
    src_file = tmp_path / "raw.jsonl"
    src_file.write_text("\n".join(json.dumps(row) for row in [
        {"trace_id": "t1", "outcome": "success", "cost": 0.01},
        {"trace_id": "t2", "outcome": "failure", "cost": 0.02},
    ]) + "\n")

    # Create source config
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "local.yml").write_text(
        f"local-data:\n  source: jsonl\n  path: {src_file}\n"
    )

    cache_dir = tmp_path / "cache"
    result = runner.invoke(main, [
        "pull", "@local-data",
        "--sources-dir", str(sources_dir),
        "--cache-dir", str(cache_dir),
    ])
    assert result.exit_code == 0
    assert "Loaded 2 traces" in result.output
    assert (cache_dir / "local-data.jsonl").exists()


def test_pull_jsonl_source_with_overrides(runner, tmp_path):
    """JSONL source with outcome override applies overrides to cached file."""
    import json

    src_file = tmp_path / "raw.jsonl"
    src_file.write_text("\n".join(json.dumps(row) for row in [
        {"trace_id": "t1", "metadata": {"status": "pass"}},
        {"trace_id": "t2", "metadata": {"status": "fail"}},
    ]) + "\n")

    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "local.yml").write_text(
        f"local-data:\n"
        f"  source: jsonl\n"
        f"  path: {src_file}\n"
        f"  outcome:\n"
        f"    field: metadata.status\n"
        f"    success: [pass]\n"
        f"    failure: [fail]\n"
    )

    cache_dir = tmp_path / "cache"
    result = runner.invoke(main, [
        "pull", "@local-data",
        "--sources-dir", str(sources_dir),
        "--cache-dir", str(cache_dir),
    ])
    assert result.exit_code == 0
    assert "50.0%" in result.output  # 1 success out of 2


def test_compare_jsonl_source(runner, tmp_path):
    """compare with @name referencing source: jsonl works end-to-end."""
    import json

    src_file = tmp_path / "data.jsonl"
    src_file.write_text("\n".join(json.dumps(row) for row in [
        {"trace_id": "t1", "outcome": "success", "cost": 0.01},
        {"trace_id": "t2", "outcome": "failure", "cost": 0.02},
    ]) + "\n")

    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "s.yml").write_text(
        f"mydata:\n  source: jsonl\n  path: {src_file}\n"
    )

    cache_dir = tmp_path / "cache"
    result = runner.invoke(main, [
        "compare",
        "--baseline", "@mydata",
        "--current", "@mydata",
        "--sources", str(sources_dir),
        "--cache-dir", str(cache_dir),
    ])
    assert result.exit_code == 0


def test_jsonl_source_missing_path_errors():
    """JSONL source without path field raises ValueError."""
    from kalibra.config import SourceConfig
    import pytest

    with pytest.raises(ValueError, match="requires a 'path'"):
        SourceConfig.from_dict({"source": "jsonl"})


def test_jsonl_source_config_parsing():
    """JSONL source parses path and doesn't require project."""
    from kalibra.config import SourceConfig

    src = SourceConfig.from_dict({"source": "jsonl", "path": "/data/traces.jsonl"})
    assert src.source == "jsonl"
    assert src.path == "/data/traces.jsonl"
    assert src.project == ""


# ── _print_pull_summary uses in-memory traces (#6) ──────────────────────────

def test_print_pull_summary_does_not_reread_file(runner, tmp_path, monkeypatch):
    """When traces are passed in-memory, _print_pull_summary does not re-read the file."""
    import json
    from kalibra.display import pull_summary as _print_pull_summary
    from kalibra.converters.base import Trace, make_span

    spans = [make_span("a", "t1", "s1", None, 0, int(1e9), {})]
    traces = [Trace("t1", spans, outcome="success")]

    load_calls = []
    original_load = None

    def mock_load(path):
        load_calls.append(path)
        return traces

    # Patch load_json_traces — should NOT be called when traces are passed
    import kalibra.cli
    monkeypatch.setattr("kalibra.converters.generic.load_json_traces", mock_load)

    _print_pull_summary(traces=traces)
    assert len(load_calls) == 0  # should not re-read
