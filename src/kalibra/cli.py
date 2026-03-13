"""Kalibra CLI."""

from __future__ import annotations

import os
from pathlib import Path

import click

DEFAULT_CACHE_DIR = "cached_sources"


@click.group()
@click.version_option()
def main():
    """Kalibra — agent evaluation and regression detection."""


# ── compare ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--baseline", required=True,
              help="Baseline traces: file/dir path or @name from sources.yml.")
@click.option("--current", required=True,
              help="Current traces: file/dir path or @name from sources.yml.")
@click.option("--format", "out_format",
              type=click.Choice(["terminal", "markdown", "json"]),
              default="terminal", help="Output format (default: terminal).")
@click.option("--require", "-r", multiple=True,
              help="Threshold expression, e.g. 'success_rate_delta >= -2'.")
@click.option("--config", "config_path", default=None, type=click.Path(),
              help="Path to compare config file (default: config/compare.yml).")
@click.option("--sources", "sources_dir", default=None, type=click.Path(),
              help="Sources directory (default: config/sources/).")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write output to file instead of stdout.")
@click.option("--refresh", is_flag=True, default=False,
              help="Re-pull @name sources even if a local cache exists.")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, type=click.Path(),
              help="Directory for cached pulled traces.", show_default=True)
@click.option("--outcome-field", default=None,
              help="Override outcome from this trace field (e.g. metadata.result). "
                   "Matches default keywords (success/failure). "
                   "For custom keywords, use a source config with outcome.success/failure lists.")
@click.option("--cost-attr", default=None,
              help="Override cost from this span attribute (e.g. custom.cost_usd).")
def compare(baseline: str, current: str, out_format: str, require: tuple,
            config_path: str | None, sources_dir: str | None, output: str | None,
            refresh: bool, cache_dir: str,
            outcome_field: str | None, cost_attr: str | None):
    """Compare two trace datasets — regression detection, statistical diff.

    \b
    --baseline and --current accept:
      - A file or directory path (SWE-bench, JSONL — auto-detected)
      - @name  — a named source from sources.yml (pulls and caches automatically)

    \b
    --config accepts:
      - A bare name (e.g. myteam) → resolved from ~/.config/kalibra/myteam.yml or myteam/
      - A file path  → used directly as compare.yml
      - A directory  → uses <dir>/compare.yml and <dir>/sources/

    \b
    Examples:
      kalibra compare --baseline ./baseline/ --current ./current/
      kalibra compare --baseline @baseline --current @current
      kalibra compare --baseline @baseline --current @current --config /data/configs/prod.yml
      kalibra compare --baseline @baseline --current @current --config ~/my-configs/
    """
    from kalibra.config import CompareConfig, load_sources
    from kalibra.report import render

    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise click.UsageError(f"Config file not found: {config_path}")
        if p.is_dir():
            raise click.UsageError(f"--config must be a file, not a directory: {config_path}")

    if sources_dir is not None:
        s = Path(sources_dir)
        if not s.exists() or not s.is_dir():
            raise click.UsageError(f"Sources path is not a directory: {sources_dir}")

    config = CompareConfig.load(config_path)
    sources = load_sources(sources_dir)
    baseline_path = _resolve_source(baseline, sources, refresh, cache_dir=cache_dir)
    current_path = _resolve_source(current, sources, refresh, cache_dir=cache_dir)

    # Early file-exists check — fail fast with a clear message
    for label, p in [("Baseline", baseline_path), ("Current", current_path)]:
        if not Path(p).exists():
            raise click.UsageError(
                f"{label} path does not exist: {p}\n\n"
                f"  If this is a named source, use @name syntax:\n"
                f"    kalibra compare --baseline @my-source --current @other-source\n\n"
                f"  If this is a file path, check the path is correct."
            )

    # CLI override flags → apply overrides to loaded traces before comparison
    if outcome_field or cost_attr:
        from kalibra.collection import TraceCollection
        from kalibra.compare import compare_collections
        from kalibra.config import CostConfig, OutcomeConfig, SourceConfig
        from kalibra.converters import load_traces
        from kalibra.converters.base import apply_overrides

        override_cfg = SourceConfig(
            source="", project="",
            outcome=OutcomeConfig(field=outcome_field) if outcome_field else None,
            cost=CostConfig(attr=cost_attr) if cost_attr else None,
        )

        click.echo(f"Loading {baseline_path}")
        b_traces = load_traces(baseline_path)
        apply_overrides(b_traces, override_cfg)

        click.echo(f"Loading {current_path}")
        c_traces = load_traces(current_path)
        apply_overrides(c_traces, override_cfg)

        baseline_col = TraceCollection.from_traces(b_traces, source=baseline_path)
        current_col = TraceCollection.from_traces(c_traces, source=current_path)
        result = compare_collections(
            baseline_col, current_col,
            require=list(require) or None, config=config,
        )
    else:
        from kalibra.collection import TraceCollection
        from kalibra.compare import compare_collections

        click.echo(f"Loading {baseline_path}")
        baseline_col = TraceCollection.from_path(baseline_path)

        click.echo(f"Loading {current_path}")
        current_col = TraceCollection.from_path(current_path)

        result = compare_collections(
            baseline_col, current_col,
            require=list(require) or None, config=config,
        )

    text = render(result, out_format)

    if output:
        with open(output, "w") as f:
            f.write(text)
        click.echo(f"Report written to {output}")
    else:
        click.echo(text)

    if not result.thresholds_passed:
        raise SystemExit(1)


# ── pull ───────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("name", required=False, default=None,
                metavar="[@NAME]")
@click.option("--source", type=click.Choice(["langfuse", "langsmith"]),
              help="Trace store to pull from.")
@click.option("--project",
              help="Project name (Langfuse project_id or LangSmith project name).")
@click.option("--since", default="7d", help="Time window: 7d, 24h, 2h, or ISO date (default: 7d).")
@click.option("--limit", default=5000, type=int, help="Max traces to fetch (default: 5000).")
@click.option("--output", "-o", default=None,
              help="Output JSONL file path (default: @name → cached_sources/<name>.jsonl, "
                   "else traces.jsonl).")
@click.option("--tags", multiple=True,
              help="Filter traces by tag (Langfuse). Repeatable.")
@click.option("--session", default=None,
              help="Filter traces by session ID (Langfuse).")
@click.option("--sources-dir", default=None, type=click.Path(),
              help="Directory of source configs (default: config/sources/).")
@click.option("--refresh", is_flag=True, default=False,
              help="Re-pull even if a local cache already exists (only relevant for @name).")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, type=click.Path(),
              help="Directory for cached pulled traces.", show_default=True)
def pull(name: str | None, source: str | None, project: str | None,
         since: str, limit: int, output: str | None, tags: tuple,
         session: str | None, sources_dir: str | None, refresh: bool,
         cache_dir: str):
    """Pull traces from Langfuse, LangSmith, or local JSONL and save as cached JSONL.

    \b
    Three ways to invoke:

      # Named source from config/sources/:
      kalibra pull @current
      kalibra pull @current --output my-run.jsonl
      kalibra pull @current --refresh          # force re-pull, bypass cache

      # Explicit flags:
      kalibra pull --source langfuse --project my-agent --since 7d

    \b
    Named sources support source: jsonl for local files:
      # config/sources/local.yml
      my-data:
        source: jsonl
        path: /data/exports/traces.jsonl

    \b
    Environment variables:
      Langfuse:   LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
      LangSmith:  LANGSMITH_API_KEY
    """
    from kalibra.config import load_sources

    sources = load_sources(sources_dir)

    if name:
        if not name.startswith("@"):
            raise click.UsageError(f"Positional NAME must start with @, got: {name!r}")
        src_name = name[1:]
        if src_name not in sources:
            available = list(sources)
            hint = (
                f"Source '@{src_name}' not found in sources config.\n"
                f"  Available sources: {available or '(none defined)'}\n\n"
                f"  To use a named source, define it in a YAML file under config/sources/\n"
                f"  or pass --sources-dir to point at your sources directory.\n\n"
                f"  Alternatively, pull with explicit flags:\n"
                f"    kalibra pull --source langfuse --project my-agent --since 7d"
            )
            raise click.UsageError(hint)
        src = sources[src_name]
        dest = output or str(_cache_path(src_name, cache_dir=cache_dir))

        # Check cache
        cache = Path(dest)
        if not refresh and cache.exists():
            click.echo(f"Using cached data: {dest}  (use --refresh to re-pull)")
            _print_pull_summary(dest)
            return

        # CLI flags override source config
        effective_tags = list(tags) or src.tags
        effective_session = session or src.session
        _do_pull(src.source, src.project, src.since, src.limit, dest,
                 tags=effective_tags, session_id=effective_session,
                 source_config=src)
    else:
        if not source or not project:
            raise click.UsageError(
                "Provide either @NAME (from a sources config) "
                "or both --source and --project.\n\n"
                "  Named source:    kalibra pull @my-baseline\n"
                "  Explicit flags:  kalibra pull --source langfuse --project my-agent --since 7d"
            )
        dest = output or "traces.jsonl"
        _do_pull(source, project, since, limit, dest,
                 tags=list(tags), session_id=session)


# ── validate ──────────────────────────────────────────────────────────────────

@main.command()
@click.argument("path", type=click.Path())
def validate(path: str):
    """Validate a JSONL trace file and show summary statistics.

    \b
    Examples:
      kalibra validate traces.jsonl
      kalibra validate cached_sources/baseline.jsonl
    """
    p = Path(path)
    if not p.exists():
        raise click.UsageError(f"File not found: {path}")
    if not p.is_file():
        raise click.UsageError(f"Expected a file, got directory: {path}")

    from kalibra.converters.generic import load_json_traces

    try:
        traces = load_json_traces(p)
    except ValueError as exc:
        click.echo(f"\nValidation failed:\n{exc}", err=True)
        raise SystemExit(1) from None

    n = len(traces)
    if n == 0:
        click.echo(f"\n  {path}")
        click.echo(f"  0 traces (empty file)")
        click.echo(f"\n  Valid")
        return

    n_spans = sum(len(t.spans) for t in traces)
    successes = sum(1 for t in traces if t.outcome == "success")
    failures = sum(1 for t in traces if t.outcome == "failure")
    unset = sum(1 for t in traces if t.outcome is None)

    # Detect which format was used
    fmt = "flat eval" if n_spans == n else "flat span"

    # Data coverage — check what metrics will actually work
    from kalibra.converters.base import span_cost, span_input_tokens, span_output_tokens

    has_cost = sum(1 for t in traces if any(span_cost(s) > 0 for s in t.spans))
    has_tokens = sum(
        1 for t in traces
        if any(span_input_tokens(s) > 0 or span_output_tokens(s) > 0 for s in t.spans)
    )
    has_duration = sum(1 for t in traces if t.duration > 0)

    click.echo(f"\n  {path}")
    click.echo(f"  Format:       {fmt}")
    click.echo(f"  Traces:       {n:,}")
    click.echo(f"  Spans:        {n_spans:,}")
    click.echo(f"  Outcomes:     {successes:,} success, {failures:,} failure, {unset:,} unset")
    click.echo(f"  Success rate: {successes / n:.1%}")

    click.echo(f"\n  Data coverage:")
    click.echo(f"    Cost:     {has_cost:,}/{n:,} traces")
    click.echo(f"    Tokens:   {has_tokens:,}/{n:,} traces")
    click.echo(f"    Duration: {has_duration:,}/{n:,} traces")

    # Warnings
    warnings = []
    if unset == n:
        warnings.append(
            "No outcome data — success_rate and per_task metrics will be unavailable.\n"
            "    Set outcome in your JSONL, or use --outcome-field on compare to map\n"
            "    a metadata field. Example source config:\n"
            "      outcome:\n"
            "        field: metadata.result\n"
            "        success: [pass, resolved]"
        )
    elif successes == 0 and failures == 0 and unset > 0:
        warnings.append(
            "All outcomes are null — success_rate metric will show n/a.\n"
            "    Set 'outcome' to \"success\" or \"failure\" in your JSONL rows."
        )
    if has_cost == 0:
        warnings.append("No cost data — cost metrics will be unavailable.")
    if has_tokens == 0:
        warnings.append("No token data — token_usage and token_efficiency metrics will be unavailable.")
    if has_duration == 0:
        warnings.append("No duration data — duration metrics will be unavailable.")

    if warnings:
        click.echo(f"\n  Warnings:")
        for w in warnings:
            for line in w.split("\n"):
                click.echo(f"    ! {line}")

    click.echo(f"\n  Valid")


# ── helpers ────────────────────────────────────────────────────────────────────


def _cache_path(name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.jsonl"


def _resolve_source(arg: str, sources: dict, refresh: bool,
                     cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """If arg starts with @, pull the named source (using cache when available)."""
    if not arg.startswith("@"):
        return arg

    name = arg[1:]
    if name not in sources:
        available = list(sources)
        hint = (
            f"Source '@{name}' not found in sources config.\n"
            f"  Available sources: {available or '(none defined)'}\n\n"
            f"  To use a named source, define it in a YAML file under config/sources/\n"
            f"  or pass --sources to point at your sources directory.\n\n"
            f"  Alternatively, pass a file or directory path directly:\n"
            f"    kalibra compare --baseline ./traces.jsonl --current ./traces2.jsonl"
        )
        raise click.UsageError(hint)

    src = sources[name]
    dest = _cache_path(name, cache_dir=cache_dir)

    if not refresh and dest.exists():
        click.echo(f"  @{name}: using cache ({dest})")
        return str(dest)

    _do_pull(src.source, src.project, src.since, src.limit, str(dest),
             tags=src.tags, session_id=src.session, source_config=src)
    return str(dest)


def _do_pull(source: str, project: str, since: str, limit: int, output: str,
             tags: list[str] | None = None, session_id: str | None = None,
             source_config=None) -> None:
    from kalibra.converters.base import apply_overrides
    from kalibra.converters.generic import save_jsonl

    if source == "jsonl":
        # Local JSONL source — load from path, apply overrides, save to cache
        from kalibra.converters.generic import load_json_traces

        jsonl_path = source_config.path if source_config and source_config.path else project
        if not jsonl_path:
            raise click.UsageError("JSONL source requires a path (set 'path' in source config)")
        p = Path(jsonl_path)
        if not p.exists():
            raise click.UsageError(f"JSONL source file not found: {jsonl_path}")

        click.echo(f"Loading {jsonl_path}...")
        traces = load_json_traces(p)
    else:
        # Remote connector — fetch from Langfuse/LangSmith
        from kalibra.connectors import get_connector
        from kalibra.connectors.langfuse import parse_since

        extra = ""
        if tags:
            extra += f", tags: {tags}"
        if session_id:
            extra += f", session: {session_id}"
        click.echo(f"Connecting to {source} (project: {project}, since: {since}{extra})...")
        connector = get_connector(source)
        project_key = "project_id" if source == "langfuse" else "project_name"
        fetch_kwargs: dict = {
            project_key: project,
            "since": parse_since(since),
            "limit": limit,
            "progress": True,
        }
        if tags:
            fetch_kwargs["tags"] = tags
        if session_id:
            fetch_kwargs["session_id"] = session_id
        traces = connector.fetch(**fetch_kwargs)

    # Apply outcome/cost overrides from source config
    apply_overrides(traces, source_config)

    click.echo(f"Loaded {len(traces):,} traces.")
    save_jsonl(traces, output)
    click.echo(f"Saved to {output}")
    _print_pull_summary(traces=traces)


def _print_pull_summary(path: str | None = None, traces=None) -> None:
    if traces is None:
        if path is None:
            return
        from kalibra.converters.generic import load_json_traces
        try:
            traces = load_json_traces(Path(path))
        except Exception:
            return
    if not traces:
        return
    n = len(traces)
    successes = sum(1 for t in traces if t.outcome == "success")
    failures = sum(1 for t in traces if t.outcome == "failure")
    unset = sum(1 for t in traces if t.outcome is None)
    click.echo(f"  Success rate: {successes / n:.1%}")
    if unset == n:
        click.echo(
            "  ! No outcome data detected — success_rate will be unavailable.\n"
            "    Use outcome overrides in your source config to map a metadata field:\n"
            "      outcome:\n"
            "        field: metadata.result\n"
            "        success: [pass, resolved]"
        )
