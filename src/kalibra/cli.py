"""Kalibra CLI — thin routing layer. Business logic lives in kalibra.commands.*."""

from __future__ import annotations

import click

from kalibra import display
from kalibra.commands.pull import DEFAULT_CACHE_DIR


@click.group()
@click.version_option()
def main():
    """Kalibra — agent evaluation and regression detection."""


# ── init ───────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing kalibra.yml without asking.")
def init(force):
    """Create a kalibra.yml config file interactively.

    \b
    Examples:
      kalibra init
      kalibra init --force
    """
    from kalibra.commands.init import run_init
    run_init(force=force)


# ── compare ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--baseline", default=None,
              help="Baseline traces: file/dir path or @name from sources.yml.")
@click.option("--current", default=None,
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
@click.option("--trace-id", "trace_id_field", default=None,
              help="Field name to use as trace ID (e.g. uuid, task_name).")
@click.option("--outcome", default=None,
              help="Metadata field for outcome detection (e.g. metadata.result).")
@click.option("--cost", "cost_field", default=None,
              help="Span attribute for cost (e.g. custom.cost_usd).")
@click.option("--task-id", default=None,
              help="Metadata field for per-task matching (e.g. braintrust.task_id).")
@click.option("-v", "--verbose", is_flag=True, default=False,
              help="Show detailed output — per-span breakdown, CIs, p-values.")
@click.option("--metrics", "show_metrics", is_flag=True, default=False,
              help="List all available metrics and their --require threshold fields, then exit.")
def compare(baseline, current, out_format, require, config_path, sources_dir,
            output, refresh, cache_dir, trace_id_field, outcome, cost_field,
            task_id, verbose, show_metrics):
    """Compare two trace datasets — regression detection, statistical diff.

    \b
    --baseline and --current accept:
      - A file or directory path (JSONL)
      - @name  — a named source from sources.yml (pulls and caches automatically)

    \b
    Examples:
      kalibra compare --baseline ./baseline/ --current ./current/
      kalibra compare --baseline @baseline --current @current
      kalibra compare --metrics
    """
    if show_metrics:
        display.metrics_list()
        return

    from kalibra.commands.compare import run_compare
    run_compare(
        baseline=baseline, current=current, out_format=out_format,
        require=require, config_path=config_path, sources_dir=sources_dir,
        output=output, refresh=refresh, cache_dir=cache_dir,
        trace_id_field=trace_id_field, outcome=outcome,
        cost_field=cost_field, task_id=task_id, verbose=verbose,
    )


# ── pull ───────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("name", required=False, default=None, metavar="[@NAME]")
@click.option("--source", type=click.Choice(["langfuse", "langsmith", "braintrust"]),
              help="Trace store to pull from.")
@click.option("--project", help="Project name.")
@click.option("--since", default="7d", help="Time window: 7d, 24h, or ISO date (default: 7d).")
@click.option("--limit", default=5000, type=int, help="Max traces to fetch (default: 5000).")
@click.option("--output", "-o", default=None, help="Output JSONL file path.")
@click.option("--tags", multiple=True, help="Filter traces by tag. Repeatable.")
@click.option("--session", default=None, help="Filter traces by session ID.")
@click.option("--sources-dir", default=None, type=click.Path(),
              help="Directory of source configs (default: config/sources/).")
@click.option("--refresh", is_flag=True, default=False,
              help="Re-pull even if a local cache already exists.")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, type=click.Path(),
              help="Directory for cached pulled traces.", show_default=True)
def pull(name, source, project, since, limit, output, tags, session, sources_dir,
         refresh, cache_dir):
    """Pull traces from Langfuse, LangSmith, or Braintrust and save as JSONL.

    \b
    Examples:
      kalibra pull @current
      kalibra pull @current --refresh
      kalibra pull --source langfuse --project my-agent --since 7d
    """
    from pathlib import Path

    from kalibra.commands.compare import _find_config
    from kalibra.commands.pull import cache_path, do_pull
    from kalibra.config import CompareConfig, load_sources

    if name:
        if not name.startswith("@"):
            raise click.UsageError(f"Positional NAME must start with @, got: {name!r}")
        src_name = name[1:]

        # Look up in kalibra.yml sources first, then legacy config/sources/.
        pop = None
        kalibra_yml = _find_config()
        if kalibra_yml:
            cfg = CompareConfig.load(str(kalibra_yml))
            pop = cfg.get_source(src_name)

        if pop is None:
            # Fall back to legacy sources.
            legacy = load_sources(sources_dir)
            if src_name in legacy:
                src = legacy[src_name]
                dest = output or str(cache_path(src_name, cache_dir=cache_dir))
                cache = Path(dest)
                if not refresh and cache.exists():
                    click.echo(f"Using cached data: {dest}  (use --refresh to re-pull)")
                    display.pull_summary(dest)
                    return
                do_pull(src.source, src.project, src.since, src.limit, dest,
                        tags=list(tags) or src.tags,
                        session_id=session or src.session,
                        source_config=src)
                return

            # Build available list from both.
            available = list((cfg.sources if kalibra_yml and cfg else {}).keys())
            available += list(legacy.keys())
            raise click.UsageError(
                f"Source '@{src_name}' not found.\n"
                f"  Available: {available or '(none defined)'}\n\n"
                f"  Define sources in kalibra.yml, or use explicit flags:\n"
                f"    kalibra pull --source langfuse --project my-agent --since 7d"
            )

        # Pull from kalibra.yml source.
        if not pop.source or not pop.project:
            raise click.UsageError(
                f"Source '@{src_name}' needs 'source' and 'project' fields."
            )
        dest = output or str(cache_path(src_name, cache_dir=cache_dir))
        cache = Path(dest)
        if not refresh and cache.exists():
            click.echo(f"Using cached data: {dest}  (use --refresh to re-pull)")
            display.pull_summary(dest)
            return

        effective_tags = list(tags) or pop.tags
        effective_session = session or pop.session
        do_pull(pop.source, pop.project, pop.since, pop.limit, dest,
                tags=effective_tags, session_id=effective_session)
    else:
        if not source or not project:
            raise click.UsageError(
                "Provide either @NAME or both --source and --project.\n\n"
                "  kalibra pull @my-baseline\n"
                "  kalibra pull --source langfuse --project my-agent --since 7d"
            )
        dest = output or "traces.jsonl"
        do_pull(source, project, since, limit, dest,
                tags=list(tags), session_id=session)


# ── inspect ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", "config_path", default=None, type=click.Path(),
              help="Compare config — inspect only checks fields needed by active metrics.")
def inspect(path, config_path):
    """Inspect a trace file — show data coverage, available fields, and config suggestions.

    \b
    Examples:
      kalibra inspect cached_sources/baseline.jsonl
      kalibra inspect cached_sources/baseline.jsonl --config config/examples/ci-gate.yml
    """
    from kalibra.commands.inspect import run_inspect
    run_inspect(path, config_path)
