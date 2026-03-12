"""AgentFlow CLI."""

from __future__ import annotations

import os
from pathlib import Path

import click


@click.group()
@click.version_option()
def main():
    """AgentFlow — agent evaluation and regression detection."""


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
def compare(baseline: str, current: str, out_format: str, require: tuple,
            config_path: str | None, sources_dir: str | None, output: str | None, refresh: bool):
    """Compare two trace datasets — regression detection, statistical diff.

    \b
    --baseline and --current accept:
      - A file or directory path (SWE-bench, JSONL — auto-detected)
      - @name  — a named source from sources.yml (pulls and caches automatically)

    \b
    --config accepts:
      - A bare name (e.g. myteam) → resolved from ~/.config/agentflow/myteam.yml or myteam/
      - A file path  → used directly as compare.yml
      - A directory  → uses <dir>/compare.yml and <dir>/sources/

    \b
    Examples:
      agentflow compare --baseline ./baseline/ --current ./current/
      agentflow compare --baseline @baseline --current @current
      agentflow compare --baseline @baseline --current @current --config /data/configs/prod.yml
      agentflow compare --baseline @baseline --current @current --config ~/my-configs/
    """
    from pathlib import Path

    from agentflow.compare import compare as _compare
    from agentflow.config import CompareConfig, load_sources
    from agentflow.report import render

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
    baseline_path = _resolve_source(baseline, sources, refresh)
    current_path = _resolve_source(current, sources, refresh)

    result = _compare(baseline_path, current_path, require=list(require) or None, config=config)
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
              help="Output JSONL file path (default: @name → .agentflow/<name>.jsonl, else traces.jsonl).")
@click.option("--tags", multiple=True,
              help="Filter traces by tag (Langfuse). Repeatable.")
@click.option("--session", default=None,
              help="Filter traces by session ID (Langfuse).")
@click.option("--sources-dir", default=None, type=click.Path(),
              help="Directory of source configs (default: config/sources/).")
@click.option("--refresh", is_flag=True, default=False,
              help="Re-pull even if a local cache already exists (only relevant for @name).")
def pull(name: str | None, source: str | None, project: str | None,
         since: str, limit: int, output: str | None, tags: tuple,
         session: str | None, sources_dir: str | None, refresh: bool):
    """Pull traces from Langfuse or LangSmith and save as JSONL.

    \b
    Two ways to invoke:

      # Named source from config/sources/:
      agentflow pull @current
      agentflow pull @current --output my-run.jsonl
      agentflow pull @current --refresh          # force re-pull, bypass cache

      # Explicit flags:
      agentflow pull --source langfuse --project my-agent --since 7d

    \b
    Environment variables:
      Langfuse:   LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
      LangSmith:  LANGSMITH_API_KEY
    """
    from agentflow.config import load_sources

    sources = load_sources(sources_dir)

    if name:
        if not name.startswith("@"):
            raise click.UsageError(f"Positional NAME must start with @, got: {name!r}")
        src_name = name[1:]
        if src_name not in sources:
            available = list(sources)
            raise click.UsageError(
                f"Source '@{src_name}' not in sources.yml. Available: {available or ['(none defined)']}"
            )
        src = sources[src_name]
        dest = output or str(_cache_path(src_name))

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
                 tags=effective_tags, session_id=effective_session)
    else:
        if not source or not project:
            raise click.UsageError(
                "Provide either @NAME (from sources.yml) "
                "or both --source and --project."
            )
        dest = output or "traces.jsonl"
        _do_pull(source, project, since, limit, dest,
                 tags=list(tags), session_id=session)


# ── helpers ────────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    cache_dir = Path("cached_sources")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{name}.jsonl"


def _resolve_source(arg: str, sources: dict, refresh: bool) -> str:
    """If arg starts with @, pull the named source (using cache when available)."""
    if not arg.startswith("@"):
        return arg

    name = arg[1:]
    if name not in sources:
        available = list(sources)
        raise click.UsageError(
            f"Source '@{name}' not in sources.yml. Available: {available or ['(none defined)']}"
        )

    src = sources[name]
    dest = _cache_path(name)

    if not refresh and dest.exists():
        click.echo(f"  @{name}: using cache ({dest})")
        return str(dest)

    _do_pull(src.source, src.project, src.since, src.limit, str(dest),
             tags=src.tags, session_id=src.session)
    return str(dest)


def _do_pull(source: str, project: str, since: str, limit: int, output: str,
             tags: list[str] | None = None, session_id: str | None = None) -> None:
    from agentflow.connectors import get_connector
    from agentflow.connectors.langfuse import parse_since
    from agentflow.converters.generic import save_jsonl

    extra = ""
    if tags:
        extra += f", tags: {tags}"
    if session_id:
        extra += f", session: {session_id}"
    click.echo(f"Connecting to {source} (project: {project}, since: {since}{extra})...")
    connector = get_connector(source)
    fetch_kwargs: dict = dict(
        project_id=project, since=parse_since(since), limit=limit, progress=True,
    )
    # Langfuse supports tag/session filtering; other connectors ignore these.
    if source == "langfuse":
        if tags:
            fetch_kwargs["tags"] = tags
        if session_id:
            fetch_kwargs["session_id"] = session_id
    traces = connector.fetch(**fetch_kwargs)
    click.echo(f"Fetched {len(traces):,} traces.")
    save_jsonl(traces, output)
    click.echo(f"Saved to {output}")
    _print_pull_summary(output, traces)


def _print_pull_summary(path: str, traces=None) -> None:
    if traces is None:
        from agentflow.converters.generic import load_json_traces
        try:
            traces = load_json_traces(Path(path))
        except Exception:
            return
    if traces:
        successes = sum(1 for t in traces if t.outcome == "success")
        click.echo(f"  Success rate: {successes / len(traces):.1%}")
