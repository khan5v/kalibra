"""Pull command — fetch traces from connectors and save as JSONL."""

from __future__ import annotations

from pathlib import Path

import click

from kalibra import display

DEFAULT_CACHE_DIR = "cached_sources"


def cache_path(name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.jsonl"


def resolve_source(arg: str, sources: dict, refresh: bool,
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
    dest = cache_path(name, cache_dir=cache_dir)

    if not refresh and dest.exists():
        click.echo(f"  @{name}: using cache ({dest})")
        return str(dest)

    do_pull(src.source, src.project, src.since, src.limit, str(dest),
            tags=src.tags, session_id=src.session, source_config=src)
    return str(dest)


def do_pull(source: str, project: str, since: str, limit: int, output: str,
            tags: list[str] | None = None, session_id: str | None = None,
            source_config=None) -> None:
    """Execute a pull — fetch traces and save to JSONL."""
    from kalibra.converters.base import apply_overrides
    from kalibra.converters.generic import save_jsonl

    if source == "jsonl":
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
        from kalibra.connectors import get_connector
        from kalibra.connectors.langfuse import parse_since

        extra = ""
        if tags:
            extra += f", tags: {tags}"
        if session_id:
            extra += f", session: {session_id}"
        click.echo(f"Connecting to {source} (project: {project}, since: {since}{extra})...")

        try:
            connector = get_connector(source)
        except RuntimeError as exc:
            display.connector_error(source, str(exc))
            raise SystemExit(1) from None

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

        try:
            traces = connector.fetch(**fetch_kwargs)
        except RuntimeError as exc:
            display.connector_error(source, str(exc))
            raise SystemExit(1) from None

    apply_overrides(traces, source_config)

    if not traces and source != "jsonl":
        display.no_traces_warning(source, project, tags, session_id, since)

    click.echo(f"Loaded {len(traces):,} traces.")
    save_jsonl(traces, output)
    click.echo(f"Saved to {output}")
    display.pull_summary(traces=traces)
