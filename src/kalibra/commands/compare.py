"""Compare command — load traces, validate thresholds, run comparison."""

from __future__ import annotations

from pathlib import Path

import click

from kalibra import display

CONFIG_FILENAME = "kalibra.yml"


def _find_config() -> Path | None:
    """Walk up from CWD looking for kalibra.yml. Stop at filesystem root."""
    current = Path.cwd().resolve()
    while True:
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _pull_population(pop, name: str, cache_dir: str, refresh: bool = False) -> str:
    """Pull traces for a population config and return the local file path."""
    if pop.path:
        return pop.path

    if not pop.source or not pop.project:
        raise click.UsageError(
            f"Config '{name}' needs either 'path' or 'source' + 'project'."
        )

    from kalibra.commands.pull import cache_path, do_pull

    tag_slug = "-".join(pop.tags)[:50] if pop.tags else name
    cache_name = f"{pop.project}_{tag_slug}".replace("/", "_").replace(" ", "_")
    dest = cache_path(cache_name, cache_dir=cache_dir)

    if refresh or not dest.exists():
        do_pull(
            source=pop.source,
            project=pop.project,
            since=pop.since,
            limit=pop.limit,
            output=str(dest),
            tags=pop.tags or None,
            session_id=pop.session,
        )

    return str(dest)


def run_compare(
    baseline: str | None,
    current: str | None,
    out_format: str,
    require: tuple,
    config_path: str | None,
    sources_dir: str | None,
    output: str | None,
    refresh: bool,
    cache_dir: str,
    outcome: str | None,
    cost_field: str | None,
    task_id: str | None,
) -> None:
    """Execute the compare command."""
    from kalibra.compare import ThresholdError, compare_collections, validate_require_exprs
    from kalibra.config import CompareConfig, load_sources, resolve_metrics
    from kalibra.metrics import DEFAULT_METRICS
    from kalibra.report import render

    # ── Resolve config file ───────────────────────────────────────────────
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise click.UsageError(f"Config file not found: {config_path}")
        if p.is_dir():
            raise click.UsageError(
                f"--config must be a file, not a directory: {config_path}"
            )
        click.echo(click.style(f"  Using config: {config_path}", dim=True))
    else:
        discovered = _find_config()
        if discovered:
            config_path = str(discovered)
            click.echo(click.style(f"  Using {config_path}", dim=True))

    if sources_dir is not None:
        s = Path(sources_dir)
        if not s.exists() or not s.is_dir():
            raise click.UsageError(f"Sources path is not a directory: {sources_dir}")

    config = CompareConfig.load(config_path)

    # ── CLI overrides ─────────────────────────────────────────────────────
    if task_id:
        config.fields.task_id = task_id
    if outcome:
        config.fields.outcome = outcome
    if cost_field:
        config.fields.cost = cost_field
    if require:
        config.require = list(require)

    # ── Resolve baseline/current paths ────────────────────────────────────
    # Priority: CLI flag > config default. Flag value can be:
    #   - a file path (./traces.jsonl)
    #   - a named source from kalibra.yml sources: section (prod-v1)
    #   - a legacy @name reference (@baseline)

    def _resolve_flag_or_config(flag_value: str | None, config_pop, label: str) -> str | None:
        if flag_value:
            # Check if it's a named source from config.
            named = config.get_source(flag_value)
            if named:
                return _pull_population(named, label, cache_dir, refresh)
            # Legacy @name syntax.
            if flag_value.startswith("@"):
                from kalibra.commands.pull import resolve_source
                legacy_sources = load_sources(sources_dir)
                return resolve_source(flag_value, legacy_sources, refresh, cache_dir=cache_dir)
            # Treat as file path.
            return flag_value
        if config_pop:
            return _pull_population(config_pop, label, cache_dir, refresh)
        return None

    baseline_path = _resolve_flag_or_config(baseline, config.baseline, "baseline")
    current_path = _resolve_flag_or_config(current, config.current, "current")

    if not baseline_path or not current_path:
        raise click.UsageError(
            "--baseline and --current are required when not set in kalibra.yml.\n\n"
            "  Run 'kalibra init' to create a config, or pass paths directly:\n"
            "    kalibra compare --baseline ./baseline.jsonl --current ./current.jsonl"
        )

    # Early file-exists check.
    for label, p in [("Baseline", baseline_path), ("Current", current_path)]:
        if not Path(p).exists():
            raise click.UsageError(
                f"{label} path does not exist: {p}\n\n"
                "  Check the path, or run 'kalibra init' to configure sources."
            )

    # ── Validate thresholds early ─────────────────────────────────────────
    active_metrics = resolve_metrics(config, DEFAULT_METRICS)
    known_fields: set[str] = set()
    for m in active_metrics:
        known_fields.update(m.threshold_field_names())
    try:
        parsed_require = validate_require_exprs(config.require, known_fields)
    except ThresholdError as exc:
        display.threshold_error(exc)
        ctx = click.get_current_context()
        ctx.exit(2)

    # ── Load traces ───────────────────────────────────────────────────────
    from kalibra.collection import TraceCollection
    from kalibra.converters import load_traces
    from kalibra.converters.base import apply_overrides

    click.echo(f"Loading {baseline_path}")
    b_traces = load_traces(baseline_path)

    click.echo(f"Loading {current_path}")
    c_traces = load_traces(current_path)

    # Apply field overrides from config (outcome, cost).
    if config.fields.outcome or config.fields.cost:
        from kalibra.config import CostConfig, OutcomeConfig, SourceConfig

        override_cfg = SourceConfig(
            source="", project="",
            outcome=OutcomeConfig(field=config.fields.outcome) if config.fields.outcome else None,
            cost=CostConfig(attr=config.fields.cost) if config.fields.cost else None,
        )
        apply_overrides(b_traces, override_cfg)
        apply_overrides(c_traces, override_cfg)

    baseline_col = TraceCollection.from_traces(b_traces, source=baseline_path)
    current_col = TraceCollection.from_traces(c_traces, source=current_path)

    # ── Run comparison ────────────────────────────────────────────────────
    result = compare_collections(
        baseline_col, current_col,
        config=config,
        _parsed_require=parsed_require,
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
