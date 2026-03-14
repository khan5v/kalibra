"""Compare command — load traces, validate thresholds, run comparison."""

from __future__ import annotations

from pathlib import Path

import click

from kalibra import display
from kalibra.commands.pull import resolve_source


def run_compare(
    baseline: str,
    current: str,
    out_format: str,
    require: tuple,
    config_path: str | None,
    sources_dir: str | None,
    output: str | None,
    refresh: bool,
    cache_dir: str,
    outcome_field: str | None,
    cost_attr: str | None,
    task_id: str | None,
) -> None:
    """Execute the compare command — all business logic lives here."""
    from kalibra.compare import ThresholdError, compare_collections, validate_require_exprs
    from kalibra.config import CompareConfig, load_sources, resolve_metrics
    from kalibra.metrics import DEFAULT_METRICS
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
    if task_id:
        config.task_id = task_id

    # Validate threshold expressions early — before loading any data.
    active_metrics = resolve_metrics(config, DEFAULT_METRICS)
    known_fields: set[str] = set()
    for m in active_metrics:
        known_fields.update(m.threshold_field_names())
    all_require_raw = list(config.require) + list(require)
    try:
        parsed_require = validate_require_exprs(all_require_raw, known_fields)
    except ThresholdError as exc:
        display.threshold_error(exc)
        ctx = click.get_current_context()
        ctx.exit(2)

    sources = load_sources(sources_dir)
    baseline_path = resolve_source(baseline, sources, refresh, cache_dir=cache_dir)
    current_path = resolve_source(current, sources, refresh, cache_dir=cache_dir)

    # Early file-exists check
    for label, p in [("Baseline", baseline_path), ("Current", current_path)]:
        if not Path(p).exists():
            raise click.UsageError(
                f"{label} path does not exist: {p}\n\n"
                f"  If this is a named source, use @name syntax:\n"
                f"    kalibra compare --baseline @my-source --current @other-source\n\n"
                f"  If this is a file path, check the path is correct."
            )

    from kalibra.collection import TraceCollection

    if outcome_field or cost_attr:
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
    else:
        click.echo(f"Loading {baseline_path}")
        baseline_col = TraceCollection.from_path(baseline_path)

        click.echo(f"Loading {current_path}")
        current_col = TraceCollection.from_path(current_path)

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
