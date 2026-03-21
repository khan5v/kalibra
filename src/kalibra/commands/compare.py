"""Compare command — load traces, validate thresholds, run comparison."""

from __future__ import annotations

from pathlib import Path

import click

from kalibra import display


def _status(msg: str, out_format: str, quiet: bool) -> None:
    """Print status line. Suppressed by --quiet, sent to stderr for json/markdown."""
    if quiet:
        return
    if out_format in ("json", "markdown"):
        click.echo(click.style(msg, dim=True), err=True)
    else:
        click.echo(click.style(msg, dim=True))


def run_compare(
    baseline: str | None,
    current: str | None,
    out_format: str,
    require: tuple,
    config_path: str | None,
    output: str | None,
    trace_id_field: str | None,
    outcome: str | None,
    cost_field: str | None,
    task_id: str | None,
    input_tokens_field: str | None = None,
    output_tokens_field: str | None = None,
    duration_field: str | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Execute the compare command."""
    from kalibra.config import CompareConfig, find_config
    from kalibra.engine import ThresholdError, compare, resolve_metrics
    from kalibra.loader import load_traces
    from kalibra.renderers import render

    # ── Resolve config file ───────────────────────────────────────────────
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise click.UsageError(f"Config file not found: {config_path}")
        if p.is_dir():
            raise click.UsageError(
                f"--config must be a file, not a directory: {config_path}"
            )
        _status(f"  Using config: {config_path}", out_format, quiet)
    else:
        discovered = find_config()
        if discovered:
            config_path = str(discovered)
            _status(f"  Using {config_path}", out_format, quiet)

    config = CompareConfig.load(config_path)

    # ── CLI overrides ─────────────────────────────────────────────────────
    if trace_id_field:
        config.fields.trace_id = trace_id_field
    if task_id:
        config.fields.task_id = task_id
    if outcome:
        config.fields.outcome = outcome
    if cost_field:
        config.fields.cost = cost_field
    if input_tokens_field:
        config.fields.input_tokens = input_tokens_field
    if output_tokens_field:
        config.fields.output_tokens = output_tokens_field
    if duration_field:
        config.fields.duration = duration_field
    if require:
        config.require = list(require)

    # ── Resolve baseline/current populations ─────────────────────────────
    baseline_path, b_resolved = _resolve_pop(
        baseline, config.baseline, "baseline", config,
    )
    current_path, c_resolved = _resolve_pop(
        current, config.current, "current", config,
    )

    if not baseline_path or not current_path:
        display.no_data()
        ctx = click.get_current_context()
        ctx.exit(0)

    # Early file-exists check.
    for label, p in [("Baseline", baseline_path), ("Current", current_path)]:
        if not Path(p).exists():
            display.header("Kalibra", "file not found")
            click.echo(
                f"  {click.style(label, bold=True)} path does not exist: "
                f"{click.style(p, fg='cyan')}"
            )
            click.echo()
            click.echo(f"  {display.bar()}")
            click.echo(
                f"  {click.style('Check the path in your', dim=True)} "
                f"{click.style('kalibra.yml', fg='cyan')}"
                f"{click.style(', or pass files directly:', dim=True)}"
            )
            click.echo(
                f"  kalibra compare {click.style('baseline.jsonl current.jsonl', dim=True)}"
            )
            click.echo()
            ctx = click.get_current_context()
            ctx.exit(1)

    # ── Validate thresholds early ─────────────────────────────────────────
    active_metrics = resolve_metrics(config.metrics)
    known_fields: set[str] = set()
    for m in active_metrics:
        known_fields.update(m.threshold_field_names())
    try:
        from kalibra.engine import _validate_require
        _validate_require(config.require, known_fields)
    except ThresholdError as exc:
        display.threshold_error(exc)
        ctx = click.get_current_context()
        ctx.exit(2)

    # ── Load traces ───────────────────────────────────────────────────────
    b_pop = b_resolved or config.baseline
    c_pop = c_resolved or config.current
    b_fields = config.fields.merge(b_pop.fields if b_pop else None)
    c_fields = config.fields.merge(c_pop.fields if c_pop else None)

    try:
        _status(f"Loading {baseline_path}", out_format, quiet)
        b_traces = load_traces(baseline_path, fields=b_fields)

        _status(f"Loading {current_path}", out_format, quiet)
        c_traces = load_traces(current_path, fields=c_fields)
    except ValueError as exc:
        display.load_error(baseline_path, str(exc))
        raise SystemExit(1) from None

    # ── Apply where filters ────────────────────────────────────────────
    b_traces = _apply_where(b_traces, b_pop, "baseline", out_format, quiet)
    c_traces = _apply_where(c_traces, c_pop, "current", out_format, quiet)

    # ── Build metric_config from fields ───────────────────────────────────
    metric_config: dict[str, dict] = {}
    effective_task_id = (
        b_fields.task_id or c_fields.task_id or config.fields.task_id
    )
    if effective_task_id:
        metric_config["trace_breakdown"] = {"task_id_field": effective_task_id}

    # ── Run comparison ────────────────────────────────────────────────────
    result = compare(
        b_traces, c_traces,
        metrics=config.metrics,
        require=config.require,
        baseline_source=baseline_path,
        current_source=current_path,
        noise_thresholds=config.noise_thresholds,
        metric_config=metric_config,
    )

    text = render(result, out_format, verbose=verbose)

    if output:
        with open(output, "w") as f:
            f.write(text)
        _status(f"Report written to {output}", out_format, quiet)
    else:
        click.echo(text)

    if not result.passed:
        raise SystemExit(1)


def _resolve_where_field(trace, field: str):
    """Look up a field for where-filtering: metadata first, then root span attributes."""
    val = trace.metadata.get(field)
    if val is not None:
        return val
    # Fall back to root span attributes — catches fields skipped by
    # OpenInference metadata extraction (llm.*, openinference.*, etc.).
    for s in trace.spans:
        if s.parent_id is None:
            val = s.attributes.get(field)
            if val is not None:
                return val
    return None


def _apply_where(traces, pop, label, out_format, quiet):
    """Filter traces using population where-matchers. Returns filtered list."""
    if not pop or not pop.where:
        return traces
    before = len(traces)
    filtered = [
        t for t in traces
        if all(m.matches(_resolve_where_field(t, m.field)) for m in pop.where)
    ]
    after = len(filtered)
    exprs = ", ".join(f"{m.field} {m.op} {m.value}" for m in pop.where)
    _status(
        f"  {label}: {after}/{before} traces matched (where: {exprs})",
        out_format, quiet,
    )
    return filtered


def _resolve_pop(
    flag_value: str | None,
    config_pop,
    label: str,
    config,
) -> tuple[str | None, PopulationConfig | None]:
    """Resolve a file path and population config from CLI flag or config.

    When a CLI flag references a named source, returns that source's
    PopulationConfig so its fields and where-clauses are used.
    """
    if flag_value:
        # Check if it's a named source from config.
        named = config.get_source(flag_value)
        if named and named.path:
            return named.path, named
        # Treat as file path — no population config.
        return flag_value, None
    if config_pop and config_pop.path:
        return config_pop.path, config_pop
    return None, None
