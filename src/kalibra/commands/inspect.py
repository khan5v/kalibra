"""Inspect command — show data coverage, available fields, and config suggestions."""

from __future__ import annotations

from pathlib import Path

import click

from kalibra import display


def run_inspect(path: str, config_path: str | None) -> None:
    """Execute the inspect command."""
    from kalibra.config import CompareConfig, find_config
    from kalibra.engine import resolve_metrics
    from kalibra.loader import load_traces
    from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS

    # Load config — needed for trace_id field mapping.
    if not config_path:
        discovered = find_config()
        if discovered:
            config_path = str(discovered)
    config = CompareConfig.load(config_path)

    try:
        traces = load_traces(path, fields=config.fields)
    except ValueError as exc:
        display.load_error(path, str(exc))
        raise SystemExit(1) from None

    n = len(traces)
    if n == 0:
        click.echo(f"\n  {path}: 0 traces (empty file)")
        return

    n_spans = sum(len(t.spans) for t in traces)

    # Determine active metrics.
    active = resolve_metrics(config.metrics)
    active_names = {m.name for m in active}

    # ── Data coverage ─────────────────────────────────────────────────────
    has_outcome = sum(
        1 for t in traces if t.outcome in (OUTCOME_SUCCESS, OUTCOME_FAILURE)
    )
    has_cost = sum(1 for t in traces if t.total_cost > 0)
    has_tokens = sum(1 for t in traces if t.total_tokens > 0)
    has_duration = sum(1 for t in traces if t.duration > 0)

    # Task ID: check if traces can be grouped into tasks.
    # A task_id is "extractable" if either metadata has a task_id field,
    # or trace IDs share prefixes (e.g. "task-1__model__0", "task-1__model__1").
    task_ids = set()
    has_metadata_task_id = False
    for t in traces:
        mid = t.metadata.get("task_id")
        if mid is not None:
            task_ids.add(str(mid))
            has_metadata_task_id = True
        else:
            # Try prefix extraction (strip __model__index suffix).
            parts = t.trace_id.split("__")
            if len(parts) >= 3 and parts[-1].isdigit():
                task_ids.add("__".join(parts[:-2]))
            else:
                task_ids.add(t.trace_id)
    task_id_extractable = has_metadata_task_id or len(task_ids) < n

    # ── Metadata keys ─────────────────────────────────────────────────────
    meta_keys: dict[str, int] = {}
    meta_unique: dict[str, set] = {}
    for t in traces:
        for k, v in (t.metadata or {}).items():
            meta_keys[k] = meta_keys.get(k, 0) + 1
            meta_unique.setdefault(k, set()).add(str(v)[:100])

    # ── Span attribute keys ───────────────────────────────────────────────
    attr_keys: dict[str, int] = {}
    for t in traces:
        for s in t.spans:
            for k in (s.attributes or {}):
                attr_keys[k] = attr_keys.get(k, 0) + 1

    # ── Render ────────────────────────────────────────────────────────────
    b = display.bar()
    d = display.dot()

    click.echo()
    click.echo(f"  {click.style('Kalibra Inspect', bold=True)}")
    click.echo(f"  {b}")
    click.echo(f"  {click.style('Source', dim=True)}    {path}")
    click.echo(f"  {click.style('Traces', dim=True)}    {n:,}")
    click.echo(f"  {click.style('Spans', dim=True)}     {n_spans:,} ({n_spans / n:.1f} avg/trace)")
    click.echo()

    # ── Metric readiness ──────────────────────────────────────────────────
    click.echo(f"  {click.style('Metric readiness', bold=True)}")
    if config_path or config.metrics:
        click.echo(f"  {click.style('(based on active config)', dim=True)}")
    click.echo()

    needs_outcome = active_names & {
        "success_rate", "trace_breakdown", "token_efficiency", "cost_quality",
    }
    needs_cost = active_names & {"cost", "cost_quality"}
    needs_tokens = active_names & {"token_usage", "token_efficiency"}
    needs_duration = active_names & {"duration"}
    needs_task_id = active_names & {"trace_breakdown"}

    def _coverage_line(label: str, count: int, total: int, needed_by: set[str]):
        if not needed_by:
            return
        ok = click.style("✓", fg="green") if count > 0 else click.style("✗", fg="yellow")
        count_str = f"{count}/{total}"
        metric_list = ", ".join(sorted(needed_by))
        click.echo(
            f"    {ok} {click.style(label, bold=True):<24s}"
            f"{count_str:>8s} traces"
            f"  {click.style(f'({metric_list})', dim=True)}"
        )

    _coverage_line("Outcome", has_outcome, n, needs_outcome)
    _coverage_line("Cost", has_cost, n, needs_cost)
    _coverage_line("Tokens", has_tokens, n, needs_tokens)
    _coverage_line("Duration", has_duration, n, needs_duration)

    if needs_task_id:
        if task_id_extractable:
            _coverage_line("Task ID", len(task_ids), n, {"trace_breakdown"})
        else:
            ok = click.style("✗", fg="yellow")
            click.echo(
                f"    {ok} {click.style('Task ID', bold=True):<24s}"
                f"{'0/'+str(n):>8s} traces"
                f"  {click.style('(trace_breakdown)', dim=True)}"
            )
            hint = "each trace has a unique ID — set fields.task_id in config"
            click.echo(f"      {click.style(hint, dim=True)}")

    click.echo()

    # ── Trace metadata ────────────────────────────────────────────────────
    click.echo(f"  {click.style('Trace metadata', bold=True)}")
    click.echo()

    if meta_keys:
        for key in sorted(meta_keys, key=lambda k: -meta_keys[k]):
            count = meta_keys[key]
            n_unique = len(meta_unique.get(key, set()))
            unique_str = f"{n_unique} unique" if n_unique > 1 else "1 value"
            padding = max(1, 34 - len(key))
            click.echo(
                f"    {click.style(key, fg='white')}"
                f"  {d * padding}  "
                f"{click.style(f'{count}/{n} traces, {unique_str}', dim=True)}"
            )
    else:
        click.echo(f"    {click.style('(no metadata)', dim=True)}")

    click.echo()

    # ── Span attributes ───────────────────────────────────────────────────
    click.echo(f"  {click.style('Span attributes', bold=True)}")
    click.echo()

    if attr_keys:
        for key in sorted(attr_keys, key=lambda k: -attr_keys[k]):
            count = attr_keys[key]
            padding = max(1, 34 - len(key))
            click.echo(
                f"    {click.style(key, fg='white')}"
                f"  {d * padding}  "
                f"{click.style(f'{count}/{n_spans} spans', dim=True)}"
            )
    else:
        click.echo(f"    {click.style('(no attributes)', dim=True)}")

    click.echo()

    # ── Config suggestions ────────────────────────────────────────────────
    suggestions = []
    if needs_task_id and not task_id_extractable:
        suggestions.append("task_id: <metadata key>")
    if needs_outcome and has_outcome == 0:
        suggestions.append("outcome: <metadata key>")
    if needs_cost and has_cost == 0:
        suggestions.append("cost: <span attribute>")

    if suggestions:
        click.echo(f"  {b}")
        click.echo(
            f"  {click.style('To fix missing fields, add to', dim=True)} "
            f"{click.style('kalibra.yml', fg='cyan')}{click.style(':', dim=True)}"
        )
        click.echo(click.style("    fields:", dim=True))
        for s in suggestions:
            click.echo(click.style(f"      {s}", dim=True))
        click.echo()
    else:
        click.echo(f"  {b}")
        click.echo(f"  {click.style('All active metrics have data.', fg='green')}")
        click.echo()
