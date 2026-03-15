"""Kalibra CLI — thin routing layer. Business logic lives in kalibra.commands.*."""

from __future__ import annotations

import click

from kalibra import display


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
              help="Baseline traces: JSONL file path or named source from kalibra.yml.")
@click.option("--current", default=None,
              help="Current traces: JSONL file path or named source from kalibra.yml.")
@click.option("--format", "out_format",
              type=click.Choice(["terminal", "markdown", "json"]),
              default="terminal", help="Output format (default: terminal).")
@click.option("--require", "-r", multiple=True,
              help="Threshold expression, e.g. 'success_rate_delta >= -2'.")
@click.option("--config", "config_path", default=None, type=click.Path(),
              help="Path to kalibra.yml config file (default: auto-discovered).")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write output to file instead of stdout.")
@click.option("--trace-id", "trace_id_field", default=None,
              help="Field name to use as trace ID (e.g. uuid, task_name).")
@click.option("--outcome", default=None,
              help="Field path for outcome detection (e.g. metadata.result).")
@click.option("--cost", "cost_field", default=None,
              help="Field path for cost (e.g. agent_cost.total_cost).")
@click.option("--task-id", default=None,
              help="Field path for per-task matching (e.g. metadata.task_name).")
@click.option("-v", "--verbose", is_flag=True, default=False,
              help="Show detailed output — per-span breakdown, CIs, p-values.")
@click.option("--metrics", "show_metrics", is_flag=True, default=False,
              help="List all available metrics and their --require threshold fields, then exit.")
def compare(baseline, current, out_format, require, config_path,
            output, trace_id_field, outcome, cost_field,
            task_id, verbose, show_metrics):
    """Compare two trace datasets — regression detection, statistical diff.

    \b
    --baseline and --current accept:
      - A JSONL file path
      - A named source from kalibra.yml sources: section

    \b
    Examples:
      kalibra compare --baseline ./baseline.jsonl --current ./current.jsonl
      kalibra compare                              (uses kalibra.yml)
      kalibra compare --metrics
    """
    if show_metrics:
        display.metrics_list()
        return

    from kalibra.commands.compare import run_compare
    run_compare(
        baseline=baseline, current=current, out_format=out_format,
        require=require, config_path=config_path,
        output=output,
        trace_id_field=trace_id_field, outcome=outcome,
        cost_field=cost_field, task_id=task_id, verbose=verbose,
    )


# ── inspect ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", "config_path", default=None, type=click.Path(),
              help="Compare config — inspect only checks fields needed by active metrics.")
def inspect(path, config_path):
    """Inspect a trace file — show data coverage, available fields, and config suggestions.

    \b
    Examples:
      kalibra inspect traces.jsonl
      kalibra inspect traces.jsonl --config kalibra.yml
    """
    from kalibra.commands.inspect import run_inspect
    run_inspect(path, config_path)
