"""Kalibra CLI — thin routing layer. Business logic lives in kalibra.commands.*."""

from __future__ import annotations

import click

from kalibra import display


@click.group()
@click.version_option()
def main():
    """Kalibra — agent evaluation and regression detection."""


# ── demo ──────────────────────────────────────────────────────────────────────

@main.command()
def demo():
    """Run a comparison on built-in sample data to see Kalibra in action."""
    from kalibra.commands.demo import run_demo
    run_demo()


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
@click.argument("files", nargs=-1, type=click.Path())
@click.option("--baseline", default=None,
              help="Baseline traces (alternative to positional args).")
@click.option("--current", default=None,
              help="Current traces (alternative to positional args).")
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
@click.option("--input-tokens", "input_tokens_field", default=None,
              help="Field path for input tokens (e.g. usage.prompt_tokens).")
@click.option("--output-tokens", "output_tokens_field", default=None,
              help="Field path for output tokens (e.g. usage.completion_tokens).")
@click.option("--duration", "duration_field", default=None,
              help="Field path for duration (e.g. elapsed_time).")
@click.option("-v", "--verbose", is_flag=True, default=False,
              help="Show detailed output — per-span breakdown, CIs, p-values.")
@click.option("-q", "--quiet", is_flag=True, default=False,
              help="Suppress status messages (config discovery, loading). CI-friendly.")
@click.option("--metrics", "show_metrics", is_flag=True, default=False,
              help="List all available metrics and their --require threshold fields, then exit.")
def compare(files, baseline, current, out_format, require, config_path,
            output, trace_id_field, outcome, cost_field,
            task_id, input_tokens_field, output_tokens_field, duration_field,
            verbose, quiet, show_metrics):
    """Compare two trace datasets — regression detection, statistical diff.

    \b
    Examples:
      kalibra compare baseline.jsonl current.jsonl
      kalibra compare --baseline a.jsonl --current b.jsonl
      kalibra compare                              (uses kalibra.yml)
      kalibra compare --metrics
    """
    if show_metrics:
        display.metrics_list()
        return

    # Positional args: kalibra compare baseline.jsonl current.jsonl
    if files:
        if len(files) == 2:
            baseline = baseline or files[0]
            current = current or files[1]
        elif len(files) == 1:
            raise click.UsageError(
                "Two files required: kalibra compare baseline.jsonl current.jsonl"
            )
        else:
            raise click.UsageError(
                f"Expected 2 files, got {len(files)}. "
                "Usage: kalibra compare baseline.jsonl current.jsonl"
            )

    from kalibra.commands.compare import run_compare
    run_compare(
        baseline=baseline, current=current, out_format=out_format,
        require=require, config_path=config_path,
        output=output,
        trace_id_field=trace_id_field, outcome=outcome,
        cost_field=cost_field, task_id=task_id,
        input_tokens_field=input_tokens_field,
        output_tokens_field=output_tokens_field,
        duration_field=duration_field,
        verbose=verbose, quiet=quiet,
    )


# ── inspect ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--config", "config_path", default=None, type=click.Path(),
              help="Config file for field mappings and metric selection.")
@click.option("--trace-id", "trace_id_field", default=None,
              help="Field name to use as trace ID (e.g. uuid, task_name).")
@click.option("--outcome", default=None,
              help="Field path for outcome detection (e.g. metadata.result).")
@click.option("--cost", "cost_field", default=None,
              help="Field path for cost (e.g. agent_cost.total_cost).")
@click.option("--task-id", default=None,
              help="Field path for per-task matching (e.g. metadata.task_name).")
@click.option("--input-tokens", "input_tokens_field", default=None,
              help="Field path for input tokens (e.g. usage.prompt_tokens).")
@click.option("--output-tokens", "output_tokens_field", default=None,
              help="Field path for output tokens (e.g. usage.completion_tokens).")
@click.option("--duration", "duration_field", default=None,
              help="Field path for duration (e.g. elapsed_time).")
@click.option("--suggest", is_flag=True, default=False,
              help="Suggest field mappings based on field names in the data.")
def inspect(path, config_path, trace_id_field, outcome, cost_field, task_id,
            input_tokens_field, output_tokens_field, duration_field, suggest):
    """Inspect a trace file — show data coverage, available fields, and config suggestions.

    \b
    Examples:
      kalibra inspect traces.jsonl
      kalibra inspect traces.jsonl --suggest
      kalibra inspect traces.jsonl --config kalibra.yml
    """
    from kalibra.commands.inspect import run_inspect
    run_inspect(
        path, config_path,
        trace_id_field=trace_id_field, outcome=outcome,
        cost_field=cost_field, task_id=task_id,
        input_tokens_field=input_tokens_field,
        output_tokens_field=output_tokens_field,
        duration_field=duration_field,
        suggest=suggest,
    )
