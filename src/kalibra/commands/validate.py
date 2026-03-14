"""Validate command — check a JSONL trace file and show summary statistics."""

from __future__ import annotations

from pathlib import Path

import click


def run_validate(path: str) -> None:
    """Execute the validate command."""
    from kalibra.converters.base import span_cost, span_input_tokens, span_output_tokens
    from kalibra.converters.generic import load_json_traces

    p = Path(path)
    if not p.exists():
        raise click.UsageError(f"File not found: {path}")
    if not p.is_file():
        raise click.UsageError(f"Expected a file, got directory: {path}")

    try:
        traces = load_json_traces(p)
    except ValueError as exc:
        click.echo(f"\nValidation failed:\n{exc}", err=True)
        raise SystemExit(1) from None

    n = len(traces)
    if n == 0:
        click.echo(f"\n  {path}")
        click.echo("  0 traces (empty file)")
        click.echo("\n  Valid")
        return

    n_spans = sum(len(t.spans) for t in traces)
    successes = sum(1 for t in traces if t.outcome == "success")
    failures = sum(1 for t in traces if t.outcome == "failure")
    unset = sum(1 for t in traces if t.outcome is None)

    fmt = "flat eval" if n_spans == n else "flat span"

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

    click.echo("\n  Data coverage:")
    click.echo(f"    Cost:     {has_cost:,}/{n:,} traces")
    click.echo(f"    Tokens:   {has_tokens:,}/{n:,} traces")
    click.echo(f"    Duration: {has_duration:,}/{n:,} traces")

    warnings = []
    if unset == n:
        warnings.append(
            "No outcome data — success_rate and per_task metrics will be unavailable.\n"
            "    Set outcome in your JSONL, or configure an outcome field.\n"
            "    Run 'kalibra inspect' to see available metadata fields."
        )
    elif successes == 0 and failures == 0 and unset > 0:
        warnings.append(
            "All outcomes are null — success_rate metric will show n/a.\n"
            "    Set 'outcome' to \"success\" or \"failure\" in your JSONL rows."
        )
    if has_cost == 0:
        warnings.append("No cost data — run 'kalibra inspect' to find cost fields.")
    if has_tokens == 0:
        warnings.append("No token data — run 'kalibra inspect' to find token fields.")
    if has_duration == 0:
        warnings.append("No duration data — duration metrics will be unavailable.")

    if warnings:
        click.echo("\n  Warnings:")
        for w in warnings:
            for line in w.split("\n"):
                click.echo(f"    ! {line}")

    click.echo("\n  Valid")
