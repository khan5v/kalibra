"""Init command — generate a kalibra.yml config file interactively."""

from __future__ import annotations

from pathlib import Path

import click

CONFIG_FILENAME = "kalibra.yml"


def run_init(force: bool = False) -> None:
    """Run the interactive init wizard."""
    config_path = Path(CONFIG_FILENAME)

    if config_path.exists() and not force:
        click.echo(f"\n  {CONFIG_FILENAME} already exists.")
        if not click.confirm("  Overwrite?", default=False):
            click.echo("  Aborted.")
            return

    click.echo()
    click.echo(f"  {click.style('Kalibra Setup', bold=True)}")
    click.echo(click.style("  " + "─" * 58, dim=True))
    click.echo()

    click.echo("  Paths to your trace files (JSONL):")
    click.echo()
    baseline_path = click.prompt(
        "  Baseline path", type=str, default="./baseline.jsonl",
    )
    current_path = click.prompt(
        "  Current path", type=str, default="./current.jsonl",
    )

    config_text = _generate_config(baseline_path, current_path)
    config_path.write_text(config_text)

    click.echo()
    click.echo(click.style("  " + "─" * 58, dim=True))
    ok = click.style("✓", fg="green")
    name = click.style(CONFIG_FILENAME, fg="cyan")
    click.echo(f"  {ok} Created {name}")
    click.echo()
    click.echo(f"  {click.style('Next:', dim=True)} kalibra compare")
    click.echo()


def _generate_config(baseline_path: str, current_path: str) -> str:
    lines = [
        "baseline:",
        f"  path: {baseline_path}",
        "",
        "current:",
        f"  path: {current_path}",
        "",
        "# ── Metrics ──────────────────────────────────────────────",
        "# All 11 built-in metrics run by default.",
        "# Remove any you don't need.",
        "",
        "metrics:",
        "  - success_rate        # task pass/fail rate + significance test",
        "  - cost                # cost per trace — median, avg, total",
        "  - steps               # steps per task — median and avg",
        "  - duration            # latency — median, avg, P95",
        "  - error_rate          # fraction of spans that error",
        "  - path_distribution   # execution path similarity",
        "  - token_usage         # token consumption — in/out/total",
        "  - token_efficiency    # tokens per successful task",
        "  - cost_quality        # cost per success",
        "  - trace_breakdown     # per-task regressions",
        "  - span_breakdown      # per-span regressions",
        "",
        "# ── Quality gates ────────────────────────────────────────",
        "# Comparison fails (exit 1) if any threshold is violated.",
        "# Run: kalibra compare --metrics  to see all available fields.",
        "",
        "require:",
        "  - success_rate_delta >= -5    # max 5 pp success rate drop",
        "  - regressions <= 10           # max 10 tasks flipped",
        "  - cost_delta_pct <= 30        # max 30% cost increase",
        "  - span_regressions <= 3       # max 3 span names regressed",
        "",
        "# ── Field mappings ───────────────────────────────────────",
        "# Set these if 'kalibra inspect' shows missing data.",
        "#",
        "# fields:",
        "#   trace_id: uuid              # which field identifies each trace",
        "#   task_id: metadata.task_id   # per-task regression matching",
        "#   outcome: metadata.result    # success/failure detection",
        "#   cost: custom.cost_usd       # cost metric source",
        "",
    ]
    return "\n".join(lines)
