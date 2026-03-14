"""Init command — generate a kalibra.yml config file interactively."""

from __future__ import annotations

import os
from pathlib import Path

import click

CONFIG_FILENAME = "kalibra.yml"

SOURCES = {
    "langfuse": {
        "env_vars": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
        "env_hint": (
            "export LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
            "export LANGFUSE_SECRET_KEY=sk-lf-..."
        ),
    },
    "langsmith": {
        "env_vars": ["LANGSMITH_API_KEY"],
        "env_hint": "export LANGSMITH_API_KEY=lsv2_pt_...",
    },
    "braintrust": {
        "env_vars": ["BRAINTRUST_API_KEY"],
        "env_hint": "export BRAINTRUST_API_KEY=sk-...",
    },
}


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

    # ── Source type ────────────────────────────────────────────────────────
    click.echo("  Where are your traces?")
    click.echo()
    for i, name in enumerate(SOURCES, 1):
        click.echo(f"    {i}. {click.style(name, fg='cyan')}")
    click.echo(f"    {len(SOURCES) + 1}. {click.style('Local JSONL files', fg='cyan')}")
    click.echo()

    choice = click.prompt("  Choose", type=int, default=1)
    source_names = list(SOURCES)

    if 1 <= choice <= len(source_names):
        source_type = source_names[choice - 1]
        is_local = False
    else:
        source_type = None
        is_local = True

    # ── Per-population config ─────────────────────────────────────────────
    if is_local:
        click.echo()
        click.echo("  Paths to your trace files:")
        click.echo()
        baseline_path = click.prompt(
            "  Baseline JSONL path", type=str, default="./baseline.jsonl",
        )
        current_path = click.prompt(
            "  Current JSONL path", type=str, default="./current.jsonl",
        )
        config_text = _generate_local_config(baseline_path, current_path)
    else:
        click.echo()
        project = click.prompt(
            f"  {click.style(source_type, bold=True)} project name", type=str,
        )
        click.echo()
        click.echo("  How do you split baseline vs current?")
        click.echo(click.style(
            "  Tags are most common. Leave blank to pull all traces.",
            dim=True,
        ))
        click.echo()
        baseline_raw = click.prompt(
            "  Baseline tags (comma-separated, or blank)", type=str, default="",
        )
        current_raw = click.prompt(
            "  Current tags (comma-separated, or blank)", type=str, default="",
        )
        baseline_tags = [t.strip() for t in baseline_raw.split(",") if t.strip()]
        current_tags = [t.strip() for t in current_raw.split(",") if t.strip()]
        config_text = _generate_remote_config(
            source_type, project, baseline_tags, current_tags,
        )

    config_path.write_text(config_text)

    # ── Print result ──────────────────────────────────────────────────────
    click.echo()
    click.echo(click.style("  " + "─" * 58, dim=True))
    ok = click.style("✓", fg="green")
    name = click.style(CONFIG_FILENAME, fg="cyan")
    click.echo(f"  {ok} Created {name}")

    # Check env vars.
    if not is_local:
        info = SOURCES[source_type]
        missing = [v for v in info["env_vars"] if not os.environ.get(v)]
        if missing:
            click.echo()
            click.echo(click.style("  Set credentials before running:", dim=True))
            for line in info["env_hint"].splitlines():
                click.echo(f"    {click.style(line, dim=True)}")

    click.echo()
    click.echo(f"  {click.style('Next:', dim=True)} kalibra compare")
    click.echo()


# ── Config generation ─────────────────────────────────────────────────────────

def _generate_remote_config(
    source: str, project: str, baseline_tags: list[str], current_tags: list[str],
) -> str:
    lines = []

    # Baseline
    lines.append("baseline:")
    lines.append(f"  source: {source}")
    lines.append(f"  project: {project}")
    lines.append("  since: 7d")
    if baseline_tags:
        lines.append(f"  tags: [{', '.join(baseline_tags)}]")

    lines.append("")

    # Current
    lines.append("current:")
    lines.append(f"  source: {source}")
    lines.append(f"  project: {project}")
    lines.append("  since: 7d")
    if current_tags:
        lines.append(f"  tags: [{', '.join(current_tags)}]")

    lines.append("")
    lines.extend(_common_sections())
    return "\n".join(lines)


def _generate_local_config(baseline_path: str, current_path: str) -> str:
    lines = [
        "baseline:",
        f"  path: {baseline_path}",
        "",
        "current:",
        f"  path: {current_path}",
        "",
    ]
    lines.extend(_common_sections())
    return "\n".join(lines)


def _common_sections() -> list[str]:
    return [
        "# ── Metrics ──────────────────────────────────────────────",
        "# All 10 built-in metrics run by default.",
        "# Remove any you don't need.",
        "",
        "metrics:",
        "  - success_rate",
        "  - per_task",
        "  - cost",
        "  - steps",
        "  - duration",
        "  - token_usage",
        "  - token_efficiency",
        "  - cost_quality",
        "",
        "# ── Quality gates ────────────────────────────────────────",
        "# Comparison fails (exit 1) if any threshold is violated.",
        "# Run: kalibra compare --metrics  to see all available fields.",
        "",
        "require:",
        "  - success_rate_delta >= -5    # max 5 pp success rate drop",
        "  - regressions <= 10           # max 10 tasks flipped",
        "  - cost_delta_pct <= 30        # max 30% cost increase",
        "",
        "# ── Field mappings ───────────────────────────────────────",
        "# Set these if 'kalibra inspect' shows missing data.",
        "#",
        "# fields:",
        "#   task_id: metadata.task_id   # per-task regression matching",
        "#   outcome: metadata.result    # success/failure detection",
        "#   cost: custom.cost_usd       # cost metric source",
        "",
    ]
