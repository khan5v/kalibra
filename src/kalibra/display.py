"""Styled terminal output helpers — shared across CLI commands."""

from __future__ import annotations

import re

import click


def bar() -> str:
    return click.style("─" * 58, dim=True)


def dot() -> str:
    return click.style("·", dim=True)


def header(title: str, subtitle: str | None = None) -> None:
    """Print a styled header: 'Kalibra · subtitle' or just 'Title'."""
    click.echo()
    if subtitle:
        click.echo(
            f"  {click.style('Kalibra', bold=True)}"
            f"  {click.style('·', dim=True)}  "
            f"{click.style(subtitle, fg='yellow')}"
        )
    else:
        click.echo(f"  {click.style(title, bold=True)}")
    click.echo(f"  {bar()}")


def threshold_error(exc: Exception) -> None:
    """Render threshold validation errors."""
    errors = str(exc).split("\n\n")

    header("Kalibra", "invalid threshold")

    for block in errors:
        lines = block.strip().splitlines()
        if not lines:
            continue
        click.echo(f"  {click.style('▸', fg='yellow')} {_style_error_headline(lines[0])}")
        for line in lines[1:]:
            click.echo(f"    {_style_error_detail(line.strip())}")
        click.echo()

    click.echo(f"  {bar()}")
    click.echo(
        f"  {click.style('Hint:', dim=True)} kalibra compare "
        f"{click.style('--metrics', fg='cyan')} to see all fields"
    )
    click.echo()



def load_error(path: str, message: str) -> None:
    """Render a trace loading error in styled format."""
    header("Kalibra", "failed to load traces")
    # The message from the loader includes path:line info.
    # Split on newlines and render each part.
    lines = message.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Lines with path:line info are the headline.
        if "—" in line and (":" in line.split("—")[0]):
            click.echo(f"  {click.style('▸', fg='yellow')} {line}")
        # "Available fields" / "These might be" are highlights.
        elif line.startswith("Available fields"):
            click.echo(f"    {click.style(line, fg='white')}")
        elif line.startswith("These might be"):
            click.echo(f"    {click.style(line, fg='cyan')}")
        # Config suggestions.
        elif line.startswith("Set in") or line.startswith("fields:"):
            click.echo(f"    {click.style(line, dim=True)}")
        elif line.startswith("trace_id:"):
            click.echo(f"      {click.style(line, fg='cyan')}")
        else:
            click.echo(f"    {click.style(line, dim=True)}")
    click.echo(f"  {bar()}")
    click.echo()



def metrics_list() -> None:
    """Print all available metrics and their threshold fields."""
    from kalibra.config import CompareConfig
    from kalibra.engine import resolve_metrics

    config = CompareConfig.load()
    all_metrics = resolve_metrics(config.metrics)

    d = dot()

    click.echo()
    click.echo(f"  {click.style('Kalibra Metrics', bold=True)}")
    click.echo(f"  {bar()}")
    click.echo()

    for m in all_metrics:
        click.echo(f"  {click.style(m.name, fg='cyan', bold=True)}")
        click.echo(f"  {click.style(m.description, dim=True)}")
        fields = m.threshold_field_names()
        if fields:
            for field_name, desc in fields.items():
                click.echo(
                    f"    {click.style(field_name, fg='white')}"
                    f"  {d * (34 - len(field_name))}  "
                    f"{click.style(desc, dim=True)}"
                )
        click.echo()

    click.echo(f"  {bar()}")
    click.echo()
    example = click.style('"success_rate_delta >= -2"', fg="cyan")
    click.echo(f"  {click.style('Quick:', dim=True)}  kalibra compare --require {example}")
    click.echo()
    config_file = click.style("kalibra.yml", fg="cyan")
    click.echo(f"  {click.style('Config:', dim=True)} add to {config_file}:")
    click.echo(click.style("          require:", dim=True))
    click.echo(click.style("            - success_rate_delta >= -2", dim=True))
    click.echo(click.style("            - cost_delta_pct <= 20", dim=True))
    click.echo(click.style("            - regressions <= 5", dim=True))
    click.echo()
    click.echo(click.style(
        "  Both combine — config gates always apply, --require adds more.",
        dim=True,
    ))
    click.echo()


def no_data() -> None:
    """Render a styled 'no data specified' message — the first thing new users see."""
    click.echo()
    click.echo(f"  {click.style('Kalibra', bold=True)}")
    click.echo(f"  {bar()}")
    click.echo()
    click.echo(
        f"  {click.style('Try it out', fg='cyan', bold=True)}"
        f"        {click.style('kalibra demo', fg='cyan')}"
    )
    click.echo(
        f"  {click.style('Your data', fg='white', bold=True)}"
        f"          {click.style('kalibra compare', fg='white')}"
        f" {click.style('baseline.jsonl current.jsonl', dim=True)}"
    )
    click.echo(
        f"  {click.style('Explore a file', fg='white', bold=True)}"
        f"     {click.style('kalibra inspect', fg='white')}"
        f" {click.style('traces.jsonl --suggest', dim=True)}"
    )
    click.echo(
        f"  {click.style('Set up config', fg='white', bold=True)}"
        f"      {click.style('kalibra init', fg='white')}"
    )
    click.echo()
    click.echo(f"  {bar()}")
    click.echo(
        f"  {click.style('Docs:', dim=True)} "
        f"{click.style('https://github.com/khan5v/kalibra', dim=True)}"
    )
    click.echo()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _style_error_headline(line: str) -> str:
    def _bold_quotes(m: re.Match) -> str:
        return click.style(m.group(0), bold=True)
    return re.sub(r"'[^']*'", _bold_quotes, line)


def _style_error_detail(line: str) -> str:
    if line.startswith("Did you mean:"):
        prefix = click.style("Did you mean: ", dim=True)
        suggestions = line[len("Did you mean: "):]
        return prefix + click.style(suggestions, fg="cyan")
    return click.style(line, dim=True)
