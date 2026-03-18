"""Demo command — interactive walkthrough with built-in sample data."""

from __future__ import annotations

import shutil
from pathlib import Path

import click

from kalibra import display


_LOGO = r"""
      ___           ___           ___                       ___           ___           ___
     /\__\         /\  \         /\__\          ___        /\  \         /\  \         /\  \
    /:/  /        /::\  \       /:/  /         /\  \      /::\  \       /::\  \       /::\  \
   /:/__/        /:/\:\  \     /:/  /          \:\  \    /:/\:\  \     /:/\:\  \     /:/\:\  \
  /::\__\____   /::\~\:\  \   /:/  /           /::\__\  /::\~\:\__\   /: \~\:\  \   /: \~\:\  \
 /:/\:::::\__\ /:/\:\ \:\__\ /:/__/         __/:/\/__/ /:/\:\ \:|__| /:/\:\ \:\__\ /:/\:\ \:\__\
 \/_|:|~~|~    \/__\:\/:/  / \:\  \        /\/:/  /    \:\~\:\/:/  / \/_|::\/:/  / \/__\:\/:/  /
    |:|  |          \::/  /   \:\  \       \::/__/      \:\ \::/  /     |:|::/  /       \::/  /
    |:|  |          /:/  /     \:\  \       \:\__\       \:\/:/  /      |:|\/__/        /:/  /
    |:|  |         /:/  /       \:\__\       \/__/        \::/__/       |:|  |         /:/  /
     \|__|         \/__/         \/__/                     ~~            \|__|         \/__/
"""


def _print_logo() -> None:
    """Print the Kalibra ASCII logo in cyan."""
    click.echo()
    for line in _LOGO.strip("\n").splitlines():
        click.echo(f"  {click.style(line, fg='cyan')}")
    click.echo()


def _prompt_continue(message: str = "Press Enter to continue, q to quit") -> bool:
    """Prompt user to continue. Only Enter or q accepted."""
    while True:
        try:
            val = click.prompt(
                f"  {click.style(message, dim=True)}",
                default="", show_default=False, prompt_suffix="",
            )
            val = val.strip().lower()
            if val == "":
                return True
            if val == "q":
                return False
            # Anything else — ask again
        except (click.Abort, EOFError):
            return False


def run_demo() -> None:
    """Interactive demo: copy sample data, run compare, explain findings."""
    from kalibra.engine import compare
    from kalibra.loader import load_traces
    from kalibra.renderers import render

    b = display.bar()

    # ── Beat 1: Intro ────────────────────────────────────────────────────
    _print_logo()
    click.echo(f"  {b}")
    click.echo()
    click.echo(
        f"  This will create a {click.style('kalibra-demo/', fg='cyan')} directory"
        f" with sample traces"
    )
    click.echo(
        f"  and run a comparison showing what Kalibra detects."
        f" {click.style('(feel free to delete it after)', dim=True)}"
    )
    click.echo()

    if not _prompt_continue():
        click.echo()
        return

    # ── Copy demo data ───────────────────────────────────────────────────
    src_dir = Path(__file__).resolve().parent.parent / "demo_data"
    dest_dir = Path("kalibra-demo")

    if not (src_dir / "baseline.jsonl").exists():
        display.header("Kalibra", "demo data missing")
        click.echo(
            f"  Built-in sample data not found. Try reinstalling:"
        )
        click.echo(
            f"  {click.style('pip install --force-reinstall kalibra', fg='cyan')}"
        )
        click.echo()
        raise SystemExit(1)

    dest_dir.mkdir(exist_ok=True)
    shutil.copy2(src_dir / "baseline.jsonl", dest_dir / "baseline.jsonl")
    shutil.copy2(src_dir / "current.jsonl", dest_dir / "current.jsonl")
    fetch_src = src_dir / "fetch_huggingface.py"
    if fetch_src.exists():
        shutil.copy2(fetch_src, dest_dir / "fetch_huggingface.py")

    ok = click.style("✓", fg="green")
    click.echo()
    click.echo(f"  {ok} Created {click.style('kalibra-demo/baseline.jsonl', fg='cyan')}")
    click.echo(f"  {ok} Created {click.style('kalibra-demo/current.jsonl', fg='cyan')}")
    if fetch_src.exists():
        click.echo(f"  {ok} Created {click.style('kalibra-demo/fetch_huggingface.py', fg='cyan')}")
    click.echo()
    click.echo(
        f"  {click.style('Running:', dim=True)} "
        f"kalibra compare kalibra-demo/baseline.jsonl kalibra-demo/current.jsonl -v"
    )

    import time
    time.sleep(2)

    # ── Beat 2: Run compare ──────────────────────────────────────────────
    baseline = load_traces(str(dest_dir / "baseline.jsonl"))
    current = load_traces(str(dest_dir / "current.jsonl"))

    result = compare(
        baseline, current,
        baseline_source="kalibra-demo/baseline.jsonl",
        current_source="kalibra-demo/current.jsonl",
        metric_config={"trace_breakdown": {"task_id_field": "task_id"}},
    )

    click.echo(render(result, "terminal", verbose=True))

    # ── Beat 3: Findings + next steps ──────────────────────────────────
    if not _prompt_continue("Press Enter to see what Kalibra found, q to quit"):
        click.echo()
        return

    click.echo()
    click.echo(f"  {b}")
    click.echo(
        f"  {click.style('Demo', fg='cyan', bold=True)}"
        f" {click.style('— what Kalibra found in this sample data', dim=True)}"
    )
    click.echo()

    click.echo(
        f"  {click.style('▸', fg='yellow')} "
        f"5 tasks regressed 5/5 → 0/5 despite overall success rate improving."
    )
    click.echo(
        f"    {click.style('Aggregates masked a targeted regression. trace_breakdown caught it.', dim=True)}"
    )
    click.echo()

    click.echo(
        f"  {click.style('▸', fg='yellow')} "
        f"Cost dropped 40% but duration doubled."
    )
    click.echo(
        f"    {click.style('A model swap improved cost but introduced latency. Both matter.', dim=True)}"
    )
    click.echo()

    click.echo(
        f"  {click.style('▸', fg='yellow')} "
        f"'search' span errors spiked while aggregate error rate barely moved."
    )
    click.echo(
        f"    {click.style('Per-span breakdown reveals problems that per-trace metrics hide.', dim=True)}"
    )

    # ── Getting started ──────────────────────────────────────────────────
    click.echo()
    click.echo(f"  {b}")
    click.echo(f"  {click.style('Getting started with your data', bold=True)}")
    click.echo()
    click.echo(
        f"  Kalibra reads JSONL — one JSON object per line, one trace per line."
    )
    click.echo(
        f"  {click.style('See kalibra-demo/*.jsonl for the exact format.', dim=True)}"
    )
    click.echo()
    click.echo(
        f"  If your traces are in a platform (Langfuse, Braintrust, LangSmith)"
    )
    click.echo(
        f"  or a different format, export them to JSONL first."
    )
    click.echo(
        f"  {click.style('See kalibra-demo/fetch_huggingface.py as a starting point.', dim=True)}"
    )
    click.echo()
    click.echo(
        f"  If your JSONL uses different field names, Kalibra can help:"
    )
    click.echo(
        f"    {click.style('kalibra inspect mydata.jsonl --suggest', fg='cyan')}"
    )

    # ── Next steps ───────────────────────────────────────────────────────
    click.echo()
    click.echo(f"  {b}")
    click.echo(f"  {click.style('Next steps', bold=True)}")
    click.echo()
    click.echo(f"  {click.style('Quick start', fg='cyan', bold=True)}")
    click.echo(
        f"    {click.style('kalibra compare baseline.jsonl current.jsonl', dim=True)}"
    )
    click.echo()
    click.echo(f"  {click.style('Full setup', fg='cyan', bold=True)}")
    click.echo(
        f"    kalibra inspect mydata.jsonl --suggest"
        f"    {click.style('# discover field mappings', dim=True)}"
    )
    click.echo(
        f"    kalibra init"
        f"                                  {click.style('# generate kalibra.yml', dim=True)}"
    )
    click.echo(
        f"    kalibra compare"
        f"                              {click.style('# runs from config', dim=True)}"
    )
    click.echo()
    click.echo(f"  {click.style('Add quality gates', fg='cyan', bold=True)}")
    click.echo(
        f"    kalibra compare ... --require {click.style('\"success_rate_delta >= -5\"', fg='cyan')}"
    )
    click.echo(
        f"    kalibra compare --metrics"
        f"                  {click.style('# see all available gate fields', dim=True)}"
    )
    click.echo()
