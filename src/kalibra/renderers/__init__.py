"""Renderers — produce text output from structured CompareResult data.

Each renderer reads Observation dicts and formats them for a specific output.
Metrics produce data. Renderers produce text. Never the other way around.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kalibra.engine import CompareResult

# Shared display names — consistent across all renderers.
METRIC_LABEL: dict[str, str] = {
    "success_rate": "Success rate",
    "cost": "Cost",
    "steps": "Steps",
    "duration": "Duration",
    "error_rate": "Error rate",
    "token_usage": "Token usage",
    "token_efficiency": "Token efficiency",
    "cost_quality": "Cost / quality",
    "trace_breakdown": "Per trace",
    "span_breakdown": "Per span",
}


def render(result: CompareResult, fmt: str, verbose: bool = False) -> str:
    """Render a CompareResult to the requested format."""
    if fmt == "terminal":
        from kalibra.renderers.terminal import render_terminal
        return render_terminal(result, verbose=verbose)
    if fmt == "markdown":
        from kalibra.renderers.markdown import render_markdown
        return render_markdown(result, verbose=verbose)
    if fmt == "json":
        from kalibra.renderers.json_renderer import render_json
        return render_json(result)
    raise ValueError(f"Unknown format: {fmt!r}")
