"""Markdown renderer — for PR comments and documentation.

Produces a Markdown table of metrics with direction badges,
plus optional breakdown sections and gate results.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kalibra.engine import CompareResult

from kalibra.metrics import Direction, Observation
from kalibra.renderers import METRIC_LABEL

_BADGE = {
    Direction.UPGRADE: "▲",
    Direction.SAME: "—",
    Direction.DEGRADATION: "▼",
    Direction.INCONCLUSIVE: "⚠",
    Direction.NA: "–",
}

_LABEL = {
    Direction.UPGRADE: "Improved",
    Direction.SAME: "Unchanged",
    Direction.DEGRADATION: "Regressed",
    Direction.INCONCLUSIVE: "Mixed",
    Direction.NA: "N/A",
}

_TRACE_METRICS = {
    "success_rate", "cost", "steps", "duration",
    "error_rate",
    "token_usage", "token_efficiency", "cost_quality",
}

# Sort priority for span breakdown: regressions first.
_DIRECTION_ORDER = {"regressed": 0, "mixed": 1, "improved": 2, "unchanged": 3}


def render_markdown(result: CompareResult, verbose: bool = False) -> str:
    """Render a CompareResult as Markdown."""
    lines: list[str] = []

    # Header
    badge = _BADGE[result.direction]
    label = _LABEL[result.direction]
    lines.append(f"## {badge} Kalibra Compare — {label}")
    lines.append("")
    lines.append(
        f"**Baseline:** {result.baseline_count:,} traces ({result.baseline_source})  "
    )
    lines.append(
        f"**Current:** {result.current_count:,} traces ({result.current_source})"
    )
    lines.append("")

    # Warnings
    if result.warnings:
        for w in result.warnings:
            lines.append(f"> :warning: {w}")
        lines.append("")

    # Metrics table
    trace_obs = [
        o for o in result.observations.values() if o.name in _TRACE_METRICS
    ]
    if trace_obs:
        lines.append("### Metrics")
        lines.append("")
        lines.append("| Metric | Direction | Delta | Detail |")
        lines.append("|--------|-----------|-------|--------|")
        for obs in trace_obs:
            lines.append(_metric_row(obs))
        lines.append("")

    # Trace breakdown
    tb = result.observations.get("trace_breakdown")
    if tb and tb.direction != Direction.NA:
        n_reg = tb.metadata.get("n_regressions", 0)
        n_imp = tb.metadata.get("n_improvements", 0)
        lines.append("### Trace Breakdown")
        lines.append("")
        lines.append(f"**{n_reg}** regressions, **{n_imp}** improvements")
        lines.append("")
        if n_reg > 0:
            lines.append("<details><summary>Regressions</summary>")
            lines.append("")
            for item in tb.metadata.get("regressions", []):
                tid = item["task_id"]
                b, c = item["baseline"], item["current"]
                lines.append(
                    f"- **{tid}**: {b['success']}/{b['total']}"
                    f" → {c['success']}/{c['total']}"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")
        if verbose and n_imp > 0:
            lines.append("<details><summary>Improvements</summary>")
            lines.append("")
            for item in tb.metadata.get("improvements", []):
                tid = item["task_id"]
                b, c = item["baseline"], item["current"]
                lines.append(
                    f"- **{tid}**: {b['success']}/{b['total']}"
                    f" → {c['success']}/{c['total']}"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # Span breakdown
    sb = result.observations.get("span_breakdown")
    if sb and sb.direction != Direction.NA:
        n_reg = sb.metadata.get("n_regressions", 0)
        n_imp = sb.metadata.get("n_improvements", 0)
        n_matched = sb.metadata.get("matched", 0)
        lines.append("### Span Breakdown")
        lines.append("")
        n_mix = sb.metadata.get("n_mixed", 0)
        parts = []
        if n_reg:
            parts.append(f"**{n_reg}** regressed")
        if n_imp:
            parts.append(f"**{n_imp}** improved")
        if n_mix:
            parts.append(f"**{n_mix}** mixed")
        summary = f"{n_matched} matched — {', '.join(parts)}" if parts else f"{n_matched} matched"
        lines.append(summary)
        lines.append("")
        per_span = sb.metadata.get("per_span", {})
        changed = {
            name: entry for name, entry in per_span.items()
            if entry.get("direction") != "unchanged"
        }
        if changed:
            lines.append("| Span | Direction | Duration | Cost | Tokens |")
            lines.append("|------|-----------|----------|------|--------|")
            for name in sorted(changed, key=lambda n: _DIRECTION_ORDER.get(
                changed[n].get("direction", "unchanged"), 2
            )):
                entry = changed[name]
                d = entry.get("direction", "unchanged")
                d_badge = {"regressed": "▼", "improved": "▲", "mixed": "⚠"}.get(d, "—")
                b = entry.get("baseline", {})
                c = entry.get("current", {})
                deltas = entry.get("deltas", {})
                dur = _delta_str(deltas.get("duration_pct", 0))
                cost = _delta_str(deltas.get("cost_pct", 0))
                tok = _delta_str(deltas.get("tokens_pct", 0))
                lines.append(f"| {name} | {d_badge} | {dur} | {cost} | {tok} |")
            lines.append("")

    # Gates
    if result.gates:
        lines.append("### Quality Gates")
        lines.append("")
        lines.append("| Gate | Result | Actual |")
        lines.append("|------|--------|--------|")
        for g in result.gates:
            if g.warning:
                icon = "⚠ SKIP"
            elif g.passed:
                icon = "✅ PASS"
            else:
                icon = "❌ FAIL"
            actual = f"{g.actual:.2f}" if not math.isnan(g.actual) else "n/a"
            lines.append(f"| `{g.expr}` | {icon} | {actual} |")
        lines.append("")

    # Verdict
    if result.gates:
        if result.passed:
            lines.append("✅ **All quality gates passed**")
        else:
            lines.append("❌ **Quality gate violation**")
    lines.append("")

    return "\n".join(lines)


def _metric_row(obs: Observation) -> str:
    badge = _BADGE[obs.direction]
    name = METRIC_LABEL.get(obs.name, obs.name)
    delta = _format_delta(obs)
    detail = _format_detail(obs)
    return f"| {name} | {badge} | {delta} | {detail} |"


def _format_delta(obs: Observation) -> str:
    if obs.delta is None:
        return "—"
    sign = "+" if obs.delta >= 0 else ""
    if obs.name in ("success_rate", "error_rate"):
        return f"{sign}{obs.delta:.1f} pp"
    return f"{sign}{obs.delta:.1f}%"


def _format_detail(obs: Observation) -> str:
    b, c = obs.baseline, obs.current
    if not b:
        return "n/a"

    if obs.name == "success_rate":
        return f"{b.get('rate', 0):.1%} → {c.get('rate', 0):.1%}"
    if obs.name == "cost":
        return f"${b.get('median', 0):.4f} → ${c.get('median', 0):.4f} median"
    if obs.name == "duration":
        return f"{b.get('median', 0):.1f}s → {c.get('median', 0):.1f}s median"
    if obs.name == "steps":
        return f"{b.get('median', 0):.0f} → {c.get('median', 0):.0f} steps"
    if obs.name == "token_usage":
        return f"{b.get('median', 0):,.0f} → {c.get('median', 0):,.0f} tokens"
    if obs.name == "error_rate":
        return f"{b.get('rate', 0):.1f}% → {c.get('rate', 0):.1f}%"
    if obs.name == "token_efficiency":
        return (
            f"{b.get('tokens_per_success', 0):,.0f}"
            f" → {c.get('tokens_per_success', 0):,.0f} tok/success"
        )
    if obs.name == "cost_quality":
        return (
            f"${b.get('cost_per_success', 0):.4f}"
            f" → ${c.get('cost_per_success', 0):.4f} /success"
        )
    return obs.description


def _delta_str(pct: float) -> str:
    """Format a percentage delta for table cells."""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"
