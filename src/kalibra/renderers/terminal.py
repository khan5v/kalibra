"""Terminal renderer — styled, colored output for the CLI.

Each metric has a format function that reads the Observation's structured data
and returns (headline, detail_lines). The renderer handles badges, colors,
indentation, and section grouping.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from kalibra.engine import CompareResult

from kalibra.metrics import Direction, Observation
from kalibra.metrics._stats import pct_delta
from kalibra.renderers import METRIC_LABEL

_INDENT = "  "
_DETAIL = "                      "  # 22 chars — aligns under metric detail

_BADGE = {
    Direction.UPGRADE: "▲",
    Direction.SAME: "≈",
    Direction.DEGRADATION: "▼",
    Direction.INCONCLUSIVE: "~",
    Direction.NA: "—",
}

_LABEL = {
    Direction.UPGRADE: "IMPROVED",
    Direction.SAME: "UNCHANGED",
    Direction.DEGRADATION: "REGRESSED",
    Direction.INCONCLUSIVE: "MIXED",
    Direction.NA: "N/A",
}

_COLOR = {
    Direction.UPGRADE: "green",
    Direction.SAME: "cyan",
    Direction.DEGRADATION: "red",
    Direction.INCONCLUSIVE: "yellow",
    Direction.NA: "white",
}

# Which metrics go in which section.
_TRACE_METRICS = {
    "success_rate", "cost", "steps", "duration",
    "error_rate",
    "token_usage", "token_efficiency", "cost_quality",
}
_TRACE_BREAKDOWN = {"trace_breakdown"}
_SPAN_BREAKDOWN = {"span_breakdown"}


def render_terminal(result: CompareResult, verbose: bool = False) -> str:
    bar = click.style("─" * 58, dim=True)
    badge = _styled_badge(result.direction)
    dir_label = _styled_label(result.direction)

    b_src = click.style(f"({result.baseline_source})", dim=True)
    c_src = click.style(f"({result.current_source})", dim=True)

    lines = [
        "",
        f"{_INDENT}{click.style('Kalibra Compare', bold=True)}",
        f"{_INDENT}{bar}",
        (f"{_INDENT}{click.style('Baseline', dim=True)}"
         f"  {result.baseline_count:>8,} traces   {b_src}"),
        (f"{_INDENT}{click.style('Current', dim=True)}"
         f"   {result.current_count:>8,} traces   {c_src}"),
        (f"{_INDENT}{click.style('Direction', dim=True)}"
         f" {badge} {dir_label}"),
        "",
    ]

    if result.warnings:
        for w in result.warnings:
            icon = click.style("!", fg="yellow", bold=True)
            lines.append(f"{_INDENT}{icon}  {click.style(w, fg='yellow')}")
        lines.append("")

    # Group observations into sections.
    obs = result.observations
    trace_obs = [o for o in obs.values() if o.name in _TRACE_METRICS]
    trace_bd = [o for o in obs.values() if o.name in _TRACE_BREAKDOWN]
    span_bd = [o for o in obs.values() if o.name in _SPAN_BREAKDOWN]
    other = [
        o for o in obs.values()
        if o.name not in _TRACE_METRICS | _TRACE_BREAKDOWN | _SPAN_BREAKDOWN
    ]

    # ── Trace metrics ─────────────────────────────────────────────────
    if trace_obs:
        _section_header(lines, "Trace metrics")
        for o in trace_obs:
            _render_metric(lines, o, verbose)
        if not verbose:
            lines.append("")

    # ── Trace breakdown ───────────────────────────────────────────────
    if trace_bd:
        _section_header(lines, "Trace breakdown")
        for o in trace_bd:
            _render_breakdown(lines, o, verbose)
        lines.append("")

    # ── Span breakdown ────────────────────────────────────────────────
    if span_bd:
        _section_header(lines, "Span breakdown")
        for o in span_bd:
            _render_breakdown(lines, o, verbose)
        lines.append("")

    # ── Other (plugins) ───────────────────────────────────────────────
    for o in other:
        _render_metric(lines, o, verbose)

    # ── Thresholds ────────────────────────────────────────────────────
    if result.gates:
        lines.append(f"{_INDENT}{click.style('Thresholds', bold=True)}")
        max_expr = max(len(g.expr) for g in result.gates)
        for g in result.gates:
            if g.warning:
                icon = click.style("SKIP", fg="yellow")
            elif g.passed:
                icon = click.style(" OK ", fg="green", bold=True)
            else:
                icon = click.style("FAIL", fg="red", bold=True)
            actual = f"{g.actual:.2f}" if not math.isnan(g.actual) else "n/a"
            line = (f"{_INDENT}  [{icon}] "
                    f"{g.expr:<{max_expr}}   actual: {actual}")
            if g.warning:
                warn = click.style(f"({g.warning})", fg="yellow")
                line += f"  {warn}"
            lines.append(line)
        lines.append("")

    # ── Verdict ───────────────────────────────────────────────────────
    lines.append(f"{_INDENT}{bar}")
    if result.gates:
        if result.passed:
            ok = click.style("PASSED", fg="green", bold=True)
            lines.append(f"{_INDENT}{ok} — all quality gates met")
        else:
            fail = click.style("FAILED", fg="red", bold=True)
            lines.append(
                f"{_INDENT}{fail} — quality gate violation (exit code 1)"
            )
    else:
        lines.append(
            f"{_INDENT}{badge} {dir_label} — no quality gates configured"
        )
    lines.append("")
    return "\n".join(lines)


# ── Section header ────────────────────────────────────────────────────────────

def _section_header(lines: list[str], title: str) -> None:
    lines.append(f"{_INDENT}{click.style(title, bold=True, dim=True)}")
    lines.append("")


# ── Metric rendering ─────────────────────────────────────────────────────────

def _render_metric(lines: list[str], obs: Observation, verbose: bool) -> None:
    badge = _styled_badge(obs.direction)
    label = METRIC_LABEL.get(obs.name, obs.name)
    label_s = click.style(f"{label:<18}", bold=True)

    headline, details = _format_metric(obs, verbose)
    lines.append(f"{_INDENT}{badge} {label_s}{headline}")

    if verbose:
        for d in details:
            lines.append(f"{_DETAIL}{click.style(d, dim=True)}")

    for w in obs.warnings:
        icon = click.style("!", fg="yellow", bold=True)
        lines.append(f"{_DETAIL}{icon}  {click.style(w, fg='yellow')}")

    if verbose:
        lines.append("")


def _render_breakdown(
    lines: list[str], obs: Observation, verbose: bool,
) -> None:
    badge = _styled_badge(obs.direction)
    label = METRIC_LABEL.get(obs.name, obs.name)
    label_s = click.style(f"{label:<18}", bold=True)

    headline = _format_breakdown_headline(obs)
    lines.append(f"{_INDENT}{badge} {label_s}{headline}")

    if verbose:
        for detail in _format_breakdown_details(obs):
            if detail.startswith(("▼", "▲", "≈", "~")):
                ch = detail[0]
                color = {"▼": "red", "▲": "green", "≈": "cyan", "~": "yellow"}[ch]
                styled = click.style(ch, fg=color, bold=True)
                name = click.style(detail[2:], bold=True)
                lines.append(f"{_DETAIL}{styled} {name}")
            elif detail.startswith("  !"):
                lines.append(
                    f"{_DETAIL}  {click.style(detail.strip(), fg='yellow')}"
                )
            else:
                lines.append(
                    f"{_DETAIL}  {click.style(detail.strip(), dim=True)}"
                )

    for w in obs.warnings:
        icon = click.style("!", fg="yellow", bold=True)
        lines.append(f"{_DETAIL}{icon}  {click.style(w, fg='yellow')}")


# ── Per-metric formatters ─────────────────────────────────────────────────────
# Each returns (headline_str, list_of_detail_lines).


def _delta_str(delta: float | None, unit: str = "%") -> str:
    """Format a delta value, handling None (undefined from-zero change)."""
    if delta is None:
        return "(new)"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}{unit}"


def _format_metric(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    """Dispatch to the right formatter."""
    fn = _FORMATTERS.get(obs.name, _format_generic)
    return fn(obs, verbose)


def _format_success_rate(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    b, c = obs.baseline, obs.current
    if b.get("rate") is None:
        return "n/a — no outcome data", []
    sign = "+" if obs.delta >= 0 else ""
    p = obs.metadata.get("pvalue")
    p_str = f"   (p={p:.3f})" if p is not None else ""
    headline = f"{b['rate']:.1%} → {c['rate']:.1%}  {sign}{obs.delta:.1f} pp{p_str}"
    details = []
    if verbose:
        sig = obs.metadata.get("significant", False)
        details.append(
            f"p={p:.3f} — {'statistically significant' if sig else 'not statistically significant'}"
        )
        details.append(f"n={b['with_outcome']}→{c['with_outcome']} traces with outcomes")
    return headline, details


def _format_cost(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    b, c = obs.baseline, obs.current
    if not b:
        return "n/a — no cost data", []
    headline = f"${b['median']:.4f} → ${c['median']:.4f} median  {_delta_str(obs.delta)}"
    details = []
    if verbose:
        avg_delta = pct_delta(b["avg"], c["avg"])
        details.append(f"${b['avg']:.4f} → ${c['avg']:.4f} avg  {_delta_str(avg_delta)}")
        details.append(f"${b['total']:.2f} → ${c['total']:.2f} total")
        ci = obs.metadata.get("ci_95")
        if ci:
            details.append(f"95% CI [{ci[0]:+.1f}%, {ci[1]:+.1f}%]")

    return headline, details


def _format_steps(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    b, c = obs.baseline, obs.current
    if not b:
        return "n/a", []
    headline = (
        f"{b['median']:.0f} → {c['median']:.0f} steps/trace (median)"
        f"  {_delta_str(obs.delta)}"
    )
    details = []
    if verbose:
        avg_delta = pct_delta(b["avg"], c["avg"])
        details.append(f"{b['avg']:.1f} → {c['avg']:.1f} avg  {_delta_str(avg_delta)}")
        ci = obs.metadata.get("ci_95")
        if ci:
            details.append(f"95% CI [{ci[0]:+.1f}%, {ci[1]:+.1f}%]")

    return headline, details


def _format_duration(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    b, c = obs.baseline, obs.current
    if not b:
        return "n/a", []
    headline = f"{b['median']:.1f}s → {c['median']:.1f}s median  {_delta_str(obs.delta)}"
    details = []
    if verbose:
        avg_delta = pct_delta(b["avg"], c["avg"])
        details.append(f"{b['avg']:.1f}s → {c['avg']:.1f}s avg  {_delta_str(avg_delta)}")
        p95_delta = obs.metadata.get("p95_delta_pct")
        details.append(f"{b['p95']:.1f}s → {c['p95']:.1f}s P95  {_delta_str(p95_delta)}")
        ci = obs.metadata.get("ci_95")
        if ci:
            details.append(f"95% CI [{ci[0]:+.1f}%, {ci[1]:+.1f}%]")

    return headline, details


def _format_error_rate(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    b, c = obs.baseline, obs.current
    if not b:
        return "n/a", []
    sign = "+" if obs.delta >= 0 else ""
    headline = f"{b['rate']:.1f}% → {c['rate']:.1f}%  {sign}{obs.delta:.1f} pp"
    details = []
    if verbose:
        p = obs.metadata.get("pvalue")
        if p is not None:
            sig = obs.metadata.get("significant", False)
            details.append(
                f"p={p:.3f} — {'statistically significant' if sig else 'not statistically significant'}"
            )
    return headline, details



def _format_token_usage(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    b, c = obs.baseline, obs.current
    if not b:
        return "n/a — no token data", []
    headline = (
        f"{b['median']:,.0f} → {c['median']:,.0f} tokens/trace (median)"
        f"  {_delta_str(obs.delta)}"
    )
    details = []
    if verbose:
        avg_delta = pct_delta(b["avg"], c["avg"])
        details.append(f"{b['avg']:,.0f} → {c['avg']:,.0f} avg  {_delta_str(avg_delta)}")
        details.append(
            f"in: {b.get('input_tokens', 0):,.0f} → {c.get('input_tokens', 0):,.0f}  |  "
            f"out: {b.get('output_tokens', 0):,.0f} → {c.get('output_tokens', 0):,.0f}"
        )
        ci = obs.metadata.get("ci_95")
        if ci:
            details.append(f"95% CI [{ci[0]:+.1f}%, {ci[1]:+.1f}%]")

    return headline, details


def _format_token_eff(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    b, c = obs.baseline, obs.current
    if not b:
        return "n/a", []
    headline = (
        f"{b['tokens_per_success']:,.0f} → {c['tokens_per_success']:,.0f}"
        f" tokens/success (median)  {_delta_str(obs.delta)}"
    )
    details = []
    if verbose:
        details.append(f"{b['successes']} → {c['successes']} successful traces")
        ci = obs.metadata.get("ci_95")
        if ci:
            details.append(f"95% CI [{ci[0]:+.1f}%, {ci[1]:+.1f}%]")
    return headline, details


def _format_cost_quality(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    b, c = obs.baseline, obs.current
    if not b:
        return "n/a", []
    headline = (
        f"${b['cost_per_success']:.4f} → ${c['cost_per_success']:.4f}"
        f" per success (median)  {_delta_str(obs.delta)}"
    )
    details = []
    if verbose:
        details.append(
            f"{b['successes']} → {c['successes']} successful traces, "
            f"${b['total_cost']:.2f} → ${c['total_cost']:.2f} total cost"
        )
        ci = obs.metadata.get("ci_95")
        if ci:
            details.append(f"95% CI [{ci[0]:+.1f}%, {ci[1]:+.1f}%]")
    return headline, details


def _format_generic(obs: Observation, verbose: bool) -> tuple[str, list[str]]:
    """Fallback for unknown metrics."""
    if obs.delta is not None:
        return _delta_str(obs.delta), []
    return obs.description, []


# ── Breakdown formatters ──────────────────────────────────────────────────────

def _format_breakdown_headline(obs: Observation) -> str:
    meta = obs.metadata
    n_imp = meta.get("n_improvements", 0)
    n_reg = meta.get("n_regressions", 0)
    n_mix = meta.get("n_mixed", 0)
    # span_breakdown has "matched", trace_breakdown has n_unchanged
    n_matched = meta.get("matched", n_imp + n_reg + n_mix + meta.get("n_unchanged", 0))
    parts = []
    if n_imp:
        parts.append(f"✓ {n_imp} improved")
    if n_reg:
        parts.append(f"✗ {n_reg} regressed")
    if n_mix:
        parts.append(f"~ {n_mix} mixed")
    return f"{n_matched} matched — {', '.join(parts)}" if parts else f"{n_matched} matched"


def _format_breakdown_details(obs: Observation) -> list[str]:
    """Format per-item detail lines for trace/span breakdown."""
    if obs.name == "trace_breakdown":
        return _format_trace_breakdown_details(obs)
    if obs.name == "span_breakdown":
        return _format_span_breakdown_details(obs)
    return []


def _format_trace_breakdown_details(obs: Observation) -> list[str]:
    details: list[str] = []
    for item in obs.metadata.get("regressions", []):
        tid = item["task_id"]
        b, c = item["baseline"], item["current"]
        details.append(f"▼ {tid}")
        details.append(
            f"  succeeded: {b['success']}/{b['total']}"
            f" → {c['success']}/{c['total']}"
        )
    for item in obs.metadata.get("improvements", []):
        tid = item["task_id"]
        b, c = item["baseline"], item["current"]
        details.append(f"▲ {tid}")
        details.append(
            f"  succeeded: {b['success']}/{b['total']}"
            f" → {c['success']}/{c['total']}"
        )
    return details


# Sort priority: regressions first, then mixed, then improved, then unchanged.
_DIRECTION_ORDER = {"regressed": 0, "mixed": 1, "improved": 2, "unchanged": 3}


def _format_span_breakdown_details(obs: Observation) -> list[str]:
    details: list[str] = []
    per_span = obs.metadata.get("per_span", {})

    for name in sorted(per_span, key=lambda n: _DIRECTION_ORDER.get(
        per_span[n].get("direction", "unchanged"), 2
    )):
        entry = per_span[name]
        d = entry.get("direction", "unchanged")
        badge_map = {"regressed": "▼", "improved": "▲", "mixed": "~", "unchanged": "≈"}
        badge = badge_map.get(d, "≈")
        details.append(f"{badge} {name}")

        b = entry.get("baseline", {})
        c = entry.get("current", {})
        deltas = entry.get("deltas", {})

        if b.get("median_duration") or c.get("median_duration"):
            sign = "+" if deltas.get("duration_pct", 0) >= 0 else ""
            details.append(
                f"  {b['median_duration']:.1f}s → {c['median_duration']:.1f}s duration"
                f"  {sign}{deltas.get('duration_pct', 0):.0f}%"
            )
        if b.get("median_cost") or c.get("median_cost"):
            sign = "+" if deltas.get("cost_pct", 0) >= 0 else ""
            details.append(
                f"  ${b['median_cost']:.4f} → ${c['median_cost']:.4f} cost"
                f"  {sign}{deltas.get('cost_pct', 0):.0f}%"
            )
        if b.get("median_tokens") or c.get("median_tokens"):
            sign = "+" if deltas.get("tokens_pct", 0) >= 0 else ""
            details.append(
                f"  {b['median_tokens']:,.0f} → {c['median_tokens']:,.0f} tokens"
                f"  {sign}{deltas.get('tokens_pct', 0):.0f}%"
            )
        if b.get("error_rate") or c.get("error_rate"):
            sign = "+" if deltas.get("error_rate_pp", 0) >= 0 else ""
            details.append(
                f"  err {b['error_rate']:.0f}% → {c['error_rate']:.0f}%"
                f"  {sign}{deltas.get('error_rate_pp', 0):.1f} pp"
            )

        details.append(f"  n={b.get('count', 0)} → {c.get('count', 0)} spans")

        warning = entry.get("warning")
        if warning:
            details.append(f"  ! {warning}")

    return details


# ── Styling helpers ───────────────────────────────────────────────────────────

def _styled_badge(direction: Direction) -> str:
    return click.style(_BADGE[direction], fg=_COLOR[direction], bold=True)


def _styled_label(direction: Direction) -> str:
    return click.style(_LABEL[direction], fg=_COLOR[direction], bold=True)


# ── Formatter registry ────────────────────────────────────────────────────────

_FORMATTERS = {
    "success_rate": _format_success_rate,
    "cost": _format_cost,
    "steps": _format_steps,
    "duration": _format_duration,
    "error_rate": _format_error_rate,
    "token_usage": _format_token_usage,
    "token_efficiency": _format_token_eff,
    "cost_quality": _format_cost_quality,
}
