"""Multi-format report rendering for CompareResult."""

from __future__ import annotations

import json

import click

from kalibra.compare import CompareResult
from kalibra.metrics import Direction

_DIRECTION_BADGE = {
    Direction.UPGRADE: "▲",
    Direction.SAME: "≈",
    Direction.DEGRADATION: "▼",
    Direction.INCONCLUSIVE: "~",
    Direction.NA: "—",
}

_DIRECTION_LABEL = {
    Direction.UPGRADE: "IMPROVED",
    Direction.SAME: "UNCHANGED",
    Direction.DEGRADATION: "REGRESSED",
    Direction.INCONCLUSIVE: "MIXED",
    Direction.NA: "N/A",
}

_DIRECTION_COLOR = {
    Direction.UPGRADE: "green",
    Direction.SAME: "cyan",
    Direction.DEGRADATION: "red",
    Direction.INCONCLUSIVE: "yellow",
    Direction.NA: "white",
}


def _styled_badge(direction: Direction) -> str:
    """Return a colored direction badge for terminal output."""
    color = _DIRECTION_COLOR[direction]
    return click.style(_DIRECTION_BADGE[direction], fg=color, bold=True)


def _styled_label(direction: Direction) -> str:
    """Return a colored direction label for terminal output."""
    color = _DIRECTION_COLOR[direction]
    return click.style(_DIRECTION_LABEL[direction], fg=color, bold=True)


_LABELS = {
    "success_rate": "Success rate",
    "cost": "Cost",
    "steps": "Steps",
    "duration": "Duration",
    "tool_error_rate": "Tool errors",
    "path_distribution": "Path dist.",
    "token_usage": "Token usage",
    "token_efficiency": "Token eff.",
    "cost_quality": "Cost / quality",
    "per_task": "Per-task",
}

_INDENT = "  "
_DETAIL_INDENT = "                    "  # 20 chars — aligns under metric detail


def render(result: CompareResult, fmt: str) -> str:
    """Render a CompareResult to the requested format string."""
    if fmt == "terminal":
        return _terminal(result)
    if fmt == "markdown":
        return _markdown(result)
    if fmt == "json":
        return _json(result)
    raise ValueError(f"Unknown format: {fmt!r}")


# ── Terminal ───────────────────────────────────────────────────────────────────


def _terminal(r: CompareResult) -> str:
    bar = click.style("─" * 58, dim=True)

    direction = r.comparison.direction
    badge = _styled_badge(direction)
    dir_label = _styled_label(direction)

    b_src = click.style(f"({r.baseline_source})", dim=True)
    c_src = click.style(f"({r.current_source})", dim=True)

    lines = [
        "",
        f"{_INDENT}{click.style('Kalibra Compare', bold=True)}",
        f"{_INDENT}{bar}",
        (f"{_INDENT}{click.style('Baseline', dim=True)}  {r.baseline_count:>8,} traces   {b_src}"),
        (f"{_INDENT}{click.style('Current', dim=True)}   {r.current_count:>8,} traces   {c_src}"),
        (f"{_INDENT}{click.style('Direction', dim=True)} {badge} {dir_label}"),
        "",
    ]

    if r.warnings:
        for w in r.warnings:
            warn_icon = click.style("!", fg="yellow", bold=True)
            lines.append(f"{_INDENT}{warn_icon}  {click.style(w, fg='yellow')}")
        lines.append("")

    for obs in r.comparison.observations.values():
        badge_m = _styled_badge(obs.direction)
        label_m = _LABELS.get(obs.name, obs.name)
        label_styled = click.style(f"{label_m:<16}", bold=True)

        if obs.name == "per_task":
            meta = obs.metadata
            if meta["matched"] == 0 and not obs.warnings:
                continue
            lines.append(f"{_INDENT}{badge_m} {label_styled}{obs.formatted}")
            n_imp = meta["n_improvements"]
            n_reg = meta["n_regressions"]
            if n_reg > 0:
                sample = ", ".join(meta["regressions"][:3])
                more = f" (+{n_reg - 3} more)" if n_reg > 3 else ""
                reg = click.style("regressed:", fg="red")
                lines.append(f"{_DETAIL_INDENT}{reg} {sample}{more}")
            if n_imp > 0:
                sample = ", ".join(meta["improvements"][:3])
                more = f" (+{n_imp - 3} more)" if n_imp > 3 else ""
                imp = click.style("improved:", fg="green")
                lines.append(f"{_DETAIL_INDENT}{imp}  {sample}{more}")
        else:
            lines.append(f"{_INDENT}{badge_m} {label_styled}{obs.formatted}")
            for detail in obs.detail_lines:
                lines.append(f"{_DETAIL_INDENT}{click.style(detail, dim=True)}")

        for w in obs.warnings:
            warn_icon = click.style("!", fg="yellow", bold=True)
            warn_text = click.style(w, fg="yellow")
            lines.append(f"{_DETAIL_INDENT}{warn_icon}  {warn_text}")
        lines.append("")

    if r.validation.gates:
        lines.append(f"{_INDENT}{click.style('Thresholds', bold=True)}")
        max_expr = max(len(g.expr) for g in r.validation.gates)
        for g in r.validation.gates:
            if g.warning:
                icon = click.style("SKIP", fg="yellow")
            elif g.passed:
                icon = click.style(" OK ", fg="green", bold=True)
            else:
                icon = click.style("FAIL", fg="red", bold=True)
            # nan-safe actual value
            actual = f"{g.actual:.2f}" if g.actual == g.actual else "n/a"
            line = f"{_INDENT}  [{icon}] {g.expr:<{max_expr}}   actual: {actual}"
            if g.warning:
                warn = click.style(f"({g.warning})", fg="yellow")
                line += f"  {warn}"
            lines.append(line)
        lines.append("")

    lines.append(f"{_INDENT}{bar}")
    if r.validation.gates:
        if r.validation.passed:
            ok = click.style("PASSED", fg="green", bold=True)
            lines.append(f"{_INDENT}{ok} — all quality gates met")
        else:
            fail = click.style("FAILED", fg="red", bold=True)
            lines.append(f"{_INDENT}{fail} — quality gate violation (exit code 1)")
    else:
        lines.append(f"{_INDENT}{badge} {dir_label} — no quality gates configured")
    lines.append("")
    return "\n".join(lines)


# ── Markdown ───────────────────────────────────────────────────────────────────


def _markdown(r: CompareResult) -> str:
    direction = r.comparison.direction
    badge = _DIRECTION_BADGE[direction]
    dir_label = _DIRECTION_LABEL[direction]

    if r.validation.gates:
        verdict = "PASSED" if r.validation.passed else "FAILED"
    else:
        verdict = dir_label

    lines = [
        "## Kalibra: Agent Quality Report\n",
        f"**Baseline:** `{r.baseline_source}` ({r.baseline_count:,} traces)  ",
        f"**Current:** `{r.current_source}` ({r.current_count:,} traces)  ",
        f"**Verdict:** {badge} **{verdict}** ({dir_label})\n",
    ]

    if r.warnings:
        for w in r.warnings:
            lines.append(f"> {w}")
        lines.append("")

    # Metrics as sections, not a table — tables can't handle multi-line
    for obs in r.comparison.observations.values():
        if obs.name == "per_task":
            continue
        label_m = _LABELS.get(obs.name, obs.name)
        badge_m = _DIRECTION_BADGE[obs.direction]
        lines.append(f"**{badge_m} {label_m}** — {obs.formatted}  ")
        for detail in obs.detail_lines:
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;{detail}  ")
        for w in obs.warnings:
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;_Warning: {w}_  ")
        lines.append("")

    pt = r.comparison.observations.get("per_task")
    if pt and pt.metadata["matched"] > 0:
        meta = pt.metadata
        n_imp = meta["n_improvements"]
        n_reg = meta["n_regressions"]
        lines.append(f"**{meta['matched']:,} tasks matched**\n")
        if n_reg > 0:
            lines.append(f"**{n_reg} regressed** (success -> failure)")
            for t in meta["regressions"][:5]:
                lines.append(f"- `{t}`")
            if n_reg > 5:
                lines.append(f"- _...and {n_reg - 5} more_")
            lines.append("")
        if n_imp > 0:
            lines.append(f"**{n_imp} improved** (failure -> success)")
            for t in meta["improvements"][:5]:
                lines.append(f"- `{t}`")
            if n_imp > 5:
                lines.append(f"- _...and {n_imp - 5} more_")
            lines.append("")

    if r.validation.gates:
        lines.append("### Thresholds\n")
        for g in r.validation.gates:
            icon = "pass" if g.passed else "**FAIL**"
            actual = f"{g.actual:.2f}" if g.actual == g.actual else "n/a"
            lines.append(f"- [{icon}] `{g.expr}` — actual: `{actual}`")
        lines.append("")

    if r.validation.gates:
        lines.append(
            "> **FAILED** — one or more thresholds not met"
            if not r.validation.passed
            else "> **PASSED** — all thresholds met"
        )
    else:
        lines.append(f"> {badge} {dir_label} (no quality gates configured)")

    lines.append("\n_Generated by [Kalibra](https://github.com/vorekhov/kalibra)_")
    return "\n".join(lines)


# ── JSON ───────────────────────────────────────────────────────────────────────


def _json(r: CompareResult) -> str:
    def _ser(v):
        if isinstance(v, set):
            return list(v)
        return v

    payload = {
        "baseline": {"source": r.baseline_source, "count": r.baseline_count},
        "current": {"source": r.current_source, "count": r.current_count},
        "warnings": r.warnings,
        "comparison": {
            "direction": r.comparison.direction.value,
            "observations": {
                name: {
                    "description": obs.description,
                    "direction": obs.direction.value,
                    "baseline": _ser(obs.baseline),
                    "current": _ser(obs.current),
                    "delta": obs.delta,
                    "formatted": obs.formatted,
                    "detail_lines": obs.detail_lines,
                    "metadata": {k: _ser(v) for k, v in obs.metadata.items()},
                    "warnings": obs.warnings,
                }
                for name, obs in r.comparison.observations.items()
            },
        },
        "validation": {
            "passed": r.validation.passed,
            "gates": [
                {
                    "expr": g.expr,
                    "passed": g.passed,
                    "actual": g.actual,
                    "metric_name": g.metric_name,
                }
                for g in r.validation.gates
            ],
        },
    }
    return json.dumps(payload, indent=2)
