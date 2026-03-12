"""Multi-format report rendering for CompareResult."""

from __future__ import annotations

import json

from agentflow.compare import CompareResult
from agentflow.metrics import Direction


_DIRECTION_BADGE = {
    Direction.UPGRADE:      "▲",
    Direction.SAME:         "≈",
    Direction.DEGRADATION:  "▼",
    Direction.INCONCLUSIVE: "~",
    Direction.NA:           "—",
}

_DIRECTION_LABEL = {
    Direction.UPGRADE:      "IMPROVED",
    Direction.SAME:         "UNCHANGED",
    Direction.DEGRADATION:  "REGRESSED",
    Direction.INCONCLUSIVE: "MIXED",
    Direction.NA:           "N/A",
}

_LABELS = {
    "success_rate":       "Success rate",
    "cost":               "Cost",
    "steps":              "Steps",
    "duration":           "Duration",
    "tool_error_rate":    "Tool errors",
    "path_distribution":  "Path dist.",
    "token_usage":        "Token usage",
    "token_efficiency":   "Token eff.",
    "cost_quality":       "Cost / quality",
    "per_task":           "Per-task",
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
    bar = "─" * 58

    direction = r.comparison.direction
    badge = _DIRECTION_BADGE[direction]
    dir_label = _DIRECTION_LABEL[direction]

    lines = [
        "",
        f"{_INDENT}AgentFlow Compare",
        f"{_INDENT}{bar}",
        f"{_INDENT}Baseline  {r.baseline_count:>8,} traces   ({r.baseline_source})",
        f"{_INDENT}Current   {r.current_count:>8,} traces   ({r.current_source})",
        f"{_INDENT}Direction {badge} {dir_label}",
        "",
    ]

    if r.warnings:
        for w in r.warnings:
            lines.append(f"{_INDENT}!  {w}")
        lines.append("")

    for obs in r.comparison.observations.values():
        badge_m = _DIRECTION_BADGE[obs.direction]
        label_m = _LABELS.get(obs.name, obs.name)

        if obs.name == "per_task":
            meta = obs.metadata
            if meta["matched"] == 0 and not obs.warnings:
                continue
            lines.append(f"{_INDENT}{badge_m} {label_m:<16}{obs.formatted}")
            n_imp = meta["n_improvements"]
            n_reg = meta["n_regressions"]
            # Show up to 3 task IDs as examples, never flood the terminal
            if n_reg > 0:
                sample = ", ".join(meta["regressions"][:3])
                suffix = f" (+{n_reg - 3} more)" if n_reg > 3 else ""
                lines.append(f"{_DETAIL_INDENT}regressed: {sample}{suffix}")
            if n_imp > 0:
                sample = ", ".join(meta["improvements"][:3])
                suffix = f" (+{n_imp - 3} more)" if n_imp > 3 else ""
                lines.append(f"{_DETAIL_INDENT}improved:  {sample}{suffix}")
        else:
            lines.append(f"{_INDENT}{badge_m} {label_m:<16}{obs.formatted}")
            for detail in obs.detail_lines:
                lines.append(f"{_DETAIL_INDENT}{detail}")

        for w in obs.warnings:
            lines.append(f"{_DETAIL_INDENT}!  {w}")
        lines.append("")

    if r.validation.gates:
        lines.append(f"{_INDENT}Thresholds")
        # Align expressions to the widest one
        max_expr = max(len(g.expr) for g in r.validation.gates)
        for g in r.validation.gates:
            icon = "SKIP" if g.warning else ("OK" if g.passed else "FAIL")
            actual = f"{g.actual:.2f}" if g.actual == g.actual else "n/a"  # nan-safe
            line = f"{_INDENT}  [{icon:>4}] {g.expr:<{max_expr}}   actual: {actual}"
            if g.warning:
                line += f"  ({g.warning})"
            lines.append(line)
        lines.append("")

    lines.append(f"{_INDENT}{bar}")
    if r.validation.gates:
        if r.validation.passed:
            lines.append(f"{_INDENT}PASSED — all quality gates met")
        else:
            lines.append(f"{_INDENT}FAILED — quality gate violation (exit code 1)")
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
        verdict_icon = "pass" if r.validation.passed else "fail"
    else:
        verdict = dir_label
        verdict_icon = "info"

    lines = [
        "## AgentFlow: Agent Quality Report\n",
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
            if not r.validation.passed else
            "> **PASSED** — all thresholds met"
        )
    else:
        lines.append(f"> {badge} {dir_label} (no quality gates configured)")

    lines.append("\n_Generated by [AgentFlow](https://github.com/khan5v/agentflow)_")
    return "\n".join(lines)


# ── JSON ───────────────────────────────────────────────────────────────────────

def _json(r: CompareResult) -> str:
    def _ser(v):
        if isinstance(v, set):
            return list(v)
        return v

    payload = {
        "baseline": {"source": r.baseline_source, "count": r.baseline_count},
        "current":  {"source": r.current_source,  "count": r.current_count},
        "warnings": r.warnings,
        "comparison": {
            "direction": r.comparison.direction.value,
            "observations": {
                name: {
                    "description": obs.description,
                    "direction":   obs.direction.value,
                    "baseline":    _ser(obs.baseline),
                    "current":     _ser(obs.current),
                    "delta":       obs.delta,
                    "formatted":   obs.formatted,
                    "detail_lines": obs.detail_lines,
                    "metadata":    {k: _ser(v) for k, v in obs.metadata.items()},
                    "warnings":    obs.warnings,
                }
                for name, obs in r.comparison.observations.items()
            },
        },
        "validation": {
            "passed": r.validation.passed,
            "gates": [
                {"expr": g.expr, "passed": g.passed, "actual": g.actual,
                 "metric_name": g.metric_name}
                for g in r.validation.gates
            ],
        },
    }
    return json.dumps(payload, indent=2)
