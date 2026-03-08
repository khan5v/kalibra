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
    Direction.UPGRADE:      "upgrade",
    Direction.SAME:         "same",
    Direction.DEGRADATION:  "degradation",
    Direction.INCONCLUSIVE: "inconclusive",
    Direction.NA:           "n/a",
}

_LABELS = {
    "success_rate":      "Success rate",
    "cost":              "Avg cost",
    "steps":             "Avg steps",
    "duration":          "Duration",
    "tool_error_rate":   "Tool error rate",
    "path_distribution": "Path distribution",
}


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
    label = _DIRECTION_LABEL[direction].upper()
    lines = [
        "",
        "  AgentFlow Compare",
        "  " + bar,
        f"  Baseline  {r.baseline_count:>8,} traces   ({r.baseline_source})",
        f"  Current   {r.current_count:>8,} traces   ({r.current_source})",
        f"  Overall   {badge} {label}",
        "",
    ]

    if r.warnings:
        for w in r.warnings:
            lines.append(f"  ⚠  {w}")
        lines.append("")

    for obs in r.comparison.observations.values():
        badge_m = _DIRECTION_BADGE[obs.direction]
        if obs.name == "per_task":
            meta = obs.metadata
            if meta["matched"] == 0 and not obs.warnings:
                continue
            lines.append(f"  {badge_m} Per-task      {obs.formatted}")
            if meta["improvements"]:
                items = ", ".join(meta["improvements"][:5])
                more = f" …+{len(meta['improvements'])-5}" if len(meta["improvements"]) > 5 else ""
                lines.append(f"    ✓ Improved:  {items}{more}")
            if meta["regressions"]:
                items = ", ".join(meta["regressions"][:5])
                more = f" …+{len(meta['regressions'])-5}" if len(meta["regressions"]) > 5 else ""
                lines.append(f"    ✗ Regressed: {items}{more}")
        else:
            label_m = _LABELS.get(obs.name, obs.name)
            lines.append(f"  {badge_m} {label_m:<18}{obs.formatted}")
        for w in obs.warnings:
            lines.append(f"    ⚠  {w}")

    if r.validation.gates:
        lines.append("")
        lines.append("  Thresholds")
        for g in r.validation.gates:
            icon = "✓" if g.passed else "✗"
            actual = f"{g.actual:.2f}" if g.actual == g.actual else "n/a"  # nan-safe
            lines.append(f"    {icon} {g.expr}   (actual: {actual})")

    lines += ["", "  " + bar]
    lines.append("  FAILED: one or more thresholds not met" if not r.validation.passed
                 else "  All checks passed")
    lines.append("")
    return "\n".join(lines)


# ── Markdown ───────────────────────────────────────────────────────────────────

def _markdown(r: CompareResult) -> str:
    direction = r.comparison.direction
    badge = _DIRECTION_BADGE[direction]
    label = _DIRECTION_LABEL[direction]
    lines = [
        "## AgentFlow: Agent Quality Report\n",
        f"**Baseline:** `{r.baseline_source}` ({r.baseline_count:,} traces)  ",
        f"**Current:** `{r.current_source}` ({r.current_count:,} traces)  ",
        f"**Overall:** {badge} {label}\n",
    ]

    if r.warnings:
        for w in r.warnings:
            lines.append(f"> ⚠️ {w}")
        lines.append("")

    lines += ["| Metric | Dir | Result | Notes |", "|--------|-----|--------|-------|"]

    for obs in r.comparison.observations.values():
        if obs.name == "per_task":
            continue
        label_m = _LABELS.get(obs.name, obs.name)
        badge_m = _DIRECTION_BADGE[obs.direction]
        notes = " ".join(f"⚠️ {w}" for w in obs.warnings) if obs.warnings else ""
        lines.append(f"| {label_m} | {badge_m} | {obs.formatted} | {notes} |")

    lines.append("")

    pt = r.comparison.observations.get("per_task")
    if pt and pt.metadata["matched"] > 0:
        meta = pt.metadata
        lines.append(f"**{meta['matched']:,} tasks matched**\n")
        if meta["improvements"]:
            lines.append(f"✅ **{len(meta['improvements'])} improved** (failure → success)")
            for t in meta["improvements"][:10]:
                lines.append(f"- `{t}`")
            if len(meta["improvements"]) > 10:
                lines.append(f"- _…and {len(meta['improvements']) - 10} more_")
            lines.append("")
        if meta["regressions"]:
            lines.append(f"⚠️ **{len(meta['regressions'])} regressed** (success → failure)")
            for t in meta["regressions"][:10]:
                lines.append(f"- `{t}`")
            if len(meta["regressions"]) > 10:
                lines.append(f"- _…and {len(meta['regressions']) - 10} more_")
            lines.append("")

    if r.validation.gates:
        lines.append("**Thresholds**\n")
        for g in r.validation.gates:
            icon = "✅" if g.passed else "❌"
            actual = f"{g.actual:.2f}" if g.actual == g.actual else "n/a"
            lines.append(f"- {icon} `{g.expr}` — actual: `{actual}`")
        lines.append("")

    lines.append(
        "> ❌ **CI gate failed** — one or more thresholds not met"
        if not r.validation.passed else
        "> ✅ All checks passed"
    )
    lines.append("\n_Generated by [AgentFlow](https://github.com/vorekhov/agentflow)_")
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
