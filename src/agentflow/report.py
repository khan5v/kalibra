"""Multi-format report rendering for CompareResult."""

from __future__ import annotations

import json

from agentflow.compare import CompareResult


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
    lines = [
        "",
        "  AgentFlow Compare",
        "  " + bar,
        f"  Baseline  {r.baseline_count:>8,} traces   ({r.baseline_source})",
        f"  Current   {r.current_count:>8,} traces   ({r.current_source})",
        "",
    ]

    if r.warnings:
        for w in r.warnings:
            lines.append(f"  ⚠  {w}")
        lines.append("")

    for m in r.metrics.values():
        if m.name == "per_task":
            meta = m.metadata
            if meta["matched"] == 0 and not m.warnings:
                continue
            lines.append(f"  Per-task      {m.formatted}")
            if meta["improvements"]:
                items = ", ".join(meta["improvements"][:5])
                more = f" …+{len(meta['improvements'])-5}" if len(meta["improvements"]) > 5 else ""
                lines.append(f"    ✓ Improved:  {items}{more}")
            if meta["regressions"]:
                items = ", ".join(meta["regressions"][:5])
                more = f" …+{len(meta['regressions'])-5}" if len(meta["regressions"]) > 5 else ""
                lines.append(f"    ✗ Regressed: {items}{more}")
        else:
            label = _LABELS.get(m.name, m.name)
            lines.append(f"  {label:<18}{m.formatted}")
        for w in m.warnings:
            lines.append(f"    ⚠  {w}")

    if r.threshold_results:
        lines.append("")
        lines.append("  Thresholds")
        for t in r.threshold_results:
            icon = "✓" if t["passed"] else "✗"
            actual = f"{t['actual']:.2f}" if t["actual"] is not None else "n/a"
            lines.append(f"    {icon} {t['expr']}   (actual: {actual})")

    lines += ["", "  " + bar]
    lines.append("  FAILED: one or more thresholds not met" if not r.thresholds_passed
                 else "  All checks passed")
    lines.append("")
    return "\n".join(lines)


_LABELS = {
    "success_rate":    "Success rate",
    "cost":            "Avg cost",
    "steps":           "Avg steps",
    "duration":        "Duration",
    "tool_error_rate": "Tool error rate",
    "path_distribution": "Path distribution",
}


# ── Markdown ───────────────────────────────────────────────────────────────────

def _markdown(r: CompareResult) -> str:
    lines = [
        "## AgentFlow: Agent Quality Report\n",
        f"**Baseline:** `{r.baseline_source}` ({r.baseline_count:,} traces)  ",
        f"**Current:** `{r.current_source}` ({r.current_count:,} traces)\n",
    ]

    if r.warnings:
        for w in r.warnings:
            lines.append(f"> ⚠️ {w}")
        lines.append("")

    lines += ["| Metric | Result | Notes |", "|--------|--------|-------|"]

    for m in r.metrics.values():
        if m.name == "per_task":
            continue
        label = _LABELS.get(m.name, m.name)
        icon = _delta_icon(m.name, m.delta)
        notes = " ".join(f"⚠️ {w}" for w in m.warnings) if m.warnings else ""
        lines.append(f"| {label} | {icon} {m.formatted} | {notes} |")

    lines.append("")

    pt = r.metrics.get("per_task")
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

    if r.threshold_results:
        lines.append("**Thresholds**\n")
        for t in r.threshold_results:
            icon = "✅" if t["passed"] else "❌"
            actual = f"{t['actual']:.2f}" if t["actual"] is not None else "n/a"
            lines.append(f"- {icon} `{t['expr']}` — actual: `{actual}`")
        lines.append("")

    lines.append(
        "> ❌ **CI gate failed** — one or more thresholds not met"
        if not r.thresholds_passed else
        "> ✅ All checks passed"
    )
    lines.append("\n_Generated by [AgentFlow](https://github.com/vorekhov/agentflow)_")
    return "\n".join(lines)


def _delta_icon(metric_name: str, delta: float | None) -> str:
    """Return ✅/⚠️/— based on whether a delta is good or bad."""
    if delta is None:
        return ""
    lower_is_better = {"cost", "steps", "duration", "tool_error_rate"}
    if metric_name in lower_is_better:
        return "✅" if delta <= 0 else "⚠️"
    if metric_name == "path_distribution":
        return "✅" if delta >= 0.8 else "⚠️"
    return "✅" if delta >= 0 else "⚠️"


# ── JSON ───────────────────────────────────────────────────────────────────────

def _json(r: CompareResult) -> str:
    def _serialise(v):
        if isinstance(v, set):
            return list(v)
        return v

    payload = {
        "baseline": {"source": r.baseline_source, "count": r.baseline_count},
        "current":  {"source": r.current_source,  "count": r.current_count},
        "metrics": {
            name: {
                "description": m.description,
                "baseline": _serialise(m.baseline),
                "current":  _serialise(m.current),
                "delta":    m.delta,
                "formatted": m.formatted,
                "metadata": {k: _serialise(v) for k, v in m.metadata.items()},
            }
            for name, m in r.metrics.items()
        },
        "thresholds": {
            "passed": r.thresholds_passed,
            "results": r.threshold_results,
        },
    }
    return json.dumps(payload, indent=2)
