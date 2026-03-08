"""Comparison metrics — protocol, result type, and built-in implementations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar

from agentflow.collection import TraceCollection
from agentflow.converters.base import span_is_error

# Minimum sample sizes for reliable metric computation.
_MIN_N = 30       # below this, any metric is suspect
_MIN_P95_N = 100  # below this, percentile stats are unreliable


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    """Outcome of comparing one metric across baseline and current."""
    name: str
    description: str
    baseline: Any                        # raw summary from summarize()
    current: Any                         # raw summary from summarize()
    delta: float | None = None           # primary scalar delta (pp or %)
    formatted: str = ""                  # one-line human-readable summary
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ── Base class ────────────────────────────────────────────────────────────────

class ComparisonMetric(ABC):
    """Base class for all comparison metrics.

    Subclass this and implement ``summarize`` and ``compare``.
    Optionally override ``threshold_fields`` to expose named values
    that users can reference in ``--require`` expressions.

    Example::

        class MyMetric(ComparisonMetric):
            name = "my_metric"
            description = "Something useful"

            def summarize(self, col: TraceCollection) -> float:
                return sum(len(t.spans) for t in col.all_traces())

            def compare(self, baseline: float, current: float) -> MetricResult:
                delta = _pct_delta(baseline, current)
                return MetricResult(
                    name=self.name, description=self.description,
                    baseline=baseline, current=current,
                    delta=delta, formatted=f"{baseline:.0f} → {current:.0f}  {delta:+.1f}%",
                )

            def threshold_fields(self, result: MetricResult) -> dict[str, float]:
                return {"my_metric_delta": result.delta}
    """

    name: ClassVar[str]
    description: ClassVar[str]

    @abstractmethod
    def summarize(self, col: TraceCollection) -> Any:
        """Reduce a collection to a summary value for this metric."""

    @abstractmethod
    def compare(self, baseline: Any, current: Any) -> MetricResult:
        """Compute a MetricResult from two summary values."""

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        """Named scalar values exposed for ``--require`` threshold expressions."""
        return {}


# ── Built-in metrics ──────────────────────────────────────────────────────────

class SuccessRateMetric(ComparisonMetric):
    name = "success_rate"
    description = "Task success rate delta with statistical significance"

    def summarize(self, col: TraceCollection) -> dict:
        traces = col.all_traces()
        successes = sum(1 for t in traces if t.outcome == "success")
        failures  = sum(1 for t in traces if t.outcome == "failure")
        with_outcome = successes + failures
        # Rate denominator is traces with a known outcome, not all traces.
        # Traces with outcome=None are excluded — they carry no signal.
        return {
            "total": len(traces),
            "with_outcome": with_outcome,
            "successes": successes,
            "failures": failures,
            "rate": successes / with_outcome if with_outcome else None,
        }

    def compare(self, baseline: dict, current: dict) -> MetricResult:
        warnings: list[str] = []

        b_rate, c_rate = baseline["rate"], current["rate"]

        if b_rate is None or c_rate is None:
            side = "baseline" if b_rate is None else "current"
            if b_rate is None and c_rate is None:
                side = "both datasets"
            warnings.append(f"No outcome data in {side} — success rate is unavailable")
            return MetricResult(
                name=self.name, description=self.description,
                baseline=baseline, current=current,
                formatted="n/a — no outcome data",
                warnings=warnings,
            )

        delta_pp = (c_rate - b_rate) * 100
        _, pval = _two_proportion_ztest(
            baseline["with_outcome"], baseline["successes"],
            current["with_outcome"],  current["successes"],
        )
        significant = pval < 0.05
        sig = f"✓ significant (p={pval:.3f})" if significant else f"~ not significant (p={pval:.3f})"
        sign = "+" if delta_pp >= 0 else ""
        formatted = f"{b_rate:.1%} → {c_rate:.1%}  {sign}{delta_pp:.1f} pp  {sig}"

        if baseline["with_outcome"] < _MIN_N or current["with_outcome"] < _MIN_N:
            small = min(baseline["with_outcome"], current["with_outcome"])
            warnings.append(
                f"Only {small} traces with known outcomes — recommend ≥{_MIN_N} for reliable rates"
            )

        return MetricResult(
            name=self.name, description=self.description,
            baseline=baseline, current=current,
            delta=round(delta_pp, 2), formatted=formatted,
            metadata={"pvalue": pval, "significant": significant},
            warnings=warnings,
        )

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {
            "success_rate_delta": result.delta,
            "success_rate": (result.current["rate"] or 0.0) * 100,
        }


class PerTaskMetric(ComparisonMetric):
    name = "per_task"
    description = "Per-task regression and improvement detection"

    def summarize(self, col: TraceCollection) -> dict[str, str]:
        """Returns {task_id: outcome} for all traces with a known outcome."""
        outcomes: dict[str, str] = {}
        for t in col.all_traces():
            if t.outcome in ("success", "failure"):
                outcomes.setdefault(_extract_task_id(t.trace_id), t.outcome)
        return outcomes

    def compare(self, baseline: dict, current: dict) -> MetricResult:
        warnings: list[str] = []
        matched = {t for t in baseline if t in current}
        regressions  = sorted(t for t in matched if baseline[t] == "success" and current[t] == "failure")
        improvements = sorted(t for t in matched if baseline[t] == "failure" and current[t] == "success")
        formatted = (
            f"{len(matched):,} tasks matched — "
            f"✓ {len(improvements)} improved, ✗ {len(regressions)} regressed"
        )

        # Warn when matched tasks are a small fraction of what's available.
        available = len(baseline) + len(current)
        if available > 0 and len(matched) == 0:
            warnings.append(
                "No tasks matched between datasets — task IDs may differ or no outcome data present"
            )
        elif len(baseline) > 0 and len(current) > 0:
            match_rate = len(matched) / min(len(baseline), len(current))
            if 0 < match_rate < 0.1:
                warnings.append(
                    f"Only {len(matched)} of {min(len(baseline), len(current))} tasks matched "
                    f"({match_rate:.0%}) — per-task results may not be representative"
                )

        return MetricResult(
            name=self.name, description=self.description,
            baseline=len(baseline), current=len(current),
            formatted=formatted,
            metadata={
                "matched": len(matched),
                "regressions":  regressions[:20],
                "improvements": improvements[:20],
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        return {
            "regressions":  len(result.metadata["regressions"]),
            "improvements": len(result.metadata["improvements"]),
        }


class CostMetric(ComparisonMetric):
    name = "cost"
    description = "Average cost per trace"

    def summarize(self, col: TraceCollection) -> dict:
        costs = [t.total_cost for t in col.all_traces()]
        return {
            "avg": _mean(costs),
            "total": sum(costs),
            "all_zero": all(c == 0.0 for c in costs),
            "n": len(costs),
        }

    def compare(self, baseline: dict, current: dict) -> MetricResult:
        warnings: list[str] = []
        if baseline["all_zero"] and current["all_zero"]:
            warnings.append("All span costs are 0 — cost data may not be populated in your traces")
            return MetricResult(
                name=self.name, description=self.description,
                baseline=baseline, current=current,
                formatted="n/a — no cost data",
                warnings=warnings,
            )

        delta = _pct_delta(baseline["avg"], current["avg"])
        sign = "+" if delta >= 0 else ""
        formatted = f"${baseline['avg']:.4f} → ${current['avg']:.4f}  {sign}{delta:.1f}%"
        return MetricResult(
            name=self.name, description=self.description,
            baseline=baseline, current=current,
            delta=delta, formatted=formatted,
            warnings=warnings,
        )

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {"cost_delta_pct": result.delta}


class StepsMetric(ComparisonMetric):
    name = "steps"
    description = "Average steps (spans) per trace"

    def summarize(self, col: TraceCollection) -> dict:
        steps = [len(t.spans) for t in col.all_traces()]
        return {"avg": _mean(steps), "n": len(steps)}

    def compare(self, baseline: dict, current: dict) -> MetricResult:
        delta = _pct_delta(baseline["avg"], current["avg"])
        sign = "+" if delta >= 0 else ""
        formatted = f"{baseline['avg']:.1f} → {current['avg']:.1f} steps  {sign}{delta:.1f}%"
        return MetricResult(
            name=self.name, description=self.description,
            baseline=baseline, current=current,
            delta=delta, formatted=formatted,
        )

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        return {"steps_delta_pct": result.delta}


class DurationMetric(ComparisonMetric):
    name = "duration"
    description = "Trace duration — average and P95 latency"

    def summarize(self, col: TraceCollection) -> dict:
        durations = sorted(t.duration for t in col.all_traces())
        return {"avg": _mean(durations), "p95": _percentile(durations, 95), "n": len(durations)}

    def compare(self, baseline: dict, current: dict) -> MetricResult:
        warnings: list[str] = []
        avg_delta = _pct_delta(baseline["avg"], current["avg"])
        p95_delta = _pct_delta(baseline["p95"], current["p95"])
        sign_avg = "+" if avg_delta >= 0 else ""
        sign_p95 = "+" if p95_delta >= 0 else ""
        formatted = (
            f"avg {baseline['avg']:.1f}s → {current['avg']:.1f}s  {sign_avg}{avg_delta:.1f}%  |  "
            f"P95 {baseline['p95']:.1f}s → {current['p95']:.1f}s  {sign_p95}{p95_delta:.1f}%"
        )

        small_n = min(baseline["n"], current["n"])
        if small_n < _MIN_P95_N:
            warnings.append(
                f"P95 computed from {small_n} traces — recommend ≥{_MIN_P95_N} for stable percentiles"
            )

        return MetricResult(
            name=self.name, description=self.description,
            baseline=baseline, current=current,
            delta=avg_delta, formatted=formatted,
            metadata={"p95_delta_pct": p95_delta},
            warnings=warnings,
        )

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        return {
            "duration_delta_pct":     result.delta,
            "duration_p95_delta_pct": result.metadata["p95_delta_pct"],
        }


class ToolErrorRateMetric(ComparisonMetric):
    name = "tool_error_rate"
    description = "Fraction of tool invocations that returned an error"

    def summarize(self, col: TraceCollection) -> dict:
        spans = [s for t in col.all_traces() for s in t.spans]
        errors = sum(1 for s in spans if span_is_error(s))
        return {"rate": errors / len(spans) if spans else 0.0,
                "errors": errors, "total": len(spans)}

    def compare(self, baseline: dict, current: dict) -> MetricResult:
        warnings: list[str] = []
        delta_pp = (current["rate"] - baseline["rate"]) * 100
        sign = "+" if delta_pp >= 0 else ""
        formatted = f"{baseline['rate']:.1%} → {current['rate']:.1%}  {sign}{delta_pp:.1f} pp"

        small_n = min(baseline["total"], current["total"])
        if small_n < _MIN_N:
            warnings.append(
                f"Only {small_n} tool invocations — error rate estimate may be noisy"
            )

        return MetricResult(
            name=self.name, description=self.description,
            baseline=baseline, current=current,
            delta=round(delta_pp, 2), formatted=formatted,
            warnings=warnings,
        )

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        return {"tool_error_rate_delta": result.delta}


class PathDistributionMetric(ComparisonMetric):
    name = "path_distribution"
    description = "Jaccard similarity of top execution paths"

    def summarize(self, col: TraceCollection) -> dict:
        traces = col.all_traces()
        path_counts = Counter(_trace_path(t) for t in traces)
        return {
            "top_paths": {p for p, _ in path_counts.most_common(20)},
            "n": len(traces),
            "unique_paths": len(path_counts),
        }

    def compare(self, baseline: dict, current: dict) -> MetricResult:
        warnings: list[str] = []
        b, c = baseline["top_paths"], current["top_paths"]
        jaccard = len(b & c) / len(b | c) if (b | c) else 1.0
        new_paths     = list(c - b)[:10]
        dropped_paths = list(b - c)[:10]
        formatted = (
            f"Jaccard {jaccard:.2f}  "
            f"(+{len(new_paths)} new, −{len(dropped_paths)} dropped)"
        )

        small_n = min(baseline["n"], current["n"])
        if small_n < _MIN_N:
            warnings.append(
                f"Path Jaccard computed from {small_n} traces — "
                f"small samples inflate apparent path divergence"
            )

        return MetricResult(
            name=self.name, description=self.description,
            baseline=len(b), current=len(c),
            delta=round(jaccard, 3), formatted=formatted,
            metadata={
                "jaccard": round(jaccard, 3),
                "new_paths": new_paths,
                "dropped_paths": dropped_paths,
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        return {"path_jaccard": result.metadata["jaccard"]}


# ── Default metric set ─────────────────────────────────────────────────────────

DEFAULT_METRICS: list[ComparisonMetric] = [
    SuccessRateMetric(),
    PerTaskMetric(),
    CostMetric(),
    StepsMetric(),
    DurationMetric(),
    ToolErrorRateMetric(),
    PathDistributionMetric(),
]


# ── Math helpers ───────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * pct / 100), len(sorted_values) - 1)
    return sorted_values[idx]


def _pct_delta(base: float, curr: float) -> float:
    if base == 0:
        return 0.0
    return round((curr - base) / base * 100, 1)


def _two_proportion_ztest(n1: int, s1: int, n2: int, s2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p_pool = (s1 + s2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return 0.0, 1.0
    z = (s2 / n2 - s1 / n1) / denom
    return round(z, 4), round(math.erfc(abs(z) / math.sqrt(2)), 6)


def _trace_path(trace) -> str:
    return " -> ".join(s.name for s in sorted(trace.spans, key=lambda s: s.start_time))


def _extract_task_id(trace_id: str) -> str:
    """SWE-bench: instance_id__model__row_idx → strip last two __ segments."""
    parts = trace_id.split("__")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "__".join(parts[:-2])
    return trace_id
