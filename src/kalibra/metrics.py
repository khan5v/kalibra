"""Comparison metrics — protocol, result type, and built-in implementations."""

from __future__ import annotations

import math
import random
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from kalibra.collection import TraceCollection
from kalibra.converters.base import (
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    span_input_tokens,
    span_is_error,
    span_output_tokens,
)

# Optional scipy for statistical tests on continuous metrics.
try:
    from scipy.stats import mannwhitneyu as _mannwhitneyu

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

_SCIPY_HINT_SHOWN = False

# Minimum sample sizes for reliable metric computation.
_MIN_N = 30  # below this, any metric is suspect
_MIN_P95_N = 100  # below this, percentile stats are unreliable


# ── Direction ─────────────────────────────────────────────────────────────────


class Direction(str, Enum):
    """Comparison signal for an observation or a rolled-up comparison."""

    UPGRADE = "upgrade"  # meaningfully better
    SAME = "same"  # within noise
    DEGRADATION = "degradation"  # meaningfully worse
    INCONCLUSIVE = "inconclusive"  # metrics pull in opposite directions
    NA = "n/a"  # no data to compare


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class Observation:
    """Outcome of comparing one metric across baseline and current."""

    name: str
    description: str
    direction: Direction
    baseline: Any  # raw summary from summarize()
    current: Any  # raw summary from summarize()
    delta: float | None = None  # primary scalar delta (pp or %)
    formatted: str = ""  # headline: delta + primary comparison
    detail_lines: list[str] = field(default_factory=list)  # sub-lines for breakdown
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# Backwards compatibility alias
MetricResult = Observation


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
            noise_threshold = 2.0
            higher_is_better = True

            def summarize(self, col: TraceCollection) -> float:
                return sum(len(t.spans) for t in col.all_traces())

            def compare(self, baseline: float, current: float) -> Observation:
                delta = _pct_delta(baseline, current)
                direction = _direction_from_delta(
                    delta, self.noise_threshold, self.higher_is_better,
                )
                return Observation(
                    name=self.name, description=self.description,
                    direction=direction,
                    baseline=baseline, current=current,
                    delta=delta,
                    formatted=f"{baseline:.0f} → {current:.0f}  {delta:+.1f}%",
                )

            def threshold_fields(self, result: Observation) -> dict[str, float]:
                return {"my_metric_delta": result.delta}
    """

    name: ClassVar[str]
    description: ClassVar[str]
    noise_threshold: ClassVar[float] = 0.5
    higher_is_better: ClassVar[bool] = True

    #: Single source of truth for threshold field names and descriptions.
    #: Override in subclasses. ``threshold_field_names()`` reads from this,
    #: and ``threshold_fields()`` must only return keys defined here.
    _fields: ClassVar[dict[str, str]] = {}

    @abstractmethod
    def summarize(self, col: TraceCollection) -> Any:
        """Reduce a collection to a summary value for this metric."""

    @abstractmethod
    def compare(self, baseline: Any, current: Any) -> Observation:
        """Compute an Observation from two summary values."""

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        """Named scalar values exposed for ``--require`` threshold expressions."""
        return {}

    @classmethod
    def threshold_field_names(cls) -> dict[str, str]:
        """Return {field_name: description} for all threshold fields this metric exposes."""
        return cls._fields


# ── Direction helper ──────────────────────────────────────────────────────────


def _direction_from_delta(
    delta: float | None,
    noise_threshold: float,
    higher_is_better: bool = True,
) -> Direction:
    if delta is None:
        return Direction.NA
    if abs(delta) <= noise_threshold:
        return Direction.SAME
    return Direction.UPGRADE if (delta > 0) == higher_is_better else Direction.DEGRADATION


# ── Mann-Whitney U helper ─────────────────────────────────────────────────────


def _maybe_scipy_hint() -> None:
    """Show a one-time hint about installing scipy for better statistical tests."""
    global _SCIPY_HINT_SHOWN
    if HAS_SCIPY or _SCIPY_HINT_SHOWN:
        return
    _SCIPY_HINT_SHOWN = True
    warnings.warn(
        "scipy not installed — continuous metrics use threshold-based comparison only. "
        "For Mann-Whitney U significance tests: pip install kalibra[stats]",
        stacklevel=3,
    )


def _mannwhitney(
    baseline_values: list[float],
    current_values: list[float],
    higher_is_better: bool,
) -> dict | None:
    """Run Mann-Whitney U test if scipy is available and samples are large enough.

    Returns {"U": float, "pvalue": float, "significant": bool} or None.
    """
    if not HAS_SCIPY:
        return None
    if len(baseline_values) < 2 or len(current_values) < 2:
        return None
    # All identical values → no test needed (mannwhitneyu raises on zero variance).
    if len(set(baseline_values)) <= 1 and len(set(current_values)) <= 1:
        if baseline_values[0] == current_values[0]:
            return {"U": 0.0, "pvalue": 1.0, "significant": False}
        # Different constants — clearly significant.
        return {"U": 0.0, "pvalue": 0.0, "significant": True}
    try:
        stat, pval = _mannwhitneyu(baseline_values, current_values, alternative="two-sided")
        return {
            "U": round(float(stat), 2),
            "pvalue": round(float(pval), 6),
            "significant": bool(pval < 0.05),
        }
    except ValueError:
        return None


# ── Built-in metrics ──────────────────────────────────────────────────────────


class SuccessRateMetric(ComparisonMetric):
    name = "success_rate"
    description = "Task success rate delta with statistical significance"
    noise_threshold = 0.5
    higher_is_better = True
    _fields = {
        "success_rate_delta": "Change in success rate (percentage points)",
        "success_rate": "Current success rate (%)",
    }

    def summarize(self, col: TraceCollection) -> dict:
        traces = col.all_traces()
        successes = sum(1 for t in traces if t.outcome == OUTCOME_SUCCESS)
        failures = sum(1 for t in traces if t.outcome == OUTCOME_FAILURE)
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

    def compare(self, baseline: dict, current: dict) -> Observation:
        warnings: list[str] = []

        b_rate, c_rate = baseline["rate"], current["rate"]

        if b_rate is None or c_rate is None:
            side = "baseline" if b_rate is None else "current"
            if b_rate is None and c_rate is None:
                side = "both datasets"
            warnings.append(f"No outcome data in {side} — success rate is unavailable")
            return Observation(
                name=self.name,
                description=self.description,
                direction=Direction.NA,
                baseline=baseline,
                current=current,
                formatted="n/a — no outcome data",
                warnings=warnings,
            )

        delta_pp = (c_rate - b_rate) * 100
        _, pval = _two_proportion_ztest(
            baseline["with_outcome"],
            baseline["successes"],
            current["with_outcome"],
            current["successes"],
        )
        significant = pval < 0.05
        sign = "+" if delta_pp >= 0 else ""
        formatted = f"{b_rate:.1%} → {c_rate:.1%}  {sign}{delta_pp:.1f} pp"

        details: list[str] = []
        if significant:
            details.append(f"p={pval:.3f} — statistically significant")
        else:
            details.append(f"p={pval:.3f} — not statistically significant")
        details.append(
            f"n={baseline['with_outcome']}→{current['with_outcome']} traces with outcomes"
        )

        if baseline["with_outcome"] < _MIN_N or current["with_outcome"] < _MIN_N:
            small = min(baseline["with_outcome"], current["with_outcome"])
            warnings.append(
                f"Only {small} traces with known outcomes — recommend ≥{_MIN_N} for reliable rates"
            )

        direction = _direction_from_delta(delta_pp, self.noise_threshold, self.higher_is_better)
        if not significant:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=baseline,
            current=current,
            delta=round(delta_pp, 2),
            formatted=formatted,
            detail_lines=details,
            metadata={"pvalue": pval, "significant": significant},
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {
            "success_rate_delta": result.delta,
            "success_rate": (result.current["rate"] or 0.0) * 100,
        }


class PerTaskMetric(ComparisonMetric):
    name = "per_task"
    description = "Per-task regression and improvement detection"
    noise_threshold = 0.0
    higher_is_better = True
    _fields = {
        "regressions": "Number of regressed tasks",
        "improvements": "Number of improved tasks",
    }

    def __init__(self):
        self.task_id_field: str | None = None

    def summarize(self, col: TraceCollection) -> dict[str, str]:
        """Returns {task_id: outcome} for all traces with a known outcome."""
        outcomes: dict[str, str] = {}
        for t in col.all_traces():
            if t.outcome in (OUTCOME_SUCCESS, OUTCOME_FAILURE):
                task_id = _extract_task_id_from_trace(t, self.task_id_field)
                outcomes.setdefault(task_id, t.outcome)
        return outcomes

    def compare(self, baseline: dict, current: dict) -> Observation:
        warnings: list[str] = []
        matched = {t for t in baseline if t in current}
        regressions = sorted(
            t for t in matched
            if baseline[t] == OUTCOME_SUCCESS and current[t] == OUTCOME_FAILURE
        )
        improvements = sorted(
            t for t in matched
            if baseline[t] == OUTCOME_FAILURE and current[t] == OUTCOME_SUCCESS
        )
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

        if not matched:
            direction = Direction.NA
        elif len(regressions) > len(improvements):
            direction = Direction.DEGRADATION
        elif len(improvements) > len(regressions):
            direction = Direction.UPGRADE
        else:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=len(baseline),
            current=len(current),
            formatted=formatted,
            metadata={
                "matched": len(matched),
                "regressions": regressions[:20],
                "improvements": improvements[:20],
                "n_regressions": len(regressions),
                "n_improvements": len(improvements),
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        return {
            "regressions": result.metadata["n_regressions"],
            "improvements": result.metadata["n_improvements"],
        }


class CostMetric(ComparisonMetric):
    name = "cost"
    description = "Cost per trace — median, average, and total"
    noise_threshold = 3.0
    higher_is_better = False
    _fields = {
        "cost_delta_pct": "Median cost change (%)",
        "total_cost": "Total cost of current run",
        "avg_cost": "Average cost per trace",
    }

    def summarize(self, col: TraceCollection) -> dict:
        costs = [t.total_cost for t in col.all_traces()]
        p25, med, p75 = _iqr(costs)
        return {
            "avg": _mean(costs),
            "median": med,
            "p25": p25,
            "p75": p75,
            "total": sum(costs),
            "ci_95": _bootstrap_ci(costs, stat_fn=_median) if len(costs) >= 2 else None,
            "all_zero": all(c == 0.0 for c in costs),
            "n": len(costs),
            "_values": costs,
        }

    def compare(self, baseline: dict, current: dict) -> Observation:
        obs_warnings: list[str] = []
        if baseline["all_zero"] and current["all_zero"]:
            obs_warnings.append(
                "All span costs are 0 — cost data may not be populated in your traces"
            )
            return Observation(
                name=self.name,
                description=self.description,
                direction=Direction.NA,
                baseline=baseline,
                current=current,
                formatted="n/a — no cost data",
                warnings=obs_warnings,
            )

        _maybe_scipy_hint()

        # Primary stat: median (robust to outliers). Delta computed on median.
        med_delta = _pct_delta(baseline["median"], current["median"])
        avg_delta = _pct_delta(baseline["avg"], current["avg"])
        sign = "+" if med_delta >= 0 else ""
        formatted = (
            f"${baseline['median']:.4f} → ${current['median']:.4f} median  {sign}{med_delta:.1f}%"
        )

        details = [
            f"${baseline['avg']:.4f} → ${current['avg']:.4f} avg"
            f"  {'+' if avg_delta >= 0 else ''}{avg_delta:.1f}%",
            f"${baseline['total']:.2f} → ${current['total']:.2f} total",
        ]
        if current["ci_95"] and baseline["ci_95"]:
            ci_lo = _pct_delta(baseline["median"], current["ci_95"][0])
            ci_hi = _pct_delta(baseline["median"], current["ci_95"][1])
            if ci_lo != ci_hi:
                details.append(f"95% CI [{ci_lo:+.1f}%, {ci_hi:+.1f}%]")

        mw = _mannwhitney(baseline["_values"], current["_values"], self.higher_is_better)
        if mw:
            if mw["significant"]:
                details.append(f"Mann-Whitney U p={mw['pvalue']:.3f} — statistically significant")
            else:
                details.append(
                    f"Mann-Whitney U p={mw['pvalue']:.3f} — not statistically significant"
                )

        direction = _direction_from_delta(med_delta, self.noise_threshold, self.higher_is_better)
        if mw and not mw["significant"]:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=baseline,
            current=current,
            delta=med_delta,
            formatted=formatted,
            detail_lines=details,
            metadata={
                "avg_delta": avg_delta,
                "ci_95": current["ci_95"],
                "baseline_ci_95": baseline["ci_95"],
                "mannwhitney": mw,
            },
            warnings=obs_warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["cost_delta_pct"] = result.delta
        fields["total_cost"] = result.current["total"]
        fields["avg_cost"] = result.current["avg"]
        return fields


class StepsMetric(ComparisonMetric):
    name = "steps"
    description = "Steps (spans) per trace — median and average"
    noise_threshold = 3.0
    higher_is_better = False
    _fields = {
        "steps_delta_pct": "Median steps change (%)",
        "avg_steps": "Average steps per trace",
        "median_steps": "Median steps per trace",
    }

    def summarize(self, col: TraceCollection) -> dict:
        steps = [float(len(t.spans)) for t in col.all_traces()]
        return {
            "avg": _mean(steps),
            "median": _median(steps),
            "ci_95": _bootstrap_ci(steps, stat_fn=_median) if len(steps) >= 2 else None,
            "n": len(steps),
            "_values": steps,
        }

    def compare(self, baseline: dict, current: dict) -> Observation:
        _maybe_scipy_hint()
        med_delta = _pct_delta(baseline["median"], current["median"])
        avg_delta = _pct_delta(baseline["avg"], current["avg"])
        sign = "+" if med_delta >= 0 else ""
        formatted = (
            f"{baseline['median']:.0f} → {current['median']:.0f}"
            f" steps/trace (median)  {sign}{med_delta:.1f}%"
        )

        details = [
            f"{baseline['avg']:.1f} → {current['avg']:.1f} avg"
            f"  {'+' if avg_delta >= 0 else ''}{avg_delta:.1f}%",
        ]
        if current["ci_95"] and baseline["ci_95"]:
            ci_lo = _pct_delta(baseline["median"], current["ci_95"][0])
            ci_hi = _pct_delta(baseline["median"], current["ci_95"][1])
            if ci_lo != ci_hi:
                details.append(f"95% CI [{ci_lo:+.1f}%, {ci_hi:+.1f}%]")

        mw = _mannwhitney(baseline["_values"], current["_values"], self.higher_is_better)
        if mw:
            if mw["significant"]:
                details.append(f"Mann-Whitney U p={mw['pvalue']:.3f} — statistically significant")
            else:
                details.append(
                    f"Mann-Whitney U p={mw['pvalue']:.3f} — not statistically significant"
                )

        direction = _direction_from_delta(med_delta, self.noise_threshold, self.higher_is_better)
        if mw and not mw["significant"]:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=baseline,
            current=current,
            delta=med_delta,
            formatted=formatted,
            detail_lines=details,
            metadata={
                "avg_delta": avg_delta,
                "ci_95": current["ci_95"],
                "baseline_ci_95": baseline["ci_95"],
                "mannwhitney": mw,
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        return {
            "steps_delta_pct": result.delta,
            "avg_steps": result.current["avg"],
            "median_steps": result.current["median"],
        }


class DurationMetric(ComparisonMetric):
    name = "duration"
    description = "Trace duration — average, median, and P95 latency"
    noise_threshold = 5.0
    higher_is_better = False
    _fields = {
        "duration_delta_pct": "Average duration change (%)",
        "duration_median_delta_pct": "Median duration change (%)",
        "duration_p95_delta_pct": "P95 duration change (%)",
        "total_duration": "Total duration of current run (s)",
    }

    def summarize(self, col: TraceCollection) -> dict:
        durations = [t.duration for t in col.all_traces()]
        sorted_d = sorted(durations)
        p25, med, p75 = _iqr(durations)
        return {
            "avg": _mean(durations),
            "median": med,
            "p25": p25,
            "p75": p75,
            "p95": _percentile(sorted_d, 95),
            "total": sum(durations),
            "ci_95": _bootstrap_ci(durations, stat_fn=_median) if len(durations) >= 2 else None,
            "n": len(durations),
            "_values": durations,
        }

    def compare(self, baseline: dict, current: dict) -> Observation:
        obs_warnings: list[str] = []
        _maybe_scipy_hint()
        avg_delta = _pct_delta(baseline["avg"], current["avg"])
        med_delta = _pct_delta(baseline["median"], current["median"])
        p95_delta = _pct_delta(baseline["p95"], current["p95"])

        # Primary stat: median (robust to tail outliers)
        sign = "+" if med_delta >= 0 else ""
        formatted = (
            f"{baseline['median']:.1f}s → {current['median']:.1f}s median  {sign}{med_delta:.1f}%"
        )

        details = [
            f"{baseline['avg']:.1f}s → {current['avg']:.1f}s avg"
            f"  {'+' if avg_delta >= 0 else ''}{avg_delta:.1f}%",
            f"{baseline['p95']:.1f}s → {current['p95']:.1f}s P95"
            f"  {'+' if p95_delta >= 0 else ''}{p95_delta:.1f}%",
        ]
        if current["ci_95"] and baseline["ci_95"]:
            ci_lo = _pct_delta(baseline["median"], current["ci_95"][0])
            ci_hi = _pct_delta(baseline["median"], current["ci_95"][1])
            if ci_lo != ci_hi:
                details.append(f"95% CI [{ci_lo:+.1f}%, {ci_hi:+.1f}%]")

        mw = _mannwhitney(baseline["_values"], current["_values"], self.higher_is_better)
        if mw:
            if mw["significant"]:
                details.append(f"Mann-Whitney U p={mw['pvalue']:.3f} — statistically significant")
            else:
                details.append(
                    f"Mann-Whitney U p={mw['pvalue']:.3f} — not statistically significant"
                )

        small_n = min(baseline["n"], current["n"])
        if small_n < _MIN_P95_N:
            obs_warnings.append(
                f"P95 computed from {small_n} traces"
                f" — recommend ≥{_MIN_P95_N} for stable percentiles"
            )

        direction = _direction_from_delta(med_delta, self.noise_threshold, self.higher_is_better)
        if mw and not mw["significant"]:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=baseline,
            current=current,
            delta=med_delta,
            formatted=formatted,
            detail_lines=details,
            metadata={
                "avg_delta_pct": avg_delta,
                "median_delta_pct": med_delta,
                "p95_delta_pct": p95_delta,
                "ci_95": current["ci_95"],
                "baseline_ci_95": baseline["ci_95"],
                "mannwhitney": mw,
            },
            warnings=obs_warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        return {
            "duration_delta_pct": result.delta,
            "duration_median_delta_pct": result.metadata["median_delta_pct"],
            "duration_p95_delta_pct": result.metadata["p95_delta_pct"],
            "total_duration": result.current["total"],
        }


class ToolErrorRateMetric(ComparisonMetric):
    name = "tool_error_rate"
    description = "Fraction of tool invocations that returned an error"
    noise_threshold = 0.5
    higher_is_better = False
    _fields = {"tool_error_rate_delta": "Error rate change (percentage points)"}

    def summarize(self, col: TraceCollection) -> dict:
        spans = [s for t in col.all_traces() for s in t.spans]
        errors = sum(1 for s in spans if span_is_error(s))
        return {
            "rate": errors / len(spans) if spans else 0.0,
            "errors": errors,
            "total": len(spans),
        }

    def compare(self, baseline: dict, current: dict) -> Observation:
        warnings: list[str] = []
        delta_pp = (current["rate"] - baseline["rate"]) * 100
        sign = "+" if delta_pp >= 0 else ""
        formatted = f"{baseline['rate']:.1%} → {current['rate']:.1%}  {sign}{delta_pp:.1f} pp"

        small_n = min(baseline["total"], current["total"])
        if small_n < _MIN_N:
            warnings.append(f"Only {small_n} tool invocations — error rate estimate may be noisy")

        direction = _direction_from_delta(delta_pp, self.noise_threshold, self.higher_is_better)
        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=baseline,
            current=current,
            delta=round(delta_pp, 2),
            formatted=formatted,
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        return {"tool_error_rate_delta": result.delta}


class PathDistributionMetric(ComparisonMetric):
    name = "path_distribution"
    description = "Jaccard similarity of top execution paths"
    noise_threshold = 0.0
    higher_is_better = True
    _fields = {"path_jaccard": "Jaccard similarity of top execution paths"}

    def summarize(self, col: TraceCollection) -> dict:
        traces = col.all_traces()
        path_counts = Counter(_trace_path(t) for t in traces)
        return {
            "top_paths": {p for p, _ in path_counts.most_common(20)},
            "n": len(traces),
            "unique_paths": len(path_counts),
        }

    def compare(self, baseline: dict, current: dict) -> Observation:
        warnings: list[str] = []
        b, c = baseline["top_paths"], current["top_paths"]
        jaccard = len(b & c) / len(b | c) if (b | c) else 1.0
        new_paths = list(c - b)[:10]
        dropped_paths = list(b - c)[:10]
        formatted = f"Jaccard {jaccard:.2f}  (+{len(new_paths)} new, −{len(dropped_paths)} dropped)"

        small_n = min(baseline["n"], current["n"])
        if small_n < _MIN_N:
            warnings.append(
                f"Path Jaccard computed from {small_n} traces — "
                f"small samples inflate apparent path divergence"
            )

        direction = Direction.SAME if jaccard >= 0.8 else Direction.INCONCLUSIVE

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=len(b),
            current=len(c),
            delta=round(jaccard, 3),
            formatted=formatted,
            metadata={
                "jaccard": round(jaccard, 3),
                "new_paths": new_paths,
                "dropped_paths": dropped_paths,
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        return {"path_jaccard": result.metadata["jaccard"]}


class TokenUsageMetric(ComparisonMetric):
    name = "token_usage"
    description = "Token consumption — input, output, and total"
    noise_threshold = 3.0
    higher_is_better = False
    _fields = {
        "token_delta_pct": "Median token usage change (%)",
        "total_tokens": "Total tokens in current run",
        "avg_tokens": "Average tokens per trace",
    }

    def summarize(self, col: TraceCollection) -> dict:
        traces = col.all_traces()
        input_tok = [sum(span_input_tokens(s) for s in t.spans) for t in traces]
        output_tok = [sum(span_output_tokens(s) for s in t.spans) for t in traces]
        totals = [i + o for i, o in zip(input_tok, output_tok)]
        return {
            "avg_input": _mean(input_tok),
            "avg_output": _mean(output_tok),
            "avg_total": _mean(totals),
            "median_total": _median(totals),
            "total": sum(totals),
            "ci_95": _bootstrap_ci(totals, stat_fn=_median) if len(totals) >= 2 else None,
            "all_zero": all(t == 0 for t in totals),
            "n": len(traces),
            "_values": totals,
        }

    def compare(self, baseline: dict, current: dict) -> Observation:
        obs_warnings: list[str] = []
        if baseline["all_zero"] and current["all_zero"]:
            obs_warnings.append(
                "All token counts are 0 — token data may not be populated in your traces"
            )
            return Observation(
                name=self.name,
                description=self.description,
                direction=Direction.NA,
                baseline=baseline,
                current=current,
                formatted="n/a — no token data",
                warnings=obs_warnings,
            )

        _maybe_scipy_hint()

        # Primary stat: median total tokens
        med_delta = _pct_delta(baseline["median_total"], current["median_total"])
        avg_delta = _pct_delta(baseline["avg_total"], current["avg_total"])
        sign = "+" if med_delta >= 0 else ""
        formatted = (
            f"{baseline['median_total']:,.0f} →"
            f" {current['median_total']:,.0f}"
            f" tokens/trace (median)  {sign}{med_delta:.1f}%"
        )
        details = [
            f"{baseline['avg_total']:,.0f} → {current['avg_total']:,.0f}"
            f" avg  {'+' if avg_delta >= 0 else ''}{avg_delta:.1f}%",
            f"in: {baseline['avg_input']:,.0f} →"
            f" {current['avg_input']:,.0f} avg"
            f"  |  out: {baseline['avg_output']:,.0f} →"
            f" {current['avg_output']:,.0f} avg",
        ]
        ci = current["ci_95"]
        if ci and baseline["ci_95"]:
            ci_lo = _pct_delta(baseline["median_total"], ci[0])
            ci_hi = _pct_delta(baseline["median_total"], ci[1])
            if ci_lo != ci_hi:
                details.append(f"95% CI [{ci_lo:+.1f}%, {ci_hi:+.1f}%]")

        mw = _mannwhitney(baseline["_values"], current["_values"], self.higher_is_better)
        if mw:
            if mw["significant"]:
                details.append(f"Mann-Whitney U p={mw['pvalue']:.3f} — statistically significant")
            else:
                details.append(
                    f"Mann-Whitney U p={mw['pvalue']:.3f} — not statistically significant"
                )

        direction = _direction_from_delta(med_delta, self.noise_threshold, self.higher_is_better)
        if mw and not mw["significant"]:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=baseline,
            current=current,
            delta=med_delta,
            formatted=formatted,
            detail_lines=details,
            metadata={
                "avg_delta": avg_delta,
                "ci_95": ci,
                "baseline_ci_95": baseline["ci_95"],
                "mannwhitney": mw,
            },
            warnings=obs_warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["token_delta_pct"] = result.delta
        fields["total_tokens"] = result.current["total"]
        fields["avg_tokens"] = result.current["avg_total"]
        return fields


class TokenEfficiencyMetric(ComparisonMetric):
    name = "token_efficiency"
    description = "Tokens per successful task"
    noise_threshold = 5.0
    higher_is_better = False
    _fields = {"token_efficiency_delta_pct": "Tokens-per-success change (%)"}

    def summarize(self, col: TraceCollection) -> dict:
        successes = [t for t in col.all_traces() if t.outcome == OUTCOME_SUCCESS]
        if not successes:
            return {"tokens_per_success": None, "n_successes": 0, "total_tokens": 0}
        total_tokens = sum(t.total_tokens for t in successes)
        return {
            "tokens_per_success": total_tokens / len(successes),
            "n_successes": len(successes),
            "total_tokens": total_tokens,
        }

    def compare(self, baseline: dict, current: dict) -> Observation:
        warnings: list[str] = []
        b_tps, c_tps = baseline["tokens_per_success"], current["tokens_per_success"]
        if b_tps is None or c_tps is None:
            side = (
                "both"
                if b_tps is None and c_tps is None
                else ("baseline" if b_tps is None else "current")
            )
            warnings.append(f"No successes in {side} — token efficiency unavailable")
            return Observation(
                name=self.name,
                description=self.description,
                direction=Direction.NA,
                baseline=baseline,
                current=current,
                formatted=f"n/a — no successes in {side}",
                warnings=warnings,
            )

        if baseline["total_tokens"] == 0 and current["total_tokens"] == 0:
            warnings.append(
                "All token counts are 0 — token data may not be populated in your traces"
            )
            return Observation(
                name=self.name,
                description=self.description,
                direction=Direction.NA,
                baseline=baseline,
                current=current,
                formatted="n/a — no token data",
                warnings=warnings,
            )

        delta = _pct_delta(b_tps, c_tps)
        sign = "+" if delta >= 0 else ""
        formatted = (
            f"{b_tps:,.0f} → {c_tps:,.0f} tokens/success  {sign}{delta:.1f}%"
            f"  ({baseline['n_successes']}→{current['n_successes']} successes)"
        )
        direction = _direction_from_delta(delta, self.noise_threshold, self.higher_is_better)
        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=baseline,
            current=current,
            delta=delta,
            formatted=formatted,
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {"token_efficiency_delta_pct": result.delta}


class CostQualityMetric(ComparisonMetric):
    name = "cost_quality"
    description = "Cost per successful task (total cost / successes)"
    noise_threshold = 5.0
    higher_is_better = False
    _fields = {
        "cost_quality_delta_pct": "Cost-per-success change (%)",
        "cost_per_success": "Current cost per successful task",
    }

    def summarize(self, col: TraceCollection) -> dict:
        traces = col.all_traces()
        total_cost = sum(t.total_cost for t in traces)
        successes = [t for t in traces if t.outcome == OUTCOME_SUCCESS]
        if not successes:
            return {
                "cost_per_success": None,
                "n_successes": 0,
                "total_cost": total_cost,
                "n_total": len(traces),
            }
        return {
            "cost_per_success": total_cost / len(successes),
            "n_successes": len(successes),
            "total_cost": total_cost,
            "n_total": len(traces),
        }

    def compare(self, baseline: dict, current: dict) -> Observation:
        warnings: list[str] = []
        b_cps, c_cps = baseline["cost_per_success"], current["cost_per_success"]
        if b_cps is None or c_cps is None:
            side = (
                "both"
                if b_cps is None and c_cps is None
                else ("baseline" if b_cps is None else "current")
            )
            warnings.append(f"No successes in {side} — cost/quality unavailable")
            return Observation(
                name=self.name,
                description=self.description,
                direction=Direction.NA,
                baseline=baseline,
                current=current,
                formatted=f"n/a — no successes in {side}",
                warnings=warnings,
            )

        if baseline["total_cost"] == 0 and current["total_cost"] == 0:
            warnings.append("All costs are $0 — cost data may not be populated in your traces")
            return Observation(
                name=self.name,
                description=self.description,
                direction=Direction.NA,
                baseline=baseline,
                current=current,
                formatted="n/a — no cost data",
                warnings=warnings,
            )

        delta = _pct_delta(b_cps, c_cps)
        sign = "+" if delta >= 0 else ""
        formatted = (
            f"${b_cps:.4f} → ${c_cps:.4f} per success  {sign}{delta:.1f}%"
            f"  ({baseline['n_successes']}/{baseline['n_total']} → "
            f"{current['n_successes']}/{current['n_total']} succeeded)"
        )
        direction = _direction_from_delta(delta, self.noise_threshold, self.higher_is_better)
        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline=baseline,
            current=current,
            delta=delta,
            formatted=formatted,
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {
            "cost_quality_delta_pct": result.delta,
            "cost_per_success": result.current["cost_per_success"] or 0.0,
        }


# ── Default metric set ─────────────────────────────────────────────────────────

DEFAULT_METRICS: list[ComparisonMetric] = [
    SuccessRateMetric(),
    PerTaskMetric(),
    CostMetric(),
    StepsMetric(),
    DurationMetric(),
    ToolErrorRateMetric(),
    PathDistributionMetric(),
    TokenUsageMetric(),
    TokenEfficiencyMetric(),
    CostQualityMetric(),
]


# ── Math helpers ───────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * pct / 100), len(sorted_values) - 1)
    return sorted_values[idx]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _iqr(values: list[float]) -> tuple[float, float, float]:
    """Return (p25, median, p75)."""
    if not values:
        return (0.0, 0.0, 0.0)
    s = sorted(values)
    return (_percentile(s, 25), _median(values), _percentile(s, 75))


def _bootstrap_ci(
    values: list[float],
    stat_fn=None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Returns (lo, hi) bounds for the statistic at ``1 - alpha`` confidence.
    Deterministic (seeded) for reproducibility.
    """
    if stat_fn is None:
        stat_fn = _mean
    if len(values) < 2:
        v = stat_fn(values)
        return (v, v)
    rng = random.Random(42)
    stats = sorted(stat_fn(rng.choices(values, k=len(values))) for _ in range(n_resamples))
    lo = max(0, int(n_resamples * alpha / 2))
    hi = min(len(stats) - 1, int(n_resamples * (1 - alpha / 2)))
    return (round(stats[lo], 6), round(stats[hi], 6))


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


def _extract_task_id_from_trace(trace, task_id_field: str | None = None) -> str:
    """Extract a stable task ID from a trace.

    If ``task_id_field`` is set (from config), looks up that metadata field.
    Otherwise falls back to parsing the trace_id string.
    """
    meta = trace.metadata or {}

    # Explicit field from config — the user told us where to look.
    if task_id_field:
        val = meta.get(task_id_field)
        if val:
            return str(val)

    # Fall back to trace_id parsing (strips __<model>__<index> suffixes).
    return _extract_task_id(trace.trace_id)


def _extract_task_id(trace_id: str) -> str:
    """Extract a stable task ID from a trace ID string.

    Strips ``__<model>__<index>`` suffixes so that traces from different runs
    of the same task are matched together for per-task comparison.
    """
    parts = trace_id.split("__")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "__".join(parts[:-2])
    return trace_id
