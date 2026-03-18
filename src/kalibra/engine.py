"""Engine — orchestrates metrics over two trace populations.

This is the v2 replacement for compare.py. It takes lists of Trace objects
(not TraceCollections), runs selected metrics, evaluates threshold gates,
and returns a flat CompareResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from kalibra.metrics import ComparisonMetric, Direction, Observation
from kalibra.model import Trace

# ── All available metrics ────────────────────────────────────────────────────

_METRIC_CLASSES: dict[str, type[ComparisonMetric]] = {}


def _register_metrics() -> dict[str, type[ComparisonMetric]]:
    """Lazily import and register all built-in metrics."""
    if _METRIC_CLASSES:
        return _METRIC_CLASSES

    from kalibra.metrics.cost import CostMetric
    from kalibra.metrics.cost_quality import CostQualityMetric
    from kalibra.metrics.duration import DurationMetric
    from kalibra.metrics.error_rate import ErrorRateMetric
    from kalibra.metrics.span_breakdown import SpanBreakdownMetric
    from kalibra.metrics.steps import StepsMetric
    from kalibra.metrics.success_rate import SuccessRateMetric
    from kalibra.metrics.token_efficiency import TokenEfficiencyMetric
    from kalibra.metrics.token_usage import TokenUsageMetric
    from kalibra.metrics.trace_breakdown import TraceBreakdownMetric

    for cls in [
        SuccessRateMetric, CostMetric, DurationMetric, StepsMetric,
        ErrorRateMetric,
        TokenUsageMetric, TokenEfficiencyMetric, CostQualityMetric,
        TraceBreakdownMetric, SpanBreakdownMetric,
    ]:
        _METRIC_CLASSES[cls.name] = cls
    return _METRIC_CLASSES


DEFAULT_METRIC_NAMES = [
    "success_rate", "cost", "duration", "steps",
    "error_rate",
    "token_usage", "token_efficiency", "cost_quality",
    "trace_breakdown", "span_breakdown",
]


def resolve_metrics(names: list[str] | None = None) -> list[ComparisonMetric]:
    """Instantiate metrics by name. None means all defaults, [] means none."""
    registry = _register_metrics()
    selected = DEFAULT_METRIC_NAMES if names is None else names
    result = []
    for name in selected:
        cls = registry.get(name)
        if cls is None:
            known = sorted(registry.keys())
            raise ValueError(
                f"Unknown metric {name!r}. Available: {', '.join(known)}"
            )
        result.append(cls())
    return result


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """A single threshold gate evaluation."""
    expr: str
    passed: bool
    actual: float
    warning: str | None = None


@dataclass
class CompareResult:
    """Full comparison result — flat structure for renderers."""
    direction: Direction
    observations: dict[str, Observation]
    baseline_source: str
    current_source: str
    baseline_count: int
    current_count: int
    warnings: list[str] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    passed: bool = True


# ── Main compare function ────────────────────────────────────────────────────

def compare(
    baseline: list[Trace],
    current: list[Trace],
    *,
    metrics: list[str] | None = None,
    require: list[str] | None = None,
    baseline_source: str = "baseline",
    current_source: str = "current",
    noise_thresholds: dict[str, float] | None = None,
    metric_config: dict[str, dict] | None = None,
) -> CompareResult:
    """Compare two trace populations and return structured results.

    Args:
        baseline: Baseline traces.
        current: Current traces.
        metrics: Metric names to run. None means all defaults.
        require: Threshold expressions (e.g. "success_rate_delta >= -2").
        baseline_source: Label for baseline source.
        current_source: Label for current source.
        noise_thresholds: Per-metric noise threshold overrides.
        metric_config: Per-metric config dicts. Keys are metric names,
            values are dicts of attribute overrides set on the metric
            instance before compare() is called. Example:
            {"trace_breakdown": {"task_id_field": "metadata.task_name"}}
    """
    active = resolve_metrics(metrics)
    noise_thresholds = noise_thresholds or {}
    metric_config = metric_config or {}

    # Parse and validate threshold expressions early.
    require = require or []
    known_fields: set[str] = set()
    for m in active:
        known_fields.update(m.threshold_field_names())
    parsed_exprs = _validate_require(require, known_fields)

    # Dataset-level warnings.
    warnings: list[str] = []
    n_b, n_c = len(baseline), len(current)
    if n_b < 30:
        warnings.append(
            f"Baseline has only {n_b} traces — recommend ≥30 for reliable results"
        )
    if n_c < 30:
        warnings.append(
            f"Current has only {n_c} traces — recommend ≥30 for reliable results"
        )
    if n_b > 0 and n_c > 0 and max(n_b, n_c) / min(n_b, n_c) > 10:
        warnings.append(
            f"Large size disparity ({n_b:,} vs {n_c:,} traces) — "
            "confidence intervals are asymmetric; treat deltas with caution"
        )

    # Run metrics.
    observations: dict[str, Observation] = {}
    threshold_values: dict[str, float] = {}

    for m in active:
        noise = noise_thresholds.get(m.name)
        if noise is not None:
            m.noise_threshold = noise
        for attr, val in metric_config.get(m.name, {}).items():
            setattr(m, attr, val)

        obs = m.compare(baseline, current)
        observations[obs.name] = obs
        threshold_values.update(m.threshold_fields(obs))

    # Evaluate gates.
    gates = _eval_gates(threshold_values, parsed_exprs)

    return CompareResult(
        direction=_rollup_direction(observations),
        observations=observations,
        baseline_source=baseline_source,
        current_source=current_source,
        baseline_count=n_b,
        current_count=n_c,
        warnings=warnings,
        gates=gates,
        passed=all(g.passed for g in gates),
    )


# ── Direction roll-up ────────────────────────────────────────────────────────

# Breakdown metrics are informational — they detect per-item regressions but
# should not dominate the overall direction. A single regressed span name
# shouldn't turn the whole result red when all aggregate metrics say UNCHANGED.
_ROLLUP_EXCLUDE = {"trace_breakdown", "span_breakdown"}


def _rollup_direction(observations: dict[str, Observation]) -> Direction:
    statuses = {
        obs.direction for obs in observations.values()
        if obs.direction != Direction.NA and obs.name not in _ROLLUP_EXCLUDE
    }
    if not statuses:
        return Direction.NA
    if Direction.INCONCLUSIVE in statuses:
        return Direction.INCONCLUSIVE
    if Direction.DEGRADATION in statuses and Direction.UPGRADE in statuses:
        return Direction.INCONCLUSIVE
    if Direction.DEGRADATION in statuses:
        return Direction.DEGRADATION
    if Direction.UPGRADE in statuses:
        return Direction.UPGRADE
    return Direction.SAME


# ── Threshold parsing and evaluation ─────────────────────────────────────────

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "=":  lambda a, b: a == b,
}

_OP_TOKENS = (">=", "<=", ">", "<", "=")


@dataclass
class _ParsedExpr:
    raw: str
    field: str
    op: str
    value: float


class ThresholdError(Exception):
    """Raised when a --require expression is invalid."""


def _parse_expr(expr: str) -> _ParsedExpr:
    expr = expr.strip()
    if not expr:
        raise ThresholdError("Empty threshold expression.")

    for op in _OP_TOKENS:
        if op in expr:
            parts = expr.split(op, 1)
            field_str = parts[0].strip()
            val_str = parts[1].strip()

            if not field_str:
                raise ThresholdError(
                    f"Missing field name in: {expr!r}\n"
                    f"  Expected: field_name {op} number\n"
                    f"  Run 'kalibra compare --metrics' to see available fields."
                )
            if not val_str:
                raise ThresholdError(
                    f"Missing threshold value in: {expr!r}\n"
                    f"  Expected: {field_str} {op} number"
                )
            try:
                value = float(val_str)
            except ValueError:
                raise ThresholdError(
                    f"Invalid threshold value {val_str!r} in: {expr!r}\n"
                    f"  The right-hand side must be a number, e.g.: {field_str} {op} 5"
                ) from None

            return _ParsedExpr(raw=expr, field=field_str, op=op, value=value)

    raise ThresholdError(
        f"No operator found in: {expr!r}\n"
        f"  Expected format: field_name >= number\n"
        f"  Operators: >=  <=  >  <  ="
    )


def _validate_require(
    exprs: list[str],
    known_fields: set[str],
) -> list[_ParsedExpr]:
    parsed = []
    errors = []

    for expr in exprs:
        expr = expr.strip()
        if not expr:
            continue
        try:
            p = _parse_expr(expr)
        except ThresholdError as exc:
            errors.append(str(exc))
            continue

        if p.field not in known_fields:
            msg = f"Unknown field {p.field!r} in: {p.raw!r}"
            matches = get_close_matches(p.field, sorted(known_fields), n=3, cutoff=0.5)
            if matches:
                suggestions = ", ".join(matches)
                msg += f"\n  Did you mean: {suggestions}"
            errors.append(msg)
            continue

        parsed.append(p)

    if errors:
        raise ThresholdError("\n\n".join(errors))

    return parsed


def _eval_gates(
    values: dict[str, float],
    parsed_exprs: list[_ParsedExpr],
) -> list[GateResult]:
    gates = []
    for p in parsed_exprs:
        if p.field not in values:
            gates.append(GateResult(
                expr=p.raw,
                passed=True,
                actual=float("nan"),
                warning=f"Metric produced no data for {p.field!r} — gate skipped",
            ))
            continue
        actual = values[p.field]
        gates.append(GateResult(
            expr=p.raw,
            passed=_OPS[p.op](actual, p.value),
            actual=round(actual, 4),
        ))
    return gates
