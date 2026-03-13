"""Comparison engine — orchestrates metrics over two TraceCollections."""

from __future__ import annotations

from dataclasses import dataclass, field

from kalibra.collection import TraceCollection
from kalibra.config import CompareConfig, resolve_metrics
from kalibra.metrics import ComparisonMetric, DEFAULT_METRICS, Direction, Observation


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Gate:
    """A single threshold gate evaluation."""
    expr: str
    passed: bool
    actual: float
    metric_name: str | None = None
    warning: str | None = None


@dataclass
class ComparisonResult:
    """What changed? Rolled-up direction + per-metric observations."""
    direction: Direction
    observations: dict[str, Observation]


@dataclass
class ValidationResult:
    """Did it pass? Gate evaluations."""
    passed: bool
    gates: list[Gate]


@dataclass
class CompareResult:
    # Metadata
    baseline_source: str
    current_source: str
    baseline_count: int
    current_count: int
    warnings: list[str]
    # The two independent trees
    comparison: ComparisonResult
    validation: ValidationResult

    # Backwards-compat shims
    @property
    def thresholds_passed(self) -> bool:
        return self.validation.passed

    @property
    def metrics(self) -> dict[str, Observation]:
        return self.comparison.observations

    def __getitem__(self, name: str) -> Observation:
        return self.comparison.observations[name]

    @property
    def threshold_results(self) -> list[dict]:
        return [
            {"expr": g.expr, "passed": g.passed, "actual": g.actual}
            for g in self.validation.gates
        ]


def compare(
    baseline_path: str,
    current_path: str,
    metrics: list[ComparisonMetric] | None = None,
    require: list[str] | None = None,
    config: CompareConfig | None = None,
) -> CompareResult:
    """Compare two trace datasets loaded from disk.

    Args:
        baseline_path: Path to baseline traces (dir or file, auto-detected).
        current_path:  Path to current traces.
        metrics:       Explicit metric list. Overrides config and DEFAULT_METRICS.
        require:       Threshold expressions appended to any defined in ``config``.
        config:        Metric selection, thresholds, and plugins. Auto-loaded from
                       ``kalibra.yml`` in cwd if omitted.

    Returns:
        A ``CompareResult`` containing all metric results and threshold outcomes.
    """
    if config is None:
        config = CompareConfig.load()
    baseline = TraceCollection.from_path(baseline_path)
    current  = TraceCollection.from_path(current_path)
    return compare_collections(baseline, current, metrics=metrics, require=require, config=config)


def compare_collections(
    baseline: TraceCollection,
    current: TraceCollection,
    metrics: list[ComparisonMetric] | None = None,
    require: list[str] | None = None,
    config: CompareConfig | None = None,
) -> CompareResult:
    """Compare two in-memory TraceCollections — the programmatic entry point.

    Use this when you already have traces in memory (e.g. from a connector,
    a test harness, or live evaluation) and don't want to write files to disk.

    Args:
        baseline:  Baseline trace collection.
        current:   Current trace collection.
        metrics:   Explicit metric list. Overrides config and DEFAULT_METRICS.
        require:   Threshold expressions appended to any defined in ``config``.
        config:    Metric selection, thresholds, and plugins. Pass an explicit
                   ``CompareConfig`` to avoid auto-loading ``kalibra.yml``.

    Returns:
        A ``CompareResult`` containing all metric results and threshold outcomes.

    Example::

        from kalibra import compare_collections, CompareConfig, TraceCollection

        baseline = TraceCollection.from_traces(run_agent(baseline_prompt), source="v1")
        current  = TraceCollection.from_traces(run_agent(new_prompt),      source="v2")

        result = compare_collections(
            baseline, current,
            config=CompareConfig(
                metrics=["success_rate", "cost"],
                require=["success_rate_delta >= -2"],
            ),
        )
        for name, obs in result.comparison.observations.items():
            print(f"{name}: {obs.direction.value}  {obs.formatted}")
    """
    if config is None:
        config = CompareConfig.load()

    if metrics is not None:
        active: list[ComparisonMetric] = metrics
    else:
        active = resolve_metrics(config, DEFAULT_METRICS)

    # Dataset-level warnings — before running any metric.
    result_warnings: list[str] = []
    n_b, n_c = len(baseline), len(current)
    if n_b < 30:
        result_warnings.append(
            f"Baseline has only {n_b} traces — recommend ≥30 for reliable results"
        )
    if n_c < 30:
        result_warnings.append(
            f"Current has only {n_c} traces — recommend ≥30 for reliable results"
        )
    if n_b > 0 and n_c > 0 and max(n_b, n_c) / min(n_b, n_c) > 10:
        result_warnings.append(
            f"Large size disparity ({n_b:,} vs {n_c:,} traces) — "
            "confidence intervals are asymmetric; treat deltas with caution"
        )

    observations: dict[str, Observation] = {}
    threshold_values: dict[str, float] = {}

    for m in active:
        # Apply per-metric noise threshold override from config.
        noise = config.noise_thresholds.get(m.name)
        if noise is not None:
            m.noise_threshold = noise  # instance-level override

        b_summary = m.summarize(baseline)
        c_summary = m.summarize(current)
        obs = m.compare(b_summary, c_summary)
        observations[obs.name] = obs
        threshold_values.update(m.threshold_fields(obs))

    all_require = list(config.require) + list(require or [])
    gates = _eval_thresholds(threshold_values, all_require)

    return CompareResult(
        baseline_source=baseline.source,
        current_source=current.source,
        baseline_count=len(baseline),
        current_count=len(current),
        warnings=result_warnings,
        comparison=ComparisonResult(
            direction=_rollup_direction(observations),
            observations=observations,
        ),
        validation=ValidationResult(
            passed=all(g.passed for g in gates),
            gates=gates,
        ),
    )


# ── Direction roll-up ─────────────────────────────────────────────────────────

def _rollup_direction(observations: dict[str, Observation]) -> Direction:
    statuses = {obs.direction for obs in observations.values() if obs.direction != Direction.NA}
    if not statuses:
        return Direction.NA
    if Direction.DEGRADATION in statuses and Direction.UPGRADE in statuses:
        return Direction.INCONCLUSIVE
    if Direction.DEGRADATION in statuses:
        return Direction.DEGRADATION
    if Direction.UPGRADE in statuses:
        return Direction.UPGRADE
    return Direction.SAME


# ── Threshold evaluation ───────────────────────────────────────────────────────

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "=":  lambda a, b: a == b,
}


def _eval_thresholds(values: dict[str, float], exprs: list[str]) -> list[Gate]:
    gates = []
    for expr in exprs:
        expr = expr.strip()
        if not expr:
            continue
        gates.append(_eval_expr(expr, values))
    return gates


def _eval_expr(expr: str, values: dict[str, float]) -> Gate:
    for op in (">=", "<=", ">", "<", "="):
        if op in expr:
            field, val_str = (s.strip() for s in expr.split(op, 1))
            try:
                threshold = float(val_str)
            except ValueError:
                return Gate(expr=expr, passed=False, actual=float("nan"))
            if field not in values:
                return Gate(
                    expr=expr,
                    passed=True,
                    actual=float("nan"),
                    warning=f"Field {field!r} not available — metric produced no data, gate skipped",
                )
            actual = values[field]
            return Gate(
                expr=expr,
                passed=_OPS[op](actual, threshold),
                actual=round(actual, 4),
                metric_name=field.split("_delta")[0] if "_delta" in field else None,
            )
    raise ValueError(f"Cannot parse threshold expression: {expr!r}")
