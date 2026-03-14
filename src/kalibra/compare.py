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
    *,
    _parsed_require: list[ParsedExpr] | None = None,
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

    # Parse and validate threshold expressions — skip if caller pre-validated.
    if _parsed_require is not None:
        parsed_exprs = _parsed_require
    else:
        all_require_raw = list(config.require) + list(require or [])
        known_fields: set[str] = set()
        for m in active:
            known_fields.update(m.threshold_field_names())
        parsed_exprs = validate_require_exprs(all_require_raw, known_fields)

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
        # Apply per-metric config overrides.
        noise = config.noise_thresholds.get(m.name)
        if noise is not None:
            m.noise_threshold = noise
        if hasattr(m, "task_id_field") and config.task_id:
            m.task_id_field = config.task_id

        b_summary = m.summarize(baseline)
        c_summary = m.summarize(current)
        obs = m.compare(b_summary, c_summary)
        observations[obs.name] = obs
        threshold_values.update(m.threshold_fields(obs))

    gates = _eval_thresholds(threshold_values, parsed_exprs)

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


# ── Threshold parsing and evaluation ──────────────────────────────────────────

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "=":  lambda a, b: a == b,
}

_OP_TOKENS = (">=", "<=", ">", "<", "=")


@dataclass
class ParsedExpr:
    """A parsed threshold expression: field op value."""
    raw: str
    field: str
    op: str
    value: float


class ThresholdError(Exception):
    """Raised when a --require expression is invalid."""


def parse_require_expr(expr: str) -> ParsedExpr:
    """Parse a single threshold expression like ``'success_rate_delta >= -2'``.

    Raises ThresholdError with an actionable message on bad syntax.
    """
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

            return ParsedExpr(raw=expr, field=field_str, op=op, value=value)

    raise ThresholdError(
        f"No operator found in: {expr!r}\n"
        f"  Expected format: field_name >= number\n"
        f"  Operators: >=  <=  >  <  ="
    )


def validate_require_exprs(
    exprs: list[str],
    known_fields: set[str],
) -> list[ParsedExpr]:
    """Parse and validate all threshold expressions against known fields.

    Raises ThresholdError on syntax errors or unrecognized field names.
    Returns parsed expressions on success.
    """
    from difflib import get_close_matches

    parsed = []
    errors = []

    for expr in exprs:
        expr = expr.strip()
        if not expr:
            continue
        try:
            p = parse_require_expr(expr)
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


def _eval_thresholds(
    values: dict[str, float],
    parsed_exprs: list[ParsedExpr],
) -> list[Gate]:
    """Evaluate pre-parsed threshold expressions against computed metric values."""
    gates = []
    for p in parsed_exprs:
        if p.field not in values:
            gates.append(Gate(
                expr=p.raw,
                passed=True,
                actual=float("nan"),
                warning=f"Metric produced no data for {p.field!r} — gate skipped",
            ))
            continue
        actual = values[p.field]
        gates.append(Gate(
            expr=p.raw,
            passed=_OPS[p.op](actual, p.value),
            actual=round(actual, 4),
            metric_name=p.field.split("_delta")[0] if "_delta" in p.field else None,
        ))
    return gates
