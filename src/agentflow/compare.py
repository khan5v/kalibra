"""Comparison engine — orchestrates metrics over two TraceCollections."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentflow.collection import TraceCollection
from agentflow.config import CompareConfig, resolve_metrics
from agentflow.metrics import ComparisonMetric, DEFAULT_METRICS, MetricResult


@dataclass
class CompareResult:
    baseline_source: str
    current_source: str
    baseline_count: int
    current_count: int
    metrics: dict[str, MetricResult]
    threshold_results: list[dict] = field(default_factory=list)
    thresholds_passed: bool = True
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> MetricResult:
        return self.metrics[name]


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
                       ``agentflow.yml`` in cwd if omitted.

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
                   ``CompareConfig`` to avoid auto-loading ``agentflow.yml``.

    Returns:
        A ``CompareResult`` containing all metric results and threshold outcomes.

    Example::

        from agentflow import compare_collections, CompareConfig, TraceCollection

        baseline = TraceCollection.from_traces(run_agent(baseline_prompt), source="v1")
        current  = TraceCollection.from_traces(run_agent(new_prompt),      source="v2")

        result = compare_collections(
            baseline, current,
            config=CompareConfig(
                metrics=["success_rate", "cost"],
                require=["success_rate_delta >= -2"],
            ),
        )
        for name, m in result.metrics.items():
            print(f"{name}: {m.formatted}")
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

    metric_results: dict[str, MetricResult] = {}
    threshold_values: dict[str, float] = {}

    for m in active:
        b_summary = m.summarize(baseline)
        c_summary = m.summarize(current)
        result = m.compare(b_summary, c_summary)
        metric_results[result.name] = result
        threshold_values.update(m.threshold_fields(result))

    all_require = list(config.require) + list(require or [])
    threshold_results, thresholds_passed = _eval_thresholds(threshold_values, all_require)

    return CompareResult(
        baseline_source=baseline.source,
        current_source=current.source,
        baseline_count=len(baseline),
        current_count=len(current),
        metrics=metric_results,
        threshold_results=threshold_results,
        thresholds_passed=thresholds_passed,
        warnings=result_warnings,
    )


# ── Threshold evaluation ───────────────────────────────────────────────────────

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "=":  lambda a, b: a == b,
}


def _eval_thresholds(values: dict[str, float], exprs: list[str]) -> tuple[list[dict], bool]:
    all_passed = True
    results = []
    for expr in exprs:
        expr = expr.strip()
        if not expr:
            continue
        entry = _eval_expr(expr, values)
        if not entry["passed"]:
            all_passed = False
        results.append(entry)
    return results, all_passed


def _eval_expr(expr: str, values: dict[str, float]) -> dict:
    for op in (">=", "<=", ">", "<", "="):
        if op in expr:
            field, val_str = (s.strip() for s in expr.split(op, 1))
            try:
                threshold = float(val_str)
            except ValueError:
                return {"expr": expr, "passed": False, "actual": None, "threshold": val_str}
            if field not in values:
                available = sorted(values)
                raise ValueError(f"Unknown threshold field: {field!r}. Available: {available}")
            actual = values[field]
            return {
                "expr": expr,
                "passed": _OPS[op](actual, threshold),
                "actual": round(actual, 4),
                "threshold": threshold,
            }
    raise ValueError(f"Cannot parse threshold expression: {expr!r}")
