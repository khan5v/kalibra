"""Kalibra — agent evaluation and regression detection.

Programmatic API::

    import kalibra

    # From files (same paths the CLI accepts)
    result = kalibra.compare("baseline.jsonl", "current.jsonl")

    # From in-memory collections (no files needed)
    baseline = kalibra.TraceCollection.from_traces(my_traces, source="v1")
    current  = kalibra.TraceCollection.from_traces(new_traces, source="v2")
    result   = kalibra.compare_collections(baseline, current)

    # Inspect results generically — works for any metric configuration
    for name, metric in result.metrics.items():
        print(f"{name}: {metric.formatted}")
        if metric.delta is not None:
            direction = "▲" if metric.delta > 0 else "▼"
            print(f"  {direction} {metric.delta:+.2f}")

    # Check threshold gates
    print("passed:", result.thresholds_passed)
    for gate in result.threshold_results:
        status = "✓" if gate["passed"] else "✗"
        print(f"  {status} {gate['expr']}  (actual: {gate['actual']})")
"""

__version__ = "0.1.0"

from opentelemetry.sdk.trace import ReadableSpan

from kalibra.collection import TraceCollection
from kalibra.compare import (
    CompareResult,
    ComparisonResult,
    Gate,
    ThresholdError,
    ValidationResult,
    compare,
    compare_collections,
)
from kalibra.config import CompareConfig
from kalibra.converters.base import Trace, make_span
from kalibra.metrics import (
    ComparisonMetric,
    CostQualityMetric,
    Direction,
    MetricResult,
    Observation,
    TokenEfficiencyMetric,
    TokenUsageMetric,
)

__all__ = [
    # Entry points
    "compare",
    "compare_collections",
    # Result types
    "CompareResult",
    "ComparisonResult",
    "ValidationResult",
    "Gate",
    "ThresholdError",
    "Direction",
    "Observation",
    "MetricResult",  # backwards compat alias for Observation
    # Configuration
    "CompareConfig",
    # Data model
    "TraceCollection",
    "Trace",
    "ReadableSpan",
    "make_span",
    # Extension point
    "ComparisonMetric",
    # New metrics
    "TokenUsageMetric",
    "TokenEfficiencyMetric",
    "CostQualityMetric",
]
