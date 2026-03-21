"""Kalibra — agent evaluation and regression detection.

Programmatic API::

    import kalibra

    # From files (same paths the CLI accepts)
    from kalibra.loader import load_traces
    baseline = load_traces("baseline.jsonl")
    current  = load_traces("current.jsonl")
    result   = kalibra.compare(baseline, current)

    # Inspect results
    for name, obs in result.observations.items():
        print(f"{name}: {obs.direction.value}")

    # Check threshold gates
    print("passed:", result.passed)
"""

__version__ = "0.2.1"

from kalibra.config import CompareConfig
from kalibra.engine import (
    CompareResult,
    GateResult,
    ThresholdError,
    compare,
    resolve_metrics,
)
from kalibra.metrics import ComparisonMetric, Direction, Observation
from kalibra.model import Span, Trace

__all__ = [
    # Entry points
    "compare",
    "resolve_metrics",
    # Result types
    "CompareResult",
    "GateResult",
    "ThresholdError",
    "Direction",
    "Observation",
    # Configuration
    "CompareConfig",
    # Data model
    "Trace",
    "Span",
    # Extension point
    "ComparisonMetric",
]
