"""AgentFlow — agent evaluation and regression detection.

Programmatic API::

    import agentflow

    # From files (same paths the CLI accepts)
    result = agentflow.compare("baseline.jsonl", "current.jsonl")

    # From in-memory collections (no files needed)
    baseline = agentflow.TraceCollection.from_traces(my_traces, source="v1")
    current  = agentflow.TraceCollection.from_traces(new_traces, source="v2")
    result   = agentflow.compare_collections(baseline, current)

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

from agentflow.collection import TraceCollection
from agentflow.compare import CompareResult, compare, compare_collections
from agentflow.config import CompareConfig
from agentflow.converters.base import Span, Trace
from agentflow.metrics import ComparisonMetric, MetricResult

__all__ = [
    # Entry points
    "compare",
    "compare_collections",
    # Result types
    "CompareResult",
    "MetricResult",
    # Configuration
    "CompareConfig",
    # Data model
    "TraceCollection",
    "Trace",
    "Span",
    # Extension point
    "ComparisonMetric",
]
