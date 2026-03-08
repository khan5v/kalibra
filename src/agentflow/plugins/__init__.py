"""Node-level metric plugin registry for AgentFlow.

Built-in metrics live in ``agentflow.plugins.builtin``. To add your own:

1. Create a Python module (e.g. ``my_metrics.py``).
2. Import ``register`` and decorate your function::

       from agentflow.plugins import register

       @register("p95_duration", "95th-percentile span duration in seconds")
       def p95_duration(node: str, traces: list) -> float:
           durations = sorted(
               s.end_time - s.start_time
               for t in traces for s in t.spans if s.name == node
           )
           if not durations:
               return 0.0
           idx = int(len(durations) * 0.95)
           return round(durations[min(idx, len(durations) - 1)], 3)

3. Either name your file ``agentflow_metrics.py`` and drop it in your project
   root (auto-discovered), or reference it in ``agentflow.yml`` under ``plugins:``.
"""

from agentflow.plugins.registry import (
    DEFAULT_NODE_METRICS,
    MetricDef,
    NodeMetricFn,
    Registry,
    _default,
)

# Module-level convenience API — delegates to the default Registry instance.
register = _default.register
available = _default.available
compute = _default.compute

# Import builtin AFTER the default instance exists so @register calls resolve.
import agentflow.plugins.builtin  # noqa: E402, F401

__all__ = [
    "DEFAULT_NODE_METRICS",
    "MetricDef",
    "NodeMetricFn",
    "Registry",
    "register",
    "available",
    "compute",
]
