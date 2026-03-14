"""Node-level metric registry — Registry class and module-level default instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kalibra.converters.base import Trace

NodeMetricFn = Callable[[str, list[Trace]], float]

# Names computed when no explicit selection is passed to Registry.compute().
DEFAULT_NODE_METRICS = ["retry_rate", "error_rate", "cost_share", "token_intensity"]


@dataclass(frozen=True)
class MetricDef:
    name: str
    description: str
    fn: NodeMetricFn


class Registry:
    """Registry of node-level metric functions.

    Instantiate directly for isolated use (e.g. in tests), or use the
    module-level ``register`` / ``available`` / ``compute`` helpers that
    delegate to the default instance.

    Example — custom registry in tests::

        from kalibra.plugins.registry import Registry
        reg = Registry()

        @reg.register("my_metric", "Does something")
        def my_metric(node, traces):
            return 42.0

        assert reg.compute("node", [], ["my_metric"]) == {"my_metric": 42.0}
    """

    def __init__(self) -> None:
        self._metrics: dict[str, MetricDef] = {}

    def register(self, name: str, description: str = "") -> Callable[[NodeMetricFn], NodeMetricFn]:
        """Decorator — register a node-level metric function.

        The decorated function receives ``(node: str, traces: list[Trace])``
        and must return a single ``float``.
        """

        def decorator(fn: NodeMetricFn) -> NodeMetricFn:
            self._metrics[name] = MetricDef(name=name, description=description, fn=fn)
            return fn

        return decorator

    def available(self) -> list[str]:
        """Names of all registered metrics."""
        return list(self._metrics)

    def compute(
        self,
        node: str,
        traces: list[Trace],
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute metrics for a single node.

        Args:
            node:    Node name to evaluate.
            traces:  Full trace dataset.
            metrics: Metric names to compute. Defaults to ``DEFAULT_NODE_METRICS``.

        Returns:
            ``{metric_name: value}``
        """
        names = metrics if metrics is not None else DEFAULT_NODE_METRICS
        return {n: self._metrics[n].fn(node, traces) for n in names if n in self._metrics}


# ── Default instance ───────────────────────────────────────────────────────────
# Module-level helpers delegate to this instance.
# Import and use these directly; do not access _default from outside this module.

_default = Registry()
