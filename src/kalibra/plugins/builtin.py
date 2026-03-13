"""Built-in node-level metrics for Kalibra.

This module serves as the reference implementation for custom metrics.
Each function is registered via ``@register`` and receives
``(node: str, traces: list[Trace])`` — returning a single float.

To add your own metrics, create a new module following the same pattern,
then reference it in ``kalibra.yml`` under ``plugins:`` or name it
``kalibra_metrics.py`` for zero-config auto-discovery.
"""

from __future__ import annotations

from kalibra.converters.base import Trace, span_cost, span_input_tokens, span_is_error, span_output_tokens
from kalibra.plugins.registry import _default

register = _default.register


@register("retry_rate", "Fraction of traces where this node runs more than once")
def retry_rate(node: str, traces: list[Trace]) -> float:
    seen = [t for t in traces if any(s.name == node for s in t.spans)]
    retried = [t for t in seen if sum(s.name == node for s in t.spans) > 1]
    return round(len(retried) / len(seen), 4) if seen else 0.0


@register("error_rate", "Fraction of invocations of this node that returned an error")
def error_rate(node: str, traces: list[Trace]) -> float:
    spans = [s for t in traces for s in t.spans if s.name == node]
    return round(sum(span_is_error(s) for s in spans) / len(spans), 4) if spans else 0.0


@register("cost_share", "This node's share of total cost across all traces")
def cost_share(node: str, traces: list[Trace]) -> float:
    node_cost = sum(span_cost(s) for t in traces for s in t.spans if s.name == node)
    total_cost = sum(span_cost(s) for t in traces for s in t.spans)
    return round(node_cost / total_cost, 4) if total_cost else 0.0


@register("token_intensity", "Average tokens per invocation of this node")
def token_intensity(node: str, traces: list[Trace]) -> float:
    tokens = [span_input_tokens(s) + span_output_tokens(s) for t in traces for s in t.spans if s.name == node]
    return round(sum(tokens) / len(tokens), 1) if tokens else 0.0
