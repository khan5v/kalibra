"""JSON renderer — machine-readable output for CI pipelines.

Serializes the CompareResult to a JSON string. All Observation data
is included; no formatting or truncation.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kalibra.engine import CompareResult


def render_json(result: CompareResult) -> str:
    """Render a CompareResult as a JSON string."""
    return json.dumps(_serialize(result), indent=2)


def _serialize(result: CompareResult) -> dict:
    return {
        "direction": result.direction.value,
        "passed": result.passed,
        "baseline": {
            "source": result.baseline_source,
            "count": result.baseline_count,
        },
        "current": {
            "source": result.current_source,
            "count": result.current_count,
        },
        "warnings": result.warnings,
        "metrics": {
            name: _serialize_observation(obs)
            for name, obs in result.observations.items()
        },
        "gates": [
            {
                "expr": g.expr,
                "passed": g.passed,
                "actual": g.actual if not math.isnan(g.actual) else None,
                "warning": g.warning,
            }
            for g in result.gates
        ],
    }


def _serialize_observation(obs) -> dict:
    return {
        "name": obs.name,
        "description": obs.description,
        "direction": obs.direction.value,
        "delta": obs.delta,
        "baseline": _clean(obs.baseline),
        "current": _clean(obs.current),
        "metadata": _clean(obs.metadata),
        "warnings": obs.warnings,
    }


def _clean(value: Any) -> Any:
    """Make values JSON-safe: handle NaN, tuples, and recurse into containers."""
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value
