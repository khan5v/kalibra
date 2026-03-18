"""Steps metric — compares number of spans (steps) per trace.

Computation:
    For each trace, count len(trace.spans) — the number of execution steps.
    Compare distributions across baseline and current.

Statistical approach:
    Headline: median step count change (resistant to outlier traces with retries).
    Detail: mean steps, bootstrap 95% CI on median.
    Bootstrap CI for confidence on the median delta.
    Direction: fewer steps is better (higher_is_better = False).
    Noise threshold: 3% — step count changes below this are noise.

Threshold fields:
    steps_delta_pct: median steps change (%)
    avg_steps: average steps per trace in current
    median_steps: median steps per trace in current
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Observation
from kalibra.metrics._stats import (
    bootstrap_ci,
    mean,
    median,
    pct_delta,
)
from kalibra.model import Trace


class StepsMetric(ComparisonMetric):
    name = "steps"
    description = "Steps per trace — median and average span count"
    noise_threshold = 3.0  # % — ignores minor fluctuations in primary metrics
    higher_is_better = False
    _fields = {
        "steps_delta_pct": "Median steps change (%)",
        "avg_steps": "Average steps per trace (current)",
        "median_steps": "Median steps per trace (current)",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        # Only count traces that have span data. Empty spans = no step info.
        b_steps = [float(len(t.spans)) for t in baseline if t.spans]
        c_steps = [float(len(t.spans)) for t in current if t.spans]

        if not b_steps or not c_steps:
            return self._no_data(
                "no step data",
                "No span data found",
            )


        b_med = median(b_steps)
        c_med = median(c_steps)
        delta = pct_delta(b_med, c_med)
        ci = bootstrap_ci(b_steps, c_steps, stat_fn=median)

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta, ci),
            delta=delta,
            baseline={
                "median": b_med,
                "avg": mean(b_steps),
            },
            current={
                "median": c_med,
                "avg": mean(c_steps),
            },
            metadata={
                "ci_95": ci,
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["steps_delta_pct"] = result.delta
        if result.current:
            fields["avg_steps"] = result.current.get("avg", 0)
            fields["median_steps"] = result.current.get("median", 0)
        return fields
