"""Duration metric — compares trace wall-clock duration distributions.

Computation:
    For each trace, compute wall-clock duration (max end - min start across spans).

Statistical approach:
    Headline: median duration change (resistant to outliers from retries/timeouts).
    Detail: mean duration, bootstrap 95% CI on median.
    Bootstrap CI for confidence on the median delta.
        Appropriate because duration distributions are typically right-skewed.
    Direction: lower duration is better (higher_is_better = False).
    Noise threshold: 5% — duration changes below this are treated as unchanged.

Threshold fields:
    duration_delta_pct: median duration change (%)
    total_duration: total wall-clock duration of current run (seconds)
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


class DurationMetric(ComparisonMetric):
    name = "duration"
    description = "Trace duration — median and average"
    noise_threshold = 5.0  # % — looser for duration which has high natural variance
    higher_is_better = False
    _fields = {
        "duration_delta_pct": "Median duration change (%)",
        "total_duration": "Total duration of current run (seconds)",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        # Filter to traces with timing data. None = not measured.
        b_durs = [t.duration for t in baseline if t.duration is not None]
        c_durs = [t.duration for t in current if t.duration is not None]

        if not b_durs or not c_durs:
            return self._no_data(
                "no duration data",
                "No duration data found",
            )

        b_med = median(b_durs)
        c_med = median(c_durs)
        delta = pct_delta(b_med, c_med)
        ci = bootstrap_ci(b_durs, c_durs, stat_fn=median)

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta, ci),
            delta=delta,
            baseline={
                "median": b_med,
                "avg": mean(b_durs),
                "total": sum(b_durs),
            },
            current={
                "median": c_med,
                "avg": mean(c_durs),
                "total": sum(c_durs),
            },
            metadata={
                "ci_95": ci,
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["duration_delta_pct"] = result.delta
        if result.current:
            fields["total_duration"] = result.current.get("total", 0)
        return fields
