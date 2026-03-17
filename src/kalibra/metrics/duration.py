"""Duration metric — compares trace wall-clock duration distributions.

Computation:
    For each trace, compute wall-clock duration (max end - min start across spans).
    Also compute P95 from sorted per-span durations.

Statistical approach:
    Headline: median duration change (resistant to outliers from retries/timeouts).
    Detail: mean duration, P95, bootstrap 95% CI on median.
    Optional: Mann-Whitney U test — non-parametric, no normality assumption.
        Appropriate because duration distributions are typically right-skewed.
    Direction: lower duration is better (higher_is_better = False).
    Noise threshold: 5% — duration changes below this are treated as unchanged.
    Warning: if < 100 traces, P95 estimate may be unreliable.

Threshold fields:
    duration_delta_pct: median duration change (%)
    duration_median_delta_pct: same as above (explicit alias)
    duration_p95_delta_pct: P95 duration change (%)
    total_duration: total wall-clock duration of current run (seconds)
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Observation
from kalibra.metrics._stats import (
    bootstrap_ci,
    mannwhitney,
    mean,
    median,
    pct_delta,
    percentile,
)
from kalibra.model import Trace

_MIN_P95_N = 100


class DurationMetric(ComparisonMetric):
    name = "duration"
    description = "Trace duration — median, average, and P95"
    noise_threshold = 5.0
    higher_is_better = False
    _fields = {
        "duration_delta_pct": "Median duration change (%)",
        "duration_p95_delta_pct": "P95 duration change (%)",
        "total_duration": "Total duration of current run (seconds)",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        # Filter to traces with timing data. None = not measured.
        b_durs = [t.duration for t in baseline
                  if t.duration is not None and t.duration > 0]
        c_durs = [t.duration for t in current
                  if t.duration is not None and t.duration > 0]

        if not b_durs or not c_durs:
            return self._no_data(
                "no duration data",
                "No duration data found",
            )


        b_med = median(b_durs)
        c_med = median(c_durs)
        delta = pct_delta(b_med, c_med)
        mw = mannwhitney(b_durs, c_durs)
        ci = bootstrap_ci(b_durs, c_durs, stat_fn=median)

        b_sorted = sorted(b_durs)
        c_sorted = sorted(c_durs)
        b_p95 = percentile(b_sorted, 95)
        c_p95 = percentile(c_sorted, 95)
        p95_delta = pct_delta(b_p95, c_p95)

        warnings: list[str] = []
        small = min(len(b_durs), len(c_durs))
        if small < _MIN_P95_N:
            warnings.append(
                f"Only {small} traces — P95 estimate may be unreliable, "
                f"recommend ≥{_MIN_P95_N}"
            )

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta, mw),
            delta=delta,
            baseline={
                "median": b_med,
                "avg": mean(b_durs),
                "p95": b_p95,
                "total": sum(b_durs),
            },
            current={
                "median": c_med,
                "avg": mean(c_durs),
                "p95": c_p95,
                "total": sum(c_durs),
            },
            metadata={
                "ci_95": ci,
                "mannwhitney": mw,
                "p95_delta_pct": p95_delta,
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["duration_delta_pct"] = result.delta
        if result.metadata.get("p95_delta_pct") is not None:
            fields["duration_p95_delta_pct"] = result.metadata["p95_delta_pct"]
        if result.current:
            fields["total_duration"] = result.current.get("total", 0)
        return fields
