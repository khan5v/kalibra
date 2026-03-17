"""Error rate metric — compares per-trace error span rates.

Computation:
    For each trace, compute error_rate = count(error spans) / total_spans.
    Compare distributions of per-trace error rates across baseline and current.

Statistical approach:
    Headline: mean error rate change (percentage points).
    Two-proportion z-test on aggregate counts (total error spans vs total spans).
    Direction: lower error rate is better (higher_is_better = False).
    Noise threshold: 0.5 pp — error rate changes below this are noise.

Threshold fields:
    error_rate_delta: error rate change (percentage points)
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Direction, Observation
from kalibra.metrics._stats import two_proportion_ztest
from kalibra.model import Trace


class ErrorRateMetric(ComparisonMetric):
    name = "error_rate"
    description = "Per-trace error span rate"
    noise_threshold = 0.5
    higher_is_better = False
    _fields = {
        "error_rate_delta": "Error rate change (percentage points)",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        b_errors = sum(1 for t in baseline for s in t.spans if s.error)
        b_total = sum(len(t.spans) for t in baseline)
        c_errors = sum(1 for t in current for s in t.spans if s.error)
        c_total = sum(len(t.spans) for t in current)

        if b_total == 0 or c_total == 0:
            return self._no_data(
                "no span data",
                "No span data found",
            )

        b_rate = b_errors / b_total * 100
        c_rate = c_errors / c_total * 100
        delta_pp = round(c_rate - b_rate, 2)

        _, pval = two_proportion_ztest(b_total, b_errors, c_total, c_errors)
        significant = pval < 0.05

        direction = self._classify(delta_pp)
        if not significant:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            delta=delta_pp,
            baseline={
                "rate": b_rate,
                "errors": b_errors,
                "total_spans": b_total,
            },
            current={
                "rate": c_rate,
                "errors": c_errors,
                "total_spans": c_total,
            },
            metadata={
                "pvalue": pval,
                "significant": significant,
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {"error_rate_delta": result.delta}
