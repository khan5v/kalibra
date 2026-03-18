"""Success rate metric — compares task pass/fail rates.

Computation:
    Count traces with outcome = "success" vs "failure" in each population.
    Compute success rate = successes / traces_with_outcome.

Statistical approach:
    Two-proportion z-test: tests H0 that both populations have the same
    success rate. Appropriate for binary outcomes with independent samples.
    Direction: higher success rate is better.
    Noise threshold: 0.5 pp — success rate changes below this are noise.

Threshold fields:
    success_rate_delta: change in success rate (percentage points)
    success_rate: current absolute success rate (%)
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Direction, Observation
from kalibra.metrics._stats import two_proportion_ztest
from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS, Trace

# CLT heuristic: n≥30 ensures the normal approximation in the z-test
# is adequate when n*p and n*(1-p) are both ≥5 (holds for rates 17-83%).
_MIN_N = 30


class SuccessRateMetric(ComparisonMetric):
    name = "success_rate"
    description = "Task success rate delta with statistical significance"
    noise_threshold = 0.5  # pp — sensitive for binary rates, configurable via config
    higher_is_better = True
    _fields = {
        "success_rate_delta": "Change in success rate (percentage points)",
        "success_rate": "Current success rate (%)",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        b_succ = sum(1 for t in baseline if t.outcome == OUTCOME_SUCCESS)
        b_fail = sum(1 for t in baseline if t.outcome == OUTCOME_FAILURE)
        c_succ = sum(1 for t in current if t.outcome == OUTCOME_SUCCESS)
        c_fail = sum(1 for t in current if t.outcome == OUTCOME_FAILURE)

        b_with = b_succ + b_fail
        c_with = c_succ + c_fail

        b_rate = b_succ / b_with if b_with else None
        c_rate = c_succ / c_with if c_with else None

        warnings: list[str] = []

        if b_rate is None or c_rate is None:
            side = "both datasets" if (b_rate is None and c_rate is None) else (
                "baseline" if b_rate is None else "current"
            )
            return self._no_data(
                "no outcome data",
                f"No outcome data in {side}",
            )

        delta_pp = round((c_rate - b_rate) * 100, 2)
        _, pval = two_proportion_ztest(b_with, b_succ, c_with, c_succ)
        significant = pval < 0.05

        if b_with < _MIN_N or c_with < _MIN_N:
            small = min(b_with, c_with)
            warnings.append(
                f"Only {small} traces with known outcomes "
                f"— recommend ≥{_MIN_N} for reliable rates"
            )

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
                "successes": b_succ,
                "with_outcome": b_with,
                "total": len(baseline),
            },
            current={
                "rate": c_rate,
                "successes": c_succ,
                "with_outcome": c_with,
                "total": len(current),
            },
            metadata={
                "pvalue": pval,
                "significant": significant,
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {
            "success_rate_delta": result.delta,
            "success_rate": (result.current.get("rate") or 0.0) * 100,
        }
