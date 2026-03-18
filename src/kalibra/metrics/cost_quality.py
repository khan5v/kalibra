"""Cost-quality metric — cost per successful trace.

Computation:
    For each successful trace, get total_cost. Compare the per-trace
    cost distributions across baseline and current populations.

Statistical approach:
    Headline: median cost per successful trace, with bootstrap CI.
    Direction: lower cost per success is better (higher_is_better = False).
    Noise threshold: 5% — changes below this are noise.
    Handles no-successes and no-cost cases gracefully.

Threshold fields:
    cost_quality_delta_pct: change in cost per success (%)
    cost_per_success: median cost per success in current run
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Observation
from kalibra.metrics._stats import bootstrap_ci, median, pct_delta
from kalibra.model import OUTCOME_SUCCESS, Trace


class CostQualityMetric(ComparisonMetric):
    name = "cost_quality"
    description = "Cost per successful trace"
    noise_threshold = 5.0  # % — looser for derived metrics with higher natural variance
    higher_is_better = False
    _fields = {
        "cost_quality_delta_pct": "Change in cost per success (%)",
        "cost_per_success": "Median cost per success in current run",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        # Per-trace cost for successful traces only.
        b_values = [t.total_cost for t in baseline
                    if t.outcome == OUTCOME_SUCCESS
                    and t.total_cost is not None]
        c_values = [t.total_cost for t in current
                    if t.outcome == OUTCOME_SUCCESS
                    and t.total_cost is not None]

        if not b_values or not c_values:
            b_succ = sum(1 for t in baseline if t.outcome == OUTCOME_SUCCESS)
            c_succ = sum(1 for t in current if t.outcome == OUTCOME_SUCCESS)
            if b_succ == 0 or c_succ == 0:
                side = "both populations" if (b_succ == 0 and c_succ == 0) else (
                    "baseline" if b_succ == 0 else "current"
                )
                return self._no_data(
                    "no successes",
                    f"No successful traces in {side}",
                )
            return self._no_data("no cost data", "No cost data found")

        b_med = median(b_values)
        c_med = median(c_values)
        delta = pct_delta(b_med, c_med)
        ci = bootstrap_ci(b_values, c_values, stat_fn=median)

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta, ci),
            delta=delta,
            baseline={
                "cost_per_success": b_med,
                "total_cost": sum(b_values),
                "successes": len(b_values),
            },
            current={
                "cost_per_success": c_med,
                "total_cost": sum(c_values),
                "successes": len(c_values),
            },
            metadata={
                "ci_95": ci,
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["cost_quality_delta_pct"] = result.delta
        if result.current:
            fields["cost_per_success"] = result.current.get("cost_per_success", 0)
        return fields
