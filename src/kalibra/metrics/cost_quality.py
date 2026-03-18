"""Cost-quality metric — cost per successful trace.

Computation:
    Sum total cost across all traces, count traces with outcome = "success".
    Compute cost_per_success = total_cost / n_successes for each population.

Statistical approach:
    Headline: percentage change in cost-per-success ratio.
    Direction: lower cost per success is better (higher_is_better = False).
    Noise threshold: 5% — changes below this are noise.
    Handles no-successes and no-cost cases gracefully.

Threshold fields:
    cost_quality_delta_pct: change in cost per success (%)
    cost_per_success: absolute cost per success in current run
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Observation
from kalibra.metrics._stats import pct_delta
from kalibra.model import OUTCOME_SUCCESS, Trace


class CostQualityMetric(ComparisonMetric):
    name = "cost_quality"
    description = "Cost per successful trace"
    noise_threshold = 5.0  # % — looser for derived metrics with higher natural variance
    higher_is_better = False
    _fields = {
        "cost_quality_delta_pct": "Change in cost per success (%)",
        "cost_per_success": "Cost per success in current run",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        b_succ = sum(1 for t in baseline if t.outcome == OUTCOME_SUCCESS)
        c_succ = sum(1 for t in current if t.outcome == OUTCOME_SUCCESS)

        if b_succ == 0 or c_succ == 0:
            side = "both populations" if (b_succ == 0 and c_succ == 0) else (
                "baseline" if b_succ == 0 else "current"
            )
            return self._no_data(
                "no successes",
                f"No successful traces in {side} — cost-quality is unavailable",
            )

        b_cost = sum(t.total_cost for t in baseline if t.total_cost is not None)
        c_cost = sum(t.total_cost for t in current if t.total_cost is not None)

        if b_cost == 0 and c_cost == 0:
            has_any = any(t.total_cost is not None for t in baseline) or \
                      any(t.total_cost is not None for t in current)
            msg = "All costs are $0" if has_any else "No cost data found"
            return self._no_data("no cost data", msg)

        b_cps = b_cost / b_succ
        c_cps = c_cost / c_succ
        delta = pct_delta(b_cps, c_cps)

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta),
            delta=delta,
            baseline={
                "cost_per_success": b_cps,
                "total_cost": b_cost,
                "successes": b_succ,
            },
            current={
                "cost_per_success": c_cps,
                "total_cost": c_cost,
                "successes": c_succ,
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["cost_quality_delta_pct"] = result.delta
        if result.current:
            fields["cost_per_success"] = result.current.get("cost_per_success", 0)
        return fields
