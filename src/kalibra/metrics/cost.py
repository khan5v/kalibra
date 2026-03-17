"""Cost metric — compares per-trace cost distributions.

Computation:
    For each trace, sum span costs → trace total cost.
    Compare distributions of trace total costs across baseline and current.

Statistical approach:
    Headline: median cost change (resistant to outliers in cost data,
        where a few expensive traces can skew the mean).
    Detail: mean cost, total cost, bootstrap 95% CI on median.
    Optional: Mann-Whitney U test — non-parametric, no normality assumption.
        Appropriate because cost distributions are typically right-skewed.
    Direction: lower cost is better (higher_is_better = False).
    Noise threshold: 3% — cost changes below this are treated as unchanged.

Threshold fields:
    cost_delta_pct: median cost change (%)
    total_cost: absolute total cost of current run (USD)
    avg_cost: average cost per trace (USD)
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Observation
from kalibra.metrics._stats import (
    bootstrap_ci,
    mannwhitney,
    mean,
    median,
    pct_delta,
)
from kalibra.model import Trace


class CostMetric(ComparisonMetric):
    name = "cost"
    description = "Cost per trace — median, average, and total"
    noise_threshold = 3.0
    higher_is_better = False
    _fields = {
        "cost_delta_pct": "Median cost change (%)",
        "total_cost": "Total cost of current run",
        "avg_cost": "Average cost per trace",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        # Filter to traces with cost data. None = not measured, 0 = free.
        b_costs = [t.total_cost for t in baseline
                   if t.total_cost is not None and t.total_cost > 0]
        c_costs = [t.total_cost for t in current
                   if t.total_cost is not None and t.total_cost > 0]

        if not b_costs and not c_costs:
            has_any = any(t.total_cost is not None for t in baseline) or \
                      any(t.total_cost is not None for t in current)
            msg = "All trace costs are $0" if has_any else "No cost data found"
            return self._no_data("no cost data", msg)

        warnings: list[str] = []
        b_coverage = len(b_costs) / len(baseline) if baseline else 0
        c_coverage = len(c_costs) / len(current) if current else 0
        if b_coverage < 0.5 or c_coverage < 0.5:
            warnings.append(
                f"Cost data in {len(b_costs)}/{len(baseline)} baseline, "
                f"{len(c_costs)}/{len(current)} current traces"
            )


        b_med = median(b_costs)
        c_med = median(c_costs)
        delta = pct_delta(b_med, c_med)
        mw = mannwhitney(b_costs, c_costs)
        ci = bootstrap_ci(b_costs, c_costs, stat_fn=median)

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta, mw),
            delta=delta,
            baseline={
                "median": b_med,
                "avg": mean(b_costs),
                "total": sum(b_costs),
            },
            current={
                "median": c_med,
                "avg": mean(c_costs),
                "total": sum(c_costs),
            },
            metadata={
                "ci_95": ci,
                "mannwhitney": mw,
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["cost_delta_pct"] = result.delta
        if result.current:
            fields["total_cost"] = result.current.get("total", 0)
            fields["avg_cost"] = result.current.get("avg", 0)
        return fields
