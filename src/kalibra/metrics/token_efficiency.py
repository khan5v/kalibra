"""Token efficiency metric — tokens consumed per successful trace.

Computation:
    For each successful trace, get total_tokens. Compare the per-trace
    token distributions across baseline and current populations.

Statistical approach:
    Headline: median tokens per successful trace, with bootstrap CI.
    Direction: fewer tokens per success is better (higher_is_better = False).
    Noise threshold: 5% — changes below this are noise.
    Handles no-successes case: returns n/a if either population has zero successes.

Threshold fields:
    token_efficiency_delta_pct: change in tokens per success (%)
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Observation
from kalibra.metrics._stats import bootstrap_ci, median, pct_delta
from kalibra.model import OUTCOME_SUCCESS, Trace


class TokenEfficiencyMetric(ComparisonMetric):
    name = "token_efficiency"
    description = "Tokens per successful trace"
    noise_threshold = 5.0  # % — looser for derived metrics with higher natural variance
    higher_is_better = False
    _fields = {
        "token_efficiency_delta_pct": "Change in tokens per success (%)",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        # Per-trace tokens for successful traces only.
        b_values = [float(t.total_tokens) for t in baseline
                    if t.outcome == OUTCOME_SUCCESS
                    and t.total_tokens is not None]
        c_values = [float(t.total_tokens) for t in current
                    if t.outcome == OUTCOME_SUCCESS
                    and t.total_tokens is not None]

        if not b_values or not c_values:
            # Distinguish "no successes" from "no token data"
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
            return self._no_data("no token data", "No token data found")

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
                "tokens_per_success": b_med,
                "successes": len(b_values),
            },
            current={
                "tokens_per_success": c_med,
                "successes": len(c_values),
            },
            metadata={
                "ci_95": ci,
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {"token_efficiency_delta_pct": result.delta}
