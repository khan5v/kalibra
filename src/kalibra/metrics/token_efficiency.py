"""Token efficiency metric — tokens consumed per successful trace.

Computation:
    Sum total tokens across all traces, count traces with outcome = "success".
    Compute tokens_per_success = total_tokens / n_successes for each population.

Statistical approach:
    Headline: percentage change in tokens-per-success ratio.
    Direction: fewer tokens per success is better (higher_is_better = False).
    Noise threshold: 5% — changes below this are noise.
    Handles no-successes case: returns n/a if either population has zero successes.

Threshold fields:
    token_efficiency_delta_pct: change in tokens per success (%)
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Observation
from kalibra.metrics._stats import pct_delta
from kalibra.model import OUTCOME_SUCCESS, Trace


class TokenEfficiencyMetric(ComparisonMetric):
    name = "token_efficiency"
    description = "Tokens per successful trace"
    noise_threshold = 5.0
    higher_is_better = False
    _fields = {
        "token_efficiency_delta_pct": "Change in tokens per success (%)",
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
                f"No successful traces in {side} — token efficiency is unavailable",
            )

        b_tokens = sum(t.total_tokens for t in baseline if t.total_tokens is not None)
        c_tokens = sum(t.total_tokens for t in current if t.total_tokens is not None)

        if b_tokens == 0 and c_tokens == 0:
            return self._no_data(
                "no token data",
                "All token counts are 0 — token efficiency unavailable",
            )

        b_tps = b_tokens / b_succ
        c_tps = c_tokens / c_succ
        delta = pct_delta(b_tps, c_tps)

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta),
            delta=delta,
            baseline={
                "tokens_per_success": b_tps,
                "total_tokens": b_tokens,
                "successes": b_succ,
            },
            current={
                "tokens_per_success": c_tps,
                "total_tokens": c_tokens,
                "successes": c_succ,
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        if result.delta is None:
            return {}
        return {"token_efficiency_delta_pct": result.delta}
