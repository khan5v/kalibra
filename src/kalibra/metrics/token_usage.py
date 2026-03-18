"""Token usage metric — compares total token consumption per trace.

Computation:
    For each trace, sum input_tokens + output_tokens across all spans.
    Also tracks input and output token breakdowns separately.

Statistical approach:
    Headline: median total-token change (resistant to outlier traces).
    Detail: mean tokens, input/output breakdown, bootstrap 95% CI on median.
    Bootstrap CI for confidence on the median delta.
    Direction: fewer tokens is better (higher_is_better = False).
    Noise threshold: 3% — token changes below this are noise.
    Handles no-data case: returns n/a if either population has no token data.

Threshold fields:
    token_delta_pct: median total token change (%)
    total_tokens: total tokens in current run
    avg_tokens: average tokens per trace in current
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


class TokenUsageMetric(ComparisonMetric):
    name = "token_usage"
    description = "Token usage per trace — median, average, and input/output breakdown"
    noise_threshold = 3.0  # % — ignores minor fluctuations in primary metrics
    higher_is_better = False
    _fields = {
        "token_delta_pct": "Median total token change (%)",
        "total_tokens": "Total tokens in current run",
        "avg_tokens": "Average tokens per trace (current)",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        # Filter to traces with token data. None = not measured.
        b_total = [float(t.total_tokens) for t in baseline
                   if t.total_tokens is not None]
        c_total = [float(t.total_tokens) for t in current
                   if t.total_tokens is not None]

        if not b_total or not c_total:
            return self._no_data("no token data", "No token data found")

        warnings: list[str] = []
        b_coverage = len(b_total) / len(baseline) if baseline else 0
        c_coverage = len(c_total) / len(current) if current else 0
        if b_coverage < 0.5 or c_coverage < 0.5:
            warnings.append(
                f"Token data in {len(b_total)}/{len(baseline)} baseline, "
                f"{len(c_total)}/{len(current)} current traces"
            )

        b_med = median(b_total)
        c_med = median(c_total)
        delta = pct_delta(b_med, c_med)
        ci = bootstrap_ci(b_total, c_total, stat_fn=median)

        b_input = sum(s.input_tokens for t in baseline for s in t.spans
                      if s.input_tokens is not None)
        b_output = sum(s.output_tokens for t in baseline for s in t.spans
                       if s.output_tokens is not None)
        c_input = sum(s.input_tokens for t in current for s in t.spans
                      if s.input_tokens is not None)
        c_output = sum(s.output_tokens for t in current for s in t.spans
                       if s.output_tokens is not None)

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta, ci),
            delta=delta,
            baseline={
                "median": b_med,
                "avg": mean(b_total),
                "total": sum(b_total),
                "input_tokens": b_input,
                "output_tokens": b_output,
            },
            current={
                "median": c_med,
                "avg": mean(c_total),
                "total": sum(c_total),
                "input_tokens": c_input,
                "output_tokens": c_output,
            },
            metadata={
                "ci_95": ci,
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        fields: dict[str, float] = {}
        if result.delta is not None:
            fields["token_delta_pct"] = result.delta
        if result.current:
            fields["total_tokens"] = result.current.get("total", 0)
            fields["avg_tokens"] = result.current.get("avg", 0)
        return fields
