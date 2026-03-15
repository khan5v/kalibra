"""Span breakdown metric — per-span-name regression/improvement detection.

Computation:
    Group all spans across all traces by span name.
    For each span name present in both populations, compute:
        median duration, median cost, median tokens, error rate.
    Compare per-span-name medians between baseline and current.
    A span name "regressed" if any dimension worsened beyond noise threshold.

Statistical approach:
    Uses median (not mean) for duration/cost/tokens — consistent with trace-level
    metrics and resistant to outliers from retries or stuck spans.
    Error rate is a proportion (errors / total).
    Direction: INCONCLUSIVE if both regressions and improvements exist.
    Noise threshold: 5% (higher than trace-level metrics) because per-span-name
        samples are typically smaller, amplifying variance.
    Warning: span names with < 30 occurrences may give unreliable results.

Threshold fields:
    span_regressions: number of span names that regressed
    span_improvements: number of span names that improved
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Direction, Observation
from kalibra.metrics._stats import median, pct_delta
from kalibra.model import Span, Trace

_MIN_SPAN_COUNT = 30


class SpanBreakdownMetric(ComparisonMetric):
    name = "span_breakdown"
    description = "Per-span-name regression and improvement detection"
    noise_threshold = 5.0
    higher_is_better = True
    _fields = {
        "span_regressions": "Number of span names that regressed",
        "span_improvements": "Number of span names that improved",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        b_groups = _group_spans(baseline)
        c_groups = _group_spans(current)

        matched_names = sorted(set(b_groups) & set(c_groups))
        if not matched_names:
            return self._no_data(
                "no matching spans",
                "No span names matched between datasets",
            )

        regressions: list[dict] = []
        improvements: list[dict] = []
        mixed: list[dict] = []
        unchanged: list[str] = []
        warnings: list[str] = []
        per_span: dict[str, dict] = {}

        for name in matched_names:
            b_spans = b_groups[name]
            c_spans = c_groups[name]
            b_count = len(b_spans)
            c_count = len(c_spans)

            small = min(b_count, c_count)
            if small < _MIN_SPAN_COUNT:
                warnings.append(
                    f"Span '{name}' has only {small} occurrences "
                    f"— recommend ≥{_MIN_SPAN_COUNT}"
                )

            b_stats = _span_stats(b_spans)
            c_stats = _span_stats(c_spans)

            # Compute deltas. Lower is better for all dimensions.
            # pct_delta returns None when base=0 and curr!=0 (undefined %).
            # Treat None as 0 for threshold comparison — no baseline to regress from.
            dur_delta = pct_delta(b_stats["median_duration"], c_stats["median_duration"])
            cost_delta = pct_delta(b_stats["median_cost"], c_stats["median_cost"])
            tok_delta = pct_delta(b_stats["median_tokens"], c_stats["median_tokens"])
            err_delta_pp = c_stats["error_rate"] - b_stats["error_rate"]

            threshold = self.noise_threshold
            d = dur_delta or 0
            co = cost_delta or 0
            tk = tok_delta or 0

            # Check each dimension independently.
            has_regression = (
                d > threshold or co > threshold
                or tk > threshold or err_delta_pp > 1.0
            )
            has_improvement = (
                d < -threshold or co < -threshold
                or tk < -threshold or err_delta_pp < -1.0
            )

            span_entry = {
                "span_name": name,
                "baseline": {**b_stats, "count": b_count},
                "current": {**c_stats, "count": c_count},
                "deltas": {
                    "duration_pct": dur_delta or 0,
                    "cost_pct": cost_delta or 0,
                    "tokens_pct": tok_delta or 0,
                    "error_rate_pp": round(err_delta_pp, 1),
                },
            }

            # Classify: mixed if both regressed and improved dimensions exist.
            # Spans below minimum count are "unchanged" — too few samples to judge.
            if small < _MIN_SPAN_COUNT:
                span_entry["direction"] = "unchanged"
                unchanged.append(name)
            elif has_regression and has_improvement:
                span_entry["direction"] = "mixed"
                mixed.append(span_entry)
            elif has_regression:
                span_entry["direction"] = "regressed"
                regressions.append(span_entry)
            elif has_improvement:
                span_entry["direction"] = "improved"
                improvements.append(span_entry)
            else:
                span_entry["direction"] = "unchanged"
                unchanged.append(name)

            per_span[name] = span_entry

        n_reg = len(regressions)
        n_imp = len(improvements)
        n_mix = len(mixed)

        if n_reg > 0 and n_imp > 0:
            direction = Direction.INCONCLUSIVE
        elif n_reg > 0:
            direction = Direction.DEGRADATION
        elif n_imp > 0:
            direction = Direction.UPGRADE
        elif n_mix > 0:
            direction = Direction.INCONCLUSIVE
        else:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            baseline={"span_names": len(b_groups), "matched": len(matched_names)},
            current={"span_names": len(c_groups), "matched": len(matched_names)},
            metadata={
                "matched": len(matched_names),
                "regressions": regressions,
                "improvements": improvements,
                "mixed": mixed,
                "n_regressions": n_reg,
                "n_improvements": n_imp,
                "n_mixed": n_mix,
                "per_span": per_span,
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        return {
            "span_regressions": float(result.metadata.get("n_regressions", 0)),
            "span_improvements": float(result.metadata.get("n_improvements", 0)),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _group_spans(traces: list[Trace]) -> dict[str, list[Span]]:
    """Group all spans by name across traces."""
    groups: dict[str, list[Span]] = {}
    for t in traces:
        for s in t.spans:
            if s.name:
                groups.setdefault(s.name, []).append(s)
    return groups


def _span_stats(spans: list[Span]) -> dict:
    """Compute median stats for a list of spans with the same name."""
    durations = [s.duration_s for s in spans]
    costs = [s.cost for s in spans]
    tokens = [float(s.total_tokens) for s in spans]
    errors = sum(1 for s in spans if s.error)
    total = len(spans)
    return {
        "median_duration": median(durations),
        "median_cost": median(costs),
        "median_tokens": median(tokens),
        "error_rate": round(errors / total * 100, 1) if total else 0,
    }
