"""Tests for new and enhanced metrics — token usage, efficiency, cost/quality,
bootstrap CIs, median/IQR, and absolute values.

Trace taxonomy used in these tests:
  - "cheap_fast":    low cost, low tokens, few steps, success
  - "expensive_slow": high cost, high tokens, many steps, success
  - "failed":         medium cost, medium tokens, failure
  - "empty":          no cost/token data at all (outcome=None)
  - "mixed_N":        varying profiles for statistical distribution tests

Each trace is designed so that exact expected values are easy to reason about.
"""

from kalibra.collection import TraceCollection
from kalibra.compare import compare_collections
from kalibra.config import CompareConfig
from kalibra.converters.base import (
    AF_COST,
    GEN_AI_INPUT_TOKENS,
    GEN_AI_OUTPUT_TOKENS,
    Trace,
    make_span,
)
from kalibra.metrics import (
    CostMetric,
    CostQualityMetric,
    Direction,
    DurationMetric,
    HAS_SCIPY,
    StepsMetric,
    TokenEfficiencyMetric,
    TokenUsageMetric,
    _bootstrap_ci,
    _iqr,
    _mannwhitney,
    _median,
)


# ── Rich trace builder ────────────────────────────────────────────────────────
# Extends the original _trace helper to support tokens and metadata.

def _span_rich(
    name: str,
    trace_id: str,
    i: int = 0,
    cost: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_s: float = 1.0,
    error: bool = False,
):
    return make_span(
        name=name,
        trace_id=trace_id,
        span_id=f"{name}_{i}",
        parent_span_id=None,
        start_ns=int(i * duration_s * 1e9),
        end_ns=int((i + 1) * duration_s * 1e9),
        attributes={
            AF_COST: cost,
            GEN_AI_INPUT_TOKENS: input_tokens,
            GEN_AI_OUTPUT_TOKENS: output_tokens,
        },
        error=error,
    )


def _trace_rich(
    trace_id: str,
    steps: list[dict],
    outcome: str = "success",
    metadata: dict | None = None,
) -> Trace:
    """Build a trace from a list of step dicts.

    Each step dict: {"name": str, "cost": float, "in_tok": int, "out_tok": int,
                     "duration_s": float, "error": bool}
    Only "name" is required; rest default to sensible zeros/ones.
    """
    spans = []
    for i, s in enumerate(steps):
        spans.append(_span_rich(
            name=s["name"],
            trace_id=trace_id,
            i=i,
            cost=s.get("cost", 0.0),
            input_tokens=s.get("in_tok", 0),
            output_tokens=s.get("out_tok", 0),
            duration_s=s.get("duration_s", 1.0),
            error=s.get("error", False),
        ))
    return Trace(trace_id=trace_id, spans=spans, outcome=outcome,
                 metadata=metadata or {})


def _col(*traces: Trace) -> TraceCollection:
    return TraceCollection.from_traces(list(traces))


# ── Canonical trace fixtures ──────────────────────────────────────────────────

def cheap_fast(tid="cheap"):
    """2 steps, low cost, low tokens, fast, success."""
    return _trace_rich(tid, [
        {"name": "plan", "cost": 0.001, "in_tok": 100, "out_tok": 20, "duration_s": 0.5},
        {"name": "act",  "cost": 0.002, "in_tok": 150, "out_tok": 30, "duration_s": 0.5},
    ], outcome="success")


def expensive_slow(tid="expensive"):
    """5 steps, high cost, high tokens, slow, success."""
    return _trace_rich(tid, [
        {"name": "plan",   "cost": 0.05, "in_tok": 2000, "out_tok": 500, "duration_s": 3.0},
        {"name": "search", "cost": 0.03, "in_tok": 1500, "out_tok": 300, "duration_s": 2.0},
        {"name": "edit",   "cost": 0.04, "in_tok": 1800, "out_tok": 400, "duration_s": 2.5},
        {"name": "test",   "cost": 0.02, "in_tok": 1000, "out_tok": 200, "duration_s": 4.0},
        {"name": "submit", "cost": 0.01, "in_tok": 500,  "out_tok": 100, "duration_s": 1.0},
    ], outcome="success")


def failed_trace(tid="failed"):
    """3 steps, medium cost, medium tokens, failure."""
    return _trace_rich(tid, [
        {"name": "plan",  "cost": 0.01,  "in_tok": 500,  "out_tok": 100, "duration_s": 1.0},
        {"name": "act",   "cost": 0.02,  "in_tok": 800,  "out_tok": 200, "duration_s": 2.0},
        {"name": "retry", "cost": 0.015, "in_tok": 600,  "out_tok": 150, "duration_s": 1.5, "error": True},
    ], outcome="failure")


def empty_trace(tid="empty"):
    """No cost or token data, no outcome."""
    return _trace_rich(tid, [
        {"name": "noop", "cost": 0.0, "in_tok": 0, "out_tok": 0, "duration_s": 1.0},
    ], outcome=None)


# ── Math helpers tests ────────────────────────────────────────────────────────

class TestMedian:
    def test_odd(self):
        assert _median([1.0, 2.0, 3.0]) == 2.0

    def test_even(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_single(self):
        assert _median([42.0]) == 42.0

    def test_empty(self):
        assert _median([]) == 0.0

    def test_unsorted_input(self):
        assert _median([5.0, 1.0, 3.0]) == 3.0


class TestIQR:
    def test_basic(self):
        p25, med, p75 = _iqr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert med == 4.5
        assert p25 == 3.0  # _percentile(sorted, 25) → idx 2
        assert p75 == 7.0  # _percentile(sorted, 75) → idx 6

    def test_empty(self):
        assert _iqr([]) == (0.0, 0.0, 0.0)


class TestBootstrapCI:
    def test_deterministic(self):
        """Same input always produces same output (seeded RNG)."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = _bootstrap_ci(vals)
        ci2 = _bootstrap_ci(vals)
        assert ci1 == ci2

    def test_single_value(self):
        lo, hi = _bootstrap_ci([42.0])
        assert lo == 42.0
        assert hi == 42.0

    def test_ci_contains_mean(self):
        vals = [float(x) for x in range(1, 101)]
        lo, hi = _bootstrap_ci(vals)
        mean = sum(vals) / len(vals)  # 50.5
        assert lo <= mean <= hi

    def test_ci_narrows_with_low_variance(self):
        narrow = [10.0, 10.0, 10.0, 10.0, 10.1]
        wide = [1.0, 5.0, 10.0, 50.0, 100.0]
        ci_narrow = _bootstrap_ci(narrow)
        ci_wide = _bootstrap_ci(wide)
        assert (ci_narrow[1] - ci_narrow[0]) < (ci_wide[1] - ci_wide[0])

    def test_empty(self):
        lo, hi = _bootstrap_ci([])
        assert lo == 0.0 and hi == 0.0


# ── TokenUsageMetric ─────────────────────────────────────────────────────────

class TestTokenUsageMetric:
    def test_summarize_basic(self):
        m = TokenUsageMetric()
        col = _col(cheap_fast(), expensive_slow())
        s = m.summarize(col)
        # cheap: in=250, out=50 → total=300
        # expensive: in=6800, out=1500 → total=8300
        assert s["total"] == 300 + 8300
        assert s["avg_total"] == (300 + 8300) / 2
        assert s["avg_input"] == (250 + 6800) / 2
        assert s["avg_output"] == (50 + 1500) / 2
        assert s["all_zero"] is False
        assert s["n"] == 2
        assert s["ci_95"] is not None

    def test_summarize_all_zero(self):
        m = TokenUsageMetric()
        col = _col(empty_trace("e1"), empty_trace("e2"))
        s = m.summarize(col)
        assert s["all_zero"] is True
        assert s["total"] == 0

    def test_compare_no_data(self):
        m = TokenUsageMetric()
        b = m.summarize(_col(empty_trace("e1"), empty_trace("e2")))
        c = m.summarize(_col(empty_trace("e3"), empty_trace("e4")))
        obs = m.compare(b, c)
        assert obs.direction == Direction.NA
        assert obs.warnings

    def test_compare_decrease(self):
        m = TokenUsageMetric()
        b = m.summarize(_col(expensive_slow("b1"), expensive_slow("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        assert obs.delta < 0  # tokens went down
        assert obs.direction == Direction.UPGRADE  # lower is better

    def test_compare_increase(self):
        m = TokenUsageMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(expensive_slow("c1"), expensive_slow("c2")))
        obs = m.compare(b, c)
        assert obs.delta > 0
        assert obs.direction == Direction.DEGRADATION

    def test_threshold_fields(self):
        m = TokenUsageMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        fields = m.threshold_fields(obs)
        assert "token_delta_pct" in fields
        assert "total_tokens" in fields
        assert "avg_tokens" in fields
        assert fields["total_tokens"] == 300 + 300  # two cheap traces

    def test_single_trace_no_ci(self):
        """CI is None when only 1 trace (can't bootstrap)."""
        m = TokenUsageMetric()
        s = m.summarize(_col(cheap_fast()))
        assert s["ci_95"] is None

    def test_formatted_output_contains_breakdown(self):
        m = TokenUsageMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(expensive_slow("c1"), expensive_slow("c2")))
        obs = m.compare(b, c)
        assert "tokens" in obs.formatted
        # input/output breakdown is now in detail_lines
        all_text = " ".join(obs.detail_lines)
        assert "in:" in all_text
        assert "out:" in all_text


# ── TokenEfficiencyMetric ────────────────────────────────────────────────────

class TestTokenEfficiencyMetric:
    def test_basic(self):
        m = TokenEfficiencyMetric()
        col = _col(cheap_fast("s1"), cheap_fast("s2"), failed_trace("f1"))
        s = m.summarize(col)
        # 2 successes, each 300 tokens → 600 total / 2 = 300 tokens/success
        assert s["n_successes"] == 2
        assert s["tokens_per_success"] == 300.0

    def test_no_successes(self):
        m = TokenEfficiencyMetric()
        col = _col(failed_trace("f1"), failed_trace("f2"))
        s = m.summarize(col)
        assert s["tokens_per_success"] is None
        assert s["n_successes"] == 0

    def test_compare_improvement(self):
        """Fewer tokens per success = upgrade."""
        m = TokenEfficiencyMetric()
        b = m.summarize(_col(expensive_slow("b1"), expensive_slow("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        assert obs.delta < 0
        assert obs.direction == Direction.UPGRADE

    def test_compare_no_successes_either_side(self):
        m = TokenEfficiencyMetric()
        b = m.summarize(_col(failed_trace("f1")))
        c = m.summarize(_col(failed_trace("f2")))
        obs = m.compare(b, c)
        assert obs.direction == Direction.NA
        assert "both" in obs.formatted

    def test_compare_no_successes_baseline_only(self):
        m = TokenEfficiencyMetric()
        b = m.summarize(_col(failed_trace("f1")))
        c = m.summarize(_col(cheap_fast("c1")))
        obs = m.compare(b, c)
        assert obs.direction == Direction.NA
        assert "baseline" in obs.formatted

    def test_compare_no_successes_current_only(self):
        m = TokenEfficiencyMetric()
        b = m.summarize(_col(cheap_fast("b1")))
        c = m.summarize(_col(failed_trace("f1")))
        obs = m.compare(b, c)
        assert obs.direction == Direction.NA
        assert "current" in obs.formatted

    def test_threshold_fields(self):
        m = TokenEfficiencyMetric()
        b = m.summarize(_col(cheap_fast("b1")))
        c = m.summarize(_col(cheap_fast("c1")))
        obs = m.compare(b, c)
        fields = m.threshold_fields(obs)
        assert "token_efficiency_delta_pct" in fields

    def test_threshold_fields_no_data(self):
        m = TokenEfficiencyMetric()
        b = m.summarize(_col(failed_trace("f1")))
        c = m.summarize(_col(failed_trace("f2")))
        obs = m.compare(b, c)
        assert m.threshold_fields(obs) == {}

    def test_formatted_shows_success_count(self):
        m = TokenEfficiencyMetric()
        b = m.summarize(_col(cheap_fast("b1"), failed_trace("bf")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        assert "1→2 successes" in obs.formatted


# ── CostQualityMetric ────────────────────────────────────────────────────────

class TestCostQualityMetric:
    def test_basic(self):
        m = CostQualityMetric()
        # 2 traces: 1 success ($0.003), 1 failure ($0.045)
        col = _col(cheap_fast("s1"), failed_trace("f1"))
        s = m.summarize(col)
        # total_cost = 0.003 + 0.045 = 0.048, 1 success → cost_per_success = 0.048
        assert s["n_successes"] == 1
        assert abs(s["total_cost"] - 0.048) < 1e-9
        assert abs(s["cost_per_success"] - 0.048) < 1e-9

    def test_all_successes(self):
        m = CostQualityMetric()
        col = _col(cheap_fast("s1"), cheap_fast("s2"))
        s = m.summarize(col)
        # total_cost = 0.006, 2 successes → 0.003 each
        assert s["n_successes"] == 2
        assert abs(s["cost_per_success"] - 0.003) < 1e-9

    def test_no_successes(self):
        m = CostQualityMetric()
        col = _col(failed_trace("f1"), failed_trace("f2"))
        s = m.summarize(col)
        assert s["cost_per_success"] is None

    def test_compare_improvement(self):
        """Lower cost per success = upgrade."""
        m = CostQualityMetric()
        # baseline: expensive traces, current: cheap traces
        b = m.summarize(_col(expensive_slow("b1"), expensive_slow("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        assert obs.delta < 0
        assert obs.direction == Direction.UPGRADE

    def test_compare_degradation(self):
        """Higher cost per success = degradation."""
        m = CostQualityMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(expensive_slow("c1"), expensive_slow("c2")))
        obs = m.compare(b, c)
        assert obs.delta > 0
        assert obs.direction == Direction.DEGRADATION

    def test_compare_no_successes(self):
        m = CostQualityMetric()
        b = m.summarize(_col(failed_trace("f1")))
        c = m.summarize(_col(failed_trace("f2")))
        obs = m.compare(b, c)
        assert obs.direction == Direction.NA

    def test_threshold_fields(self):
        m = CostQualityMetric()
        b = m.summarize(_col(cheap_fast("b1")))
        c = m.summarize(_col(cheap_fast("c1")))
        obs = m.compare(b, c)
        fields = m.threshold_fields(obs)
        assert "cost_quality_delta_pct" in fields
        assert "cost_per_success" in fields

    def test_formatted_shows_success_ratio(self):
        m = CostQualityMetric()
        b = m.summarize(_col(cheap_fast("b1"), failed_trace("bf")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2"), failed_trace("cf")))
        obs = m.compare(b, c)
        assert "1/2" in obs.formatted  # baseline: 1 of 2 succeeded
        assert "2/3" in obs.formatted  # current: 2 of 3 succeeded

    def test_includes_failure_cost_in_numerator(self):
        """Total cost includes failures — the real economics."""
        m = CostQualityMetric()
        # 1 success at $0.003, 1 failure at $0.045 → cost_per_success = $0.048
        col = _col(cheap_fast("s1"), failed_trace("f1"))
        s = m.summarize(col)
        assert s["cost_per_success"] > cheap_fast().total_cost


# ── Enhanced CostMetric ──────────────────────────────────────────────────────

class TestEnhancedCostMetric:
    def test_summarize_has_new_fields(self):
        m = CostMetric()
        col = _col(cheap_fast("t1"), expensive_slow("t2"))
        s = m.summarize(col)
        assert "median" in s
        assert "p25" in s
        assert "p75" in s
        assert "ci_95" in s
        assert "total" in s

    def test_total_cost_exposed(self):
        m = CostMetric()
        col = _col(cheap_fast("t1"), expensive_slow("t2"))
        s = m.summarize(col)
        # cheap: 0.003, expensive: 0.15
        assert abs(s["total"] - 0.153) < 1e-9

    def test_median_cost(self):
        m = CostMetric()
        t1 = cheap_fast("t1")   # cost = 0.003
        t2 = cheap_fast("t2")   # cost = 0.003
        t3 = expensive_slow("t3")  # cost = 0.15
        col = _col(t1, t2, t3)
        s = m.summarize(col)
        assert s["median"] == 0.003  # 2 of 3 are cheap

    def test_formatted_shows_median_and_details(self):
        m = CostMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        assert "median" in obs.formatted
        all_detail = " ".join(obs.detail_lines)
        assert "total" in all_detail
        assert "avg" in all_detail

    def test_threshold_fields_include_absolutes(self):
        m = CostMetric()
        b = m.summarize(_col(cheap_fast("b1")))
        c = m.summarize(_col(cheap_fast("c1")))
        obs = m.compare(b, c)
        fields = m.threshold_fields(obs)
        assert "total_cost" in fields
        assert "avg_cost" in fields
        assert "cost_delta_pct" in fields

    def test_ci_in_metadata(self):
        m = CostMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2"), expensive_slow("b3")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2"), expensive_slow("c3")))
        obs = m.compare(b, c)
        assert "ci_95" in obs.metadata

    def test_no_data_still_works(self):
        m = CostMetric()
        b = m.summarize(_col(empty_trace("e1"), empty_trace("e2")))
        c = m.summarize(_col(empty_trace("e3"), empty_trace("e4")))
        obs = m.compare(b, c)
        assert obs.direction == Direction.NA

    def test_single_trace_no_ci(self):
        m = CostMetric()
        s = m.summarize(_col(cheap_fast("t1")))
        assert s["ci_95"] is None


# ── Enhanced DurationMetric ──────────────────────────────────────────────────

class TestEnhancedDurationMetric:
    def test_summarize_has_new_fields(self):
        m = DurationMetric()
        col = _col(cheap_fast("t1"), expensive_slow("t2"))
        s = m.summarize(col)
        assert "median" in s
        assert "p25" in s
        assert "p75" in s
        assert "total" in s
        assert "ci_95" in s

    def test_total_duration(self):
        m = DurationMetric()
        col = _col(cheap_fast("t1"), expensive_slow("t2"))
        s = m.summarize(col)
        # cheap: start=0, end=1.0s → 1.0s
        # expensive: start=0, end=16.0s (5 spans, max end = i=3 at 4*4=16s)
        assert abs(s["total"] - (1.0 + 16.0)) < 0.1

    def test_median_duration(self):
        m = DurationMetric()
        # 3 traces with durations 1.0, 1.0, 12.5
        col = _col(cheap_fast("t1"), cheap_fast("t2"), expensive_slow("t3"))
        s = m.summarize(col)
        assert s["median"] == 1.0

    def test_formatted_shows_median(self):
        m = DurationMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        assert "median" in obs.formatted

    def test_threshold_fields_include_new(self):
        m = DurationMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        fields = m.threshold_fields(obs)
        assert "duration_median_delta_pct" in fields
        assert "total_duration" in fields
        assert "duration_delta_pct" in fields
        assert "duration_p95_delta_pct" in fields


# ── Enhanced StepsMetric ─────────────────────────────────────────────────────

class TestEnhancedStepsMetric:
    def test_summarize_has_new_fields(self):
        m = StepsMetric()
        col = _col(cheap_fast("t1"), expensive_slow("t2"))
        s = m.summarize(col)
        assert "median" in s
        assert "ci_95" in s

    def test_median_steps(self):
        m = StepsMetric()
        col = _col(cheap_fast("t1"), cheap_fast("t2"), expensive_slow("t3"))
        s = m.summarize(col)
        # steps: 2, 2, 5 → median = 2
        assert s["median"] == 2.0

    def test_formatted_shows_steps_and_details(self):
        m = StepsMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        assert "steps" in obs.formatted
        assert "avg" in " ".join(obs.detail_lines)

    def test_threshold_fields_include_new(self):
        m = StepsMetric()
        b = m.summarize(_col(cheap_fast("b1"), cheap_fast("b2")))
        c = m.summarize(_col(cheap_fast("c1"), cheap_fast("c2")))
        obs = m.compare(b, c)
        fields = m.threshold_fields(obs)
        assert "steps_delta_pct" in fields
        assert "avg_steps" in fields
        assert "median_steps" in fields


# ── Integration: all new metrics appear in full compare ──────────────────────

class TestIntegration:
    def test_new_metrics_in_default_set(self):
        """All new metrics appear in a default compare_collections run."""
        baseline = _col(
            cheap_fast("b1"), cheap_fast("b2"), expensive_slow("b3"), failed_trace("bf"),
        )
        current = _col(
            cheap_fast("c1"), cheap_fast("c2"), cheap_fast("c3"), failed_trace("cf"),
        )
        # Pass empty config so all DEFAULT_METRICS run (not filtered by compare.yml)
        result = compare_collections(baseline, current, config=CompareConfig())
        obs_names = set(result.comparison.observations.keys())
        assert "token_usage" in obs_names
        assert "token_efficiency" in obs_names
        assert "cost_quality" in obs_names

    def test_threshold_gates_on_new_metrics(self):
        """New metric threshold fields work in --require expressions."""
        baseline = _col(cheap_fast("b1"), cheap_fast("b2"))
        current = _col(cheap_fast("c1"), cheap_fast("c2"))
        result = compare_collections(
            baseline, current,
            config=CompareConfig(require=["total_cost <= 1.0"]),
        )
        assert result.validation.passed is True

    def test_threshold_gate_fail_on_total_cost(self):
        """Gate fails when total cost exceeds threshold."""
        baseline = _col(cheap_fast("b1"))
        current = _col(expensive_slow("c1"), expensive_slow("c2"))
        result = compare_collections(
            baseline, current,
            config=CompareConfig(require=["total_cost <= 0.01"]),
        )
        assert result.validation.passed is False

    def test_threshold_gate_on_token_efficiency(self):
        """Can gate on token efficiency delta."""
        baseline = _col(expensive_slow("b1"), expensive_slow("b2"))
        current = _col(cheap_fast("c1"), cheap_fast("c2"))
        result = compare_collections(
            baseline, current,
            config=CompareConfig(require=["token_efficiency_delta_pct <= 0"]),
        )
        # Current uses fewer tokens per success → delta is negative → passes
        assert result.validation.passed is True


# ── Edge cases and robustness ────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_empty_traces(self):
        """Metrics handle collections of empty traces gracefully."""
        baseline = _col(empty_trace("e1"), empty_trace("e2"))
        current = _col(empty_trace("e3"), empty_trace("e4"))
        result = compare_collections(baseline, current, config=CompareConfig())
        # Should not crash — token_usage and cost should be NA
        assert result.comparison.observations["token_usage"].direction == Direction.NA
        assert result.comparison.observations["cost"].direction == Direction.NA

    def test_mixed_outcomes(self):
        """Metrics handle mix of success, failure, and None outcomes."""
        col = _col(cheap_fast("s1"), failed_trace("f1"), empty_trace("e1"))
        m = TokenEfficiencyMetric()
        s = m.summarize(col)
        assert s["n_successes"] == 1
        assert s["tokens_per_success"] == 300.0  # cheap_fast tokens

    def test_single_trace_metrics(self):
        """All metrics work with a single trace per collection."""
        baseline = _col(cheap_fast("b1"))
        current = _col(expensive_slow("c1"))
        result = compare_collections(baseline, current, config=CompareConfig())
        # Should complete without error
        assert result.comparison.direction is not None

    def test_identical_collections(self):
        """All deltas are zero/same for identical collections."""
        traces = [cheap_fast("t1"), expensive_slow("t2"), failed_trace("t3")]
        baseline = _col(*traces)
        current = _col(*traces)
        result = compare_collections(baseline, current, config=CompareConfig())
        for name, obs in result.comparison.observations.items():
            if obs.direction != Direction.NA:
                assert obs.direction == Direction.SAME, f"{name} should be SAME, got {obs.direction}"

    def test_large_population_ci(self):
        """CI narrows with larger populations."""
        small_b = _col(*[cheap_fast(f"s{i}") for i in range(5)])
        small_c = _col(*[cheap_fast(f"s{i}") for i in range(5, 10)])
        large_b = _col(*[cheap_fast(f"l{i}") for i in range(100)])
        large_c = _col(*[cheap_fast(f"l{i}") for i in range(100, 200)])

        m = CostMetric()
        s_small = m.summarize(small_b)
        s_large = m.summarize(large_b)

        # Larger sample → tighter CI
        if s_small["ci_95"] and s_large["ci_95"]:
            small_width = s_small["ci_95"][1] - s_small["ci_95"][0]
            large_width = s_large["ci_95"][1] - s_large["ci_95"][0]
            assert large_width <= small_width


# ── Mann-Whitney U tests ─────────────────────────────────────────────────────

import pytest


class TestMannWhitneyHelper:
    def test_returns_none_with_tiny_samples(self):
        """Need at least 2 values per side."""
        assert _mannwhitney([1.0], [2.0, 3.0], higher_is_better=True) is None
        assert _mannwhitney([1.0, 2.0], [3.0], higher_is_better=True) is None

    def test_identical_constants(self):
        """All-same values on both sides → p=1, not significant."""
        result = _mannwhitney([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], higher_is_better=True)
        assert result is not None
        assert result["significant"] is False
        assert result["pvalue"] == 1.0

    def test_different_constants(self):
        """Clearly distinct constant populations → significant."""
        result = _mannwhitney([1.0, 1.0, 1.0], [9.0, 9.0, 9.0], higher_is_better=True)
        assert result is not None
        assert result["significant"] is True

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_clearly_different_distributions(self):
        """Well-separated distributions should be significant."""
        low = [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.0, 2.0]
        high = [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0]
        result = _mannwhitney(low, high, higher_is_better=True)
        assert result is not None
        assert result["significant"] is True
        assert result["pvalue"] < 0.01

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_overlapping_distributions_not_significant(self):
        """Heavily overlapping distributions should not be significant."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        result = _mannwhitney(a, b, higher_is_better=True)
        assert result is not None
        assert result["significant"] is False


class TestMannWhitneyInMetrics:
    """Test that Mann-Whitney integrates into continuous metrics correctly."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_cost_metric_has_mannwhitney(self):
        m = CostMetric()
        b = m.summarize(_col(*[cheap_fast(f"b{i}") for i in range(10)]))
        c = m.summarize(_col(*[expensive_slow(f"c{i}") for i in range(10)]))
        obs = m.compare(b, c)
        assert obs.metadata.get("mannwhitney") is not None
        assert obs.metadata["mannwhitney"]["significant"] is True
        assert any("Mann-Whitney" in line for line in obs.detail_lines)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_steps_metric_has_mannwhitney(self):
        m = StepsMetric()
        b = m.summarize(_col(*[cheap_fast(f"b{i}") for i in range(10)]))
        c = m.summarize(_col(*[expensive_slow(f"c{i}") for i in range(10)]))
        obs = m.compare(b, c)
        assert obs.metadata.get("mannwhitney") is not None
        assert any("Mann-Whitney" in line for line in obs.detail_lines)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_duration_metric_has_mannwhitney(self):
        m = DurationMetric()
        b = m.summarize(_col(*[cheap_fast(f"b{i}") for i in range(10)]))
        c = m.summarize(_col(*[expensive_slow(f"c{i}") for i in range(10)]))
        obs = m.compare(b, c)
        assert obs.metadata.get("mannwhitney") is not None
        assert any("Mann-Whitney" in line for line in obs.detail_lines)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_token_usage_metric_has_mannwhitney(self):
        m = TokenUsageMetric()
        b = m.summarize(_col(*[cheap_fast(f"b{i}") for i in range(10)]))
        c = m.summarize(_col(*[expensive_slow(f"c{i}") for i in range(10)]))
        obs = m.compare(b, c)
        assert obs.metadata.get("mannwhitney") is not None
        assert any("Mann-Whitney" in line for line in obs.detail_lines)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_identical_data_not_significant(self):
        """Identical populations → Mann-Whitney says not significant → direction=SAME."""
        m = CostMetric()
        traces = [cheap_fast(f"t{i}") for i in range(20)]
        b = m.summarize(_col(*traces))
        c = m.summarize(_col(*traces))
        obs = m.compare(b, c)
        assert obs.direction == Direction.SAME
        mw = obs.metadata.get("mannwhitney")
        assert mw is not None
        assert mw["significant"] is False

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_mannwhitney_overrides_threshold_direction(self):
        """When delta exceeds noise threshold but Mann-Whitney says not significant,
        direction should be SAME (statistical test takes precedence)."""
        m = CostMetric()
        # Create populations with overlapping distributions but slightly different medians
        import random as rng
        rng.seed(99)
        # Both populations drawn from similar range — noise, not signal
        b_traces = []
        c_traces = []
        for i in range(30):
            b_traces.append(_trace_rich(f"b{i}", [
                {"name": "step", "cost": rng.uniform(0.01, 0.10), "duration_s": 1.0},
            ], outcome="success"))
            c_traces.append(_trace_rich(f"c{i}", [
                {"name": "step", "cost": rng.uniform(0.01, 0.10), "duration_s": 1.0},
            ], outcome="success"))
        b = m.summarize(_col(*b_traces))
        c = m.summarize(_col(*c_traces))
        obs = m.compare(b, c)
        # If Mann-Whitney says not significant, direction should be SAME regardless of delta
        if obs.metadata["mannwhitney"]["significant"] is False:
            assert obs.direction == Direction.SAME

    def test_single_trace_no_mannwhitney(self):
        """Mann-Whitney needs ≥2 per side — should gracefully skip."""
        m = CostMetric()
        b = m.summarize(_col(cheap_fast("b1")))
        c = m.summarize(_col(expensive_slow("c1")))
        obs = m.compare(b, c)
        # Should still work — just no Mann-Whitney in metadata
        assert obs.metadata.get("mannwhitney") is None
        assert obs.direction is not None

    def test_summarize_stores_raw_values(self):
        """All continuous metrics store _values in their summarize output."""
        for MetricClass in [CostMetric, StepsMetric, DurationMetric, TokenUsageMetric]:
            m = MetricClass()
            s = m.summarize(_col(cheap_fast("t1"), expensive_slow("t2")))
            assert "_values" in s, f"{MetricClass.__name__} missing _values"
            assert len(s["_values"]) == 2
