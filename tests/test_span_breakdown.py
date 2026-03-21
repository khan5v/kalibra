"""Tests for span breakdown metric — CI classification, error rate z-test, mixed gating."""

from __future__ import annotations

from kalibra.metrics.span_breakdown import SpanBreakdownMetric
from kalibra.model import Span, Trace

# Realistic nanosecond base timestamp.
_BASE = 1_700_000_000_000_000_000


def _traces(n, name="step", dur_ns=1_000_000_000, error=False, tokens=None, cost=None):
    """Build n traces, each with one span."""
    spans_kwargs = {
        "start_ns": _BASE,
        "end_ns": _BASE + dur_ns,
        "error": error,
    }
    if tokens is not None:
        spans_kwargs["input_tokens"] = tokens
        spans_kwargs["output_tokens"] = 0
    if cost is not None:
        spans_kwargs["cost"] = cost
    return [
        Trace(trace_id=f"t{i}", spans=[Span(span_id=f"s{i}", name=name, **spans_kwargs)])
        for i in range(n)
    ]


class TestSpanCIClassification:
    """CI-based direction should match _classify logic: significant AND above noise."""

    def _direction(self, baseline, current):
        m = SpanBreakdownMetric()
        obs = m.compare(baseline, current)
        return obs.metadata["per_span"]["step"]["direction"]

    def test_clear_regression(self):
        """100% duration increase, no variance → CI entirely above threshold."""
        assert self._direction(
            _traces(30, dur_ns=1_000_000_000),
            _traces(30, dur_ns=2_000_000_000),
        ) == "regressed"

    def test_clear_improvement(self):
        """50% duration decrease → CI entirely below -threshold."""
        assert self._direction(
            _traces(30, dur_ns=2_000_000_000),
            _traces(30, dur_ns=1_000_000_000),
        ) == "improved"

    def test_tiny_change_below_noise(self):
        """1% change — real (CI tight) but below 5% noise threshold."""
        assert self._direction(
            _traces(30, dur_ns=1_000_000_000),
            _traces(30, dur_ns=1_010_000_000),
        ) == "unchanged"

    def test_ci_crosses_zero_unchanged(self):
        """Noisy data where CI includes zero → not significant."""
        import random
        rng = random.Random(99)
        baseline = [
            Trace(trace_id=f"t{i}", spans=[
                Span(span_id=f"s{i}", name="step",
                     start_ns=_BASE, end_ns=_BASE + int(1e9 + rng.gauss(0, 3e8)))
            ]) for i in range(30)
        ]
        current = [
            Trace(trace_id=f"t{i}", spans=[
                Span(span_id=f"s{i}", name="step",
                     start_ns=_BASE, end_ns=_BASE + int(1.05e9 + rng.gauss(0, 3e8)))
            ]) for i in range(30)
        ]
        assert self._direction(baseline, current) == "unchanged"

    def test_below_min_span_count_always_unchanged(self):
        """Even with a 10x regression, n<30 → unchanged with warning."""
        m = SpanBreakdownMetric()
        obs = m.compare(
            _traces(5, dur_ns=1_000_000_000),
            _traces(5, dur_ns=10_000_000_000),
        )
        step = obs.metadata["per_span"]["step"]
        assert step["direction"] == "unchanged"
        assert step["warning"] is not None
        assert "30" in step["warning"]

    def test_ci_stored_in_entry(self):
        """CI values are present in per-span data."""
        m = SpanBreakdownMetric()
        obs = m.compare(
            _traces(30, dur_ns=1_000_000_000),
            _traces(30, dur_ns=2_000_000_000),
        )
        ci = obs.metadata["per_span"]["step"]["ci_95"]
        assert ci["duration"] is not None
        assert ci["duration"][0] > 0  # entire CI positive

    def test_ci_none_when_insufficient_data(self):
        """CI is None when n < 2."""
        m = SpanBreakdownMetric()
        obs = m.compare(
            _traces(1, dur_ns=1_000_000_000),
            _traces(1, dur_ns=2_000_000_000),
        )
        ci = obs.metadata["per_span"]["step"]["ci_95"]
        assert ci["duration"] is None


class TestSpanErrorRateZTest:
    """Error rate uses two-proportion z-test, not a fixed threshold."""

    def _direction(self, baseline, current):
        m = SpanBreakdownMetric()
        obs = m.compare(baseline, current)
        return obs.metadata["per_span"]["step"]["direction"]

    def test_small_error_change_not_significant(self):
        """3/30 errors (10%) — z-test says p=0.076, not significant at p<0.05."""
        baseline = _traces(30, error=False)
        current = [
            Trace(trace_id=f"t{i}", spans=[
                Span(span_id=f"s{i}", name="step",
                     start_ns=_BASE, end_ns=_BASE + 1_000_000_000,
                     error=(i < 3))
            ]) for i in range(30)
        ]
        assert self._direction(baseline, current) == "unchanged"

    def test_large_error_change_significant(self):
        """10/100 errors (10%) — z-test says p=0.001, significant."""
        baseline = _traces(100, error=False)
        current = [
            Trace(trace_id=f"t{i}", spans=[
                Span(span_id=f"s{i}", name="step",
                     start_ns=_BASE, end_ns=_BASE + 1_000_000_000,
                     error=(i < 10))
            ]) for i in range(100)
        ]
        assert self._direction(baseline, current) == "regressed"

    def test_no_errors_unchanged(self):
        """0% → 0% — no errors, no change."""
        assert self._direction(
            _traces(30, error=False),
            _traces(30, error=False),
        ) == "unchanged"

    def test_error_improvement_significant(self):
        """Error rate drops significantly → improved."""
        baseline = [
            Trace(trace_id=f"t{i}", spans=[
                Span(span_id=f"s{i}", name="step",
                     start_ns=_BASE, end_ns=_BASE + 1_000_000_000,
                     error=(i < 20))
            ]) for i in range(100)
        ]
        current = _traces(100, error=False)
        assert self._direction(baseline, current) == "improved"


class TestMixedSpanGating:
    """Mixed spans count toward span_regressions for gate purposes."""

    def test_mixed_counts_as_regression_for_gate(self):
        """A span that regresses in cost but improves in duration is 'mixed'
        but counts toward span_regressions for gate threshold."""
        m = SpanBreakdownMetric()
        # Duration improves (2s → 1s), cost regresses ($0.01 → $0.05)
        baseline = _traces(30, dur_ns=2_000_000_000, cost=0.01)
        current = _traces(30, dur_ns=1_000_000_000, cost=0.05)
        obs = m.compare(baseline, current)

        step = obs.metadata["per_span"]["step"]
        assert step["direction"] == "mixed"

        # Gate field includes mixed spans.
        fields = m.threshold_fields(obs)
        assert fields["span_regressions"] >= 1

    def test_pure_regression_counts_for_gate(self):
        """A clean regression counts for gates too."""
        m = SpanBreakdownMetric()
        obs = m.compare(
            _traces(30, dur_ns=1_000_000_000),
            _traces(30, dur_ns=3_000_000_000),
        )
        fields = m.threshold_fields(obs)
        assert fields["span_regressions"] >= 1

    def test_unchanged_does_not_count(self):
        """Unchanged spans don't count toward regressions."""
        m = SpanBreakdownMetric()
        obs = m.compare(
            _traces(30, dur_ns=1_000_000_000),
            _traces(30, dur_ns=1_000_000_000),
        )
        fields = m.threshold_fields(obs)
        assert fields["span_regressions"] == 0
