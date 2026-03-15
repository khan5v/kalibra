"""Shared statistical functions for metrics.

All metrics use these building blocks. No metric reimplements basic stats.

Statistical notes:
- Median is preferred over mean for headline comparisons (resistant to outliers).
- Bootstrap CI gives distribution-free confidence intervals on any statistic.
- Mann-Whitney U is the non-parametric test for "are these two distributions different?"
  No normality assumption. Works with any sample size ≥ 2.
- Two-proportion z-test is for binary outcomes (success rate).
"""

from __future__ import annotations

import math
import random
import warnings


def mean(values: list[float]) -> float:
    """Arithmetic mean. Returns 0.0 for empty list."""
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    """Median. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def percentile(sorted_values: list[float], pct: int) -> float:
    """Percentile from pre-sorted values. Returns 0.0 for empty list."""
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * pct / 100), len(sorted_values) - 1)
    return sorted_values[idx]


def pct_delta(base: float, curr: float) -> float | None:
    """Percentage change from base to curr.

    Returns None when base is 0 and curr is not — the percentage change
    is mathematically undefined (division by zero). Callers should handle
    None as 'went from nothing to something' rather than displaying 0%.
    Returns 0.0 when both are 0 (no change).
    """
    if base == 0:
        if curr == 0:
            return 0.0
        return None
    return round((curr - base) / base * 100, 1)


def bootstrap_ci(
    baseline: list[float],
    current: list[float],
    stat_fn=None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Bootstrap confidence interval on the percentage delta between two populations.

    Resamples both populations independently, computes the statistic on each
    resample, then computes pct_delta between them. Returns (lo, hi) bounds
    on the percentage change at ``1 - alpha`` confidence.

    This answers: "how confident are we in the observed delta?"
    E.g. CI of [-30%, -10%] means we're 95% confident the true change is
    between -30% and -10%.

    Returns None if either population has < 2 values.
    Deterministic (seeded) for reproducibility.
    """
    if stat_fn is None:
        stat_fn = median
    if len(baseline) < 2 or len(current) < 2:
        return None
    rng = random.Random(42)
    raw = [
        pct_delta(
            stat_fn(rng.choices(baseline, k=len(baseline))),
            stat_fn(rng.choices(current, k=len(current))),
        )
        for _ in range(n_resamples)
    ]
    # Drop None values (base=0 resamples where pct_delta is undefined).
    deltas = sorted(d for d in raw if d is not None)
    if not deltas:
        return None
    lo = max(0, int(n_resamples * alpha / 2))
    hi = min(len(deltas) - 1, int(n_resamples * (1 - alpha / 2)))
    return (round(deltas[lo], 1), round(deltas[hi], 1))


def mannwhitney(
    baseline: list[float],
    current: list[float],
) -> dict | None:
    """Mann-Whitney U test (non-parametric).

    Tests whether two independent samples come from the same distribution.
    No normality assumption. Works with ordinal data.

    Returns {"U": float, "pvalue": float, "significant": bool} or None
    if scipy is not installed or samples are too small.
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return None

    if len(baseline) < 2 or len(current) < 2:
        return None

    # All identical values → no test needed.
    if len(set(baseline)) <= 1 and len(set(current)) <= 1:
        if baseline[0] == current[0]:
            return {"U": 0.0, "pvalue": 1.0, "significant": False}
        return {"U": 0.0, "pvalue": 0.0, "significant": True}

    try:
        stat, pval = mannwhitneyu(baseline, current, alternative="two-sided")
        return {
            "U": round(float(stat), 2),
            "pvalue": round(float(pval), 6),
            "significant": bool(pval < 0.05),
        }
    except ValueError:
        return None


def two_proportion_ztest(
    n1: int, s1: int, n2: int, s2: int,
) -> tuple[float, float]:
    """Two-proportion z-test for comparing success rates.

    Tests H0: p1 = p2 (same success rate).

    Args:
        n1, s1: total and successes in baseline.
        n2, s2: total and successes in current.

    Returns:
        (z_statistic, p_value)
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p_pool = (s1 + s2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return 0.0, 1.0
    z = (s2 / n2 - s1 / n1) / denom
    return round(z, 4), round(math.erfc(abs(z) / math.sqrt(2)), 6)


def maybe_scipy_hint() -> None:
    """Show a one-time hint about installing scipy."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        warnings.warn(
            "scipy not installed — continuous metrics use threshold-based "
            "comparison only. For Mann-Whitney U significance tests: "
            "pip install kalibra[stats]",
            stacklevel=3,
        )
