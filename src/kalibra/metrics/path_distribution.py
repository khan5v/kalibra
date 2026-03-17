"""Path distribution metric — compares execution path similarity.

Computation:
    For each trace, build an execution path = tuple of span names in order.
    Collect the set of top paths (by frequency) in each population.
    Compute Jaccard similarity = |intersection| / |union| of top path sets.

Statistical approach:
    Jaccard similarity ranges from 0.0 (completely different paths) to
    1.0 (identical path distributions). No statistical test — this is a
    structural similarity measure.
    Direction: higher similarity is better (higher_is_better = True).
    Noise threshold: 0.0 — any change in path distribution is reported.
    Warning: if < 30 traces, path distribution may not be representative.

Threshold fields:
    path_jaccard: Jaccard similarity of top execution paths (0-1)
"""

from __future__ import annotations

from collections import Counter

from kalibra.metrics import ComparisonMetric, Observation
from kalibra.model import Trace

_MIN_TRACES = 30

# Compare only the top-K most frequent paths. Without this, datasets with
# thousands of unique paths (each trace takes a slightly different route)
# would always show near-zero Jaccard even when the dominant patterns are
# identical. Top-K focuses the comparison on structurally significant paths.
_TOP_K = 20


class PathDistributionMetric(ComparisonMetric):
    name = "path_distribution"
    description = "Execution path similarity (Jaccard index of top paths)"
    noise_threshold = 0.0
    higher_is_better = True
    _fields = {
        "path_jaccard": "Jaccard similarity of top execution paths (0-1)",
    }

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        if not baseline or not current:
            return self._no_data(
                "no traces",
                "No trace data found",
            )

        b_paths = self._extract_paths(baseline)
        c_paths = self._extract_paths(current)

        b_top = set(self._top_paths(b_paths))
        c_top = set(self._top_paths(c_paths))

        if not b_top and not c_top:
            return self._no_data(
                "no paths",
                "No span data found",
            )

        union = b_top | c_top
        intersection = b_top & c_top
        jaccard = len(intersection) / len(union) if union else 1.0
        jaccard = round(jaccard, 4)

        # Delta: 1.0 = identical, 0.0 = completely different
        # For classification, treat (jaccard - 1.0) * 100 as the "delta from perfect"
        delta = round((jaccard - 1.0) * 100, 1)

        warnings: list[str] = []
        small = min(len(baseline), len(current))
        if small < _MIN_TRACES:
            warnings.append(
                f"Only {small} traces — path distribution may not be representative, "
                f"recommend ≥{_MIN_TRACES}"
            )

        return Observation(
            name=self.name,
            description=self.description,
            direction=self._classify(delta),
            delta=delta,
            baseline={
                "unique_paths": len(b_top),
                "total_traces": len(baseline),
            },
            current={
                "unique_paths": len(c_top),
                "total_traces": len(current),
            },
            metadata={
                "jaccard": jaccard,
                "shared_paths": len(intersection),
                "union_paths": len(union),
            },
            warnings=warnings,
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        jaccard = result.metadata.get("jaccard")
        if jaccard is None:
            return {}
        return {"path_jaccard": jaccard}

    @staticmethod
    def _extract_paths(traces: list[Trace]) -> list[tuple[str, ...]]:
        """Extract execution path (span name sequence) per trace."""
        paths: list[tuple[str, ...]] = []
        for t in traces:
            names = tuple(s.name for s in t.spans if s.name)
            if names:
                paths.append(names)
        return paths

    @staticmethod
    def _top_paths(paths: list[tuple[str, ...]], k: int = _TOP_K) -> list[tuple[str, ...]]:
        """Return the top-k most frequent paths."""
        counts = Counter(paths)
        return [p for p, _ in counts.most_common(k)]
