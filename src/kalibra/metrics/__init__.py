"""Metrics package — self-contained comparison metrics.

Each metric is one file with one class. The class implements:
- compare(baseline_traces, current_traces) → Observation

The metric has full access to Trace and Span objects. It extracts what it needs,
computes statistics, classifies direction, and returns structured data.
No formatting — renderers handle display.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from kalibra.model import Trace

# ── Direction ─────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    UPGRADE = "upgrade"
    SAME = "same"
    DEGRADATION = "degradation"
    INCONCLUSIVE = "inconclusive"
    NA = "n/a"


# ── Observation (pure data, no formatting) ────────────────────────────────────

@dataclass
class Observation:
    """Result of comparing one metric across baseline and current.

    Contains only structured data. No formatted strings.
    Renderers read this to produce terminal/markdown/JSON output.
    """

    name: str
    description: str
    direction: Direction = Direction.NA
    delta: float | None = None
    baseline: dict = field(default_factory=dict)
    current: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ── Base class ────────────────────────────────────────────────────────────────

class ComparisonMetric(ABC):
    """Base class for all comparison metrics.

    Subclass this to add a new metric. Implement compare() to produce
    an Observation from two populations of traces.

    The metric has full access to Trace objects — it extracts values,
    computes statistics, and classifies direction. All in one method.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    noise_threshold: float = 0.5  # can be overridden per-instance via engine config
    higher_is_better: ClassVar[bool] = True

    #: {field_name: description} — for --metrics display and --require validation.
    _fields: ClassVar[dict[str, str]] = {}

    @abstractmethod
    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        """Compare two trace populations and return structured result.

        The metric owns the full pipeline:
        1. Extract values from traces/spans
        2. Compute statistics (median, CI, significance tests)
        3. Classify direction
        4. Return Observation with structured data (no formatting)
        """

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        """Return gateable threshold fields from this metric's result."""
        return {}

    @classmethod
    def threshold_field_names(cls) -> dict[str, str]:
        """Return {field_name: description} for all threshold fields."""
        return cls._fields

    # ── Common helpers ────────────────────────────────────────────────────

    def _classify(
        self,
        delta: float | None,
        ci: tuple[float, float] | None = None,
    ) -> Direction:
        """Classify direction from delta and optional bootstrap CI.

        CI-based significance: if the confidence interval on the percentage
        delta includes zero, the change is not significant — even if the
        point estimate is large. This directly tests "did the median shift?"
        which is the claim we're making.

        Noise threshold: if the delta is real but trivially small (e.g. 0.3%
        cost change), classify as SAME regardless of significance.
        """
        if delta is None:
            return Direction.NA
        if ci is not None and ci[0] <= 0 <= ci[1]:
            return Direction.SAME
        if abs(delta) <= self.noise_threshold:
            return Direction.SAME
        if (delta > 0) == self.higher_is_better:
            return Direction.UPGRADE
        return Direction.DEGRADATION

    def _no_data(self, msg: str, warning: str) -> Observation:
        """Return an n/a observation for missing data."""
        return Observation(
            name=self.name,
            description=self.description,
            direction=Direction.NA,
            warnings=[warning],
            metadata={"no_data_reason": msg},
        )
