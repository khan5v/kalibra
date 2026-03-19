"""Kalibra configuration — unified config from kalibra.yml."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

CONFIG_FILENAME = "kalibra.yml"


# ── Field mappings ────────────────────────────────────────────────────────────

@dataclass
class FieldsConfig:
    """Maps trace/span fields to what Kalibra metrics expect."""

    trace_id: str | None = None
    task_id: str | None = None
    outcome: str | None = None
    cost: str | None = None
    input_tokens: str | None = None
    output_tokens: str | None = None
    duration: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> FieldsConfig:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            trace_id=data.get("trace_id"),
            task_id=data.get("task_id"),
            outcome=data.get("outcome"),
            cost=data.get("cost"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            duration=data.get("duration"),
        )

    def merge(self, override: FieldsConfig | None) -> FieldsConfig:
        """Return a new FieldsConfig with non-None values from override winning."""
        if override is None:
            return self
        return FieldsConfig(
            trace_id=override.trace_id or self.trace_id,
            task_id=override.task_id or self.task_id,
            outcome=override.outcome or self.outcome,
            cost=override.cost or self.cost,
            input_tokens=override.input_tokens or self.input_tokens,
            output_tokens=override.output_tokens or self.output_tokens,
            duration=override.duration or self.duration,
        )


# ── Population config (baseline / current) ────────────────────────────────────

@dataclass
class PopulationConfig:
    """Where to get traces for one side of the comparison.

    ``path`` points to a local JSONL file.
    ``fields`` optionally overrides global field mappings for this source.
    """

    path: str | None = None
    fields: FieldsConfig | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> PopulationConfig | None:
        if data is None or not isinstance(data, dict):
            return None
        fields_data = data.get("fields")
        return cls(
            path=data.get("path"),
            fields=FieldsConfig.from_dict(fields_data) if fields_data else None,
        )


def _resolve_population(
    value: dict | str | None,
    sources: dict[str, PopulationConfig],
) -> PopulationConfig | None:
    """Resolve a baseline/current value — inline dict or string source reference."""
    if value is None:
        return None
    if isinstance(value, str):
        if value in sources:
            return sources[value]
        raise ValueError(
            f"Unknown source {value!r} in baseline/current. "
            f"Available sources: {list(sources)}"
        )
    if isinstance(value, dict):
        return PopulationConfig.from_dict(value)
    return None


# ── Main config ───────────────────────────────────────────────────────────────

@dataclass
class CompareConfig:
    """Unified Kalibra configuration — parsed from kalibra.yml.

    Holds everything: named sources, population selection, metrics, gates,
    and field mappings.
    """

    # Named sources — reusable definitions referenced by baseline/current.
    sources: dict[str, PopulationConfig] = dc_field(default_factory=dict)

    # Population configs — inline or string reference to a named source.
    baseline: PopulationConfig | None = None
    current: PopulationConfig | None = None

    # Field mappings
    fields: FieldsConfig = dc_field(default_factory=FieldsConfig)

    # Metric selection (None = all built-ins)
    metrics: list[str] | None = None

    # Quality gates
    require: list[str] = dc_field(default_factory=list)

    # Per-metric noise threshold overrides
    noise_thresholds: dict[str, float] = dc_field(default_factory=dict)

    @property
    def task_id(self) -> str | None:
        return self.fields.task_id

    @task_id.setter
    def task_id(self, value: str | None) -> None:
        self.fields.task_id = value

    def get_source(self, name: str) -> PopulationConfig | None:
        """Look up a named source."""
        return self.sources.get(name)

    @classmethod
    def from_dict(cls, data: dict) -> CompareConfig:
        raw_noise = data.get("noise_thresholds") or {}
        fields_data = data.get("fields") or {}
        if not fields_data.get("task_id") and data.get("task_id"):
            fields_data["task_id"] = data["task_id"]

        # Parse named sources.
        sources: dict[str, PopulationConfig] = {}
        for name, src_data in (data.get("sources") or {}).items():
            if isinstance(src_data, dict):
                pop = PopulationConfig.from_dict(src_data)
                if pop:
                    sources[name] = pop

        # Parse baseline/current — can be inline dict or string reference.
        baseline = _resolve_population(data.get("baseline"), sources)
        current = _resolve_population(data.get("current"), sources)

        return cls(
            sources=sources,
            baseline=baseline,
            current=current,
            fields=FieldsConfig.from_dict(fields_data),
            metrics=data.get("metrics") or None,
            require=[r for r in (data.get("require") or []) if r],
            noise_thresholds={k: float(v) for k, v in raw_noise.items()},
        )

    @classmethod
    def load(cls, path: str | None = None) -> CompareConfig:
        """Load config from a file or discover kalibra.yml by walking up from CWD."""
        import yaml

        if path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with p.open() as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)

        # Walk-up discovery.
        discovered = find_config()
        if discovered:
            with discovered.open() as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)

        return cls()


def find_config() -> Path | None:
    """Walk up from CWD looking for kalibra.yml."""
    current = Path.cwd().resolve()
    while True:
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent
