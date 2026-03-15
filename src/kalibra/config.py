"""Kalibra configuration — unified config from kalibra.yml."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

_AUTO_PLUGIN_FILE = "kalibra_metrics.py"
_LEGACY_CONFIG_FILE = "config/compare.yml"
_LEGACY_SOURCES_DIR = "config/sources"


# ── Population config (baseline / current) ────────────────────────────────────

@dataclass
class PopulationConfig:
    """Where to get traces for one side of the comparison.

    Either ``path`` (local JSONL) or ``source`` + ``project`` (remote pull).
    """

    # Local file path — if set, no remote pull needed.
    path: str | None = None

    # Remote source — langfuse, langsmith, braintrust.
    source: str | None = None
    project: str | None = None
    since: str = "7d"
    limit: int = 5000
    tags: list[str] = dc_field(default_factory=list)
    session: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> PopulationConfig | None:
        if data is None or not isinstance(data, dict):
            return None
        raw_tags = data.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        return cls(
            path=data.get("path"),
            source=data.get("source"),
            project=data.get("project"),
            since=str(data.get("since", "7d")),
            limit=int(data.get("limit", 5000)),
            tags=[str(t) for t in raw_tags],
            session=data.get("session"),
        )


# ── Field mappings ────────────────────────────────────────────────────────────

@dataclass
class FieldsConfig:
    """Maps trace/span fields to what Kalibra metrics expect."""

    trace_id: str | None = None
    task_id: str | None = None
    outcome: str | None = None
    cost: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> FieldsConfig:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            trace_id=data.get("trace_id"),
            task_id=data.get("task_id"),
            outcome=data.get("outcome"),
            cost=data.get("cost"),
        )


# ── Outcome / cost overrides (legacy, still used by connectors) ──────────────

@dataclass
class OutcomeConfig:
    field: str | None = None
    success: list[str] = dc_field(default_factory=lambda: ["success"])
    failure: list[str] = dc_field(default_factory=lambda: ["failure", "error", "failed"])

    @classmethod
    def from_dict(cls, data) -> OutcomeConfig | None:
        if data is None or not isinstance(data, dict):
            return None
        return cls(
            field=data.get("field"),
            success=data.get("success", ["success"]),
            failure=data.get("failure", ["failure", "error", "failed"]),
        )


@dataclass
class CostConfig:
    attr: str | None = None

    @classmethod
    def from_dict(cls, data) -> CostConfig | None:
        if data is None or not isinstance(data, dict):
            return None
        return cls(attr=data.get("attr"))


@dataclass
class SourceConfig:
    """Legacy named source config (from config/sources/*.yml)."""

    source: str
    project: str = ""
    path: str | None = None
    since: str = "7d"
    limit: int = 5000
    tags: list[str] = dc_field(default_factory=list)
    session: str | None = None
    outcome: OutcomeConfig | None = None
    cost: CostConfig | None = None

    @classmethod
    def from_dict(cls, data: dict) -> SourceConfig:
        source = data.get("source", "")
        if source == "jsonl" and "path" not in data:
            raise ValueError("JSONL source requires a 'path' field")
        if source != "jsonl" and "project" not in data:
            raise ValueError(f"Source '{source}' requires a 'project' field")
        raw_tags = data.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        return cls(
            source=source,
            project=data.get("project", ""),
            path=data.get("path"),
            since=str(data.get("since", "7d")),
            limit=int(data.get("limit", 5000)),
            tags=[str(t) for t in raw_tags],
            session=data.get("session") or None,
            outcome=OutcomeConfig.from_dict(data.get("outcome")),
            cost=CostConfig.from_dict(data.get("cost")),
        )


def load_sources(directory: str | None = None) -> dict[str, SourceConfig]:
    """Load legacy named sources from config/sources/*.yml."""
    import yaml

    d = Path(directory) if directory else Path(_LEGACY_SOURCES_DIR)
    if not d.is_dir():
        return {}
    result: dict[str, SourceConfig] = {}
    for p in sorted(d.glob("*.yml")):
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        for name, cfg in data.items():
            if isinstance(cfg, dict):
                result[name] = SourceConfig.from_dict(cfg)
    return result


def _resolve_population(
    value: dict | str | None,
    sources: dict[str, PopulationConfig],
) -> PopulationConfig | None:
    """Resolve a baseline/current value — inline dict or string source reference."""
    if value is None:
        return None
    if isinstance(value, str):
        # String reference to a named source.
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

    def get_source(self, name: str) -> PopulationConfig | None:
        """Look up a named source."""
        return self.sources.get(name)

    @classmethod
    def load(cls, path: str | None = None) -> CompareConfig:
        """Load config from a file.

        Tries ``path`` first, then falls back to the legacy ``config/compare.yml``.
        Returns an empty config if no file is found.
        """
        import yaml

        if path:
            p = Path(path)
            if p.exists():
                with p.open() as f:
                    data = yaml.safe_load(f) or {}
                return cls.from_dict(data)

        # Legacy fallback.
        legacy = Path(_LEGACY_CONFIG_FILE)
        if legacy.exists():
            with legacy.open() as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)

        return cls()


# ── Metric resolution ─────────────────────────────────────────────────────────

def resolve_metrics(config: CompareConfig, defaults: list) -> list:
    """Return the active metric list for a comparison run."""
    auto_extras = _load_auto_plugin()

    if config.metrics is None:
        return list(defaults) + auto_extras

    by_name = {m.name: m for m in defaults}
    active: list = []
    for entry in config.metrics:
        if entry in by_name:
            active.append(by_name[entry])
        elif "." in entry:
            active.extend(_load_module_metrics(entry))
    active.extend(auto_extras)
    return active


def _load_auto_plugin() -> list:
    auto = Path(_AUTO_PLUGIN_FILE)
    if not auto.exists():
        return []
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        mod = importlib.import_module(auto.stem)
    except ImportError as e:
        raise ImportError(
            f"Auto-plugin {_AUTO_PLUGIN_FILE!r} could not be imported: {e}"
        ) from e
    return list(getattr(mod, "METRICS", []))


def _load_module_metrics(dotted: str) -> list:
    try:
        mod = importlib.import_module(dotted)
    except ImportError as e:
        raise ImportError(
            f"Plugin module {dotted!r} could not be imported: {e}"
        ) from e
    return list(getattr(mod, "METRICS", []))
