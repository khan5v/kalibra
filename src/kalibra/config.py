"""Compare configuration — metric selection, thresholds, and plugin loading."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

# Auto-discovered in cwd if present (no config entry needed).
_AUTO_PLUGIN_FILE = "kalibra_metrics.py"
_CONFIG_FILE = "config/compare.yml"
_SOURCES_DIR = "config/sources"
_CACHE_DIR = "cached_sources"


@dataclass
class OutcomeConfig:
    """Override outcome detection for a source.

    Looks up ``field`` in trace metadata and matches the value against
    ``success`` / ``failure`` keyword lists.  If the field is missing or
    the value doesn't match either list, the connector's default heuristic
    is kept.

    Example YAML::

        outcome:
          field: metadata.evaluation_result
          success: [pass, resolved]
          failure: [fail, timeout]
    """

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
    """Override which span attribute is used for cost.

    Example YAML::

        cost:
          attr: custom.cost_usd
    """

    attr: str | None = None

    @classmethod
    def from_dict(cls, data) -> CostConfig | None:
        if data is None or not isinstance(data, dict):
            return None
        return cls(attr=data.get("attr"))


@dataclass
class SourceConfig:
    """Named pull configuration, referenced as ``@name`` in compare/pull commands.

    Attributes:
        source:  Connector to use — ``"langfuse"``, ``"langsmith"``, or ``"jsonl"``.
        project: Project name or ID in the trace store (not used for jsonl).
        path:    Local file path (required for ``source: jsonl``).
        since:   Time window to fetch — e.g. ``"7d"``, ``"24h"``, ``"2026-01-01"``.
        limit:   Max traces to fetch (default 5000).
        outcome: Override outcome detection (see ``OutcomeConfig``).
        cost:    Override cost computation (see ``CostConfig``).
    """

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
    def from_dict(cls, data: dict) -> "SourceConfig":
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
    """Load named pull configs from all ``*.yml`` files in ``config/sources/``.

    Files are merged in alphabetical order; later files override earlier ones
    for the same name. Returns an empty dict if the directory does not exist.
    """
    import yaml

    d = Path(directory) if directory else Path(_SOURCES_DIR)
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


@dataclass
class CompareConfig:
    """Controls which metrics run and what threshold gates are applied.

    Attributes:
        metrics:           Metric names or dotted module paths to load. ``None`` runs all
                           DEFAULT_METRICS plus any auto-discovered plugins.
                           Built-in names: ``"success_rate"``, ``"cost"``, etc.
                           Dotted paths: ``"myproject.custom_metrics"`` — module is
                           imported and its ``METRICS: list[ComparisonMetric]`` is run.
        require:           Threshold expressions evaluated after comparison.
                           Each is a string like ``"success_rate_delta >= -2"``.
                           The compare command exits with code 1 if any expression fails.
        noise_thresholds:  Per-metric noise threshold overrides. Keyed by metric name.
                           E.g. ``{"success_rate": 1.0, "cost": 5.0}``.
                           Overrides the class-level default for that metric.
    """

    metrics: list[str] | None = None
    require: list[str] = dc_field(default_factory=list)
    noise_thresholds: dict[str, float] = dc_field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "CompareConfig":
        raw_noise = data.get("noise_thresholds") or {}
        return cls(
            metrics=data.get("metrics") or None,
            require=[r for r in (data.get("require") or []) if r],
            noise_thresholds={k: float(v) for k, v in raw_noise.items()},
        )

    @classmethod
    def load(cls, path: str | None = None) -> "CompareConfig":
        """Load from ``config/compare.yml`` (or *path*).

        Returns an empty (all-defaults) config if the file does not exist.
        """
        import yaml

        p = Path(path) if path else Path(_CONFIG_FILE)
        if not p.exists():
            return cls()
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


def resolve_metrics(config: CompareConfig, defaults: list) -> list:
    """Return the active metric list for a comparison run.

    Resolution order:
    1. If ``config.metrics`` is ``None``, all ``defaults`` are used.
    2. Otherwise each entry in ``config.metrics`` is resolved:
       - Known built-in name → the corresponding default metric instance.
       - Dotted path (e.g. ``"myproject.custom"```) → module is imported and
         its ``METRICS: list[ComparisonMetric]`` is appended.
       - Unknown string with no dot → silently skipped.
    3. ``kalibra_metrics.py`` in the current directory is always auto-loaded
       if it exists (zero-config extension point, like pytest's conftest.py).
    """
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
        # unknown bare name: skip silently
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
        raise ImportError(f"Auto-plugin {_AUTO_PLUGIN_FILE!r} could not be imported: {e}") from e
    return list(getattr(mod, "METRICS", []))


def _load_module_metrics(dotted: str) -> list:
    try:
        mod = importlib.import_module(dotted)
    except ImportError as e:
        raise ImportError(f"Plugin module {dotted!r} could not be imported: {e}") from e
    return list(getattr(mod, "METRICS", []))
