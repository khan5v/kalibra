"""Trace converters — transform various trace formats into a common representation."""

from pathlib import Path

from kalibra.converters.base import ReadableSpan, Trace


def load_traces(path: str, trace_format: str = "auto", progress: bool = False) -> list[Trace]:
    """Load traces from a file or directory, auto-detecting format if needed.

    Supported formats:
    - ``swebench`` — SWE-bench ``.traj`` files or parquet
    - ``json``     — kalibra JSONL (from ``kalibra pull``)
    - ``auto``     — detect from file/directory structure
    """
    p = Path(path)

    if trace_format == "auto":
        trace_format = _detect_format(p)

    if trace_format == "swebench":
        from kalibra.converters.swebench import load_swebench_traces
        return load_swebench_traces(p, progress=progress)

    if trace_format == "json":
        from kalibra.converters.generic import load_json_traces
        return load_json_traces(p)

    raise ValueError(f"Unknown trace format: {trace_format!r}. Supported: swebench, json")


def _detect_format(path: Path) -> str:
    if path.suffix in (".jsonl", ".json") and path.is_file():
        return "json"
    # SWE-bench: directory with .traj files or parquet
    if path.is_dir() or path.suffix in (".traj", ".parquet"):
        return "swebench"
    return "json"


__all__ = ["load_traces", "Trace", "ReadableSpan"]
