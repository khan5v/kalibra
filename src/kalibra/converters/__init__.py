"""Trace converters — transform various trace formats into a common representation."""

from pathlib import Path

from kalibra.converters.base import ReadableSpan, Trace


def load_traces(path: str, trace_format: str = "auto", progress: bool = False) -> list[Trace]:
    """Load traces from a file or directory, auto-detecting format if needed.

    Supported formats:
    - ``json``  — kalibra JSONL (from ``kalibra pull``)
    - ``auto``  — detect from file extension
    """
    p = Path(path)

    if trace_format == "auto":
        trace_format = "json"

    if trace_format == "json":
        from kalibra.converters.generic import load_json_traces
        return load_json_traces(p)

    raise ValueError(f"Unknown trace format: {trace_format!r}. Supported: json")


__all__ = ["load_traces", "Trace", "ReadableSpan"]
