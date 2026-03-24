"""Trace loader registry — pluggable loaders for different trace formats."""

from __future__ import annotations

from pathlib import Path

from kalibra.model import Trace


class TraceLoader:
    """Base class for trace format loaders.

    Each subclass handles one trace format (OpenInference, OTel GenAI, etc.).
    Subclasses must implement ``detect`` and ``load``.
    """

    name: str = ""

    def detect(self, item: dict) -> bool:
        """Return True if this dict looks like a span/trace in our format."""
        raise NotImplementedError

    def load(self, path: Path) -> list[Trace]:
        """Load traces from a JSONL file in this format."""
        raise NotImplementedError


# Backward compat alias — external code may reference TraceFormat.
TraceFormat = TraceLoader
