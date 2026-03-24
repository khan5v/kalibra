"""Shared utilities for trace format loaders.

These functions are used by multiple format implementations and by the
main loader module. Centralised here to avoid circular imports.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS


_SUCCESS_VALUES = {"success", "true", "1", "pass", "passed", "resolved"}
_FAILURE_VALUES = {"failure", "false", "0", "fail", "failed", "error"}


def _safe_float(val) -> float | None:
    """Convert to float, rejecting NaN and Infinity."""
    if val is None:
        return None
    try:
        res = float(val)
        if math.isnan(res) or math.isinf(res):
            return None
        return res
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    """Convert to int, rejecting NaN and Infinity and handling float overflow."""
    if val is None:
        return None
    try:
        res = float(val)
        if math.isnan(res) or math.isinf(res):
            return None
        return int(res)
    except (ValueError, TypeError, OverflowError):
        return None


def _iso_to_ns(ts: str) -> int:
    """Parse ISO 8601 timestamp to nanoseconds.

    Handles: Z suffix, +HH:MM offsets, -HH:MM offsets, fractional seconds.
    """
    if not ts:
        return 0
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1e9)
    except (ValueError, TypeError):
        return 0


def _parse_ts_to_ns(val) -> int:
    """Parse a timestamp to nanoseconds."""
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return 0
        if val < 1e12:
            return int(val * 1e9)
        if val < 1e15:
            return int(val * 1e6)
        if val < 1e18:
            return int(val * 1e3)
        try:
            return int(val)
        except OverflowError:
            return 0
    if isinstance(val, str):
        return _iso_to_ns(val)
    return 0



def _flatten_dict(d: dict, prefix: str, out: dict) -> None:
    """Flatten a nested dict into dot-notation keys.

    Lists are excluded — trace attributes often contain large message
    arrays (prompts, responses) that would bloat metadata. Scalar values
    only. If list-membership filtering is needed later, this is where
    to add it.
    """
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, key, out)
        elif v is not None and not isinstance(v, list):
            out[key] = v


def _group_by_trace_id(raw_spans: list[dict]) -> dict[str, list[dict]]:
    """Group flat OTel spans by context.trace_id.

    Skips non-dict items and items without a valid context.trace_id.
    Shared by all OTel-based loaders (OpenInference, OTel GenAI).
    """
    groups: dict[str, list[dict]] = {}
    for item in raw_spans:
        if not isinstance(item, dict):
            continue
        context = item.get("context")
        if not isinstance(context, dict):
            continue
        trace_id = str(context.get("trace_id") or "")
        if not trace_id:
            continue
        groups.setdefault(trace_id, []).append(item)
    return groups


def _find_root_span(spans: list[dict]) -> dict | None:
    """Find the root span (no parent_id) from a list of spans in one trace.

    If multiple roots exist, picks the one with the earliest start_time.
    Spans missing start_time are deprioritized (not chosen over spans with timestamps).
    """
    root = None
    for s in spans:
        pid = s.get("parent_id")
        if pid is None or pid == "":
            if root is None:
                root = s
            else:
                s_time = s.get("start_time") or ""
                root_time = root.get("start_time") or ""
                # Only prefer s over root if s has a real timestamp that's earlier,
                # or if root has no timestamp but s does.
                if s_time and (not root_time or s_time < root_time):
                    root = s
    return root


def _resolve_attr(attrs: dict, dot_path: str):
    """Resolve an attribute by dot-path, handling both flat and nested layouts.

    Tries flat key first (attrs["gen_ai.usage.input_tokens"]),
    then nested traversal (attrs["gen_ai"]["usage"]["input_tokens"]).
    """
    if dot_path in attrs:
        return attrs[dot_path]
    parts = dot_path.split(".")
    current = attrs
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


# OTel status codes: 0=UNSET, 1=OK, 2=ERROR.
_OTEL_STATUS_MAP = {0: "", 1: "OK", 2: "ERROR"}


def _normalize_status(raw) -> str:
    """Normalize OTel status code to uppercase string."""
    if raw is None:
        return ""
    if isinstance(raw, int):
        return _OTEL_STATUS_MAP.get(raw, "")
    return str(raw).upper()


def _auto_parse_json_strings(obj):
    """Recursively parse JSON strings embedded in field values."""
    if isinstance(obj, dict):
        return {k: _auto_parse_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_auto_parse_json_strings(v) for v in obj]
    if isinstance(obj, str) and obj and obj.lstrip()[0:1] in ("{", "["):
        try:
            return _auto_parse_json_strings(json.loads(obj))
        except (json.JSONDecodeError, ValueError):
            pass
    return obj


def _classify_outcome(val) -> str | None:
    """Classify a raw value as success/failure outcome."""
    if isinstance(val, bool):
        return OUTCOME_SUCCESS if val else OUTCOME_FAILURE
    val_str = str(val).lower().strip()
    if val_str in _SUCCESS_VALUES:
        return OUTCOME_SUCCESS
    if val_str in _FAILURE_VALUES:
        return OUTCOME_FAILURE
    return None


def _resolve_dot_path(obj: dict, path: str):
    """Resolve a dot-notation path against a dict.

    Tries two strategies:
    1. Direct flat key: obj["agent_cost.total_cost"] (flattened attributes)
    2. Nested traversal: obj["agent_cost"]["total_cost"] (nested dicts)
    """
    if path in obj:
        return obj[path]
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _error(
    path: Path, line_no: int, msg: str, *, hint: str | None = None,
) -> None:
    """Raise a formatted parse error."""
    parts = [f"\n  {path}:{line_no} — {msg}"]
    if hint:
        for line in hint.split("\n"):
            parts.append(f"  {line}")
    raise ValueError("\n".join(parts))
