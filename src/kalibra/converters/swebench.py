"""Converter for SWE-bench / SWE-agent trajectory files.

Supports two formats:
1. GitHub demo format (.traj files): {"trajectory": [{"action", "observation", ...}], "info": {...}}
2. Nebius/HuggingFace parquet format: chat messages with role=system/user/ai, text field
"""

import json
import re
from pathlib import Path
from hashlib import md5

import pandas as pd

from kalibra.converters.base import Trace, make_span

# ── Action classification ──

_ACTION_PATTERNS = [
    (r"^(create|touch)\b", "create_file"),
    (r"^(edit|insert|replace|sed|patch)\b", "edit_file"),
    (r"^(open|cat|head|tail|less|more|goto|scroll_up|scroll_down)\b", "read_file"),
    (r"^(find|grep|search|ls|find_file|search_dir|search_file|rg|ag)\b", "search"),
    (r"^(python|python3)\b", "run_python"),
    (r"^(pip|conda|apt|brew|npm)\b", "install_deps"),
    (r"^(git|cd)\b", "navigate"),
    (r"^submit\b", "submit"),
    (r"^(test|pytest|make\s+test|tox|unittest|nosetests)\b", "run_tests"),
]


def _classify_action(raw_action: str) -> str:
    """Classify a raw action string into a high-level action type."""
    action = raw_action.strip().split("\n")[0]  # first line only
    for pattern, action_type in _ACTION_PATTERNS:
        if re.match(pattern, action, re.IGNORECASE):
            return action_type
    return "other"


def _extract_command(ai_text: str) -> str:
    """Extract the command from an AI turn's text (chat format).

    SWE-agent AI turns contain DISCUSSION + ```command``` blocks.
    """
    if not ai_text:
        return ""
    # Look for code block
    match = re.search(r"```\s*\n?(.*?)\n?```", ai_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: last line often is the command
    lines = [l.strip() for l in ai_text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


# ── Loader dispatch ──

def load_swebench_traces(path: Path, progress: bool = False) -> list[Trace]:
    """Load SWE-bench trajectories from .traj files or parquet directory."""
    if path.is_dir():
        parquet_files = list(path.glob("*.parquet"))
        if parquet_files:
            return _load_from_parquet(path, progress=progress)
        traj_files = list(path.glob("**/*.traj"))
        if traj_files:
            return _load_from_traj_files(traj_files, progress=progress)
    elif path.suffix == ".parquet":
        return _load_from_parquet(path, progress=progress)
    elif path.suffix == ".traj":
        return _load_from_traj_files([path])

    raise FileNotFoundError(f"No trajectory data found in {path}")


# ── Parquet loader (Nebius/HuggingFace format) ──

def _load_from_parquet(path: Path, progress: bool = False) -> list[Trace]:
    """Load all parquet files into Traces."""
    if path.is_file():
        df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(path)  # pandas handles directory of parquets

    total = len(df)
    if progress:
        import click
        click.echo(f"  Processing {total:,} rows from parquet...")

    traces = []
    for i, (_, row) in enumerate(df.iterrows()):
        trace = _parse_chat_trajectory(row)
        if trace:
            traces.append(trace)
        if progress and (i + 1) % 500 == 0:
            import click
            click.echo(f"  Converted {i + 1:,} / {total:,} ({(i + 1) * 100 // total}%)")

    return traces


def _parse_chat_trajectory(row) -> Trace | None:
    """Parse a Nebius-format row (chat messages) into a Trace."""
    traj = row.get("trajectory")
    if traj is None or len(traj) == 0:
        return None

    instance_id = row.get("instance_id", "unknown")
    model_name = row.get("model_name", "")
    # Use index from row name if available to ensure uniqueness
    row_idx = getattr(row, 'name', '') if hasattr(row, 'name') else ''
    trace_id = f"{instance_id}__{model_name}__{row_idx}".replace("/", "_")

    # Extract ai turns — each ai turn is one agent step
    spans = []
    t = 0.0
    step_idx = 0

    for i, msg in enumerate(traj):
        role = msg.get("role", "")
        text = msg.get("text") or ""

        if role == "ai":
            command = _extract_command(text)
            action_type = _classify_action(command) if command else "think"

            # Get the next user message as observation
            obs = ""
            if i + 1 < len(traj) and traj[i + 1].get("role") == "user":
                obs = traj[i + 1].get("text") or ""

            duration = max(0.5, len(obs) / 1000)
            obs_lower = obs.lower()
            is_error = any(kw in obs_lower for kw in (
                "traceback", "exception", "syntaxerror", "nameerror",
                "typeerror", "valueerror", "indexerror", "keyerror",
                "filenotfounderror", "importerror",
            ))

            span_id = md5(f"{trace_id}:{step_idx}".encode()).hexdigest()[:16]
            spans.append(make_span(
                name=action_type,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=None,
                start_ns=int(t * 1e9),
                end_ns=int((t + duration) * 1e9),
                attributes={
                    "command": command[:200],
                    "thought": text[:200],
                    "observation_length": len(obs),
                },
                error=is_error,
            ))
            t += duration
            step_idx += 1

    if not spans:
        return None

    # Use 'target' field (bool) for outcome — this is whether the issue was actually resolved
    target = row.get("target")
    if target is True:
        outcome = "success"
    elif target is False:
        outcome = "failure"
    else:
        outcome = None

    return Trace(
        trace_id=trace_id,
        spans=spans,
        outcome=outcome,
        metadata={
            "source": "swebench_nebius",
            "instance_id": instance_id,
            "model_name": model_name,
            "exit_status": row.get("exit_status", ""),
        },
    )


# ── .traj file loader (GitHub demo format) ──

def _load_from_traj_files(files: list[Path], progress: bool = False) -> list[Trace]:
    """Load .traj JSON files into Traces."""
    total = len(files)
    traces = []
    for i, f in enumerate(sorted(files)):
        trace = _parse_traj_file(f)
        if trace:
            traces.append(trace)
        if progress and (i + 1) % 500 == 0:
            import click
            click.echo(f"  Converted {i + 1:,} / {total:,} ({(i + 1) * 100 // total}%)")
    return traces


def _parse_traj_file(path: Path) -> Trace | None:
    """Parse a .traj file — handles both action/observation and chat formats."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict):
        return None

    steps = data.get("trajectory", [])
    if not steps:
        return None

    trace_id = path.stem
    info = data.get("info", {})

    # Detect format: action/observation vs chat messages
    first = steps[0]
    if "action" in first:
        return _parse_action_format(trace_id, steps, info)
    elif "role" in first:
        # Chat format — wrap as a pseudo-row and reuse parquet parser
        row = {
            "trajectory": steps,
            "instance_id": info.get("instance_id", trace_id),
            "model_name": info.get("model_name", ""),
            "exit_status": info.get("exit_status", ""),
            "target": info.get("resolved"),
        }
        return _parse_chat_trajectory(row)

    return None


def _parse_action_format(trace_id: str, steps: list[dict], info: dict) -> Trace:
    """Parse GitHub demo format with action/observation fields."""
    model_stats = info.get("model_stats", {})
    spans = []
    t = 0.0

    for i, step in enumerate(steps):
        raw_action = step.get("action", "")
        action_type = _classify_action(raw_action)
        span_id = md5(f"{trace_id}:{i}".encode()).hexdigest()[:16]

        obs = step.get("observation", "")
        duration = max(0.5, len(obs) / 1000)
        obs_lower = obs.lower()
        is_error = any(kw in obs_lower for kw in (
            "traceback", "exception", "syntaxerror", "nameerror",
            "typeerror", "valueerror", "indexerror", "keyerror",
        ))

        attrs: dict = {
            "command": raw_action[:200],
            "thought": step.get("thought", "")[:200],
            "observation_length": len(obs),
        }
        if model_stats.get("model"):
            from kalibra.converters.base import GEN_AI_MODEL, GEN_AI_INPUT_TOKENS, GEN_AI_OUTPUT_TOKENS, AF_COST
            attrs[GEN_AI_MODEL]         = model_stats["model"]
            attrs[GEN_AI_INPUT_TOKENS]  = model_stats.get("tokens_sent", 0) // max(len(steps), 1)
            attrs[GEN_AI_OUTPUT_TOKENS] = model_stats.get("tokens_received", 0) // max(len(steps), 1)
            attrs[AF_COST]              = model_stats.get("total_cost", 0.0) / max(len(steps), 1)
        spans.append(make_span(
            name=action_type,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            start_ns=int(t * 1e9),
            end_ns=int((t + duration) * 1e9),
            attributes=attrs,
            error=is_error,
        ))
        t += duration

    exit_status = info.get("exit_status", "")
    if exit_status == "submitted":
        outcome = "success"
    elif exit_status in ("failed", "error", "timeout"):
        outcome = "failure"
    else:
        outcome = None

    return Trace(
        trace_id=trace_id,
        spans=spans,
        outcome=outcome,
        metadata={
            "source": "swebench",
            "file": trace_id,
            "exit_status": exit_status,
        },
    )
