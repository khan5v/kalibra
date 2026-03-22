#!/usr/bin/env python3
"""Build multiple demo scenarios and run Kalibra on each.

Each scenario demonstrates a failure mode that aggregate metrics (and
crewai test's LLM-as-judge scores) would miss, but Kalibra's per-task
or per-span breakdown catches.

Scenarios:
  1. REDISTRIBUTION — aggregate success rate identical, but different tasks fail
  2. COST EXPLOSION — success rate improves, but tokens double

Requires traces_raw.jsonl (gitignored) as seed data. To obtain it:
  1. Run generate_traces_live.py with an Anthropic API key, OR
  2. Use the pre-built scenario_*.jsonl files directly (no regeneration needed).

Usage:
    python examples/crewai/scenarios.py
"""

from __future__ import annotations

import copy
import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR / "traces_raw.jsonl"

COMPLEX_TASKS = {"compare_dbs", "design_auth", "debug_guide", "migration_plan", "review_arch"}
SIMPLE_TASKS = {"define_api", "list_benefits", "explain_git", "define_ci", "explain_cache"}

rng = random.Random(42)


# ── Helpers ────────────────────────────────────────────────────────


def new_trace_id() -> str:
    return uuid.uuid4().hex


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def jitter(value: int | float, lo: float = 0.90, hi: float = 1.10) -> int:
    return max(1, int(value * rng.uniform(lo, hi)))


def load_raw_traces() -> dict[str, list[dict]]:
    """Load raw traces grouped by trace_id."""
    spans = [json.loads(line) for line in RAW_PATH.open()]
    traces: dict[str, list[dict]] = {}
    for s in spans:
        tid = s["context"]["trace_id"]
        traces.setdefault(tid, []).append(s)
    return traces


def get_root_attrs(trace_spans: list[dict]) -> dict:
    for s in trace_spans:
        if s["parent_id"] is None:
            return s.get("attributes", {})
    return {}


def strip_span(span: dict) -> dict:
    """Remove large text fields, keep structure + metrics."""
    s = copy.deepcopy(span)
    attrs = s.get("attributes", {})

    for key in list(attrs.keys()):
        if key in ("input.value", "input.mime_type", "llm.system", "llm.invocation_parameters"):
            del attrs[key]
        elif key.startswith("llm.input_messages") or key.startswith("llm.output_messages"):
            del attrs[key]

    if span["span_kind"] == "LLM" and "output.value" in attrs:
        try:
            output = json.loads(attrs["output.value"])
            stripped = {
                "id": output.get("id", "msg_demo"),
                "type": "message",
                "role": "assistant",
                "stop_reason": output.get("stop_reason", "end_turn"),
                "content": [{"type": "text", "text": "[stripped]"}],
            }
            attrs["output.value"] = json.dumps(stripped)
        except (json.JSONDecodeError, TypeError):
            attrs.pop("output.value", None)
    else:
        attrs.pop("output.value", None)
        attrs.pop("output.mime_type", None)

    s["attributes"] = attrs
    s["events"] = []
    return s


def normalize_name(name: str) -> str:
    if name.startswith("Crew_") and ".kickoff" in name:
        return "Crew.kickoff"
    return name


def is_writer_llm(span: dict, all_spans: list[dict]) -> bool:
    if span["span_kind"] != "LLM":
        return False
    for s in all_spans:
        if s["context"]["span_id"] == span["parent_id"]:
            return "Writer" in s["name"]
    return False


def build_trace(
    raw_spans: list[dict],
    variant: str,
    time_offset: timedelta,
    token_overrides: dict | None = None,
    stop_reason_override: str | None = None,
    writer_tokens: int | None = None,
    writer_stop: str | None = None,
    duration_scale: float = 1.0,
) -> list[dict]:
    """Build one variant trace from raw spans.

    token_overrides: {span_kind: (lo, hi)} for random token assignment
    writer_tokens: override writer LLM completion tokens specifically
    writer_stop: override writer LLM stop_reason specifically
    duration_scale: multiply LLM span durations (>1 = slower, <1 = faster)
    """
    new_tid = new_trace_id()
    id_map = {}
    for s in raw_spans:
        id_map[s["context"]["span_id"]] = new_span_id()

    result = []
    for s in raw_spans:
        stripped = strip_span(s)
        stripped["name"] = normalize_name(stripped["name"])

        # Contextualize LLM span names
        if s["span_kind"] == "LLM":
            if is_writer_llm(s, raw_spans):
                stripped["name"] = "LLM (Technical Writer)"
            else:
                stripped["name"] = "LLM (Research Analyst)"

        old_sid = s["context"]["span_id"]
        stripped["context"] = {"trace_id": new_tid, "span_id": id_map[old_sid]}
        if s["parent_id"] and s["parent_id"] in id_map:
            stripped["parent_id"] = id_map[s["parent_id"]]

        # Shift timestamps
        for field in ("start_time", "end_time"):
            if stripped.get(field):
                dt = datetime.fromisoformat(stripped[field])
                stripped[field] = (dt + time_offset).isoformat()

        # Set variant on root
        if s["parent_id"] is None:
            stripped["attributes"]["variant"] = variant

        # Token overrides for LLM spans
        if s["span_kind"] == "LLM":
            attrs = stripped["attributes"]
            is_writer = is_writer_llm(s, raw_spans)

            if is_writer and writer_tokens is not None:
                attrs["llm.token_count.completion"] = writer_tokens
                if writer_stop:
                    try:
                        ov = json.loads(attrs.get("output.value", "{}"))
                        ov["stop_reason"] = writer_stop
                        attrs["output.value"] = json.dumps(ov)
                    except (json.JSONDecodeError, TypeError):
                        pass
            elif token_overrides and s["span_kind"] in token_overrides:
                lo, hi = token_overrides[s["span_kind"]]
                attrs["llm.token_count.completion"] = rng.randint(lo, hi)

            if stop_reason_override and not (is_writer and writer_stop):
                try:
                    ov = json.loads(attrs.get("output.value", "{}"))
                    ov["stop_reason"] = stop_reason_override
                    attrs["output.value"] = json.dumps(ov)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Jitter prompt tokens
            orig = s["attributes"].get("llm.token_count.prompt", 500)
            attrs["llm.token_count.prompt"] = jitter(orig)

            # Scale LLM span duration (e.g., more tokens → slower)
            if duration_scale != 1.0 and stripped.get("start_time") and stripped.get("end_time"):
                start = datetime.fromisoformat(stripped["start_time"])
                end = datetime.fromisoformat(stripped["end_time"])
                dur = (end - start).total_seconds()
                new_dur = dur * duration_scale * rng.uniform(0.9, 1.1)
                stripped["end_time"] = (start + timedelta(seconds=max(0.5, new_dur))).isoformat()

        result.append(stripped)

    # Fix parent span end_times: if any child ends after its parent,
    # extend the parent. Walk bottom-up by sorting on end_time descending.
    if duration_scale != 1.0:
        end_times: dict[str, datetime] = {}
        for s in result:
            if s.get("end_time"):
                end_times[s["context"]["span_id"]] = datetime.fromisoformat(s["end_time"])

        for s in result:
            parent_id = s.get("parent_id")
            if not parent_id or parent_id not in end_times:
                continue
            child_end = end_times.get(s["context"]["span_id"])
            if child_end and child_end > end_times[parent_id]:
                end_times[parent_id] = child_end

        for s in result:
            sid = s["context"]["span_id"]
            if sid in end_times and s.get("end_time"):
                s["end_time"] = end_times[sid].isoformat()

    return result


# ── Scenario builders ──────────────────────────────────────────────


def build_scenario_redistribution(raw_traces: dict) -> tuple[list[dict], str]:
    """Scenario 1: Same aggregate success rate, different tasks fail.

    Baseline: compare_dbs and design_auth always fail (truncated).
              Everything else succeeds.
    Current:  compare_dbs and design_auth now succeed.
              But migration_plan and review_arch now fail.
    Aggregate: 80% → 80%. Identical.
    Kalibra: 2 regressions + 2 improvements. Gate fails.
    crewai test: "Score 7.5 → 7.4. Within noise."
    """
    BASELINE_FAIL = {"compare_dbs", "design_auth"}
    CURRENT_FAIL = {"migration_plan", "review_arch"}

    all_spans = []
    for tid, raw_spans in raw_traces.items():
        attrs = get_root_attrs(raw_spans)
        task_id = attrs.get("task.id", "")

        # Baseline
        fails = task_id in BASELINE_FAIL
        bl = build_trace(
            raw_spans, "baseline", timedelta(0),
            token_overrides={"LLM": (400, 1000)},
            stop_reason_override="end_turn",
            writer_tokens=rng.randint(130, 160) if fails else None,
            writer_stop="max_tokens" if fails else None,
        )
        all_spans.extend(bl)

        # Current
        fails = task_id in CURRENT_FAIL
        cur = build_trace(
            raw_spans, "current", timedelta(hours=4),
            token_overrides={"LLM": (400, 1000)},
            stop_reason_override="end_turn",
            writer_tokens=rng.randint(130, 160) if fails else None,
            writer_stop="max_tokens" if fails else None,
        )
        all_spans.extend(cur)

    config = """# Scenario 1: Redistribution
# Aggregate success rate is identical (80% → 80%).
# But WHICH tasks fail shifted completely.

sources:
  baseline:
    path: {path}
    where: [variant == baseline]
  current:
    path: {path}
    where: [variant == current]

baseline: baseline
current: current
fields:
  task_id: task.id
require:
  - regressions <= 0
"""
    return all_spans, config


def build_scenario_cost_explosion(raw_traces: dict) -> tuple[list[dict], str]:
    """Scenario 2: Success rate improves, but token usage doubles.

    Baseline: 75% success (complex tasks fail 50% of the time).
              Normal token usage (~800 per LLM call).
    Current:  90% success (complex tasks now mostly succeed).
              But token usage doubled (~1600 per LLM call) — model is
              doing chain-of-thought internally, producing verbose output.
    Aggregate: Looks like pure improvement!
    Kalibra: token_delta_pct gate fails (tokens +100%).
    crewai test: "Score 8.0 → 8.5. Improved!"
    """
    all_spans = []
    for tid, raw_spans in raw_traces.items():
        attrs = get_root_attrs(raw_spans)
        task_id = attrs.get("task.id", "")
        run_id = attrs.get("task.run_id", 0)
        is_complex = task_id in COMPLEX_TASKS

        # Baseline: complex tasks fail 50% (run_id=0 fails)
        bl_writer_tok = None
        bl_writer_stop = None
        if is_complex and run_id == 0:
            bl_writer_tok = rng.randint(130, 160)
            bl_writer_stop = "max_tokens"

        bl = build_trace(
            raw_spans, "baseline", timedelta(0),
            token_overrides={"LLM": (600, 1000)},
            stop_reason_override="end_turn",
            writer_tokens=bl_writer_tok,
            writer_stop=bl_writer_stop,
        )
        all_spans.extend(bl)

        # Current: almost all succeed, but tokens doubled
        cur_writer_tok = None
        cur_writer_stop = None
        # 2 complex tasks still fail on run_id=0 (keeps current at 85%)
        if task_id in ("review_arch", "debug_guide") and run_id == 0:
            cur_writer_tok = rng.randint(130, 160)
            cur_writer_stop = "max_tokens"

        # Scale duration for LLM spans proportional to token increase
        cur = build_trace(
            raw_spans, "current", timedelta(hours=4),
            token_overrides={"LLM": (1200, 2000)},
            stop_reason_override="end_turn",
            writer_tokens=cur_writer_tok,
            writer_stop=cur_writer_stop,
            duration_scale=1.6,  # more tokens → slower
        )
        all_spans.extend(cur)

    config = """# Scenario 2: Cost explosion
# Success improved (75% → 97.5%). Looks great!
# But tokens doubled. The model is being verbose.

sources:
  baseline:
    path: {path}
    where: [variant == baseline]
  current:
    path: {path}
    where: [variant == current]

baseline: baseline
current: current
fields:
  task_id: task.id
require:
  - token_delta_pct <= 20
  - regressions <= 0
"""
    return all_spans, config



# ── Main ───────────────────────────────────────────────────────────


@dataclass
class Scenario:
    name: str
    description: str
    crewai_test_sees: str
    kalibra_catches: str
    builder: callable
    traces_path: str
    config_path: str


SCENARIOS = [
    Scenario(
        name="redistribution",
        description="Same success rate, different tasks fail",
        crewai_test_sees="Score 7.5 → 7.4. Average quality unchanged. Ship it.",
        kalibra_catches="2 tasks regressed, 2 improved. Failure distribution shifted entirely.",
        builder=build_scenario_redistribution,
        traces_path="scenario_redistribution.jsonl",
        config_path="scenario_redistribution.yml",
    ),
    Scenario(
        name="cost_explosion",
        description="Success improved, but tokens doubled",
        crewai_test_sees="Score 8.0 → 8.5. Quality improved. Ship it.",
        kalibra_catches="Tokens +35%. A third more cost for a statistically insignificant improvement.",
        builder=build_scenario_cost_explosion,
        traces_path="scenario_cost_explosion.jsonl",
        config_path="scenario_cost_explosion.yml",
    ),
]


def main():
    if not RAW_PATH.exists():
        print(f"Error: {RAW_PATH} not found.")
        return

    raw_traces = load_raw_traces()
    print(f"Loaded {sum(len(v) for v in raw_traces.values())} raw spans across {len(raw_traces)} traces\n")

    for scenario in SCENARIOS:
        print(f"{'='*60}")
        print(f"Scenario: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"{'='*60}")

        # Build traces
        spans, config_template = scenario.builder(raw_traces)
        traces_path = SCRIPT_DIR / scenario.traces_path
        config_path = SCRIPT_DIR / scenario.config_path

        with traces_path.open("w") as f:
            for span in spans:
                f.write(json.dumps(span) + "\n")

        config_text = config_template.replace("{path}", f"examples/crewai/{scenario.traces_path}")
        config_path.write_text(config_text)

        size_kb = traces_path.stat().st_size / 1024
        print(f"  Wrote {len(spans)} spans → {traces_path.name} ({size_kb:.0f} KB)")

        # Run Kalibra
        import subprocess
        result = subprocess.run(
            ["kalibra", "compare", "--config", str(config_path), "-v"],
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr

        # Extract key lines (strip ANSI codes for matching)
        import re
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        for line in output.split("\n"):
            clean = ansi_re.sub("", line).strip()
            if any(kw in clean for kw in [
                "Success rate", "Token usage", "Per trace", "Per span",
                "Quality gates", "[ OK ]", "[FAIL]", "[SKIP]",
                "PASSED", "FAILED", "Gates",
                "succeeded:", "regressed", "improved",
            ]):
                print(f"    {clean}")

        print()
        print(f"  crewai test would say: {scenario.crewai_test_sees}")
        print(f"  Kalibra catches:       {scenario.kalibra_catches}")
        print()


if __name__ == "__main__":
    main()
