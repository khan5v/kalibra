#!/usr/bin/env python3
"""Generate synthetic agent traces and upload to Langfuse or LangSmith.

  # Langfuse (default):
  python3 scripts/synth_traces.py --mode baseline
  python3 scripts/synth_traces.py --mode current

  # LangSmith:
  python3 scripts/synth_traces.py --mode baseline --target langsmith
  python3 scripts/synth_traces.py --mode current  --target langsmith

  # Custom tags and count:
  python3 scripts/synth_traces.py --mode baseline --target langsmith --count 50 --tags myteam,v2

  # Then pull + compare:
  agentflow pull @baseline && agentflow pull @current
  agentflow compare --baseline @baseline --current @current
"""

import argparse
import base64
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

# ── Task pool ────────────────────────────────────────────────────────────────
# Stable task IDs shared between baseline and current so per_task matching works.

TASK_POOL = [
    "django__django-14787",
    "django__django-15061",
    "django__django-15498",
    "flask__flask-4935",
    "flask__flask-5012",
    "requests__requests-6028",
    "requests__requests-6179",
    "sympy__sympy-21847",
    "sympy__sympy-22714",
    "sympy__sympy-23262",
    "scikit-learn__sklearn-25638",
    "scikit-learn__sklearn-25747",
    "matplotlib__matplotlib-25311",
    "matplotlib__matplotlib-25442",
    "pandas__pandas-52847",
    "pandas__pandas-53112",
    "sphinx__sphinx-11445",
    "sphinx__sphinx-11502",
    "astropy__astropy-14182",
    "pytest__pytest-11143",
    "pytest__pytest-11178",
    "black__black-3820",
    "mypy__mypy-16141",
    "pylint__pylint-8929",
    "tornado__tornado-6901",
    "httpx__httpx-2879",
    "pydantic__pydantic-8504",
    "fastapi__fastapi-10178",
    "celery__celery-8226",
    "sqlalchemy__sqla-10371",
]

# ── Execution path templates ─────────────────────────────────────────────────
# Modeled from real SWE-bench / Nebius traces:
#   - Solved traces:   median 13 steps, edit-heavy, fewer navigation loops
#   - Unsolved traces: median 19 steps, navigation-heavy, deep retry loops
# Step names match real SWE-agent actions: edit, open, python, search_dir,
# search_file, find_file, create, ls, grep, scroll_down, goto, submit.

BASELINE_PATHS = {
    # 13 steps — classic retry-heavy: navigate, fail, navigate more, re-edit
    "retry_loop": [
        "find_file", "open", "scroll_down", "edit", "python",
        "search_dir", "open", "edit", "python", "edit", "python",
        "edit", "python",
    ],
    # 19 steps — deep search with lots of navigation (unsolved pattern)
    "deep_search": [
        "find_file", "open", "scroll_down", "scroll_down", "goto",
        "search_dir", "search_file", "open", "scroll_down", "edit",
        "python", "ls", "open", "edit", "python", "edit", "python",
        "grep", "open",
    ],
    # 24 steps — brute force: many edit-test cycles, reproduction scripts
    "brute_force": [
        "find_file", "open", "scroll_down", "create", "python",
        "search_dir", "open", "edit", "python", "search_file",
        "open", "scroll_down", "goto", "edit", "python",
        "ls", "open", "edit", "python", "edit", "python",
        "edit", "python", "edit",
    ],
    # 8 steps — quick fail: minimal exploration, early exit
    "quick_fail": [
        "find_file", "open", "scroll_down", "edit", "python",
        "ls", "open", "edit",
    ],
    # 15 steps — navigation heavy: lots of reading, little editing
    "nav_heavy": [
        "find_file", "open", "scroll_down", "scroll_down", "goto",
        "search_dir", "ls", "open", "scroll_down", "search_file",
        "open", "scroll_down", "edit", "python", "edit",
    ],
    # 10 steps — minimally viable solve attempt
    "minimal": [
        "find_file", "open", "scroll_down", "edit", "python",
        "edit", "python", "grep", "open", "edit",
    ],
}

CURRENT_PATHS = {
    # 11 steps — targeted: find, read, fix, test, submit
    "targeted": [
        "find_file", "open", "scroll_down", "edit", "python",
        "edit", "python", "submit",
        "open", "edit", "submit",
    ],
    # 8 steps — fast success: minimal steps, clean solve
    "fast_success": [
        "find_file", "open", "edit", "python",
        "edit", "python", "submit",
        "open",
    ],
    # 14 steps — one retry: read, edit, fail test, re-edit, pass
    "one_retry": [
        "find_file", "open", "scroll_down", "search_file", "open",
        "edit", "python", "edit", "python", "create", "python",
        "edit", "python", "submit",
    ],
    # 10 steps — read-then-fix: thorough reading before editing
    "read_then_fix": [
        "find_file", "open", "scroll_down", "scroll_down", "grep",
        "open", "edit", "python", "edit", "submit",
    ],
    # 6 steps — quick fail: tried and gave up
    "quick_fail": [
        "find_file", "open", "scroll_down", "edit", "python", "edit",
    ],
    # 16 steps — deep solve with reproduction script
    "repro_and_fix": [
        "find_file", "open", "scroll_down", "create", "python",
        "search_dir", "open", "scroll_down", "edit", "python",
        "edit", "python", "grep", "open", "edit", "submit",
    ],
}

# ── Per-step profiles ────────────────────────────────────────────────────────
# Names match real SWE-agent actions. Profiles model Claude API calls where
# applicable. Navigation/tool-only steps have no LLM cost.

STEP_PROFILES = {
    # — LLM-backed actions (agent reasons + issues command) —
    "edit": {
        "model": "claude-sonnet-4-20250514",
        "input_tokens": (2000, 8000),    # growing context window
        "output_tokens": (200, 1200),    # patch output
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "duration": (3.0, 12.0),
    },
    "create": {
        "model": "claude-sonnet-4-20250514",
        "input_tokens": (1500, 5000),
        "output_tokens": (100, 800),
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "duration": (2.0, 8.0),
    },
    "search_dir": {
        "model": "claude-sonnet-4-20250514",
        "input_tokens": (1000, 4000),
        "output_tokens": (50, 300),
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "duration": (2.0, 8.0),
    },
    "search_file": {
        "model": "claude-sonnet-4-20250514",
        "input_tokens": (1000, 3500),
        "output_tokens": (50, 300),
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "duration": (2.0, 6.0),
    },
    "find_file": {
        "model": "claude-sonnet-4-20250514",
        "input_tokens": (800, 2500),
        "output_tokens": (50, 200),
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "duration": (1.0, 4.0),
    },
    "grep": {
        "model": "claude-sonnet-4-20250514",
        "input_tokens": (800, 3000),
        "output_tokens": (50, 250),
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "duration": (1.0, 5.0),
    },
    "submit": {
        "model": "claude-haiku-4-5-20251001",
        "input_tokens": (200, 800),
        "output_tokens": (50, 200),
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.005,
        "duration": (1.0, 3.0),
    },

    # — Navigation / tool-only actions (no LLM cost) —
    "open": {
        "model": None,
        "input_tokens": (0, 0),
        "output_tokens": (0, 0),
        "cost_per_1k_input": 0,
        "cost_per_1k_output": 0,
        "duration": (0.5, 2.0),
    },
    "scroll_down": {
        "model": None,
        "input_tokens": (0, 0),
        "output_tokens": (0, 0),
        "cost_per_1k_input": 0,
        "cost_per_1k_output": 0,
        "duration": (0.2, 1.0),
    },
    "goto": {
        "model": None,
        "input_tokens": (0, 0),
        "output_tokens": (0, 0),
        "cost_per_1k_input": 0,
        "cost_per_1k_output": 0,
        "duration": (0.2, 1.0),
    },
    "ls": {
        "model": None,
        "input_tokens": (0, 0),
        "output_tokens": (0, 0),
        "cost_per_1k_input": 0,
        "cost_per_1k_output": 0,
        "duration": (0.2, 0.5),
    },
    "python": {
        "model": None,  # running tests/scripts — no LLM call
        "input_tokens": (0, 0),
        "output_tokens": (0, 0),
        "cost_per_1k_input": 0,
        "cost_per_1k_output": 0,
        "duration": (3.0, 30.0),
    },
}

# ── Mode configs ─────────────────────────────────────────────────────────────
# Success rates calibrated from real Nebius data:
#   llama-70b: 17.6%, llama-405b: 47.5%
#   Baseline models weaker agent (~20%), current models improved agent (~38%).
# Failure multiplier: real unsolved traces are ~2x longer than solved.

MODES = {
    "baseline": {
        "n_traces": 200,
        "success_rate": 0.20,
        "tool_error_rate": 0.10,
        "path_weights": {
            "retry_loop":   0.20,
            "deep_search":  0.20,
            "brute_force":  0.15,
            "quick_fail":   0.20,
            "nav_heavy":    0.15,
            "minimal":      0.10,
        },
        "failure_cost_multiplier": 1.8,
        "failure_duration_multiplier": 1.8,
        "model_override": None,
        "seed": 42,
    },
    "current": {
        "n_traces": 200,
        "success_rate": 0.38,
        "tool_error_rate": 0.04,
        "path_weights": {
            "targeted":       0.25,
            "fast_success":   0.20,
            "one_retry":      0.20,
            "read_then_fix":  0.15,
            "quick_fail":     0.10,
            "repro_and_fix":  0.10,
        },
        "failure_cost_multiplier": 1.5,
        "failure_duration_multiplier": 1.5,
        "model_override": None,
        "seed": 99,
    },
}


# ── Shared helpers ──────────────────────────────────────────────────────────

def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    names = list(weights.keys())
    w = [weights[n] for n in names]
    return rng.choices(names, weights=w, k=1)[0]


def _retry(fn, description: str = "request", max_retries: int = 5):
    """Call fn with exponential backoff on errors."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"{description} failed after {max_retries} retries: {exc}") from exc
            print(f"  Error ({description}, attempt {attempt + 1}/{max_retries}) — retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


def generate_trace_data(
    mode: str,
    cfg: dict,
    paths: dict[str, list[str]],
    task_id: str,
    trace_idx: int,
    rng: random.Random,
) -> dict:
    """Generate trace data (target-agnostic). Returns a dict describing one trace."""
    model_tag = "sonnet" if mode == "current" else "baseline"
    trace_name = f"{task_id}__{model_tag}__{trace_idx}"

    success = rng.random() < cfg["success_rate"]
    path_name = _weighted_choice(rng, cfg["path_weights"])
    path = paths[path_name]

    cost_mult = 1.0 if success else cfg.get("failure_cost_multiplier", 1.0)
    dur_mult = 1.0 if success else cfg.get("failure_duration_multiplier", 1.0)

    steps = []
    for step_name in path:
        profile = STEP_PROFILES[step_name]
        is_error = rng.random() < cfg["tool_error_rate"]

        in_tok = rng.randint(*profile["input_tokens"]) if profile["input_tokens"][1] > 0 else 0
        out_tok = rng.randint(*profile["output_tokens"]) if profile["output_tokens"][1] > 0 else 0

        cost = (
            in_tok / 1000 * profile["cost_per_1k_input"]
            + out_tok / 1000 * profile["cost_per_1k_output"]
        ) * cost_mult

        duration = rng.uniform(*profile["duration"]) * dur_mult
        model = cfg.get("model_override") or profile["model"]

        steps.append({
            "name": step_name,
            "model": model,
            "run_type": profile.get("run_type", "llm" if model else "tool"),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost": round(cost, 6),
            "duration": duration,
            "is_error": is_error,
        })

    return {
        "trace_name": trace_name,
        "task_id": task_id,
        "mode": mode,
        "success": success,
        "path_name": path_name,
        "steps": steps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Langfuse backend
# ═══════════════════════════════════════════════════════════════════════════════

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BATCH_SIZE = 25


def _langfuse_auth_header() -> str:
    token = base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
    ).decode()
    return f"Basic {token}"


def _langfuse_post_batch(events: list[dict]) -> None:
    body = json.dumps({"batch": events}).encode()
    max_retries = 5
    delay = 1.0
    for attempt in range(max_retries):
        req = urllib.request.Request(
            f"{LANGFUSE_HOST}/api/public/ingestion",
            data=body,
            headers={
                "Authorization": _langfuse_auth_header(),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in (200, 201, 207):
                    return
                if resp.status >= 500:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Langfuse ingestion returned HTTP {resp.status} after {max_retries} retries")
                    print(f"  Server error {resp.status} (attempt {attempt + 1}/{max_retries}) — retrying in {delay:.0f}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"Langfuse ingestion returned HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Langfuse ingestion failed (HTTP {exc.code}) after {max_retries} retries")
                wait = delay
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        wait = max(float(retry_after), delay)
                    print(f"  Rate limited — waiting {wait:.0f}s...")
                else:
                    print(f"  Server error {exc.code} (attempt {attempt + 1}/{max_retries}) — retrying in {wait:.0f}s...")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            raise
        except (urllib.error.URLError, OSError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Langfuse ingestion failed after {max_retries} retries: {exc}")
            print(f"  Connection error (attempt {attempt + 1}/{max_retries}) — retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _ts_iso(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def trace_to_langfuse_events(trace_data: dict, tags: list[str] | None = None) -> list[dict]:
    """Convert generic trace data → Langfuse ingestion events."""
    mode = trace_data["mode"]
    trace_id = trace_data["trace_name"]
    now = time.time()

    events = [
        {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": _now_iso(),
            "body": {
                "id": trace_id,
                "name": "agent-run",
                "sessionId": f"agentflow-synth-{mode}",
                "tags": tags or ["agentflow", mode],
                "output": {"outcome": "success" if trace_data["success"] else "failure"},
                "metadata": {
                    "seed_mode": mode,
                    "task_id": trace_data["task_id"],
                    "path": trace_data["path_name"],
                },
                "timestamp": _ts_iso(now),
            },
        }
    ]

    t = now
    for step in trace_data["steps"]:
        obs_body: dict = {
            "id": str(uuid.uuid4()),
            "traceId": trace_id,
            "type": "GENERATION" if step["model"] else "SPAN",
            "name": step["name"],
            "startTime": _ts_iso(t),
            "endTime": _ts_iso(t + step["duration"]),
            "level": "ERROR" if step["is_error"] else "DEFAULT",
            "calculatedTotalCost": step["cost"],
        }
        if step["model"]:
            obs_body["model"] = step["model"]
        if step["input_tokens"] > 0 or step["output_tokens"] > 0:
            obs_body["usage"] = {
                "input": step["input_tokens"],
                "output": step["output_tokens"],
                "unit": "TOKENS",
            }
        if step["is_error"]:
            obs_body["statusMessage"] = f"Tool call failed: {step['name']} returned non-zero exit code"

        events.append({
            "id": str(uuid.uuid4()),
            "type": "observation-create",
            "timestamp": _now_iso(),
            "body": obs_body,
        })
        t += step["duration"]

    return events


def send_langfuse(mode: str, cfg: dict, paths: dict, rng: random.Random,
                   count: int | None = None, tags: list[str] | None = None) -> None:
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        raise SystemExit(
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables.\n"
            "  export LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
            "  export LANGFUSE_SECRET_KEY=sk-lf-..."
        )

    n = count or cfg["n_traces"]
    tags = tags or ["agentflow", mode]
    print(f"Sending {n} {mode} traces to Langfuse ({LANGFUSE_HOST})...")
    print(f"  Tags: {tags}")
    print()

    pending: list[dict] = []
    sent = 0

    for i in range(n):
        task_id = TASK_POOL[i % len(TASK_POOL)]
        trace_data = generate_trace_data(mode, cfg, paths, task_id, i, rng)
        pending.extend(trace_to_langfuse_events(trace_data, tags=tags))

        if len(pending) >= LANGFUSE_BATCH_SIZE or i == n - 1:
            _langfuse_post_batch(pending)
            sent += len([e for e in pending if e["type"] == "trace-create"])
            pending = []
            print(f"  {sent}/{n} traces sent...")

    print(f"\nDone. {n} {mode} traces uploaded to Langfuse.")


# ═══════════════════════════════════════════════════════════════════════════════
# LangSmith backend
# ═══════════════════════════════════════════════════════════════════════════════

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_API_URL = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agentflow-synth")

def send_langsmith_trace(client, trace_data: dict, project: str, tags: list[str] | None = None) -> None:
    """Upload one trace to LangSmith using the SDK."""
    mode = trace_data["mode"]
    run_tags = tags or ["agentflow", mode]
    now = datetime.now(timezone.utc)

    root_id = uuid.uuid4()
    root_error = None if trace_data["success"] else "Agent failed to solve the task"

    _retry(
        lambda: client.create_run(
            name="agent-run",
            run_type="chain",
            id=root_id,
            project_name=project,
            start_time=now,
            inputs={"task_id": trace_data["task_id"], "path": trace_data["path_name"]},
            tags=run_tags,
            extra={"metadata": {
                "seed_mode": mode,
                "task_id": trace_data["task_id"],
                "trace_name": trace_data["trace_name"],
            }},
        ),
        "create root run",
    )

    t = now
    for step in trace_data["steps"]:
        child_id = uuid.uuid4()
        end_time = t + timedelta(seconds=step["duration"])

        extra: dict = {"metadata": {
            "agentflow_cost": step["cost"],
            "agentflow_input_tokens": step["input_tokens"],
            "agentflow_output_tokens": step["output_tokens"],
        }}
        if step["model"]:
            extra["invocation_params"] = {"model_name": step["model"]}

        child_kwargs: dict = dict(
            name=step["name"],
            run_type=step.get("run_type", "llm" if step["model"] else "tool"),
            id=child_id,
            parent_run_id=root_id,
            project_name=project,
            start_time=t,
            inputs={"step": step["name"]},
            tags=run_tags,
            extra=extra,
        )

        _retry(lambda kw=child_kwargs: client.create_run(**kw), f"create {step['name']}")

        update_kwargs: dict = dict(
            run_id=child_id,
            end_time=end_time,
            outputs={"result": f"{step['name']} completed"},
        )
        if step["input_tokens"] > 0 or step["output_tokens"] > 0:
            update_kwargs["usage_metadata"] = {
                "input_tokens": step["input_tokens"],
                "output_tokens": step["output_tokens"],
                "total_tokens": step["input_tokens"] + step["output_tokens"],
            }
        if step["is_error"]:
            update_kwargs["error"] = f"Tool call failed: {step['name']} returned non-zero exit code"

        _retry(lambda kw=update_kwargs: client.update_run(**kw), f"update {step['name']}")
        t = end_time

    _retry(
        lambda: client.update_run(
            run_id=root_id,
            end_time=t,
            outputs={"outcome": "success" if trace_data["success"] else "failure"},
            error=root_error,
        ),
        "update root run",
    )


def send_langsmith(mode: str, cfg: dict, paths: dict, rng: random.Random, project: str,
                    count: int | None = None, tags: list[str] | None = None) -> None:
    if not LANGSMITH_API_KEY:
        raise SystemExit(
            "Set LANGSMITH_API_KEY environment variable.\n"
            "  export LANGSMITH_API_KEY=lsv2_pt_..."
        )

    from langsmith import Client
    client = Client(api_key=LANGSMITH_API_KEY, api_url=LANGSMITH_API_URL)

    n = count or cfg["n_traces"]
    tags = tags or ["agentflow", mode]
    print(f"Sending {n} {mode} traces to LangSmith ({LANGSMITH_API_URL})...")
    print(f"  Project: {project}")
    print(f"  Tags: {tags}")
    print()

    for i in range(n):
        task_id = TASK_POOL[i % len(TASK_POOL)]
        trace_data = generate_trace_data(mode, cfg, paths, task_id, i, rng)
        send_langsmith_trace(client, trace_data, project, tags=tags)
        if (i + 1) % 5 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} traces sent...")

    print(f"\nDone. {n} {mode} traces uploaded to LangSmith (project: {project}).")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def send_traces(mode: str, target: str, project: str | None = None,
                 count: int | None = None, tags: list[str] | None = None) -> None:
    cfg = MODES[mode]
    paths = BASELINE_PATHS if mode == "baseline" else CURRENT_PATHS
    rng = random.Random(cfg["seed"])

    n = count or cfg["n_traces"]
    print(f"  Traces: {n}")
    print(f"  Success rate: {cfg['success_rate']:.0%}")
    print(f"  Tool error rate: {cfg['tool_error_rate']:.0%}")
    print(f"  Task pool: {len(TASK_POOL)} tasks")
    print()

    if target == "langfuse":
        send_langfuse(mode, cfg, paths, rng, count=count, tags=tags)
        source_name = "baseline" if mode == "baseline" else "current"
    else:
        ls_project = project or LANGSMITH_PROJECT
        send_langsmith(mode, cfg, paths, rng, ls_project, count=count, tags=tags)
        source_name = "ls-baseline" if mode == "baseline" else "ls-current"

    print()
    if mode == "baseline":
        print("Next steps:")
        print(f"  agentflow pull @{source_name}")
        print(f"  python3 scripts/synth_traces.py --mode current --target {target}")
    else:
        baseline_name = source_name.replace("current", "baseline")
        print("Next steps:")
        print(f"  agentflow pull @{source_name}")
        print(f"  agentflow compare --baseline @{baseline_name} --current @{source_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic agent traces and upload to Langfuse or LangSmith.",
    )
    parser.add_argument(
        "--mode", choices=["baseline", "current"], required=True,
        help="Which batch to send. Run baseline first, pull, then current.",
    )
    parser.add_argument(
        "--target", choices=["langfuse", "langsmith"], default="langfuse",
        help="Where to send traces (default: langfuse).",
    )
    parser.add_argument(
        "--project", default=None,
        help="LangSmith project name (default: LANGSMITH_PROJECT env or 'agentflow-synth').",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Number of traces to send (overrides mode default).",
    )
    parser.add_argument(
        "--tags", default=None,
        help="Comma-separated tags to attach to traces (default: agentflow,<mode>).",
    )
    args = parser.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    send_traces(args.mode, args.target, args.project, count=args.count, tags=tags)


if __name__ == "__main__":
    main()
