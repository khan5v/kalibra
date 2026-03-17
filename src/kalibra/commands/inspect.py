"""Inspect command — show data coverage, available fields, and config suggestions."""

from __future__ import annotations

from pathlib import Path

import click

from kalibra import display


def run_inspect(
    path: str,
    config_path: str | None,
    *,
    trace_id_field: str | None = None,
    outcome: str | None = None,
    cost_field: str | None = None,
    task_id: str | None = None,
    input_tokens_field: str | None = None,
    output_tokens_field: str | None = None,
    duration_field: str | None = None,
    suggest: bool = False,
) -> None:
    """Execute the inspect command."""
    from kalibra.config import CompareConfig, find_config
    from kalibra.engine import resolve_metrics
    from kalibra.loader import load_traces
    from kalibra.model import OUTCOME_FAILURE, OUTCOME_SUCCESS

    # Load config — needed for field mappings.
    if config_path:
        if not Path(config_path).exists():
            raise click.UsageError(f"Config file not found: {config_path}")
    else:
        discovered = find_config()
        if discovered:
            config_path = str(discovered)
    config = CompareConfig.load(config_path)

    # CLI flag overrides.
    if trace_id_field:
        config.fields.trace_id = trace_id_field
    if outcome:
        config.fields.outcome = outcome
    if cost_field:
        config.fields.cost = cost_field
    if input_tokens_field:
        config.fields.input_tokens = input_tokens_field
    if output_tokens_field:
        config.fields.output_tokens = output_tokens_field
    if duration_field:
        config.fields.duration = duration_field
    if task_id:
        config.fields.task_id = task_id

    try:
        traces = load_traces(path, fields=config.fields)
    except ValueError as exc:
        display.load_error(path, str(exc))
        raise SystemExit(1) from None

    n = len(traces)
    if n == 0:
        click.echo(f"\n  {path}: 0 traces (empty file)")
        return

    n_spans = sum(len(t.spans) for t in traces)

    # Determine active metrics.
    active = resolve_metrics(config.metrics)
    active_names = {m.name for m in active}

    # ── Data coverage ─────────────────────────────────────────────────────
    has_outcome = sum(
        1 for t in traces if t.outcome in (OUTCOME_SUCCESS, OUTCOME_FAILURE)
    )
    has_cost = sum(1 for t in traces if t.total_cost is not None and t.total_cost > 0)
    has_tokens = sum(1 for t in traces if t.total_tokens is not None and t.total_tokens > 0)
    has_duration = sum(1 for t in traces if t.duration is not None and t.duration > 0)

    # Task ID: check if traces can be grouped into tasks.
    # A task_id is "extractable" if either metadata has a task_id field,
    # or trace IDs share prefixes (e.g. "task-1__model__0", "task-1__model__1").
    task_ids = set()
    has_metadata_task_id = False
    for t in traces:
        mid = t.metadata.get("task_id")
        if mid is not None:
            task_ids.add(str(mid))
            has_metadata_task_id = True
        else:
            # Try prefix extraction (strip __model__index suffix).
            parts = t.trace_id.split("__")
            if len(parts) >= 3 and parts[-1].isdigit():
                task_ids.add("__".join(parts[:-2]))
            else:
                task_ids.add(t.trace_id)
    task_id_extractable = has_metadata_task_id or len(task_ids) < n

    # ── Metadata keys ─────────────────────────────────────────────────────
    meta_keys: dict[str, int] = {}
    meta_unique: dict[str, set] = {}
    for t in traces:
        for k, v in (t.metadata or {}).items():
            meta_keys[k] = meta_keys.get(k, 0) + 1
            meta_unique.setdefault(k, set()).add(str(v)[:100])

    # ── Span attribute keys ───────────────────────────────────────────────
    attr_keys: dict[str, int] = {}
    for t in traces:
        for s in t.spans:
            for k in (s.attributes or {}):
                attr_keys[k] = attr_keys.get(k, 0) + 1

    # ── Render ────────────────────────────────────────────────────────────
    b = display.bar()
    d = display.dot()

    click.echo()
    click.echo(f"  {click.style('Kalibra Inspect', bold=True)}")
    click.echo(f"  {b}")
    click.echo(f"  {click.style('Source', dim=True)}    {path}")
    click.echo(f"  {click.style('Traces', dim=True)}    {n:,}")
    click.echo(f"  {click.style('Spans', dim=True)}     {n_spans:,} ({n_spans / n:.1f} avg/trace)")
    click.echo()

    # ── Metric readiness ──────────────────────────────────────────────────
    click.echo(f"  {click.style('Metric readiness', bold=True)}")
    if config_path or config.metrics:
        click.echo(f"  {click.style('(based on active config)', dim=True)}")
    click.echo()

    needs_outcome = active_names & {
        "success_rate", "trace_breakdown", "token_efficiency", "cost_quality",
    }
    needs_cost = active_names & {"cost", "cost_quality"}
    needs_tokens = active_names & {"token_usage", "token_efficiency"}
    needs_duration = active_names & {"duration"}
    needs_task_id = active_names & {"trace_breakdown"}

    def _coverage_line(label: str, count: int, total: int, needed_by: set[str]):
        if not needed_by:
            return
        ok = click.style("✓", fg="green") if count > 0 else click.style("✗", fg="yellow")
        count_str = f"{count}/{total}"
        metric_list = ", ".join(sorted(needed_by))
        click.echo(
            f"    {ok} {click.style(label, bold=True):<24s}"
            f"{count_str:>8s} traces"
            f"  {click.style(f'({metric_list})', dim=True)}"
        )

    _coverage_line("Outcome", has_outcome, n, needs_outcome)
    _coverage_line("Cost", has_cost, n, needs_cost)
    _coverage_line("Tokens", has_tokens, n, needs_tokens)
    _coverage_line("Duration", has_duration, n, needs_duration)

    if needs_task_id:
        if task_id_extractable:
            _coverage_line("Task ID", len(task_ids), n, {"trace_breakdown"})
        else:
            ok = click.style("✗", fg="yellow")
            click.echo(
                f"    {ok} {click.style('Task ID', bold=True):<24s}"
                f"{'0/'+str(n):>8s} traces"
                f"  {click.style('(trace_breakdown)', dim=True)}"
            )
            hint = "each trace has a unique ID — set fields.task_id in config"
            click.echo(f"      {click.style(hint, dim=True)}")

    click.echo()

    # ── Trace fields ─────────────────────────────────────────────────────
    click.echo(f"  {click.style('Trace fields', bold=True)}")
    click.echo()

    # Standard fields — always show what's populated.
    std_trace = [
        ("outcome", has_outcome),
        ("cost", has_cost),
        ("tokens", has_tokens),
        ("duration", has_duration),
    ]
    for label, count in std_trace:
        if count > 0:
            padding = max(1, 34 - len(label))
            click.echo(
                f"    {click.style(label, fg='white')}"
                f"  {d * padding}  "
                f"{click.style(f'{count}/{n} traces', dim=True)}"
            )

    # Extra metadata fields beyond the standard ones.
    if meta_keys:
        for key in sorted(meta_keys, key=lambda k: -meta_keys[k]):
            count = meta_keys[key]
            n_unique = len(meta_unique.get(key, set()))
            unique_str = f"{n_unique} unique" if n_unique > 1 else "1 value"
            padding = max(1, 34 - len(key))
            click.echo(
                f"    {click.style(key, fg='white')}"
                f"  {d * padding}  "
                f"{click.style(f'{count}/{n} traces, {unique_str}', dim=True)}"
            )

    click.echo()

    # ── Span fields ──────────────────────────────────────────────────────
    if n_spans > 0:
        click.echo(f"  {click.style('Span fields', bold=True)}")
        click.echo()

        # Standard span fields — show what's populated.
        has_span_cost = sum(1 for t in traces for s in t.spans if s.cost > 0)
        has_span_tokens = sum(
            1 for t in traces for s in t.spans if s.total_tokens > 0
        )
        has_span_model = sum(
            1 for t in traces for s in t.spans if s.model
        )
        has_span_error = sum(1 for t in traces for s in t.spans if s.error)

        std_span = [
            ("name", sum(1 for t in traces for s in t.spans if s.name)),
            ("cost", has_span_cost),
            ("tokens", has_span_tokens),
            ("model", has_span_model),
            ("error", has_span_error),
        ]
        for label, count in std_span:
            if count > 0:
                padding = max(1, 34 - len(label))
                click.echo(
                    f"    {click.style(label, fg='white')}"
                    f"  {d * padding}  "
                    f"{click.style(f'{count}/{n_spans} spans', dim=True)}"
                )

        # Extra attributes beyond standard fields.
        if attr_keys:
            for key in sorted(attr_keys, key=lambda k: -attr_keys[k]):
                count = attr_keys[key]
                padding = max(1, 34 - len(key))
                click.echo(
                    f"    {click.style(key, fg='white')}"
                    f"  {d * padding}  "
                    f"{click.style(f'{count}/{n_spans} spans', dim=True)}"
                )

        click.echo()

    click.echo()

    # ── Config suggestions ────────────────────────────────────────────────
    suggestions = []
    if needs_task_id and not task_id_extractable:
        suggestions.append("task_id: <metadata key>")
    if needs_outcome and has_outcome == 0:
        suggestions.append("outcome: <metadata key>")
    if needs_cost and has_cost == 0:
        suggestions.append("cost: <span attribute>")

    if suggestions and not suggest:
        click.echo(f"  {b}")
        click.echo(
            f"  {click.style('To fix missing fields, add to', dim=True)} "
            f"{click.style('kalibra.yml', fg='cyan')}"
            f"{click.style(' or use CLI flags:', dim=True)}"
        )
        click.echo(click.style("    fields:", dim=True))
        for s in suggestions:
            click.echo(click.style(f"      {s}", dim=True))
        click.echo()
    else:
        click.echo(f"  {b}")
        click.echo(f"  {click.style('All active metrics have data.', fg='green')}")
        click.echo()

    # ── Suggest field mappings ────────────────────────────────────────────
    if suggest:
        # Collect fields with unique value counts for smarter ranking.
        field_uniques: dict[str, set] = {}
        for t in traces:
            for k, v in t.metadata.items():
                field_uniques.setdefault(k, set()).add(str(v)[:100])
            for s in t.spans:
                for k, v in s.attributes.items():
                    field_uniques.setdefault(k, set()).add(str(v)[:100])
        _print_suggestions(field_uniques, n, path, b)


# ── Suggest logic ────────────────────────────────────────────────────────────

# Known aliases per dimension — from real-world Langfuse, LangSmith,
# Braintrust, HuggingFace, and OpenAI trace exports.
_ALIASES: dict[str, list[str]] = {
    "trace_id": [
        "trace_id", "id", "uuid", "run_id", "request_id", "instance_id",
        "task_name", "traj_id", "session_id", "experiment_id", "eval_id",
        "issue_id", "issue_name", "sample_id", "example_id",
    ],
    "outcome": [
        "outcome", "result", "status", "evaluation", "success", "passed",
        "resolved", "correctness", "label", "verdict", "score", "is_correct",
        "judge", "is_resolved", "is_success",
    ],
    "cost": [
        "cost", "total_cost", "price", "total_price", "llm_cost", "api_cost",
        "run_cost", "agent_cost", "usage_cost", "token_cost",
    ],
    "input_tokens": [
        "input_tokens", "prompt_tokens", "total_input_tokens",
        "tokens_in", "input_token_count", "prompt_token_count",
    ],
    "output_tokens": [
        "output_tokens", "completion_tokens", "total_output_tokens",
        "tokens_out", "output_token_count", "completion_token_count",
    ],
    "duration": [
        "duration", "duration_s", "elapsed", "elapsed_time", "latency",
        "run_time", "execution_time", "wall_time", "time_seconds",
    ],
    "task_id": [
        "task_id", "instance_id", "task_name", "problem_id", "question_id",
        "benchmark_id", "sample_id", "example_id", "test_id", "case_id",
    ],
}

_MAX_CANDIDATES = 3

# Map suggest dimensions to CLI flag names.
_DIM_TO_FLAG = {
    "trace_id": "--trace-id",
    "task_id": "--task-id",
    "outcome": "--outcome",
    "cost": "--cost",
    "input_tokens": "--input-tokens",
    "output_tokens": "--output-tokens",
    "duration": "--duration",
}


def _score_field(field: str, aliases: list[str]) -> int:
    """Score a field name against an alias list. Higher = better match.

    3 = last segment exactly matches an alias
    2 = full field path exactly matches an alias
    1 = fuzzy match on last segment via difflib
    0 = no match

    Substring matching is intentionally excluded — "agent_cost.total_input_tokens"
    should not match the "cost" alias list just because "cost" appears in the path.
    """
    last_segment = field.rsplit(".", 1)[-1].lower()
    field_lower = field.lower()

    for alias in aliases:
        alias_lower = alias.lower()
        if last_segment == alias_lower:
            return 3
        if field_lower == alias_lower:
            return 2

    from difflib import get_close_matches
    if get_close_matches(last_segment, [a.lower() for a in aliases], n=1, cutoff=0.6):
        return 1

    return 0


def _print_suggestions(
    field_uniques: dict[str, set], n_traces: int, file_path: str, bar: str,
) -> None:
    """Print field mapping suggestions and a copy-pasteable compare command."""
    click.echo(f"  {bar}")
    click.echo(f"  {click.style('Suggested field mappings', bold=True)}")
    click.echo()

    all_fields = set(field_uniques.keys())
    best_picks: dict[str, str] = {}
    all_candidates: dict[str, list[tuple[int, str]]] = {}

    for dimension, aliases in _ALIASES.items():
        scored: list[tuple[int, str]] = []
        for field in sorted(all_fields):
            s = _score_field(field, aliases)
            if s > 0:
                n_unique = len(field_uniques.get(field, set()))
                # Penalize fields useless for this dimension.
                if dimension == "outcome" and n_unique < 2:
                    s = max(0, s - 2)
                if dimension == "trace_id" and n_unique < n_traces * 0.5:
                    s = max(0, s - 1)
                if dimension == "task_id" and n_unique >= n_traces * 0.9:
                    s = max(0, s - 2)  # every trace unique = not a grouping key
                if s > 0:
                    scored.append((s, field))
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        all_candidates[dimension] = scored[:_MAX_CANDIDATES]

    has_any = any(c for c in all_candidates.values())
    if not has_any:
        click.echo(f"  {click.style('No field mapping suggestions — standard field names detected.', dim=True)}")
        click.echo()
        return

    for dimension, candidates in all_candidates.items():
        label = click.style(dimension, bold=True)
        click.echo(f"    {label}")

        if candidates:
            for i, (score, field) in enumerate(candidates):
                if i == 0 and score >= 2:
                    star = click.style("★", fg="green")
                    name = click.style(field, fg="cyan")
                    click.echo(f"      {star} {name}")
                    best_picks[dimension] = field
                else:
                    click.echo(f"        {click.style(field, dim=True)}")
        else:
            click.echo(f"        {click.style('(no candidates found)', dim=True)}")

        click.echo()

    if not best_picks:
        return

    # Build CLI flags from confident picks.
    flags: list[str] = []
    for dim, pick in best_picks.items():
        if dim == "trace_id" and pick == "trace_id":
            continue
        flag = _DIM_TO_FLAG.get(dim)
        if flag:
            flags.append(f"{flag} {pick}")

    if not flags:
        return

    # ── Two paths: quick (flags) or persistent (config) ──────────────
    click.echo(f"  {click.style('Option 1 — quick compare with flags:', bold=True)}")
    click.echo()
    cmd = f"    kalibra compare {file_path} <current.jsonl>"
    if len(flags) <= 2:
        cmd += " " + " ".join(flags)
        click.echo(f"  {cmd}")
    else:
        click.echo(f"  {cmd} \\")
        for i, flag in enumerate(flags):
            suffix = " \\" if i < len(flags) - 1 else ""
            click.echo(f"      {flag}{suffix}")
    click.echo()

    click.echo(f"  {click.style('Option 2 — save to config (reusable, supports CI gates):', bold=True)}")
    click.echo()
    click.echo(click.style("    # kalibra init to create kalibra.yml, then add:", dim=True))
    click.echo(click.style("    fields:", dim=True))
    for dim in _ALIASES:
        pick = best_picks.get(dim)
        if pick:
            click.echo(click.style(f"      {dim}: {pick}", dim=True))
    click.echo()

    click.echo(
        f"  {click.style('Replace <current.jsonl> with your second file.', dim=True)}"
    )
    click.echo(
        f"  {click.style('Same format assumed for both files. If they differ, use config', dim=True)}"
    )
    click.echo(
        f"  {click.style('with per-source field overrides — see README.', dim=True)}"
    )
    click.echo()
