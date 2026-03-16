# Kalibra

Regression detection and CI quality gates for AI agents.

You change a prompt, swap a model, or refactor a tool — did the agent get better or worse? Kalibra compares two trace populations and tells you, with statistical rigor, at both trace and span level.

## Quickstart

```bash
pip install kalibra
```

Run your first comparison — sample data is included:

```bash
kalibra compare \
  --baseline examples/sample-baseline.jsonl \
  --current examples/sample-current.jsonl
```

```
  Kalibra Compare
  ──────────────────────────────────────────────────────────
  Baseline        30 traces   (sample-baseline.jsonl)
  Current         30 traces   (sample-current.jsonl)
  Direction ▲ IMPROVED

  Trace metrics

  ▲ Success rate      13.3% → 46.7%  +33.3 pp   (p=0.005)
  ▲ Cost              $0.1112 → $0.0787 median  -29.2%
  ▲ Duration          48.2s → 40.8s median  -15.4%
  ▲ Steps             14 → 10 steps/trace (median)  -28.6%
  ≈ Error rate        3.3% → 1.2%  -2.0 pp
  ▲ Token usage       24,772 → 17,153 tokens/trace (median)  -30.8%
  ▲ Token efficiency  177,461 → 40,839 tokens/success  -77.0%
  ▲ Cost / quality    $0.8232 → $0.1914 per success  -76.8%

  Trace breakdown

  ▲ Per trace         10 matched — ✓ 6 improved, ✗ 0 regressed

  Span breakdown

  ~ Per span          10 matched — ✓ 3 improved, ✗ 7 regressed

  ──────────────────────────────────────────────────────────
  ~ MIXED — no quality gates configured
```

Add `-v` for full detail — per-span duration/cost/token breakdowns, confidence intervals, per-task outcome changes.

---

## JSONL schema

Kalibra reads JSONL files. One line per trace. Spans are nested inside.

### Full trace with spans

```json
{
  "trace_id": "run-42",
  "outcome": "success",
  "metadata": {"task_id": "issue-123", "model": "claude-sonnet-4-20250514"},
  "spans": [
    {
      "span_id": "s1",
      "name": "plan",
      "cost": 0.003,
      "input_tokens": 500,
      "output_tokens": 200,
      "start_time": "2026-01-15T10:00:00Z",
      "end_time": "2026-01-15T10:00:05Z",
      "model": "claude-sonnet-4-20250514"
    },
    {
      "span_id": "s2",
      "name": "search",
      "parent_id": "s1",
      "start_time": "2026-01-15T10:00:05Z",
      "end_time": "2026-01-15T10:00:07Z",
      "error": true
    }
  ]
}
```

### Minimal trace (no spans)

If your data doesn't have span-level detail, just put the totals at trace level:

```json
{"trace_id": "run-42", "outcome": "success", "cost": 0.05, "input_tokens": 2000, "output_tokens": 500, "duration_s": 12.3}
```

Kalibra creates a synthetic span from these fields so all metrics still work.

### Field reference

**Trace fields:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `trace_id` | string | yes | Configurable via `fields.trace_id` |
| `outcome` | string | no | `"success"` or `"failure"` |
| `metadata` | object | no | Freeform key-value pairs |
| `spans` | array | no | If absent, trace-level fields become a single span |

**Span fields:**

| Field | Type | Notes |
|---|---|---|
| `span_id` | string | Unique within the trace |
| `name` | string | Step name (e.g. "plan", "search", "edit") |
| `parent_id` | string | Parent span ID (null = root) |
| `cost` | number | USD |
| `input_tokens` | integer | |
| `output_tokens` | integer | |
| `model` | string | Model name |
| `start_time` | string/int | ISO 8601, unix seconds, or nanoseconds |
| `end_time` | string/int | |
| `duration_s` | number | Shorthand — used if start/end not present |
| `error` | boolean | |
| `attributes` | object | Extra data (auto-flattened to dot-notation) |

**Auto-parsing**: JSON strings embedded in field values are parsed recursively. `"stats": "{\"tokens\": 500}"` becomes accessible as `stats.tokens`.

---

## Set up your project

```bash
kalibra init
```

This generates a `kalibra.yml` with all 11 metrics and starter quality gates:

```yaml
# kalibra.yml
baseline:
  path: ./baseline.jsonl

current:
  path: ./current.jsonl

metrics:
  - success_rate        # task pass/fail rate + significance test
  - cost                # cost per trace — median, avg, total
  - steps               # steps per task — median and avg
  - duration            # latency — median, avg, P95
  - error_rate          # fraction of spans that error
  - path_distribution   # execution path similarity
  - token_usage         # token consumption — in/out/total
  - token_efficiency    # tokens per successful task
  - cost_quality        # cost per success
  - trace_breakdown     # per-task regressions
  - span_breakdown      # per-span regressions

require:
  - success_rate_delta >= -5
  - regressions <= 10
  - cost_delta_pct <= 30
  - span_regressions <= 3
```

Then just:

```bash
kalibra compare
```

---

## Configuration

### Named sources

Define reusable sources and reference them by name:

```yaml
sources:
  prod-v1:
    path: ./data/v1.jsonl
  prod-v2:
    path: ./data/v2.jsonl
    fields:
      outcome: metadata.result    # per-source field override

baseline: prod-v1
current: prod-v2
```

### Quality gates

Gates are set under `require:`. Comparison fails (exit 1) if any threshold is violated — designed for CI pipelines.

```yaml
require:
  - success_rate_delta >= -2    # max 2 pp success rate drop
  - regressions <= 5            # max 5 traces flipped
  - cost_delta_pct <= 20        # max 20% cost increase
  - span_regressions <= 3       # max 3 span names regressed
```

See all available threshold fields:

```bash
kalibra compare --metrics
```

Typos are caught early with suggestions:

```
  ▸ Unknown field 'succes_rate_delta' in: 'succes_rate_delta >= -2'
    Did you mean: success_rate_delta, success_rate
```

### CLI flags override config

```bash
kalibra compare --baseline ./other.jsonl                    # override path
kalibra compare --require "success_rate_delta >= -10"       # replace gates
kalibra compare --config config/examples/ci-gate.yml        # different config
```

### Field mappings

If your traces use non-standard field names, map them:

```yaml
fields:
  trace_id: uuid                       # which field identifies each trace
  task_id: metadata.task_name          # per-task regression matching
  outcome: task_status.evaluation      # success/failure detection
  cost: agent_cost.total_cost          # cost source (dot-notation paths)
  input_tokens: key_stats.input_tokens
  output_tokens: key_stats.output_tokens
```

Each population can override fields independently:

```yaml
sources:
  langfuse-prod:
    path: ./langfuse-export.jsonl
    fields:
      outcome: metadata.result
      cost: usage.total_cost
  braintrust-prod:
    path: ./braintrust-export.jsonl
    fields:
      outcome: scores.correctness
      cost: metrics.cost

baseline: langfuse-prod
current: braintrust-prod
```

### Inspect your data

See what's in a trace file and which metrics will work:

```bash
kalibra inspect traces.jsonl
```

If your data uses non-standard field names, ask Kalibra to suggest mappings:

```bash
kalibra inspect traces.jsonl --suggest
```

This scans your field names and shows ranked candidates for each dimension (trace_id, outcome, cost, tokens, duration) with a copy-pasteable `fields:` config block.

---

## Metrics

11 built-in metrics across trace and span levels.

### Trace metrics

| Metric | What it measures | Key threshold fields |
|---|---|---|
| **success_rate** | Pass/fail rate + two-proportion z-test | `success_rate_delta`, `success_rate` |
| **cost** | Cost per trace — median, avg, total | `cost_delta_pct`, `total_cost`, `avg_cost` |
| **steps** | Steps per trace — median and avg | `steps_delta_pct`, `avg_steps`, `median_steps` |
| **duration** | Latency — median, avg, P95 | `duration_delta_pct`, `duration_p95_delta_pct` |
| **error_rate** | Span error fraction | `tool_error_rate_delta` |
| **path_distribution** | Execution path similarity (Jaccard) | `path_jaccard` |
| **token_usage** | Token consumption — in/out/total | `token_delta_pct`, `total_tokens`, `avg_tokens` |
| **token_efficiency** | Tokens per successful task | `token_efficiency_delta_pct` |
| **cost_quality** | Cost per success | `cost_quality_delta_pct`, `cost_per_success` |

### Breakdown metrics

| Metric | What it measures | Key threshold fields |
|---|---|---|
| **trace_breakdown** | Per-task regressions and improvements | `regressions`, `improvements` |
| **span_breakdown** | Per-span regressions — duration, cost, tokens, errors | `span_regressions`, `span_improvements` |

**Statistical methods**: Success rate uses a two-proportion z-test. Continuous metrics use bootstrap 95% CIs on the delta between populations. Install `scipy` for Mann-Whitney U significance tests: `pip install kalibra[stats]`.

**Compact vs verbose**: Default output shows one line per metric. Add `-v` for full detail — per-span breakdowns, per-task outcome changes, confidence intervals, p-values.

---

## Use cases

### Gate CI on agent quality

```yaml
# kalibra.yml
baseline:
  path: ./baselines/production.jsonl

current:
  path: ./eval-output/canary.jsonl

require:
  - success_rate_delta >= -2
  - regressions <= 5
  - cost_delta_pct <= 20
```

<details>
<summary>GitHub Actions example</summary>

```yaml
# .github/workflows/agent-quality.yml
name: Agent Quality Gate
on: [pull_request]

jobs:
  kalibra:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - run: pip install kalibra

      - name: Run agent evaluation
        run: python eval.py --output eval-output/canary.jsonl

      - name: Quality gate
        run: kalibra compare --format markdown --output report.md

      - name: Post PR comment
        if: always()
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: report.md
```

</details>

### Compare models

```bash
kalibra compare --baseline gpt4-traces.jsonl --current gpt4o-traces.jsonl -v
```

### Use from Python

```python
from kalibra.loader import load_traces
from kalibra.engine import compare
from kalibra.renderers import render

baseline = load_traces("baseline.jsonl")
current = load_traces("current.jsonl")

result = compare(
    baseline, current,
    require=["success_rate_delta >= -5"],
)

print(render(result, "terminal", verbose=True))
print("passed:", result.passed)
```

---

## Commands

```bash
kalibra init                          # create kalibra.yml interactively
kalibra compare                       # compare using kalibra.yml
kalibra compare --baseline a --current b   # compare two files directly
kalibra compare -v                    # verbose — per-span/per-task detail
kalibra compare -q                    # quiet — no status messages, CI-friendly
kalibra compare --format markdown     # output as GitHub PR comment
kalibra compare --format json         # machine-readable JSON
kalibra compare --metrics             # list all metrics and threshold fields
kalibra inspect traces.jsonl          # show data coverage and fields
kalibra inspect traces.jsonl --suggest  # suggest field mappings
```

---

## Development

```bash
git clone https://github.com/khan5v/kalibra.git
cd kalibra
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,stats]"
pytest
```

Python 3.11+. MIT license.
