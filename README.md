# Kalibra

Regression detection and CI quality gates for AI agents.

You change a prompt, swap a model, or refactor a tool — did the agent get better or worse? Kalibra compares two runs and tells you, with statistical rigor — at both trace and span level.

## Quickstart

```bash
pip install git+https://github.com/khan5v/kalibra.git
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
  Baseline        30 traces   Current        30 traces
  Direction ~ MIXED

  Trace metrics

  ▲ Success rate    13.3% → 46.7%  +33.3 pp   (p=0.005)
  ▲ Cost            $0.111 → $0.079 median  -29.2%
  ▲ Steps           14 → 10 median  -28.6%
  ▲ Duration        48.2s → 40.8s median  -15.4%
  ▲ Token usage     24,772 → 17,153 median  -30.8%
  ≈ Token eff.      13,612 → 13,034 tokens/success  -4.2%
  ▲ Cost / quality  $0.82 → $0.19 per success  -76.8%

  Trace breakdown

  ▲ Per trace       10 matched — ✓ 4 improved, ✗ 0 regressed

  Span breakdown

  ~ Per span        10 matched — ✓ 4 improved, ✗ 6 regressed

  ──────────────────────────────────────────────────────────
```

Add `-v` for full detail — per-span duration/cost/token/error breakdowns, confidence intervals, per-task outcome changes:

```bash
kalibra compare --baseline examples/sample-baseline.jsonl --current examples/sample-current.jsonl -v
```

---

## Set up your project

Generate a `kalibra.yml` config interactively:

```bash
kalibra init
```

This generates a complete config with all 11 metrics and starter quality gates:

```yaml
# kalibra.yml
baseline:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, v1.2]

current:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, v1.3]

metrics:
  - success_rate        # task pass/fail rate + significance test
  - cost                # cost per trace — median, avg, total
  - steps               # steps per task — median and avg
  - duration            # latency — median, avg, P95
  - tool_error_rate     # fraction of spans that error
  - path_distribution   # execution path similarity
  - token_usage         # token consumption — in/out/total
  - token_efficiency    # tokens per successful task
  - cost_quality        # cost per success
  - per_task            # trace breakdown — per-task regressions
  - span_breakdown      # span breakdown — per-step regressions

require:
  - success_rate_delta >= -5
  - regressions <= 10
  - cost_delta_pct <= 30
  - span_regressions <= 3
```

Set your credentials and run:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

kalibra compare
```

That's it. Kalibra reads `kalibra.yml`, pulls traces, compares, and gates.

---

## Configuration

### Named sources

Define reusable sources and reference them by name:

```yaml
# kalibra.yml
sources:
  prod-v1:
    source: langfuse
    project: my-agent
    since: 7d
    tags: [production, v1]
  prod-v2:
    source: langfuse
    project: my-agent
    since: 7d
    tags: [production, v2]

baseline: prod-v1
current: prod-v2
```

Pull a named source directly:

```bash
kalibra pull @prod-v1
```

### Mixed sources

Baseline and current are independent — they can use different platforms:

```yaml
baseline:
  path: ./baselines/v1.2.jsonl       # local file

current:
  source: braintrust
  project: my-agent
  since: 1d
  tags: [canary]
```

### Quality gates

Gates are set in `kalibra.yml` under `require:`. Comparison fails (exit 1) if any threshold is violated.

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
  Did you mean: success_rate_delta, success_rate, tool_error_rate_delta
```

### CLI flags override config

`--baseline` / `--current` override the config populations. `--require` replaces config gates. `--config` switches to a different config file.

```bash
kalibra compare --current staging                          # override source
kalibra compare --require "success_rate_delta >= -10"      # replace gates
kalibra compare --config config/examples/ci-gate.yml       # different config
```

### Field mappings

If `kalibra inspect` shows missing data, map your trace fields:

```yaml
fields:
  trace_id: uuid                  # which field identifies each trace
  task_id: braintrust.task_id     # per-task regression matching
  outcome: task_status.evaluation # success/failure detection (handles booleans)
  cost: agent_cost.total_cost     # cost source (supports nested paths)
  input_tokens: key_stats.input_tokens
  output_tokens: key_stats.output_tokens
```

Nested JSON strings are auto-parsed — `"stats": "{\"tokens\": 500}"` becomes `stats.tokens: 500`.

### Inspect your data

See what's in a trace file, what metrics will work, and what config to add:

```bash
kalibra inspect cached_sources/baseline.jsonl
```

---

## Supported platforms

| Platform | Auth |
|---|---|
| **Langfuse** | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` |
| **LangSmith** | `LANGSMITH_API_KEY` |
| **Braintrust** | `BRAINTRUST_API_KEY` |
| **Local JSONL** | No auth needed |

All remote requests retry with exponential backoff on rate limits and server errors.

---

## Metrics

11 built-in metrics across trace and span levels. Use `kalibra compare --metrics` for an interactive reference.

### Trace metrics

| Metric | What it measures | Key threshold fields |
|---|---|---|
| **success_rate** | Pass/fail rate + z-test | `success_rate_delta`, `success_rate` |
| **cost** | Cost per trace — median, avg, total | `cost_delta_pct`, `total_cost`, `avg_cost` |
| **steps** | Steps per trace — median and avg | `steps_delta_pct`, `avg_steps`, `median_steps` |
| **duration** | Latency — median, avg, P95 | `duration_delta_pct`, `duration_p95_delta_pct` |
| **tool_error_rate** | Span error fraction | `tool_error_rate_delta` |
| **path_distribution** | Execution path similarity | `path_jaccard` |
| **token_usage** | Token consumption — in/out/total | `token_delta_pct`, `total_tokens`, `avg_tokens` |
| **token_efficiency** | Tokens per successful task | `token_efficiency_delta_pct` |
| **cost_quality** | Cost per success | `cost_quality_delta_pct`, `cost_per_success` |

### Breakdown metrics

| Metric | What it measures | Key threshold fields |
|---|---|---|
| **per_task** | Per-trace regressions and improvements | `regressions`, `improvements` |
| **span_breakdown** | Per-span regressions — duration, cost, tokens, errors | `span_regressions`, `span_improvements` |

**Statistical methods**: Success rate uses a two-proportion z-test. Continuous metrics include bootstrap 95% CIs. Install `kalibra[stats]` for Mann-Whitney U significance tests.

**Compact vs verbose**: Default output shows one line per metric. Add `-v` for full detail — per-span breakdowns, per-task outcome changes, confidence intervals, p-values.

---

## Use cases

### Gate CI on agent quality

```yaml
# kalibra.yml
baseline:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, baseline]

current:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, canary]

require:
  - success_rate_delta >= -2
  - regressions <= 5
  - cost_delta_pct <= 20
  - span_regressions <= 3
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

      - run: pip install git+https://github.com/khan5v/kalibra.git

      - name: Quality gate
        env:
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
        run: |
          kalibra compare --format markdown --output report.md

      - name: Post PR comment
        if: always()
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: report.md
```

</details>

### Compare models

```bash
kalibra compare --baseline gpt4-run --current gpt4o-run -v
```

### Use from Python

```python
import kalibra

result = kalibra.compare("baseline.jsonl", "current.jsonl")

for name, obs in result.metrics.items():
    print(f"{name}: {obs.formatted}")
print("passed:", result.thresholds_passed)
```

---

## Custom metrics

Drop a `kalibra_metrics.py` in your project root — auto-discovered, no config needed. Implement `summarize()`, `compare()`, and `threshold_fields()` on a `ComparisonMetric` subclass, and export a `METRICS` list. Custom metrics appear in `kalibra compare --metrics` and work with `--require` like built-ins.

---

## Other commands

```bash
kalibra init                          # create kalibra.yml interactively
kalibra inspect traces.jsonl          # show data coverage and available fields
kalibra pull @source-name             # pull traces from a named source
kalibra compare --metrics             # list all metrics and threshold fields
kalibra compare -v                    # verbose — full per-span/per-task detail
kalibra compare --format markdown     # output as GitHub PR comment
kalibra compare --format json         # machine-readable output
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
