# Kalibra

Regression detection and CI quality gates for AI agents.

You change a prompt, swap a model, or refactor a tool — did the agent get better or worse? Kalibra compares two runs and tells you, with statistical rigor.

## Quickstart

```bash
pip install git+https://github.com/khan5v/kalibra.git
```

Run your first comparison — sample data is included:

```bash
kalibra compare \
  --baseline examples/sample-baseline.jsonl \
  --current examples/sample-current.jsonl \
  --require "success_rate_delta >= -5" \
  --require "regressions <= 3"
```

```
  Kalibra Compare
  ──────────────────────────────────────────────────────────
  Baseline        30 traces   (examples/sample-baseline.jsonl)
  Current         30 traces   (examples/sample-current.jsonl)
  Direction ▲ IMPROVED

  ▲ Success rate    13.3% → 46.7%  +33.3 pp
                    p=0.005 — statistically significant

  ▲ Per-task        10 matched — ✓ 4 improved, ✗ 0 regressed

  ▲ Cost            $0.1112 → $0.0787 median  -29.2%
                    95% CI [-46.0%, -12.1%]
                    Mann-Whitney U p=0.031 — statistically significant

  ▲ Steps           14 → 10 steps/trace (median)  -28.6%
  ▲ Duration        48.2s → 40.8s median  -15.4%
  ▲ Token usage     24,772 → 17,153 tokens/trace (median)  -30.8%
  ▲ Cost / quality  $0.8232 → $0.1914 per success  -76.8%
  ...

  Thresholds
    [ OK ] success_rate_delta >= -5   actual: 33.33
    [ OK ] regressions <= 3           actual: 0.00

  ──────────────────────────────────────────────────────────
  PASSED — all quality gates met
```

Quality gates exit with code 1 on violation — ready for CI.

`success_rate_delta >= -5` means: allow at most 5 percentage points drop. `regressions <= 3` means: at most 3 tasks can flip from success to failure.

To see every field you can gate on:

```bash
kalibra compare --metrics
```

---

## Connect your data

### Langfuse

Define your sources in a YAML file:

```yaml
# config/sources/production.yml
baseline:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, v1]

current:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, v2]
```

Set credentials and pull:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

kalibra pull @baseline
kalibra pull @current
kalibra compare --baseline @baseline --current @current
```

The `@name` syntax pulls traces from the named source, caches them locally as JSONL, and reuses the cache on subsequent runs. Use `--refresh` to force a re-pull.

### LangSmith

```yaml
# config/sources/langsmith.yml
baseline:
  source: langsmith
  project: my-agent
  since: 7d
  tags: [baseline]
```

```bash
export LANGSMITH_API_KEY=lsv2_pt_...

kalibra pull @baseline
```

### Local files

If you already have JSONL files, skip `pull` entirely:

```bash
kalibra compare --baseline ./before.jsonl --current ./after.jsonl
```

### Filtering

Source configs and CLI flags both support `tags`, `session`, `since` (time window like `7d`, `24h`, or ISO date), and `limit` (default: 5000). All remote requests retry with exponential backoff on rate limits and server errors.

---

## Use cases

### Gate CI on agent quality

Kalibra exits with code 1 when any gate fails — plug it into any CI system.

```yaml
# config/examples/ci-gate.yml
metrics:
  - success_rate
  - per_task
  - cost
  - duration
  - token_usage

require:
  - success_rate_delta >= -2        # at most 2 pp drop
  - regressions <= 5                # at most 5 tasks flipped
  - cost_delta_pct <= 20            # at most 20% cost increase
  - duration_p95_delta_pct <= 30    # P95 latency increase
  - token_delta_pct <= 25           # token increase
```

```bash
kalibra compare \
  --baseline @baseline --current @current \
  --config config/examples/ci-gate.yml
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

      - name: Pull traces
        env:
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
        run: |
          kalibra pull @baseline
          kalibra pull @current

      - name: Quality gate
        run: |
          kalibra compare \
            --baseline @baseline --current @current \
            --config config/examples/ci-gate.yml \
            --format markdown --output report.md

      - name: Post PR comment
        if: always()
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: report.md
```

</details>

### Evaluate a model swap

When switching models, you care about cost-effectiveness — not just raw accuracy. The `model-comparison` config focuses on cost/quality, token efficiency, and success rate without hard gates, so you get data to decide rather than a pass/fail.

```bash
kalibra compare \
  --baseline @gpt4-run --current @gpt4o-run \
  --config config/examples/model-comparison.yml
```

### Enforce cost budgets

Hard limits on spend. Fails if the current run exceeds absolute cost thresholds — total cost, per-trace cost, cost per success, and token caps.

```bash
kalibra compare \
  --baseline @baseline --current @current \
  --config config/examples/cost-control.yml
```

### Iterate on prompts locally

Fast feedback with minimal output — just success rate and cost, no gates. Edit a prompt, re-run your eval, compare.

```bash
kalibra compare \
  --baseline ./before.jsonl --current ./after.jsonl \
  --config config/examples/quick-check.yml
```

### Use from Python

```python
import kalibra

# Compare files — same as the CLI
result = kalibra.compare("baseline.jsonl", "current.jsonl")

# Or compare in-memory trace collections
result = kalibra.compare_collections(
    baseline_col, current_col,
    config=kalibra.CompareConfig(
        metrics=["success_rate", "cost"],
        require=["success_rate_delta >= -2"],
    ),
)

for name, obs in result.metrics.items():
    print(f"{name}: {obs.formatted}")
print("passed:", result.thresholds_passed)
```

---

## Metrics

All 10 built-in metrics run by default. Use `kalibra compare --metrics` for an interactive reference.

| Metric | What it measures | Key threshold fields |
|---|---|---|
| **success_rate** | Pass/fail rate + z-test | `success_rate_delta`, `success_rate` |
| **per_task** | Tasks that regressed or improved | `regressions`, `improvements` |
| **cost** | Cost per trace — median, avg, total | `cost_delta_pct`, `total_cost`, `avg_cost` |
| **steps** | Steps per trace — median and avg | `steps_delta_pct`, `avg_steps`, `median_steps` |
| **duration** | Latency — median, avg, P95 | `duration_delta_pct`, `duration_p95_delta_pct`, `total_duration` |
| **token_usage** | Token consumption — in/out/total | `token_delta_pct`, `total_tokens`, `avg_tokens` |
| **token_efficiency** | Tokens per successful task | `token_efficiency_delta_pct` |
| **cost_quality** | Cost per success | `cost_quality_delta_pct`, `cost_per_success` |
| **tool_error_rate** | Tool call error fraction | `tool_error_rate_delta` |
| **path_distribution** | Execution path similarity | `path_jaccard` |

**Statistical methods**: Success rate uses a two-proportion z-test. Continuous metrics include bootstrap 95% CIs. Install `kalibra[stats]` for Mann-Whitney U significance tests.

---

## Configuration

### Quality gates

Gates can be set via `--require` flags or in `config/compare.yml`. Both sources combine.

```yaml
# config/compare.yml
metrics:
  - success_rate
  - cost
  - duration

require:
  - success_rate_delta >= -2
  - total_cost <= 50
```

Omit `metrics` to run all 10 built-ins. Omit `require` for no gates.

Typos in field names are caught early with suggestions:

```
▸ Unknown field 'succes_rate_delta' in: 'succes_rate_delta >= -2'
  Did you mean: success_rate_delta, success_rate, tool_error_rate_delta
```

### Outcome and cost overrides

Connectors detect outcome (success/failure) using platform heuristics. Override when the defaults don't match your setup:

```yaml
# config/sources/production.yml
baseline:
  source: langfuse
  project: my-agent
  since: 7d

  outcome:
    field: metadata.evaluation_result
    success: [pass, resolved]
    failure: [fail, timeout]

  cost:
    attr: custom.cost_usd
```

### Custom metrics

Drop a `kalibra_metrics.py` in your project root — auto-discovered, no config needed. Implement `summarize()`, `compare()`, and `threshold_fields()` on a `ComparisonMetric` subclass, and export a `METRICS` list. Custom metrics appear in `kalibra compare --metrics` and work with `--require` like built-ins.

See the built-in metrics in `src/kalibra/metrics.py` for reference implementations.

---

## Other commands

```bash
# Validate a JSONL file and show summary stats
kalibra validate traces.jsonl

# Output formats
kalibra compare ... --format terminal    # default — colored table
kalibra compare ... --format markdown    # GitHub PR comments
kalibra compare ... --format json        # machine-readable
kalibra compare ... --format markdown --output report.md
```

---

## Synthetic trace generator

Generate realistic test traces for Langfuse or LangSmith — useful for testing your setup:

```bash
python3 scripts/synth_traces.py --mode baseline --target langfuse
python3 scripts/synth_traces.py --mode current  --target langfuse
```

Supports `--target langsmith`, `--count N`, and `--tags a,b,c`.

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
