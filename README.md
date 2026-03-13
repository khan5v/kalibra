# Kalibra

Regression detection and CI quality gates for AI agents.

```
Kalibra Compare
──────────────────────────────────────────────────────────
Baseline     1,240 traces   (cached_sources/baseline.jsonl)
Current      1,187 traces   (cached_sources/current.jsonl)
Direction    ▲ IMPROVED

▲ Success rate    42.3% → 46.1%  +3.8 pp  ✓ significant (p=0.003)

▲ Cost            $0.0100 → $0.0075 median  -25.0%
                  $0.0120 → $0.0090 avg  -25.0%
                  $14.88 → $10.69 total
                  95% CI [-28.1%, -21.3%]

▲ Steps           7 → 5 steps/trace (median)  -28.6%
                  7.4 → 5.2 avg  -29.7%

▲ Duration        34.1s → 28.5s median  -16.4%
                  38.4s → 31.2s avg  -18.8%
                  91.2s → 74.1s P95  -18.8%

▲ Token usage     1,100 → 950 tokens/trace (median)  -13.6%
                  1,240 → 1,080 avg  -12.9%
                  in: 890 → 720 avg  |  out: 350 → 360 avg

▲ Token eff.      8,300 → 6,200 tokens/success  -25.3%  (580→620 successes)
▲ Cost / quality  $0.048 → $0.032 per success  -33.3%  (580/1240 → 620/1187 succeeded)
≈ Per-task        1,103 matched — ✓ 31 improved, ✗ 8 regressed
                  regressed: django__django-15498, flask__flask-5012 (+6 more)

Thresholds
  [  OK] success_rate_delta >= -2     actual: 3.80
  [  OK] regressions <= 10            actual: 8.00
  [  OK] total_cost <= 50             actual: 10.69

──────────────────────────────────────────────────────────
PASSED — all quality gates met
```

## What it does

You change a prompt, swap a model, or refactor a tool. Did the agent get better or worse? Kalibra compares a **baseline** run against a **current** run and tells you:

- Success rate change with statistical significance (two-proportion z-test)
- Per-task regressions and improvements
- Cost, token, and latency deltas with bootstrap confidence intervals
- Cost-effectiveness: cost per success, tokens per success
- In CI, exit code 1 if any threshold is violated

## Install

```bash
pip install git+https://github.com/khan5v/kalibra.git
```

Python 3.11+.

---

## Quick start

```bash
# Compare local files
kalibra compare --baseline ./baseline.jsonl --current ./current.jsonl

# Compare named sources (fetched from Langfuse/LangSmith, cached locally)
kalibra compare --baseline @baseline --current @current

# Add quality gates
kalibra compare --baseline @baseline --current @current \
  --require "success_rate_delta >= -2" \
  --require "regressions <= 5" \
  --require "total_cost <= 50"

# Output formats
kalibra compare ... --format markdown    # GitHub PR comment
kalibra compare ... --format json        # machine-readable
```

---

## Metrics

All metrics run by default. Disable any by editing `config/compare.yml`.

| Metric | What it measures | Key threshold fields |
|---|---|---|
| **success_rate** | Pass/fail rate delta + z-test significance | `success_rate_delta`, `success_rate` |
| **per_task** | Individual tasks that regressed or improved | `regressions`, `improvements` |
| **cost** | Cost per trace — median, avg, total, 95% CI | `cost_delta_pct`, `total_cost`, `avg_cost` |
| **steps** | Steps per trace — median and avg, CI | `steps_delta_pct`, `avg_steps`, `median_steps` |
| **duration** | Trace duration — median, avg, P95, CI | `duration_delta_pct`, `duration_median_delta_pct`, `duration_p95_delta_pct`, `total_duration` |
| **token_usage** | Token consumption — median, avg, in/out breakdown, CI | `token_delta_pct`, `total_tokens`, `avg_tokens` |
| **token_efficiency** | Tokens per successful task | `token_efficiency_delta_pct` |
| **cost_quality** | Total cost / number of successes | `cost_quality_delta_pct`, `cost_per_success` |
| **tool_error_rate** | Fraction of tool calls returning errors | `tool_error_rate_delta` |
| **path_distribution** | Jaccard similarity of execution paths | `path_jaccard` |

**Confidence intervals**: Cost, duration, steps, and token metrics include bootstrap 95% CIs on the median. Gate on the upper bound for conservative thresholds (e.g., `cost_delta_pct <= 10` gates on the median point estimate).

**Absolute thresholds**: Gate on absolute values, not just deltas — `total_cost <= 50`, `avg_tokens <= 5000`, `total_duration <= 3600`.

---

## Pulling traces

```bash
kalibra pull @baseline                    # pull from config/sources/
kalibra pull @baseline --refresh          # force re-pull
kalibra pull --source langfuse --project my-agent --since 7d

# Filter by tags or session
kalibra pull --source langfuse --since 7d --tags kalibra --tags baseline
kalibra pull --source langfuse --since 7d --session experiment-42
```

Tag and session filters can also be set in source configs:

```yaml
# config/sources/production.yml — Langfuse
baseline:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [kalibra, baseline]
  session: experiment-42

# config/sources/synth-langsmith.yml — LangSmith
ls-baseline:
  source: langsmith
  project: kalibra-synth      # LangSmith project name
  since: 7d
  limit: 500
  tags: [kalibra, baseline]
```

All remote requests use exponential backoff (5 retries) on rate limits, server errors, and connection failures.

| Service | Environment variables |
|---|---|
| Langfuse | `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` |
| LangSmith | `LANGSMITH_API_KEY`, `LANGSMITH_API_URL` (optional) |

---

## Configuration

### `config/compare.yml`

By default, all built-in metrics run with no quality gates. Customize by listing specific metrics and adding `require` expressions.

### Example configurations

Ready-to-use configs in `config/examples/`. Use with `--config`:

| Config | Use case | What it does |
|---|---|---|
| **`ci-gate.yml`** | CI/CD pipelines | Gates on success rate, regressions, cost, latency, tokens. Exits 1 on violation. |
| **`model-comparison.yml`** | Evaluating a model swap | Cost-effectiveness focus: cost/quality, token efficiency, no hard gates. |
| **`cost-control.yml`** | Budget enforcement | Absolute limits: total run cost, per-trace cost, cost per success, token cap. |
| **`swebench.yml`** | SWE-bench benchmarks | Tight regression detection: per-task failures, token efficiency per solve. |
| **`quick-check.yml`** | Local development | Minimal metrics, no gates. Fast iteration on prompt changes. |

```bash
# Use an example config
kalibra compare --baseline @baseline --current @current \
  --config config/examples/ci-gate.yml

# Or write your own
kalibra compare --baseline @baseline --current @current \
  --config config/compare.yml
```

Example — `config/examples/ci-gate.yml`:

```yaml
metrics:
  - success_rate
  - per_task
  - cost
  - duration
  - token_usage

require:
  - success_rate_delta >= -2
  - regressions <= 5
  - cost_delta_pct <= 20
  - duration_p95_delta_pct <= 30
  - token_delta_pct <= 25
```

Omit `metrics` to run all built-ins. Omit `require` for no gates.

### `config/sources/*.yml`

```yaml
# config/sources/production.yml — Langfuse
prod-baseline:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, v1]

prod-current:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, v2]
```

```yaml
# config/sources/synth-langsmith.yml — LangSmith
ls-baseline:
  source: langsmith
  project: synth
  since: 7d
  tags: [kalibra, baseline]

ls-current:
  source: langsmith
  project: synth
  since: 7d
  tags: [kalibra, current]
```

Override locations: `--config /path/to/compare.yml`, `--sources /path/to/sources/`.

### Outcome and cost overrides

By default, connectors detect outcome (success/failure) using platform heuristics — keyword matching on output fields, error flags, feedback scores. Cost is read from the platform's reported value.

You can override both per source when the defaults don't match your setup:

```yaml
# config/sources/production.yml
prod-baseline:
  source: langfuse
  project: my-agent
  since: 7d
  tags: [production, v1]

  # Use a specific metadata field for outcome instead of keyword heuristics
  outcome:
    field: metadata.evaluation_result       # dot-path into trace metadata
    success: [pass, resolved, correct]      # values that map to "success"
    failure: [fail, timeout, incorrect]     # values that map to "failure"

  # Use a different span attribute for cost
  cost:
    attr: custom.cost_usd                   # span attribute name
```

**Outcome override** — looks up `field` in trace metadata, matches the value (case-insensitive) against `success` and `failure` keyword lists. If the field is missing or the value doesn't match either list, the connector's default heuristic is kept. Defaults for `success`/`failure` lists are `["success"]` and `["failure", "error", "failed"]`.

**Cost override** — reads cost from the specified span attribute instead of the default `kalibra.cost`. Useful when your instrumentation writes cost to a custom field.

The `field` path supports several forms:
- Bare key: `status` → `trace.metadata["status"]`
- Metadata prefix: `metadata.langfuse.eval` → `trace.metadata["langfuse.eval"]`
- Nested dicts: `metadata.eval.result` → `trace.metadata["eval"]["result"]`

Overrides are applied after the connector fetches traces and before JSONL is written, so the saved cache reflects the overridden values.

---

## Custom metrics

Drop `kalibra_metrics.py` in your project root — auto-discovered, no config needed:

```python
# kalibra_metrics.py
from kalibra import ComparisonMetric, Observation, TraceCollection

class SubmitRateMetric(ComparisonMetric):
    name = "submit_rate"
    description = "Fraction of traces that include a submit step"

    def summarize(self, col: TraceCollection) -> float:
        traces = col.all_traces()
        return sum(1 for t in traces if any(s.name == "submit" for s in t.spans)) / len(traces) if traces else 0.0

    def compare(self, baseline: float, current: float) -> Observation:
        delta = round((current - baseline) * 100, 2)
        return Observation(
            name=self.name, description=self.description,
            baseline=baseline, current=current, delta=delta,
            formatted=f"{baseline:.1%} → {current:.1%}  {delta:+.1f} pp",
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        return {"submit_rate_delta": result.delta}

METRICS = [SubmitRateMetric()]
```

Or add a dotted module path in `config/compare.yml`: `- mypackage.custom_metrics`.

---

## Programmatic API

```python
import kalibra

baseline = kalibra.TraceCollection.from_traces(run_agent(suite, model="gpt-4"))
current  = kalibra.TraceCollection.from_traces(run_agent(suite, model="gpt-4o"))

result = kalibra.compare_collections(
    baseline, current,
    config=kalibra.CompareConfig(
        metrics=["success_rate", "cost", "token_efficiency"],
        require=["success_rate_delta >= -2", "cost_per_success <= 0.05"],
    ),
)

for name, obs in result.metrics.items():
    print(f"{name}: {obs.formatted}")
print("passed:", result.thresholds_passed)
```

---

## Trace formats

Kalibra reads:
- **JSONL** — portable format, produced by `kalibra pull`
- **SWE-bench** `.traj` files and parquet directories
- **Langfuse** / **LangSmith** — via connectors with `kalibra pull`

Internally, all traces are OTel `ReadableSpan` trees using [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

---

## Synthetic trace generator

Generate realistic test traces calibrated against real SWE-bench data:

```bash
# Langfuse
python3 scripts/synth_traces.py --mode baseline --target langfuse
python3 scripts/synth_traces.py --mode current  --target langfuse

# LangSmith
python3 scripts/synth_traces.py --mode baseline --target langsmith --project synth
python3 scripts/synth_traces.py --mode current  --target langsmith --project synth

# Custom count and tags
python3 scripts/synth_traces.py --mode baseline --target langsmith --count 50 --tags myteam,v2
```

Flags: `--count N` overrides the default trace count, `--tags a,b,c` sets comma-separated tags (default: `kalibra,<mode>`).

---

## Development

```bash
git clone https://github.com/khan5v/kalibra.git
cd kalibra
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
