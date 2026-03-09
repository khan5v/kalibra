# AgentFlow

Regression detection and CI quality gates for AI agents.

```
AgentFlow Compare
──────────────────────────────────────────────────────────
Baseline    1,240 traces   (cached_sources/baseline.jsonl)
Current     1,187 traces   (cached_sources/current.jsonl)

Success rate      42.3% → 46.1%   +3.8 pp   ✓ significant (p=0.003)
Avg steps         24.1 → 21.8     -9.5%
Avg cost          $0.0120 → $0.0090  -25.0%
Duration          avg 38.4s → 31.2s  -18.8%  |  P95 91.2s → 74.1s  -18.7%
Per-task          1,103 matched — ✓ 31 improved, ✗ 8 regressed
──────────────────────────────────────────────────────────
All checks passed
```

## The problem it solves

When you change a prompt, swap a model, or refactor a tool, you need to know: did the agent get better or worse? The answer isn't obvious — success rate might improve while cost doubles, or aggregate numbers stay flat while specific tasks regress.

AgentFlow compares a **baseline** run against a **current** run and gives you a structured answer: success rate with statistical significance, exact tasks that regressed or improved, and efficiency metrics across steps, cost, and latency. In CI, it fails the build if any threshold is violated.

## Install

```bash
pip install git+https://github.com/vorekhov/agentflow.git
```

Requires Python 3.10+.

---

## Core concepts

### Traces

A **trace** is one agent run — it has an outcome (`success` or `failure`) and a sequence of **spans** (tool calls, LLM calls, sub-steps). AgentFlow reads traces from:

- SWE-bench `.traj` files or parquet directories
- JSONL files pulled from Langfuse or LangSmith via `agentflow pull`

### Sources

A **source** is a named pull configuration defined in `config/sources/`. It tells AgentFlow where to fetch traces and caches the result under `cached_sources/`. You reference sources by name with `@`.

### Checks

**Checks** are the metrics and threshold gates defined in `config/compare.yml`. Metrics compute numbers; thresholds decide whether the comparison passes or fails.

---

## Quick start

**Step 1.** Define your sources in `config/sources/myproject.yml`:

```yaml
baseline:
  source: langfuse
  project: my-agent
  since: 7d

current:
  source: langfuse
  project: my-agent
  since: 1d
```

**Step 2.** Compare:

```bash
agentflow compare --baseline @baseline --current @current
```

AgentFlow fetches both datasets (or reads from `cached_sources/` if already pulled), runs all metrics, and prints the report. Add `--refresh` to force a re-pull.

---

## Pulling traces

Pull traces from Langfuse or LangSmith and cache them locally.

```bash
# Pull a named source (reads config/sources/, caches to cached_sources/)
agentflow pull @baseline
agentflow pull @baseline --refresh      # force re-pull, ignore cache

# Pull with explicit flags (no config file needed)
agentflow pull --source langfuse --project my-agent --since 7d
agentflow pull --source langsmith --project my-agent --output run.jsonl
```

**Environment variables:**

| Service | Variables |
|---|---|
| Langfuse | `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` |
| LangSmith | `LANGSMITH_API_KEY` |

The pull command handles rate limiting automatically — it retries with exponential backoff on 429 responses.

---

## Comparing runs

### Basic comparison

```bash
# Named sources (auto-fetches and caches)
agentflow compare --baseline @baseline --current @current

# Local files or directories
agentflow compare --baseline ./baseline/ --current ./current/

# Mix — named source vs. a local file
agentflow compare --baseline @baseline --current ./fresh-run.jsonl
```

### Output formats

```bash
--format terminal    # default — readable table for the terminal
--format markdown    # for GitHub PR comments
--format json        # machine-readable, for downstream tools
--output report.md   # write to file instead of stdout
```

### Pointing to external config

By default AgentFlow looks for `config/compare.yml` and `config/sources/` relative to the current directory. Override either with flags:

```bash
# Use a config file from a different location
agentflow compare --baseline @baseline --current @current \
  --config /shared/configs/my-agent.yml

# Use a sources directory from a different location
agentflow compare --baseline @baseline --current @current \
  --sources /shared/configs/sources/

# Both together
agentflow compare --baseline @baseline --current @current \
  --config /shared/configs/my-agent.yml \
  --sources /shared/configs/sources/
```

`--config` must be a file. `--sources` must be a directory. Both error immediately if the path does not exist or is the wrong type.

### Threshold gates

Thresholds make `agentflow compare` exit with code 1 when a condition is violated. Use them in CI to enforce quality standards.

```bash
agentflow compare \
  --baseline @baseline \
  --current @current \
  --require "success_rate_delta >= -2" \
  --require "regressions <= 5"
```

Available threshold fields:

| Field | Description |
|---|---|
| `success_rate_delta` | Change in success rate (percentage points) |
| `success_rate` | Absolute success rate of the current run (%) |
| `regressions` | Tasks that went success → failure |
| `improvements` | Tasks that went failure → success |
| `cost_delta_pct` | % change in average cost per trace |
| `steps_delta_pct` | % change in average steps per trace |
| `duration_delta_pct` | % change in average duration |
| `duration_p95_delta_pct` | % change in P95 latency |
| `tool_error_rate_delta` | Change in tool error rate (percentage points) |
| `path_jaccard` | Path similarity — 1.0 is identical, 0 is completely different |

---

## Configuration

### `config/compare.yml` — metrics and gates (`--config` to override)

Controls which metrics run and what thresholds are enforced.

```yaml
# config/compare.yml

metrics:
  - success_rate      # success rate delta + two-proportion z-test
  - per_task          # per-task regression / improvement detection
  - cost              # average cost per trace
  - steps             # average steps (spans) per trace
  - duration          # average and P95 latency
  - tool_error_rate   # fraction of tool calls that returned an error
  - path_distribution # Jaccard similarity of top execution paths

require:
  - success_rate_delta >= -2    # at most 2 pp drop
  - regressions <= 5            # at most 5 task regressions
  - cost_delta_pct <= 20        # at most 20% cost increase
```

Omit `metrics` entirely to run all built-in metrics. Omit `require` for no gates.

### `config/sources/` — named pull registry (`--sources` to override)

Each `.yml` file in this directory defines named sources. All files are merged automatically — you can split them by project, environment, or team.

```yaml
# config/sources/production.yml

prod-baseline:
  source: langfuse
  project: my-agent-prod
  since: 7d
  limit: 5000

prod-current:
  source: langfuse
  project: my-agent-prod
  since: 1d
```

```bash
agentflow compare --baseline @prod-baseline --current @prod-current
```

---

## Custom metrics

### Drop-in plugin (zero config)

Create `agentflow_metrics.py` in your project root — it is auto-discovered and loaded automatically, no config entry needed.

```python
# agentflow_metrics.py
from agentflow import ComparisonMetric, MetricResult, TraceCollection

class SubmitRateMetric(ComparisonMetric):
    name = "submit_rate"
    description = "Fraction of traces that include a submit step"

    def summarize(self, col: TraceCollection) -> float:
        traces = col.all_traces()
        if not traces:
            return 0.0
        return sum(1 for t in traces if any(s.name == "submit" for s in t.spans)) / len(traces)

    def compare(self, baseline: float, current: float) -> MetricResult:
        delta = round((current - baseline) * 100, 2)
        return MetricResult(
            name=self.name,
            description=self.description,
            baseline=baseline,
            current=current,
            delta=delta,
            formatted=f"{baseline:.1%} → {current:.1%}  {delta:+.1f} pp",
        )

    def threshold_fields(self, result: MetricResult) -> dict[str, float]:
        return {"submit_rate_delta": result.delta}

METRICS = [SubmitRateMetric()]
```

### Module path (for packages)

Add a dotted module path to the `metrics` list in `config/compare.yml`:

```yaml
metrics:
  - success_rate
  - mypackage.eval.custom_metrics   # imported, its METRICS list is run
```

### Node-level metrics

Register per-span diagnostic metrics (shown in the per-node breakdown):

```python
from agentflow.plugins import register
from agentflow.converters.base import Trace

@register("p95_duration", "95th-percentile span duration in seconds")
def p95_duration(node: str, traces: list[Trace]) -> float:
    durations = sorted(
        s.end_time - s.start_time
        for t in traces for s in t.spans if s.name == node
    )
    if not durations:
        return 0.0
    idx = min(int(len(durations) * 0.95), len(durations) - 1)
    return round(durations[idx], 3)
```

Built-in node metrics: `retry_rate`, `error_rate`, `cost_share`, `token_intensity`.

---

## Programmatic API

Use `compare_collections` when you have traces in memory and don't need to write files to disk.

```python
import agentflow
from agentflow import CompareConfig, TraceCollection

baseline = TraceCollection.from_traces(run_agent(test_suite, model="gpt-4"))
current  = TraceCollection.from_traces(run_agent(test_suite, model="gpt-4o"))

result = agentflow.compare_collections(
    baseline,
    current,
    config=CompareConfig(
        metrics=["success_rate", "cost", "per_task"],
        require=["success_rate_delta >= -2", "regressions <= 5"],
    ),
)

for name, metric in result.metrics.items():
    print(f"{name}: {metric.formatted}")

print("passed:", result.thresholds_passed)
```

---

## Development

```bash
git clone https://github.com/vorekhov/agentflow.git
cd agentflow
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
