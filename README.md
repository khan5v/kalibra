<p align="center">
  <strong>Kalibra</strong><br>
  Regression detection and CI quality gates for AI agents.
</p>

<!-- badges will activate once the package is on PyPI and repo is public
<p align="center">
  <a href="https://pypi.org/project/kalibra/"><img src="https://img.shields.io/pypi/v/kalibra" alt="PyPI"></a>
  <a href="https://pypi.org/project/kalibra/"><img src="https://img.shields.io/pypi/pyversions/kalibra" alt="Python"></a>
  <a href="https://github.com/khan5v/kalibra/LICENSE"><img src="https://img.shields.io/github/license/khan5v/kalibra" alt="License"></a>
</p>
-->

---

You change a prompt, swap a model, or refactor a tool — did the agent get better or worse?

Kalibra compares two populations of traces and tells you, with statistical rigor, what changed — success rate, cost, latency, tokens, per-task regressions, per-span breakdowns. Two dependencies. One command.

```bash
pip install kalibra
kalibra demo
```

```
  Kalibra Compare
  ──────────────────────────────────────────────────────────
  Baseline       100 traces   (baseline.jsonl)
  Current        100 traces   (current.jsonl)
  Direction ~ MIXED

  Trace metrics

  ▲ Success rate      50.0% → 75.0%  +25.0 pp   (p=0.000)
  ▲ Cost              $0.0358 → $0.0213 median  -40.5%
  ▼ Duration          7.6s → 15.2s median  +99.1%
  ≈ Steps             4 → 4 steps/trace (median)  +0.0%
  ▼ Error rate        0.2% → 4.3%  +4.1 pp
  ≈ Token usage       7,746 → 7,738 tokens/trace (median)  -0.1%
  ▲ Cost / quality    $0.0385 → $0.0189 per success (median)  -51.0%

  Trace breakdown

  ~ Per trace         20 matched — ✓ 10 improved, ✗ 5 regressed

  Span breakdown

  ▼ Per span          5 matched — ✗ 1 regressed, ~ 4 mixed

  ──────────────────────────────────────────────────────────
  ~ MIXED — no quality gates configured
```

Add `-v` for per-task outcome changes, per-span breakdowns, confidence intervals, and P95s.

## Why Kalibra

Aggregate metrics hide task-level regressions. Your success rate went from 80% to 82% — great. Except five tasks that used to pass now fail, masked by eight new easy ones that pass. Kalibra catches this.

- **10 metrics** — success rate, cost, duration, steps, error rate, tokens, token efficiency, cost/quality, per-task breakdown, per-span breakdown
- **Statistical rigor** — bootstrap 95% CIs on continuous metrics, two-proportion z-test on rates, noise thresholds to ignore jitter
- **Quality gates** — `require: success_rate_delta >= -5` fails your CI pipeline (exit 1) when thresholds are violated
- **Any JSONL** — flat traces, nested spans, non-standard field names. Use `--suggest` to auto-detect field mappings
- **Three output formats** — terminal (human), markdown (PR comments), JSON (automation)
- **Two dependencies** — click + pyyaml. No scipy, no ML frameworks, no API keys

## Quickstart

**1. Install**

```bash
pip install kalibra
```

**2. Try the demo**

```bash
kalibra demo
```

This creates a `kalibra-demo/` directory with sample traces and runs a comparison. Afterwards:

```bash
kalibra compare kalibra-demo/baseline.jsonl kalibra-demo/current.jsonl -v
```

**3. Compare your own data**

If your fields don't match the defaults, let Kalibra figure it out:

```bash
kalibra inspect your-traces.jsonl --suggest
```

This scans your data and prints a copy-pasteable compare command with the right `--outcome`, `--cost`, `--trace-id` flags.

## Quality gates for CI

```yaml
# kalibra.yml
baseline:
  path: ./baselines/production.jsonl
current:
  path: ./eval-output/canary.jsonl

require:
  - success_rate_delta >= -2     # max 2pp success rate drop
  - regressions <= 5             # max 5 tasks regressed
  - cost_delta_pct <= 20         # max 20% cost increase
  - span_regressions <= 3        # max 3 span types regressed
```

```bash
kalibra compare        # reads kalibra.yml, exits 1 on failure
```

## Field mapping

Kalibra works with any JSONL shape. Map your fields in config or on the command line:

```yaml
# kalibra.yml
fields:
  outcome: metadata.result
  cost: agent_cost.total_cost
  task_id: metadata.task_name
```

```bash
kalibra compare a.jsonl b.jsonl --outcome metadata.result --cost usage.total_cost
```

Comparing files with different schemas? Override fields per source:

```yaml
baseline:
  path: ./langfuse.jsonl
  fields: { outcome: metadata.result, cost: usage.total_cost }
current:
  path: ./braintrust.jsonl
  fields: { outcome: scores.correctness, cost: metrics.cost }
```

## Python API

```python
from kalibra.loader import load_traces
from kalibra.engine import compare
from kalibra.renderers import render

baseline = load_traces("baseline.jsonl")
current = load_traces("current.jsonl")

result = compare(baseline, current, require=["success_rate_delta >= -5"])
print(render(result, "terminal", verbose=True))
print("passed:", result.passed)
```

## Commands

```
kalibra compare [a.jsonl b.jsonl]     Compare traces — flags, config, or positional args
kalibra compare -v                    Verbose — CIs, P95s, per-task/per-span detail
kalibra compare --format markdown     Markdown for PR comments
kalibra compare --format json         Machine-readable JSON
kalibra compare --metrics             List all threshold fields
kalibra inspect traces.jsonl          Show data coverage and fields
kalibra inspect traces.jsonl --suggest  Auto-detect field mappings
kalibra init                          Create kalibra.yml interactively
kalibra demo                          Run comparison on built-in sample data
```

## Development

```bash
git clone https://github.com/khan5v/kalibra.git
cd kalibra
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
