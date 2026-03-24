# Getting Started

## Install

```bash
pip install kalibra
```

## Try the demo

```bash
kalibra demo
```

This creates a `kalibra-demo/` directory with sample traces and runs an interactive comparison. Afterwards:

```bash
kalibra compare kalibra-demo/baseline.jsonl kalibra-demo/current.jsonl -v
```

## Interactive tutorials

Each notebook works without an API key using pre-recorded traces. All Kalibra analysis runs identically.

| Integration | Trace format | Demo scenario | Tutorial |
|---|---|---|---|
| **Phoenix / OpenInference** | `llm.*`, `openinference.*` | Multi-step agent with span tree aggregation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khan5v/kalibra/blob/main/examples/phoenix_kalibra_tutorial.ipynb) |
| **OTel GenAI** | `gen_ai.*` | Truncation regression hidden by aggregate improvement | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khan5v/kalibra/blob/main/examples/otel_genai/otel_genai_tutorial.ipynb) |
| **CrewAI** | Flat JSONL | Failure redistribution and cost explosion | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khan5v/kalibra/blob/main/examples/crewai/crewai_kalibra_tutorial.ipynb) |

## Compare your own data

```bash
kalibra compare baseline.jsonl current.jsonl
```

If your JSONL uses non-standard field names, let Kalibra figure it out:

```bash
kalibra inspect your-traces.jsonl --suggest
```

This scans your data and prints metric readiness, field coverage, and a copy-pasteable compare command:

```
  Metric readiness
    ✓ Outcome          200/200 traces
    ✗ Cost               0/200 traces
    ✓ Tokens           200/200 traces
    ✓ Duration         200/200 traces

  Suggested field mappings
    ★ gen_ai.usage.input_tokens
    ★ gen_ai.usage.output_tokens

  Option 1 — quick compare with flags:
      kalibra compare traces.jsonl <current.jsonl> \
      --input-tokens gen_ai.usage.input_tokens \
      --output-tokens gen_ai.usage.output_tokens
```

## Set up quality gates

Create a `kalibra.yml` to make comparisons repeatable and add CI gates:

```bash
kalibra init
```

Or write one manually:

```yaml
baseline:
  path: ./baselines/production.jsonl
current:
  path: ./eval-output/canary.jsonl

require:
  - success_rate_delta >= -2
  - regressions <= 5
  - cost_delta_pct <= 20
```

Run `kalibra compare --metrics` to see all available gate fields — `token_delta_pct`, `duration_delta_pct`, `span_regressions`, and more.

Then:

```bash
kalibra compare    # reads kalibra.yml, exits 1 on failure
```

### Filtering from a single file

If your baseline and current traces are in the same file (tagged by a field like `variant`), use `where` to split them:

```yaml
sources:
  baseline:
    path: ./all-traces.jsonl
    where:
      - variant == baseline
  current:
    path: ./all-traces.jsonl
    where:
      - variant == current

require:
  - success_rate_delta >= -2
  - regressions <= 5
```

Operators: `==` (equal), `!=` (not equal), `=~` (regex match), `!~` (regex not match). Multiple matchers are ANDed. Traces missing the field are excluded.

## Add to CI

```yaml
# .github/workflows/quality-gate.yml
name: Agent Quality Gate
on: [pull_request]

jobs:
  kalibra:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v5
      - run: python eval.py --output current.jsonl
      - uses: khan5v/kalibra-action@v1
        with:
          baseline: baselines/production.jsonl
          current: current.jsonl
          config: kalibra.yml
```

The [`kalibra-action`](https://github.com/khan5v/kalibra-action) posts a markdown report as a PR comment and exits 1 if any gate fails.
