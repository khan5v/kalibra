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

## Compare your own data

```bash
kalibra compare baseline.jsonl current.jsonl
```

If your JSONL uses non-standard field names, let Kalibra figure it out:

```bash
kalibra inspect your-traces.jsonl --suggest
```

This scans your data and prints a copy-pasteable compare command with the right `--outcome`, `--cost`, `--trace-id` flags.

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

The action posts a markdown report as a PR comment and exits 1 if any gate fails.
