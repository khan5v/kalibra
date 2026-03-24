# CrewAI

Kalibra complements `crewai test` — it covers the operational side that LLM-as-judge scoring doesn't: token cost, failure patterns, latency, and per-task regression detection.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khan5v/kalibra/blob/main/examples/crewai/crewai_kalibra_tutorial.ipynb) Interactive tutorial with two scenarios — no API key needed.

## The workflow

```
CrewAI Crew → OpenTelemetry → Phoenix → Export JSONL → Kalibra (compare)
```

**1. Run your crew with tracing enabled**

CrewAI emits OpenTelemetry spans natively. Use Phoenix (or any OTel collector) to capture them:

```python
import phoenix as px
from phoenix.otel import register

px.launch_app()
register()

# Your crew runs here — spans are captured automatically
crew.kickoff()
```

**2. Export traces**

```python
import json
from phoenix.client import Client

client = Client()
spans = client.spans.get_spans(project_identifier="default")

with open("traces.jsonl", "w") as f:
    for span in spans:
        f.write(json.dumps(span, default=str) + "\n")
```

**3. Compare with Kalibra**

=== "CLI"

    ```bash
    kalibra compare --baseline before.jsonl --current after.jsonl -v
    ```

=== "YAML config"

    ```yaml
    # kalibra.yml
    sources:
      baseline:
        path: traces.jsonl
        where:
          - variant == baseline
      current:
        path: traces.jsonl
        where:
          - variant == current

    fields:
      task_id: task.id

    require:
      - token_delta_pct <= 20
      - regressions <= 0
    ```

    ```bash
    kalibra compare --config kalibra.yml -v
    ```

## Two demo scenarios

### Failure redistribution

Model swap. `crewai test` score: 7.5 → 7.4. Aggregate success: 80% → 80%. Looks stable.

Kalibra shows: two task types went from 4/4 → 0/4 while two others went from 0/4 → 4/4. The failures shifted — regressions and improvements canceled in the aggregate.

### Cost explosion

Chain-of-thought enabled. `crewai test` score: 8.0 → 8.5. Success: 75% → 90%. Looks better.

Kalibra shows: tokens per trace jumped 35%. The success improvement isn't even statistically significant (p=0.077). You're paying a third more for a marginal, uncertain gain.

## Different tools for different problems

| | `crewai test` | Kalibra |
|---|---|---|
| What it answers | "Is the output good?" | "Did cost, success, or latency change?" |
| Method | LLM-as-judge scoring | Bootstrap CIs, p-values, per-task breakdown |
| Token / cost tracking | Not built in | Per trace and per span |
| Deterministic | No — LLM judge varies | Yes — pure computation |
| CI gates | Not built in | Exit code 1 on gate failure |

## What Kalibra sees in CrewAI traces

CrewAI emits OpenInference-compatible spans via OpenTelemetry. Each crew run produces a trace tree:

```
crew_run (CHAIN)
 └── Crew.kickoff (CHAIN)
      ├── Research Analyst._execute_core (AGENT)
      │    └── LLM (Research Analyst) — tokens, model, stop_reason
      └── Technical Writer._execute_core (AGENT)
           └── LLM (Technical Writer) — tokens, model, stop_reason
```

Kalibra extracts tokens and cost from LLM spans, computes duration from the full tree, and counts leaf spans as steps. The `variant` and `task.id` attributes on the root span drive `where` filtering and per-task breakdown.
