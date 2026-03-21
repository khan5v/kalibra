# Phoenix / OpenInference

Kalibra auto-detects [OpenInference](https://github.com/Arize-ai/openinference) trace exports from [Phoenix](https://github.com/Arize-ai/phoenix). No field mapping needed — just export your spans and compare.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khan5v/kalibra/blob/main/examples/phoenix_kalibra_tutorial.ipynb) Try the interactive tutorial — run a live agent with your Anthropic key, or use pre-recorded traces to explore instantly.

## The workflow

```
Phoenix (tracing) → Export spans as JSONL → Kalibra (compare)
```

**1. Export traces from Phoenix**

```python
from phoenix.client import Client

client = Client()
spans = client.spans.get_spans(project_identifier="default")

with open("traces.jsonl", "w") as f:
    for span in spans:
        f.write(json.dumps(span, default=str) + "\n")
```

**2. Compare with Kalibra**

=== "CLI"

    ```bash
    kalibra compare --baseline before.jsonl --current after.jsonl -v
    ```

=== "Python"

    ```python
    from kalibra.loader import load_traces
    from kalibra.engine import compare
    from kalibra.renderers import render

    baseline = load_traces("before.jsonl")
    current = load_traces("after.jsonl")

    result = compare(baseline, current, require=["token_delta_pct <= 50"])
    print(render(result, "terminal", verbose=True))
    ```

That's it. Kalibra detects the OpenInference format from the first line and handles the rest.

## What happens under the hood

OpenInference exports are flat arrays of spans — one span per line, each carrying a `context.trace_id`. Kalibra:

1. **Groups spans into traces** by `trace_id`
2. **Builds the span tree** using `parent_id` relationships
3. **Extracts tokens and cost** from LLM spans (CHAIN/AGENT/TOOL spans have `None` for these — they're excluded automatically)
4. **Counts steps** as leaf spans, not total spans — orchestration wrappers aren't actions
5. **Computes duration** as wall-clock time (`max(end) - min(start)`), not sum of span durations
6. **Parses finish reason** from `llm.output_messages` attributes or the `output.value` JSON (`stop_reason` for Anthropic, `finish_reason` for OpenAI/Google) — used by the `error_rate` and `success_rate` metrics to detect truncation and errors

## What's detected

| Format | Auto-detected | Notes |
|--------|--------------|-------|
| Phoenix JSONL export (`get_spans()`) | Yes | Nested or dot-flattened attributes |
| Phoenix JSON array export | Yes | Same detection logic |
| Flat Kalibra JSONL (one trace per line) | Yes | Default format, no OpenInference needed |
| Nested OTel (child_spans trees) | Yes | JSON array format |

