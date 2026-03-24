# OTel GenAI

Kalibra auto-detects [OTel GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) traces — any JSONL with `gen_ai.*` attributes. Validated with `opentelemetry-instrumentation-openai-v2`. Compatible with any exporter that preserves standard `gen_ai.*` span attributes. No field mapping needed — just export your spans and compare.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khan5v/kalibra/blob/main/examples/otel_genai/otel_genai_tutorial.ipynb) Try the interactive tutorial — uses pre-recorded traces, no API key needed.

## The workflow

```
Your OTel GenAI exporter → JSONL → Kalibra (compare)
```

**1. Export traces as JSONL**

Any exporter that writes OTel spans with `gen_ai.*` attributes works. The key requirement is that your instrumentor emits `gen_ai.*` attributes (not `openinference.*`). For example, `opentelemetry-instrumentation-openai-v2` does this natively.

Export from your OTel collector or tracing backend as one span per line:

```python
import json

# spans = your_collector.export()  # however your platform exports
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

    result = compare(baseline, current, require=["regressions <= 2"])
    print(render(result, "terminal", verbose=True))
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
      - token_delta_pct <= -10
      - regressions <= 2
    ```

    ```bash
    kalibra compare --config kalibra.yml -v
    ```

That's it. Kalibra detects the `gen_ai.*` attributes automatically and handles the rest.

!!! tip "Single file with both populations?"
    If your OTel collector streams all traces into one JSONL file, use `where` clauses in the YAML config to split them dynamically by any root span attribute — `variant`, `git_sha`, `branch`, etc.

## What happens under the hood

OTel GenAI exports are flat arrays of spans — one span per line, each carrying a `context.trace_id`. Kalibra:

1. **Groups spans into traces** by `trace_id`
2. **Builds the span tree** using `parent_id` relationships
3. **Extracts tokens** from `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens`
4. **Maps finish reasons to outcomes** — `gen_ai.response.finish_reasons: ["stop"]` → success, `["length"]` → failure (truncated). Also maps `max_tokens`, `content_filter`, `safety`, and `recitation` to failure
5. **Reports cost as N/A** — the OTel GenAI spec has no cost attribute. If your platform adds a vendor-specific cost field, map it via `fields.cost` in your config
6. **Counts steps** as leaf spans, not total spans
7. **Computes duration** as wall-clock time (`max(end) - min(start)`)

## Platform compatibility

Kalibra works with any exporter that preserves `gen_ai.*` span attributes in JSONL. Platforms adopting the OTel GenAI semantic conventions include:

| Platform | Expected `gen_ai.*` support | Notes |
|----------|---------------------------|-------|
| `opentelemetry-instrumentation-openai-v2` | **Validated** | What the tutorial traces use |
| PydanticAI / Logfire | Expected | Built on OTel |
| Langfuse | Expected | May include custom cost attributes — map via `fields.cost` |
| Datadog LLM Observability | Expected | Uses `gen_ai.*` conventions |
| OpenLLMetry (Traceloop) | Expected | OTel-based auto-instrumentation |

!!! note
    "Expected" means the platform uses OTel GenAI conventions based on their documentation, but Kalibra has not been tested against their specific JSONL export format. If you encounter issues, please [open an issue](https://github.com/khan5v/kalibra/issues).

## What's detected

| Format | Auto-detected | Notes |
|--------|--------------|-------|
| OTel GenAI JSONL (`gen_ai.*` attributes) | Yes | Any exporter that preserves `gen_ai.*` attributes |
| Explicit format selection | `--trace-format otel-genai` | Use when auto-detection fails |
| Flat Kalibra JSONL (one trace per line) | Yes | Fallback format, no `gen_ai.*` needed |

## Differences from OpenInference

| Aspect | OTel GenAI | OpenInference |
|--------|-----------|---------------|
| Token attributes | `gen_ai.usage.input_tokens` | `llm.token_count.prompt` |
| Cost attribute | None (spec doesn't define it) | `llm.cost.total` |
| Finish reason | `gen_ai.response.finish_reasons` (array) | Nested in `output.value` JSON |
| Span classification | `gen_ai.operation.name` | `openinference.span.kind` |
| Typical tree depth | 2 levels (agent → operation) | Arbitrary (CHAIN → LLM/TOOL → ...) |
