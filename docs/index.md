---
template: home.html
title: Kalibra — Regression detection for AI agents
hide:
  - navigation
  - toc
---

# Kalibra

**The diff tool for AI agent runs.** The CLI that catches what the dashboard misses.

```bash
pip install kalibra
kalibra demo
kalibra compare baseline.jsonl current.jsonl -v
```

---

## Start here

<div class="grid cards" markdown>

-   **[Getting started](getting-started.md)**

    Install, run the demo, compare your own data, set up a CI gate.

-   **[Methods](methods.md)**

    Two-proportion z-test, percentile bootstrap, what's on the roadmap, what Kalibra does *not* claim.

-   **Integrations**

    [Phoenix / OpenInference](phoenix.md) · [OTel GenAI](otel-genai.md) · [CrewAI](crewai.md)

</div>

---

> "Unsuccessful AI products almost always share a common root cause: a failure to create robust evaluation systems."
> — [Hamel Husain, *Your AI Product Needs Evals*](https://hamel.dev/blog/posts/evals/)

Kalibra exists for the layer below the eval: once you have two runs of trace data, did anything actually change? Aggregate success rate can stay flat while half the task types flip. Token cost can move 30% with the median untouched. Kalibra runs the statistical test, prints the verdict, and exits non-zero when a gate fails.
