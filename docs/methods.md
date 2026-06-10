# Methods

Kalibra is statistically transparent: every number it prints has a named method behind it, and every method has documented limits. This page lists them.

## Trace, task, and span

These three words appear throughout the docs. They are not synonyms.

- **Trace** — one end-to-end run of your agent on one input. One row in a JSONL file. One trace_id.
- **Task** — the logical *thing* the agent was asked to do (e.g. `"summarize_pr_42"`, `"book_flight_lhr_jfk"`). One task can be run many times — across seeds, model versions, baseline and current. Kalibra groups traces by `fields.task_id` to compare the same task across two populations. Per-task breakdown surfaces tasks that regressed even when the aggregate looks fine.
- **Span** — one operation *inside* a trace: an LLM call, a tool invocation, a chain step. A trace has 0..N spans. Span data powers steps, error rate, and span breakdown. Phoenix / OpenInference traces are trees of spans; flat JSONL traces typically have none.

A trace answers "what happened on this run." A task is the unit you group runs by. A span is the unit inside a run.

## Metrics at a glance

Every metric is one of two types: a **proportion** (binary outcomes, two-proportion z-test) or a **continuous distribution** (heavy-tailed, percentile bootstrap on the median). Breakdowns combine both.

| Metric | Method | Noise floor | Gate fields |
|---|---|---|---|
| `success_rate` | Two-proportion z-test | 0.5 pp | `success_rate_delta`, `success_rate` |
| `error_rate` | Two-proportion z-test | 0.5 pp | `error_rate_delta` |
| `cost` | Bootstrap CI on median %Δ | 3% | `cost_delta_pct`, `total_cost`, `avg_cost` |
| `duration` | Bootstrap CI on median %Δ | 5% | `duration_delta_pct`, `total_duration` |
| `token_usage` | Bootstrap CI on median %Δ | 3% | `token_delta_pct`, `total_tokens`, `avg_tokens` |
| `steps` | Bootstrap CI on median %Δ | 3% | `steps_delta_pct`, `avg_steps`, `median_steps` |
| `cost_quality` | Bootstrap CI on median %Δ | 5% | `cost_quality_delta_pct`, `cost_per_success` |
| `token_efficiency` | Bootstrap CI on median %Δ | 5% | `token_efficiency_delta_pct` |
| `trace_breakdown` | Per-task point comparison | — | `regressions`, `improvements` |
| `span_breakdown` | Bootstrap CI + z-test per span | 5% | `span_regressions`, `span_improvements` |

Use the gate fields on the right in `kalibra.yml` to set thresholds. Run `kalibra compare --metrics` to print them with descriptions.

## How a metric is classified

For each metric Kalibra reports a direction: **upgrade**, **regression**, **unchanged**, **mixed**, or **n/a**. The classification rule is the same for every continuous metric:

A continuous metric counts as **upgrade** or **regression** when **both** conditions hold:

1. The bootstrap 95% CI on the percentage delta excludes zero.
2. The absolute percentage delta exceeds the metric's noise floor (the table above).

If the CI includes zero, the change is not statistically distinguishable from noise → **unchanged**. If the delta is statistically real but smaller than the noise floor, it is practically negligible → **unchanged**. Sign and `higher_is_better` decide upgrade vs regression.

If no CI could be computed at all (fewer than 2 values per side, or the CI was suppressed — see bootstrap limits below), significance cannot be assessed and the metric is classified **mixed** (inconclusive). Kalibra never falls back to classifying on the bare point estimate.

**Gate evaluation follows the classification.** A threshold gate on a delta field (`*_delta`, `*_delta_pct`, `*_delta_pp`) is only evaluated when the metric reached a verdict. When the change is not significant (**unchanged**) or significance could not be assessed (**mixed**), the gate is skipped with an explicit warning rather than evaluated against an unverified point estimate. Gates on absolute fields (`success_rate`, `total_cost`, …) are budget checks, not significance tests — they always evaluate.

For proportion metrics (`success_rate`, `error_rate`), the same two-condition logic applies with p < 0.05 in place of the CI check and a percentage-point noise floor in place of the percentage floor.

## Success rate, error rate, and other proportions

**Two-proportion z-test (pooled).** For binary outcomes (success/failure, error/no-error), Kalibra computes the pooled-variance two-proportion z-test:

$$
z = \frac{\hat{p}_2 - \hat{p}_1}{\sqrt{\hat{p}(1-\hat{p}) \left(\tfrac{1}{n_1} + \tfrac{1}{n_2}\right)}}, \quad
\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}
$$

The p-value is two-sided, computed from the standard normal via `erfc(|z|/√2)` — equivalent to `2 · (1 − Φ(|z|))`. A delta is reported as "significant" when p < 0.05.

**Limits:**

- Small samples (n < 30 per arm) make the normal approximation unreliable. Kalibra still reports the result but emits a low-sample warning.
- No Wald confidence interval on the proportion delta is reported today — only the point estimate and p-value.
- No multiple-testing correction across metrics. Running 8 gates at α = 0.05 inflates the family-wise false positive rate. Kalibra discloses this at run time: with 3 or more significance-gated thresholds configured, a warning reports the family-wise upper bound (1 − 0.95^k) on at least one spurious gate failure.

## Cost, tokens, duration, and other continuous metrics

**Percentile bootstrap (n = 1000) for the percentage change in medians.** Kalibra resamples both populations with replacement 1,000 times, computes the median on each resample, computes the percentage change between resampled medians, and reports the 2.5th and 97.5th percentiles of those 1,000 percentage changes as the 95% CI on the delta.

The bootstrap is seeded (`random.Random(42)`) so output is reproducible across runs on the same input. Note: reordering the JSONL changes which elements the seeded indices pull, so the CI will shift — same data, different order, different CI.

**Why the median, not the mean?** Token counts and latencies are heavy-tailed. A single 100k-token outlier moves the mean meaningfully but barely shifts the median. Kalibra optimises for "did the typical run change?" not "did the worst-case run change?"

**Limits:**

- 1,000 resamples is fixed. Monte Carlo error on the 95% bounds is roughly ±0.5%.
- The percentile bootstrap is biased for skewed distributions.
- When >20% of resamples produce an undefined percentage change (baseline resampled median = 0), the CI is suppressed rather than reported on a biased remainder. This typically fires when baseline cost is mostly zero — e.g. a free local model compared against a paid API. With no CI the metric is classified mixed (inconclusive) and its delta gates are skipped.
- The seed is hard-coded.
- No power analysis or minimum detectable effect (MDE) yet.

## Per-task breakdown (`trace_breakdown`)

For each `task_id` present in both populations, Kalibra computes the baseline and current success rate from traces grouped by that task. A task is flagged "regressed" if its current rate is strictly below baseline, "improved" if strictly above. This is a point-estimate comparison — there is no per-task significance test today, so small-sample noise can produce spurious entries when a task has few runs in either arm.

**Aggregate direction:** if any task regressed and any task improved, the metric reports **mixed**. If only regressions, **regression**. If only improvements, **upgrade**.

**Limits:**

- No per-task z-test or CI. A 1-of-2 vs 2-of-2 task counts as a "regression" with no significance gate.
- No low-sample exclusion. Every task with a strict change contributes to the tally.

## Per-span breakdown (`span_breakdown`)

For each span `name` that appears in both populations, Kalibra compares median cost, median tokens, median duration, and error rate.

- **Continuous dimensions** (cost / tokens / duration) use the same classification rule as trace-level metrics: a dimension counts as regressed only when the 95% CI excludes zero AND the absolute percentage change exceeds the 5% per-span noise floor.
- **Error rate** uses the two-proportion z-test with p < 0.05 AND an absolute change above 1.0 percentage points. The dual condition exists because high-volume spans can produce statistically significant but practically negligible error-rate shifts.

A span name is classified `regressed` if any dimension regressed and none improved, `improved` if any improved and none regressed, `mixed` if both occurred on different dimensions. Mixed spans count toward the `span_regressions` gate — a span that doubles in cost while getting slightly faster should still trigger.

Span names with fewer than 30 occurrences in either arm are excluded from the regression/improvement tally — small samples produce too many false positives. They still appear in verbose output, marked with a low-sample warning.

## Data invariants

These rules exist because violating them silently corrupts statistics:

- **`None` means "not measured." `0` means "measured as zero."** Metrics filter out `None`; metrics include `0`. A trace with no recorded cost has `total_cost = None`, not `0`.
- **Empty `spans` means "no span-level data."** Span-dependent metrics (steps, span breakdown, error rate from spans) return n/a for span-less traces. Kalibra does not invent synthetic spans.
- **Empty `trace_id` means "not configured."** The loader does not guess which JSON field is the trace identifier. If `fields.trace_id` is not set and the data has no `trace_id` field, per-task breakdown finds 0 matched tasks.

When the loader drops malformed input (non-dict items, spans without a trace context, spans with empty trace IDs), it raises a `KalibraDataWarning` rather than failing silently. Filter or count these to monitor your pipeline:

```python
import warnings
from kalibra import KalibraDataWarning, load_traces

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", KalibraDataWarning)
    traces = load_traces("export.jsonl")
    for w in caught:
        print(w.message)
```

## What's on the roadmap

- **Benjamini–Hochberg FDR correction** across simultaneously evaluated metrics — controls the false discovery rate when many gates fire at once.
- **BCa bootstrap** for skewed continuous metrics — adjusts the percentile bootstrap for bias and skew, narrower and better-centred than plain percentile.
- **Per-task z-test** with low-sample exclusion and CI — replaces the current point-estimate comparison.
- **Wald CI on success-rate delta** — alongside the p-value.
- **Minimum detectable effect (MDE)** and post-hoc power for proportion gates — answers "is my sample size large enough to see the regression I care about?"
- **P95 latency metric** with bootstrap CI — tail-latency tracking, complementing median.
- **User-configurable bootstrap seed** — the seed is fixed at `42` today.

If a number on the roadmap matters to your decision today, [open an issue](https://github.com/khan5v/kalibra/issues) — knowing what's blocked helps prioritise it.

## What Kalibra does *not* claim

- It does not claim outputs are "correct" — there is no LLM judge built in.
- It does not claim a passing gate means no regression — only that the regressions it tested for are within thresholds.
- It does not claim statistical significance implies practical significance. A 0.3% success-rate drop with p < 0.05 may not be worth acting on. Gates exist to encode practical thresholds; significance tests exist to flag whether a delta is likely real.
