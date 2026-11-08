# KV Cache Precision Research Implementation Plan

Implement a Python research codebase for comparing KV-cache quantization and selective-precision methods.

The research question is:

> At a fixed KV-cache memory budget, does preserving selected context tokens at higher precision retain more quality than uniform quantization?

Read `KV Cache Precision — Experiment Design.docx` for background. Apply the methodological requirements below where they clarify or correct the document.

## 1. Keep the architecture simple

Use:

- Python with `pyproject.toml` and a committed dependency lockfile.
- MLX / MLX-LM as the first inference backend.
- YAML experiment configurations.
- Pydantic for configuration validation.
- `argparse` for the CLI.
- JSONL for per-example results.
- Small Python analysis scripts for tables and plots.
- `pytest` for correctness-critical behavior.

Avoid a plugin framework, dependency-injection framework, database, dashboard, or distributed execution system. Use small functions, dataclasses, and explicit registries.

Implement only the abstractions needed by the first working experiment.

## 2. Organize around four replaceable components

| Component | Responsibility | Adding something new |
|---|---|---|
| Backend | Model loading, prefill, cache branching, generation, continuation scoring | Add a backend adapter |
| Cache method | Convert and maintain caches under a precision policy | Add a method implementation |
| Benchmark adapter | Load examples, construct inputs, invoke official scoring | Add an adapter |
| Analysis | Compare saved results and generate reports | Add an analysis function |

Benchmark adapters must not implement quantization. Cache methods must not know which benchmark is running.

Suggested layout:

```text
pyproject.toml
README.md
LICENSE
configs/
  smoke.yaml
  baseline.yaml
  uniform_sweep.yaml
  selective_sweep.yaml
src/kvstudy/
  cli.py
  config.py
  records.py
  runner.py
  backends/
    mlx.py
  methods/
    base.py
    identity.py
    uniform.py
    selective.py
    selectors.py
  benchmarks/
    base.py
    ruler.py
    longbench.py
    pg19.py
  analysis/
    compare.py
    plots.py
tests/
docs/
  methodology.md
  adding_experiments.md
  adding_methods.md
results/                    # gitignored
```

Do not create speculative modules that have no implementation yet.

## 3. Make configurations explicit and validated

A configuration should specify the model, datasets, cache methods, execution mode, and output location.

Example schema:

```yaml
schema_version: 1
name: selective_precision_dev

model:
  backend: mlx
  id: Qwen/Qwen3-4B
  revision: "<pinned-checkpoint-revision>"
  weight_dtype: bfloat16

execution:
  mode: quality_simulated
  seed: 42

protocol:
  prefill_dtype: bfloat16
  conversion_point: after_context_before_question
  key_position: post_rope
  protected_window:
    tokens: 512
    precision_bits: 8
    overflow: demote_oldest

benchmarks:
  - name: ruler
    split_manifest: data/manifests/ruler_dev.jsonl
    tasks: [niah_single, niah_multi_key, variable_tracking]
    context_lengths: [4096, 8192, 16384]
    max_new_tokens: 128

  - name: longbench
    split_manifest: data/manifests/longbench_dev.jsonl
    tasks: [hotpotqa, qasper]
    max_new_tokens: 128

  - name: pg19
    split_manifest: data/manifests/pg19_dev.jsonl
    context_lengths: [4096, 8192, 16384]
    continuation_tokens: 512

methods:
  - id: bf16
    kind: identity

  - id: uniform_4bit
    kind: uniform
    key_bits: 4
    value_bits: 4
    group_size: 64
    axis: per_token

  - id: residual_8_4
    kind: selective
    selector: quantization_residual
    high_bits: 8
    low_bits: 4
    high_fraction: 0.20
    group_size: 64
    axis: per_token

output:
  root: results
```

This is an illustrative schema, not a claim that every requested format is supported by MLX.

Reject unknown configuration fields and unsupported combinations. Never silently substitute a different bit width, dtype, dataset, or backend.

The BF16 identity baseline remains unquantized throughout. Compressed methods share the same protected-window policy.

Initially use explicit method lists. Add a small sweep expander later if repetition becomes inconvenient; avoid configuration inheritance and templating systems.

## 4. Define a small common example format

Each benchmark adapter should produce a record containing:

```text
example_id
source_id
benchmark
task
split
context
question or reference continuation
reference answers
metadata
```

`source_id` identifies the underlying book or document for split isolation and clustered statistics.

Preserve official benchmark formatting and scoring behavior. Record token counts for the rendered model input, including template overhead.

For LongBench, specify natural length buckets and a documented truncation policy. Do not silently pad or truncate examples to create artificial length comparisons.

## 5. Implement one controlled execution procedure

For quality experiments:

```text
Load example
→ render and tokenize the full prompt consistently
→ identify the context/question boundary
→ prefill context in BF16
→ retain an untouched reference cache
→ create an independent branch for each method
→ apply cache transformation
→ process question and generate, or score continuation
→ invoke evaluator
→ save result
→ release branch
```

Handle prompt boundaries at the token level so splitting the prefill does not accidentally change tokenization or chat-template semantics.

Cache branches must not mutate the reference or one another.

For PG-19:

- Teacher-force the reference continuation.
- Record summed NLL and the number of scored tokens.
- Explicitly handle the first continuation prediction, whose prefix logits may precede cache conversion.
- Apply the same continuation-window policy across compressed methods.
- Aggregate perplexity as `exp(total_nll / total_scored_tokens)`.

Distinguish **post-prefill cache conversion** from future experiments that quantize during prefill. They are separate protocols and must have different identifiers.

## 6. Separate simulated quality from real systems measurements

Support two explicit execution modes:

- `quality_simulated`: quantize and dequantize values to study quality.
- `systems_real`: store genuinely packed quantized caches and measure execution.

In simulated mode, report estimated cache bytes and label them as estimates. Do not report the simulation’s resident memory or speed as quantized-system performance.

In real mode:

- Count actual cache buffers and metadata.
- Measure selection, conversion, prefill, and decode separately.
- Force MLX computation to complete before timing boundaries.
- Use warm-ups and repeated measurements.
- Measure deployment memory without a retained BF16 reference cache.
- Keep controlled generation lengths for speed comparisons.

Validate simulation against real execution on a small fixture before a large sweep.

## 7. Implement methods incrementally

First implement:

1. BF16 identity.
2. Uniform per-token quantization.
3. Selective precision with random selection.
4. Sinks and recency.
5. Quantization-residual selection.

Add attention-based selection and per-channel key quantization later.

Keep selection separate from storage:

```python
select_tokens(cache_view, selection_config) -> selection
```

Selection should support per-layer and per-KV-head decisions without requiring a large inheritance hierarchy.

Define exactly what the residual score measures and how K and V contributions are combined.

For memory comparisons, include:

- Quantized payload.
- Scales and biases.
- Position indices and padding.
- Protected tokens.
- Any retained auxiliary state.

Report both configured and realized high-precision fractions. Do not label different byte counts “equal memory.”

## 8. Reuse official scoring implementations

- RULER: use the official evaluator and task-specific scoring.
- LongBench v1: use the official evaluator and dataset-specific metrics.
- PG-19: implement the cache-aware NLL loop.

Pin evaluator revisions and record them in run metadata. Preserve licenses and attribution when incorporating upstream code.

Save enough predictions and reference information to rerun answer scoring without another model inference run.

Do not introduce an LLM judge for these initial experiments.

## 9. Make every result reproducible and resumable

Group results by experiment, dated run, and method. Keep methods evaluated together under the same run so paired comparisons share one configuration and dataset manifest:

```text
results/<experiment_name>/<dated_run_id>/
  resolved_config.yaml
  manifest.json
  events.jsonl
  methods/
    bf16/
      results.jsonl
      summary.json
    uniform_4bit/
      results.jsonl
      summary.json
    residual_8_4/
      results.jsonl
      summary.json
  comparisons/
    summary.json
    figures/
```

Use a filesystem-safe UTC timestamp plus a short unique suffix for `dated_run_id`, for example `2026-11-07T180000Z_a1b2c3d4`. Record the full timestamp and configuration hash in the manifest. Every fresh invocation creates a new run directory, even when its configuration matches a previous run; explicit resume reuses the original directory.

Method folder names come from validated, unique method IDs in the resolved configuration. Store per-example records only in their method folder; avoid duplicate run-level copies. The run manifest indexes method IDs and their relative result paths. Analysis discovers inputs through this manifest and writes cross-method reports to `comparisons/`.

The manifest should contain:

- Configuration hash and schema version.
- Code commit and dirty-state information.
- Model and tokenizer revisions.
- Package versions.
- Hardware and OS.
- Dataset, split, and evaluator fingerprints.
- Seeds and execution mode.

Each result must identify the run, example, method, model, protocol, and context length, and include quality values, memory fields, and relevant timing measurements. Keep these identifiers in the records even though the directory structure also identifies the run and method.

Represent failures explicitly. Never convert failed or out-of-memory runs into zero scores.

Resume only when the experiment identity matches. Skip completed records, retry failed records explicitly, and detect an interrupted final JSONL write.

## 10. Keep statistical analysis separate from inference

Analysis should consume saved results without loading the model.

Implement:

- Paired differences against BF16.
- Paired differences against a specified uniform comparator.
- Comparisons against random selection.
- Bootstrap confidence intervals.
- Clustering by source where examples are related.
- Predefined task and length aggregation.
- Optional Holm correction for declared comparison families.

Use paired bootstrap for fractional RULER scores and F1. Use McNemar only for genuinely binary outcomes.

Choose policies and comparator configurations using dev data. Require the final comparison configuration to name the selected methods explicitly; do not automatically select a winner using test results.

Generate separate quality–memory plots for RULER, LongBench, and NLL, plus latency comparisons. Do not combine incompatible metrics into one overall score.

## 11. Provide a small CLI

```bash
kvstudy validate configs/smoke.yaml
kvstudy run configs/smoke.yaml
kvstudy run configs/uniform_sweep.yaml
kvstudy resume results/<experiment_name>/<dated_run_id>
kvstudy analyze results/<experiment_name>/<dated_run_id>
kvstudy compare configs/comparison.yaml
```

Validation should summarize the expanded run count and identify unsupported configurations before expensive execution.

Dataset preparation should produce versioned split manifests. Download models and datasets explicitly rather than as an unexpected side effect of validation.

## 12. Implement in working milestones

| Milestone | Acceptance condition |
|---|---|
| A. Vertical slice | One model, a few RULER examples, BF16 inference, official scores, saved results |
| B. Uniform quantization | BF16 versus uniform simulation, independent branches, resumable runs |
| C. Evaluation coverage | LongBench and PG-19 adapters work; analysis reads all three |
| D. Selective precision | Random, recency, and residual policies; accurate memory accounting |
| E. Real storage | Packed implementations validated against simulation; real memory and timing |
| F. Publication preparation | Documented reproduction commands, frozen configs, plots, licenses, and limitations |

Test the behaviors that can invalidate the research: branch isolation, quantization accounting, official scorer parity, continuation alignment, split leakage, pairing, and resume correctness. Use CPU-friendly fixtures for unit tests and mark MLX integration tests separately.

**Definition of done:** A researcher can reproduce a small experiment from the README, add another experiment by writing YAML, add a selection policy without modifying the runner, and trace every reported figure back to saved per-example results.

Implement milestone A first and demonstrate it works before expanding the architecture.
