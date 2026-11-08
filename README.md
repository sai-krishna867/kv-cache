# kvstudy

Research harness for one question:

> At a fixed KV-cache memory budget, does preserving selected context tokens at higher precision retain more quality than uniform quantization?

Status: **milestone B**. One model (Qwen3-4B, bf16 weights), RULER examples, official RULER scoring, and two cache methods:
- a BF16 identity baseline;
- uniform per-token affine quantization (2/3/4/5/6/8-bit K and V, group size 32/64/128), simulated by quantize-then-dequantize.

Runs are resumable, and the analysis reports paired differences against the BF16 baseline with source-clustered bootstrap intervals. Selective precision, LongBench, PG-19 and real packed storage come in later milestones (see `IMPLEMENTATION_PLAN.md`).

Background and protocol: `docs/methodology.md`.

## Requirements

- Apple-silicon Mac (MLX backend)
- [uv](https://docs.astral.sh/uv/)
- About 8 GB of disk for the model

## Reproduce the smoke experiment

```bash
# 1. Environment (pinned by uv.lock)
uv sync --all-groups

# 2. Model, at the pinned revision (explicit download; nothing downloads during validate/run)
uv run hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c

# 3. RULER generator, at the pinned commit
git clone https://github.com/NVIDIA/RULER data/upstream/RULER
git -C data/upstream/RULER checkout db43f92ff634bbce1c33d1fb15858aec4bba509a

# 4. Generate RULER examples with the model's tokenizer -> versioned dev/test split manifests
uv run kvstudy prepare ruler \
  --model-id Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --tasks niah_single_1 vt --lengths 4096 8192 16384 --num-samples 25 --seed 42 \
  --out-dir data/manifests/ruler_qwen3-4b_s42

# 5. Validate (checks files, prints the expanded run count), then run
uv run kvstudy validate configs/smoke.yaml
uv run kvstudy run configs/smoke.yaml

# 6. Re-summarize saved results without the model (per-method summaries, paired diffs vs BF16 with bootstrap CIs)
uv run kvstudy analyze results/smoke/<dated_run_id>

# If a run is interrupted, continue it in the same directory
uv run kvstudy resume results/smoke/<dated_run_id>
uv run kvstudy resume results/smoke/<dated_run_id> --retry-failed   # also rerun failed/oom records
```

`resume` refuses to continue if anything that defines the experiment has changed. That covers:
- the resolved config hash;
- the experiment source code, fingerprinted from the files on disk so uncommitted edits count (`analysis/` and `cli.py` excluded);
- the model and tokenizer files;
- package versions;
- hardware and OS;
- dataset and evaluator fingerprints;
- the protocol, system prompt, or methods.

It skips (example, method) pairs that already have a record and removes an interrupted final JSONL line, logging it in `events.jsonl`. Failed records are rerun only with `--retry-failed`. A retry is appended as a new row with `attempt + 1`, and analysis uses the latest attempt.

`niah_single_1` and `vt` use RULER's synthetic noise haystack, so no essay download is needed. Upstream's generator downloads NLTK `punkt` data on first use.

## RULER dev pilot

The pilot adds multi-key, multi-value and multi-query retrieval at 4K/8K/16K. Those tasks need RULER's Paul Graham essay haystack.

```bash
# Essay haystack (upstream script; its two scraping packages are used only for this command)
(cd data/upstream/RULER/scripts/data/synthetic/json && \
  uv run --project ../../../../../../.. --with html2text==2024.2.26 --with beautifulsoup4==4.12.3 \
  python download_paulgraham_essay.py)

# Manifests: same seed and more samples. The existing manifests are only read, and
# --disjoint-from refuses any source that would change split or any reused ID with new content.
uv run kvstudy prepare ruler \
  --model-id Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --tasks niah_single_1 vt niah_multikey_1 niah_multivalue niah_multiquery \
  --lengths 4096 8192 16384 --num-samples 50 --seed 42 \
  --out-dir data/manifests/ruler_qwen3-4b_pilot_s42 \
  --disjoint-from data/manifests/ruler_qwen3-4b_s42/ruler_dev.jsonl data/manifests/ruler_qwen3-4b_s42/ruler_test.jsonl

uv run kvstudy validate configs/ruler_pilot.yaml
uv run kvstudy run configs/ruler_pilot.yaml
```

The essay text is downloaded from the web, not pinned upstream. Its sha256 is recorded in each essay example's metadata.

## Results layout

```text
results/<experiment>/<UTC timestamp>_<8 hex>/
  resolved_config.yaml   # validated config as run
  manifest.json          # config hash, code state, model/tokenizer revision, packages,
                         # hardware/OS, dataset + evaluator fingerprints, seed, mode, methods index
  events.jsonl           # run_started / prefill_failed / run_interrupted / resume_started /
                         # interrupted_write_removed / run_finished
  methods/<method_id>/
    results.jsonl        # one row per example: prediction, references, score, status,
                         # token counts, estimated cache bytes, timings, attempt
    summary.json         # per task x context length aggregates
  comparisons/summary.json   # all methods side by side + paired differences vs the identity baseline
```

Every invocation of `run` creates a new run directory. Failed or out-of-memory examples are recorded with `status` `failed`/`oom` and `score: null`, and are never counted as zero.

Timings in `quality_simulated` mode are wall-clock values for orientation only. They are not systems measurements.

## Configs

- `configs/smoke.yaml`: 3 dev examples each of `niah_single_1` and `vt` at 4K; BF16 vs uniform 4-bit.
- `configs/baseline.yaml`: all dev examples of those tasks at 4K/8K/16K; BF16 only (Phase 0 shape).
- `configs/uniform_sweep.yaml`: the Phase 1 uniform table (B0, U8, A84, A48, U5, U4g32, U4, U3, U2) on the same examples.
- `configs/ruler_pilot.yaml`: B0, U4, U3, U2 on the pilot dev split (five tasks at 4K/8K/16K).

Unknown fields and unsupported values are rejected at validation time. Examples include `execution.mode: systems_real`, `axis: per_channel`, bit widths MLX cannot quantize to, a group size that does not divide the model's head_dim, a compressed method without `protocol.protected_window`, and a window below a method's precision.

## Tests

```bash
uv run pytest -m "not mlx and not integration"   # CPU-only unit tests; never imports MLX or touches Metal
uv run pytest -m mlx                             # tiny random MLX model: split prefill, branch isolation,
                                                 # quantization accounting and format checks, protected window
uv run pytest -m integration                     # real Qwen tokenizer boundary; scorer parity vs RULER clone;
                                                 # resume after an interrupted write, explicit retries, identity check
```

## Licenses and attribution

This project is licensed under Apache-2.0 (`LICENSE`).

`src/kvstudy/benchmarks/_vendor/ruler_eval.py` contains scoring code copied verbatim from [NVIDIA/RULER](https://github.com/NVIDIA/RULER) (Apache-2.0, Copyright 2024 NVIDIA CORPORATION) at the commit pinned above. RULER data is generated by running the upstream generator from the pinned clone; it is not redistributed here.
