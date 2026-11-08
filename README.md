# kvstudy

Research harness for one question:

> At a fixed KV-cache memory budget, does preserving selected context tokens at higher precision retain more quality than uniform quantization?

Status: **implementation checkpoint; research validation is ongoing** (see `IMPLEMENTATION_PLAN.md`). One model (Qwen3-4B, bf16 weights) on MLX, and:
- benchmarks: RULER (official scoring), LongBench v1 English QA (official F1), PG-19 (cache-aware continuation NLL);
- cache methods: a BF16 identity baseline; uniform per-token affine quantization (2/3/4/5/6/8-bit K and V, group size 32/64/128); selective precision, with a fraction of context tokens per layer and KV head kept at high precision, chosen by `random`, `sinks_recency` or `quantization_residual`;
- one protected window shared by all compressed methods;
- two modes: `quality_simulated` (quantize-then-dequantize on independent branches of one BF16 prefill, estimated memory) and `systems_real` (packed caches, measured memory and timing). Packed-versus-simulated validation tests are included; verify them on the target Mac before relying on systems results.

Runs are resumable, and the analysis reports paired differences against the BF16 baseline with source-clustered bootstrap intervals, declared comparisons with Holm correction, and figures.

Background and protocol: `docs/methodology.md`. Run outputs and summaries are saved locally under `results/`; a final research report is pending. Extending: `docs/adding_experiments.md`, `docs/adding_methods.md`.

## Requirements

- Apple-silicon Mac (MLX backend)
- [uv](https://docs.astral.sh/uv/)
- About 8 GB of disk for the model

## Reproduce the smoke experiment from a fresh clone

Run these commands in a native macOS terminal with Metal GPU access. The exact synthetic RULER manifests are included in Git; you do not need to generate data or clone RULER for this quick start. Model weights are downloaded separately into the Hugging Face cache, outside this repository.

```bash
# 1. Clone and install the locked environment
git clone https://github.com/sai-krishna867/kv-cache.git
cd kv-cache
uv sync --locked --all-groups

# 2. Verify the bundled data before running
uv run python scripts/verify_datasets.py

# 3. Model, at the pinned revision (about 8 GB; explicit download)
uv run hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c

# 4. Validate, then run: 6 dev examples x 2 methods = 12 evaluations
uv run kvstudy validate configs/smoke.yaml
uv run kvstudy run configs/smoke.yaml

# 5. Replace <dated_run_id> with the directory printed by the run command
uv run kvstudy analyze results/smoke/<dated_run_id>

# If a run is interrupted, continue it in the same directory
uv run kvstudy resume results/smoke/<dated_run_id>
uv run kvstudy resume results/smoke/<dated_run_id> --retry-failed   # also rerun failed/oom records
```

To reproduce the original uniform development sweep on the bundled data:

```bash
uv run kvstudy validate configs/uniform_sweep.yaml
uv run kvstudy run configs/uniform_sweep.yaml
```

This evaluates 31 dev examples against nine configurations (279 evaluations). `configs/baseline.yaml` evaluates the same 31 examples with BF16 only. Preserve the 119 bundled test examples for the final evaluation; do not use them to select methods or hyperparameters.

Experiments in `quality_simulated` mode simulate low-bit numerical errors but store dequantized cache values in BF16. Their memory estimates are not measured savings, and their timings are not packed-cache performance measurements; those come from `systems_real` runs (`configs/systems_real.yaml`). Matching inputs does not guarantee bit-identical answers across different hardware or library versions.

See [dataset provenance and regeneration](docs/datasets.md) for what is bundled, checksums, and preparation of larger datasets. Record the Git commit used for each reproduction. Raw run outputs remain local under `results/`.

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
# Required only for generating additional RULER data, not the quick start
mkdir -p data/upstream
git clone https://github.com/NVIDIA/RULER data/upstream/RULER
git -C data/upstream/RULER checkout db43f92ff634bbce1c33d1fb15858aec4bba509a

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

## Larger datasets (not bundled)

```bash
MODEL="--model-id Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c"
M=data/manifests

# RULER QA source files (upstream download_qa_dataset.sh URLs; HotpotQA from its Hugging Face fallback)
(cd data/upstream/RULER/scripts/data/synthetic/json && \
  curl -sSfL https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -o squad.json && \
  curl -sSfL https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json -o hotpotqa.json)

# Harder RULER tasks (BF16 headroom check), then the selective-sweep task set
uv run kvstudy prepare ruler $MODEL --tasks niah_multikey_2 niah_multikey_3 qa_1 qa_2 \
  --lengths 4096 8192 16384 --num-samples 25 --seed 42 --out-dir $M/ruler_qwen3-4b_hard_s42 \
  --disjoint-from $M/ruler_qwen3-4b_{s42,pilot_s42}/ruler_{dev,test}.jsonl
uv run kvstudy prepare ruler $MODEL --tasks vt niah_multivalue niah_multiquery qa_1 qa_2 \
  --lengths 4096 8192 16384 --num-samples 50 --seed 42 --out-dir $M/ruler_qwen3-4b_selective_s42 \
  --disjoint-from $M/ruler_qwen3-4b_{s42,pilot_s42,hard_s42}/ruler_{dev,test}.jsonl

# LongBench: pinned code and data, English QA bucketed by natural rendered length (no truncation)
git clone https://github.com/THUDM/LongBench data/upstream/LongBench
git -C data/upstream/LongBench checkout 2e00731f8d0bff23dc4325161044d0ed8af94c1e
uv run hf download THUDM/LongBench data.zip --repo-type dataset --revision 5e628be450b7e67fb7ae6e201bd6d8f7056f7672
uv run kvstudy prepare longbench $MODEL --tasks hotpotqa 2wikimqa musique multifieldqa_en qasper narrativeqa \
  --lengths 4096 8192 16384 --out-dir $M/longbench_qwen3-4b

# PG-19 test books (DeepMind's public bucket, md5-verified), one window per book and length
uv run kvstudy prepare pg19 $MODEL --pg19-split test --lengths 4096 8192 16384 \
  --continuation-tokens 512 --num-books 100 --seed 42 --out-dir $M/pg19_qwen3-4b_s42

uv run python scripts/verify_datasets.py --all
```

Each regenerated earlier example must come back identical under the same ID and split; `--disjoint-from` refuses anything else. RULER `cwe` is not prepared: its word list is a Git LFS object in the clone, and the LFS host was unreachable here.

The essay text is downloaded from the web, not pinned upstream. Its sha256 is recorded in each essay example's metadata. After preparing the pilot, run `uv run python scripts/verify_datasets.py --all`: missing optional datasets are reported, and existing datasets must match the published fingerprints. A mismatch means the data is not an exact reproduction; preserve it under a new version rather than changing the published checksums.

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
  comparisons/<name>.json    # declared comparisons (kvstudy compare)
  comparisons/figures/       # kvstudy plot
```

Further commands on a saved run:

```bash
uv run kvstudy plot results/<name>/<run_id>       # comparisons/figures/: quality-memory per benchmark; systems latency/memory
uv run kvstudy compare configs/<comparison>.yaml  # declared {method, against} pairs, Holm-corrected -> comparisons/<name>.json
```

Every invocation of `run` creates a new run directory. Failed or out-of-memory examples are recorded with `status` `failed`/`oom` and `score: null`, and are never counted as zero.

Timings in `quality_simulated` mode are wall-clock values for orientation only. They are not systems measurements.

## Configs

- `configs/smoke.yaml`: 3 dev examples each of `niah_single_1` and `vt` at 4K; BF16 vs uniform 4-bit.
- `configs/baseline.yaml`: all dev examples of those tasks at 4K/8K/16K; BF16 only (Phase 0 shape).
- `configs/uniform_sweep.yaml`: the Phase 1 uniform table (B0, U8, A84, A48, U5, U4g32, U4, U3, U2) on the same examples.
- `configs/ruler_pilot.yaml`: B0, U4, U3, U2 on the pilot dev split (five tasks at 4K/8K/16K).
- `configs/ruler_hard_baseline.yaml`: BF16 headroom check on decoy-needle multi-key and QA tasks.
- `configs/selective_sweep.yaml`: uniform vs selective (residual, random, sinks+recency) at matched budgets on the RULER tasks with headroom or quantization loss.
- `configs/longbench_dev.yaml`, `configs/pg19_dev.yaml`: the same comparison on LongBench QA F1 and PG-19 NLL.
- `configs/systems_real.yaml`: packed storage; measured memory and timing for B0, U4, U2, Q82.
- `configs/comparison_selective.yaml`: declared comparisons on the selective sweep (one Holm family).

Unknown fields and unsupported values are rejected at validation time. Examples include `execution.mode: systems_real` without its measurement settings, `axis: per_channel`, a selector without its parameter, bit widths MLX cannot quantize to, a group size that does not divide the model's head_dim, a compressed method without `protocol.protected_window`, and a window below a method's precision.

## Tests

```bash
uv run pytest -m "not mlx and not integration"   # CPU-only unit tests; never imports MLX or touches Metal
uv run pytest -m mlx                             # tiny random MLX models: split prefill, branch isolation,
                                                 # quantization accounting and format checks, protected window,
                                                 # selectors, continuation NLL, packed storage vs simulation
uv run pytest -m integration                     # real Qwen tokenizer boundary; scorer parity vs RULER clone;
                                                 # resume after an interrupted write, explicit retries, identity check
```

## Licenses and attribution

This project is licensed under Apache-2.0 (`LICENSE`).

`src/kvstudy/benchmarks/_vendor/ruler_eval.py` contains scoring code copied verbatim from [NVIDIA/RULER](https://github.com/NVIDIA/RULER) (Apache-2.0, Copyright 2024 NVIDIA CORPORATION) at the commit pinned above. `src/kvstudy/benchmarks/_vendor/longbench_eval.py` contains scoring code copied verbatim from [THUDM/LongBench](https://github.com/THUDM/LongBench) (MIT, Copyright 2023 THU-KEG & Zhipu AI; license text in the file) at commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`. Bundled synthetic examples were produced with that generator; see [dataset provenance](docs/datasets.md). The project license does not replace the terms of third-party datasets. Essay-containing RULER manifests, LongBench documents, PG-19 books, and model weights are not bundled.
