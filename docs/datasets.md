# Dataset provenance and reproduction

The repository includes the exact synthetic manifests used for the initial smoke, baseline, and uniform-quantization runs. They contain generated noise, synthetic facts, and variable assignments, rather than scraped essays or source documents.

| Bundled file | Examples | Intended use |
|---|---:|---|
| `data/manifests/ruler_qwen3-4b_s42/ruler_dev.jsonl` | 31 | Development and smoke experiments |
| `data/manifests/ruler_qwen3-4b_s42/ruler_test.jsonl` | 119 | Reserved test evaluation |

Each JSONL record includes its context, question, references, source ID, split, and generation metadata. Both files have adjacent SHA-256 sidecars. The complete frozen inventory is in [dataset_fingerprints.json](dataset_fingerprints.json); this inventory contains hashes and counts, not third-party text.

## Provenance

- Generator: NVIDIA RULER, commit `db43f92ff634bbce1c33d1fb15858aec4bba509a`.
- Upstream: https://github.com/NVIDIA/RULER (Apache-2.0, Copyright 2024 NVIDIA CORPORATION). The Apache-2.0 license is included at the repository root.
- Tokenizer: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Tasks: `niah_single_1` and `vt`.
- Target lengths: 4096, 8192, 16384 tokens; actual rendered lengths are recorded during inference.
- Seed: 42; 25 examples per task and target length, 150 total.
- Dev/test assignment: deterministic source-ID hash, with approximately 20% assigned to dev. Exact realized counts are 31 and 119.

Do not regenerate or overwrite the bundled files for ordinary reproduction. Their exact bytes, rather than a fresh generation with the same seed, are the reference inputs.

## Verification

From the repository root:

```bash
uv run python scripts/verify_datasets.py
uv run python scripts/verify_datasets.py --all
```

The first command requires and checks bundled files. The second additionally checks optional local manifests against the published inventory, explicitly skipping missing optional datasets. A mismatch exits with a nonzero status. These commands do not require model weights or Metal.

For an independent regeneration, install the locked preparation dependencies and download the pinned tokenizer/model as described in the README. Clone the pinned RULER generator, then write to a new directory:

```bash
mkdir -p data/upstream
git clone https://github.com/NVIDIA/RULER data/upstream/RULER
git -C data/upstream/RULER checkout db43f92ff634bbce1c33d1fb15858aec4bba509a
uv run kvstudy prepare ruler \
  --model-id Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --tasks niah_single_1 vt --lengths 4096 8192 16384 --num-samples 25 --seed 42 \
  --out-dir data/manifests/ruler_qwen3-4b_s42_regenerated
```

Skip the clone command if the clone already exists. Compare each regenerated file's SHA-256 with the corresponding bundled file. Generator or dependency differences must be investigated rather than silently replacing the reference inputs.

## Larger datasets

The larger RULER pilot embeds Paul Graham essay text; the harder RULER set includes third-party QA data. LongBench and PG-19 also contain third-party source text. These manifests are not included in the public repository. The generator's software license alone does not establish redistribution terms for those texts.

The README contains the pinned RULER pilot generation commands. Its essay download uses live web content, so regeneration can differ from the original corpus. Published hashes detect that difference; they cannot reconstruct missing source text. Exact reproduction of those historical runs requires matching source artifacts obtained under their applicable terms.

LongBench and PG-19 preparation adapters are being integrated. Use their explicit `kvstudy prepare longbench --help` and `kvstudy prepare pg19 --help` interfaces; document and freeze their final configuration and source provenance before claiming a reproducible release. The optional fingerprints capture the local manifests available when this inventory was published, not a claim that all corresponding experiments are finished.

Model weights and raw results remain outside Git. Record the code commit, resolved configuration, dataset hash, model revision, package versions, hardware, and OS with each new run. A reproduction on another machine should create a new run; do not resume another machine's run directory.
