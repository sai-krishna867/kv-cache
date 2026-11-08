# Adding an experiment

An experiment is one YAML file in `configs/`. No code changes are needed for a new combination of model revision, benchmarks, lengths and methods that the schema already supports.

1. **Data.** Prepare a versioned manifest if the one you need does not exist. Manifests are written once; pass earlier ones to `--disjoint-from` so no source changes split:

   ```bash
   uv run kvstudy prepare ruler --model-id Qwen/Qwen3-4B --revision <rev> \
     --tasks qa_1 qa_2 --lengths 4096 8192 16384 --num-samples 50 --seed 42 \
     --out-dir data/manifests/<new_dir> --disjoint-from data/manifests/<old_dir>/ruler_*.jsonl
   uv run kvstudy prepare longbench --model-id ... --revision ... --tasks hotpotqa qasper \
     --lengths 4096 8192 16384 --out-dir data/manifests/<new_dir>
   uv run kvstudy prepare pg19 --model-id ... --revision ... --lengths 4096 8192 16384 \
     --continuation-tokens 512 --num-books 100 --out-dir data/manifests/<new_dir>
   ```

2. **Config.** Copy the closest file in `configs/` and edit it. Every field is validated; unknown fields and unsupported values are errors, never substitutions. Requested task × length cells must exist in the manifest (LongBench buckets are natural, so check which cells are populated). Use dev manifests while choosing methods; name the final methods explicitly before touching test manifests.

3. **Validate**, which prints the expanded run count and checks the model snapshot, manifests and group sizes without downloading anything:

   ```bash
   uv run kvstudy validate configs/<name>.yaml
   ```

4. **Run, resume, analyze:**

   ```bash
   uv run kvstudy run configs/<name>.yaml
   uv run kvstudy resume results/<name>/<run_id>            # after an interruption
   uv run kvstudy analyze results/<name>/<run_id>
   uv run kvstudy plot results/<name>/<run_id>               # needs: uv sync --group plots
   ```

5. **Declared comparisons.** Write a comparison file naming the run and each `{method, against}` pair, then `uv run kvstudy compare configs/<comparison>.yaml`. With `holm: true` the listed pairs form one family.

Every figure and table traces to `methods/<id>/results.jsonl` in the run directory, through `manifest.json`.
