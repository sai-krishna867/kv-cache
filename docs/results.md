# Results so far (development splits)

All numbers come from Qwen3-4B (revision `1cfa9a72…`, BF16 weights) on an M5 Pro with 24 GB, under protocol `post_prefill_conversion/v1`: the context is prefilled in BF16, the cache is converted, then the question is read and the answer generated through the converted cache. Every figure traces to `results/<run>/methods/<id>/results.jsonl`.

These are **dev-split** results, used to choose methods. The held-out test evaluation is frozen in `configs/final_test.yaml` (8,911 runs) with its declared comparisons in `configs/comparison_final.yaml`, and has not been run.

Intervals are 95% paired cluster-bootstrap intervals over `source_id` (10,000 resamples). "p_Holm" is Holm-adjusted over the declared family. A bootstrap p-value of 0 means below 1/10,000.

## Runs

| Run | Directory | Runs | Failures |
|---|---|---|---|
| Uniform sweep (Phase 1 table) | `results/uniform_sweep/2026-11-08T040811Z_e295940f` | 279 | 0 |
| RULER pilot (B0, U4, U3, U2) | `results/ruler_pilot/2026-11-08T065719Z_7c0c3fc8` | 588 | 0 |
| BF16 headroom, harder tasks | `results/ruler_hard_baseline/2026-11-08T072613Z_0c53411a` | 73 | 0 |
| Selective sweep, RULER | `results/selective_sweep/2026-11-08T074130Z_906a5c29` | 1,490 | 0 |
| PG-19 dev | `results/pg19_dev/2026-11-08T170737Z_efa1ed1e` | 364 | 0 |
| LongBench dev | `results/longbench_dev/2026-11-08T172848Z_497614a9` | 707 | 0 |
| systems_real | `results/systems_real/2026-11-08T174255Z_caf1d6c6` | 24 (×4 repeats) | 0 |

## Headroom

BF16 scores 100 on `niah_single_1`, `vt`, `niah_multikey_1/2/3` and `niah_multiquery` at 4K–16K, so those cells can only show losses. Cells inside the design's 10–95% band: `qa_1` (50–83), `qa_2` (67–83), `niah_multivalue` (83–98) and LongBench QA (F1 ≈ 48).

## Uniform quantization

Memory per element matches the design table exactly (`b + 32/g`). Pooled over the RULER pilot (147 examples):

| vs B0 | Bits | Score change | Answers changed |
|---|---|---|---|
| U4 | 4.5 | −0.5 [−1.2, 0.0] | 14% |
| U3 | 3.5 | −2.0 [−3.7, −0.7] | 20% |
| U2 | 2.5 | −27.0 [−32.8, −21.7] | 64% |

U4 is close to lossless; U3 loses a little; U2 breaks every task.

## Selective precision at matched memory

Selective methods keep a fraction of context tokens per layer and KV head at high precision; the rest stay at 2 bits. Fractions are set so each method sits at or below its uniform comparator's bytes (3.494 vs 3.5 bits, 2.997 vs 3.0).

### RULER (149 dev examples, 109 sources)

Declared comparisons (`configs/comparison_selective.yaml`):

| Comparison | Bits | Difference | p_Holm |
|---|---|---|---|
| Q82 (residual 8/2) vs U3 | 3.494 vs 3.5 | −11.0 [−16.3, −6.0] | 0.0006 |
| S82 (sinks + recency) vs U3 | 3.494 vs 3.5 | −8.0 [−13.1, −3.2] | 0.0012 |
| R82 (random) vs U3 | 3.494 vs 3.5 | −21.5 [−28.1, −15.4] | <0.0001 |
| Q42 (residual 4/2) vs U3 | 3.498 vs 3.5 | −3.2 [−7.2, +0.5] | 0.12 |
| Q82 vs R82 | equal | +10.5 [+5.0, +16.3] | <0.0001 |
| S82 vs R82 | equal | +13.5 [+8.8, +18.5] | <0.0001 |
| **Q82b3 (residual 8/2) vs U2g32** | **2.997 vs 3.0** | **+10.9 [+5.6, +16.7]** | **<0.0001** |

By task (BF16 score, then change vs BF16):

| Task | B0 | U3 | Q42 | S82 | Q82 | R82 | U2g32 | Q82b3 |
|---|---|---|---|---|---|---|---|---|
| vt | 100 | 0.0 | 0.0 | −10.7 | −0.7 | −22.0 | −32.0 | −2.0 |
| niah_multiquery | 100 | −0.8 | −0.8 | −6.2 | −0.8 | −9.4 | −17.2 | −3.9 |
| niah_multivalue | 91.7 | −6.5 | −5.6 | −13.9 | −18.5 | −24.1 | −33.3 | −18.5 |
| qa_1 | 73.3 | −6.7 | −16.7 | −20.0 | −36.7 | −40.0 | −36.7 | −43.3 |
| qa_2 | 76.7 | −3.3 | −10.0 | −6.7 | −16.7 | −30.0 | −26.7 | −23.3 |

Figure: `results/selective_sweep/…/comparisons/figures/ruler_quality_memory.png`.

### LongBench English QA (101 dev examples, 86 documents; F1)

| vs B0 (F1 48.4) | Bits | F1 change |
|---|---|---|
| U4 | 4.5 | +2.0 [−0.5, +5.2] |
| U3 | 3.5 | −0.5 [−3.8, +2.8] |
| S82 | 3.494 | −8.1 [−14.5, −1.7] |
| Q82 | 3.494 | −9.4 [−16.8, −1.8] |
| R82 | 3.494 | −13.8 [−21.7, −5.8] |
| U2 | 2.5 | −19.1 [−27.2, −10.9] |

### PG-19 (52 dev windows from 19 books; 511 scored tokens each)

| vs B0 (perplexity 16.99) | Bits | Perplexity ratio | ΔNLL per token |
|---|---|---|---|
| U4 | 4.5 | 1.013 | +0.013 [+0.010, +0.016] |
| **S82** | 3.494 | **1.033** | +0.032 [+0.023, +0.041] |
| U3 | 3.5 | 1.058 | +0.057 [+0.051, +0.062] |
| Q82 | 3.494 | 1.151 | +0.141 [+0.121, +0.159] |
| R82 | 3.494 | 1.245 | +0.219 [+0.198, +0.240] |
| U2 | 2.5 | 1.302 | +0.264 [+0.241, +0.286] |

### What the dev data says

- **The answer depends on the budget.** At 3.5 bits, uniform 3-bit beats every 8/2 selective method on RULER and LongBench. At 3.0 bits, residual-selected 8/2 beats uniform 2-bit (group 32) by 10.9 points on RULER.
- **The answer depends on the task.** On exact-string retrieval (`vt`, multi-query), residual selection is nearly lossless even at 3.0 bits. On QA, holding most document tokens at 2 bits costs much more than 3 bits everywhere does. On language modelling (PG-19), sinks + recency at 3.494 bits beats U3.
- **Which tokens are chosen matters.** At equal bytes, residual and sinks + recency both beat random selection by 10–14 points on RULER; random selection is the worst method at every budget.
- **No selector wins everywhere.** Residual is best on retrieval, sinks + recency on PG-19, and uniform on QA at 3.5 bits.

## Real packed storage (`systems_real`)

`vt` examples, 2 per length; medians over 3 repeats after 1 warm-up; fixed 64-token decode; no BF16 reference retained.

| Method | Length | Cache (measured) | Bits | Convert | Decode tok/s | MLX peak |
|---|---|---|---|---|---|---|
| B0 | 16K | 2,304 MiB | 16.07 | — | 25.1 | 10.32 GiB |
| U4 | 16K | 645 MiB | 4.500 | 0.48 s | 25.0 | 11.06 GiB |
| Q82 | 16K | 501 MiB | 3.494 | 1.33 s | 24.3 | 10.80 GiB |
| U2 | 16K | 359 MiB | 2.500 | 0.50 s | 25.3 | 10.78 GiB |

(4K and 8K rows are in each method's `summary.json`.)

- **Memory.** Measured packed buffers equal the analytic estimates exactly (4.500 / 3.494 / 2.500 bits). BF16 shows 16.07–16.27 bits because MLX allocates in 256-token steps.
- **Speed.** Packed caches do not decode faster at these lengths. Decode is dominated by reading the 8 GB of weights, and attention over packed segments uses unfused `quantized_matmul`. Selective conversion costs about 1.3 s at 16K (selection plus gathering).
- **Peak memory** is set by the BF16 prefill that precedes conversion, so it does not drop. Lowering it needs quantization during prefill, which is a different protocol.
- Outputs were identical across repeats.

## Limitations

- **Dev-split results.** Small cells (6–13 examples): treat per-cell numbers as indicative. The test split is untouched.
- **One model, one machine.** Qwen3-4B on MLX only.
- **Simulation.** Quality runs hold dequantized values in BF16. Real packed storage matched the simulation on a tiny model (identical greedy tokens, bytes exact) but was not run at scale for quality.
- **Narrow task coverage.**
  - RULER `cwe` and `fwe` are missing: `cwe`'s word list is a Git LFS object that could not be fetched, and `fwe` needs scipy.
  - LongBench covers only the six English QA datasets.
  - Many RULER cells have no BF16 headroom.
- **Unpinned essay text.** RULER essay-haystack examples use live-downloaded essay text (hash recorded).
- **Window never overflows.** The protected window never overflowed on these benchmarks, so demotion is exercised only in tests.
- **Selector parameters are fixed:** `sink_tokens` = 4, random seed 0, one high/low pair (8/2) plus 4/2. No per-layer budgets or attention-based selection.
