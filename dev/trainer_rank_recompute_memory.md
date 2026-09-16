# Recompute memory calibration, September 16, 2026

The sharded retained-activation estimate admits the requested Qwen3.8-27B
selective TP4 series (two sequences of 2k, 4k, and 8k tokens), bounds measured
forward peaks in compiled and eager execution, and refuses #913's original long
pair before execution. It accounts for the actual 48 GDN / 16 attention layers.

The [CSV](trainer_rank_recompute_memory.csv) records final estimator commit
`aef12a9e8502d181fbc614198a4bc4b438a34e9f`, plus an earlier eager-execution failure
as a negative control. Each row aggregates all ranks and two repetitions for one
model/topology/mode/shape. Peaks are maxima; available memory is the minimum;
refusals have blank measurement fields. Source and driver hashes are per row.

## Requested TP4 calibration

All values are **incremental allocated GiB above the pre-forward baseline**,
not total GPU usage or reserved memory. Model/adaptor weights are random; this
measures the native runtime's memory behavior, not pretrained correctness.

Qwen3.8-27B, 64 layers, bf16, LoRA rank 1, selective `core_attn`, SP enabled:

| Tokens per sequence | Packed tokens | Estimate | Compiled forward | Eager forward | Compiled forward + backward |
| --: | --: | --: | --: | --: | --: |
| 2,048 | 4,096 | 32.974 | 20.576 | 20.613 | 20.759 |
| 4,096 | 6,964 | 56.075 | 35.088 | 35.079 | 35.482 |
| 8,192 | 13,928 | 112.151 | 69.811 | 69.924 | 70.598 |

Each pair shares 30% of its prefix. The planner chose an unshared layout for the
2k pair and shared layouts for 4k/8k; token counts include TP padding. The compiled
run's pre-forward allocation was at most 13.007 GiB, with a usable incremental
budget of at least 120.004 GiB. The 8k prediction also fits the issue's 119.289 GiB
budget. No-recompute peaks were 20.577 / 35.090 / 69.816 GiB under the same estimates.

The original synthetic sibling geometry (19,221 + 19,222 logical tokens, sharing
5,733 prefix tokens) is refused at TP4 with a 263.403 GiB estimate for 32,712 packed
tokens. This is an admission result, **not a measured long-pair selective peak**.

## What changed in the estimate

For gradient-enabled non-full recompute, the retained floor is the packed token
count times dtype bytes times the sum of layer storage. Dense layers retain
`5H/SP + 2H + 6F/TP`, plus `(9A - 5H)/TP` for each attention layer or
`(7G - 5H)/TP` for each GDN layer. Here `H` is hidden width, `F` is FFN width,
`A = max(H, heads * head_dim)`, `G = max(H, 2*key_width + 2*value_width)`, and
`SP` is TP when sequence parallel is enabled, otherwise 1.

- The norm/residual term follows the SP distinction in
  [Megatron's activation model](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py).
  Projection and dense MLP storage receive TP discounts. The separate `2H`
  remains replicated because ART's LoRA wrappers retain gathered attention and
  MLP inputs even with SP (`return_layernorm_output_gathered` and
  `_column_parallel_lora_input` in `src/art/megatron/lora.py`).
- Four FFN widths covered compiled attention-only runs but underestimated eager
  execution. Six cover the measured eager gate/up activations and LoRA sums.
  Both execution modes use six: compilation can fall back to eager.
- GDN uses its own seven-width calibrated envelope for projection, convolution,
  and recurrent storage. Charge it only to actual GDN layers, instead of charging
  every layer the maximum of attention and GDN widths. This is an empirical
  envelope for the measured native paths, not an exact tensor-liveness proof.
- Routed/shared expert FFN and dispatch storage receive no TP/EP/ETP discount.
  Optional `mlp`/`moe` recomputation receives no discount in this revision.
- The existing static estimate and MoE FC2 estimate remain floors. Learned
  profiles can only raise the estimate; output bytes and the existing 10% safety
  margin are still applied. No-grad and full-recompute paths are unchanged.

## Additional evidence

Qwen3.8-27B selective TP8, compiled:

| Tokens per sequence | Estimate | Forward peak |
| --: | --: | --: |
| 2,048 | 19.259 | 14.524 |
| 4,096 | 32.775 | 24.612 |
| 8,192 | 65.513 | 48.947 |

No-recompute peaks were 14.525 / 24.614 / 48.950 GiB. The final estimate refuses
16k pairs (131.025 GiB) and the original long pair (153.866 GiB) at TP8. Those
selective peaks remain unmeasured; TP sharding does not remove gathered storage,
and the estimate can still over-refuse near capacity.

The eager attention-only negative control is Qwen3-1.7B TP2, pairs of
1,024 / 4,096 / 8,192 / 16,384 tokens. Before the six-FFN correction, predictions
were 2.567 / 10.262 / 20.524 / 41.046 GiB against observed
2.835 / 11.121 / 22.223 / 44.297 GiB. Re-running the final code produced the same
peaks with estimates 3.181 / 12.717 / 25.434 / 50.863 GiB. The smallest observed
headroom in the final campaign is 12.2%. Recorded peaks are regression witnesses:
restoring the old estimator fails all four, and dropping the gathered-input term
fails three.

A held-out Qwen3.5-9B model at TP2 used pairs of 3,072 / 6,144 tokens with a 60%
shared prefix, after coefficients were fixed. Estimates 23.158 / 46.305 GiB covered
forward peaks 14.155 / 28.005 GiB and forward/backward peaks 14.352 / 28.399 GiB.

Adding `mlp` to selective recomputation on the 27B at TP4 reduced the 2k / 4k /
8k forward peaks to 11.213 / 19.006 / 37.881 GiB, under the same estimates.
Qwen3.5-35B-A3B at TP4/EP4/ETP1 used 1k / 2k / 4k pairs: estimates
16.852 / 33.705 / 57.310 GiB covered default selective peaks
4.536 / 8.277 / 13.990 GiB. Adding `moe` reduced them to
2.622 / 4.536 / 7.710 GiB. These runs support keeping the current undiscounted
floor; they do not establish per-module discounts or bounds on pretrained routing
imbalance. Compiled attention-only selective and no-recompute controls also pass.

The final campaign contains **46 cells: 296 measured rank-samples and 48 refused
rank-samples**. Every measured forward was covered, every backward completed
without CUDA OOM, and every loss and adapter gradient was finite. The CSV also
includes four cells / 16 rank-samples from the earlier eager negative control;
those failed coverage checks are intentionally preserved, not final-code failures.

## Method, reproduction, and limits

NVIDIA H200, PyTorch 2.11.0+cu128, CUDA 12.8, bf16, DP/CP/PP=1. Each mode runs in a
fresh process. Every repetition clears learned memory profiles, gradients, and
unused cached allocations. Sample 0 includes any first-execution compilation and
autotuning; sample 1 is warm. Both contribute to reported maxima.

The driver plans paired hidden-state requests with gradients, tries the
minimum-memory layout if admission refuses, and executes the unsplit native plan
only if it fits. It then backpropagates a mean-square hidden-state loss. There is
no admission bypass, splitting, or optimizer step. Forward/backward peaks include
the diagnostic loss; finite-gradient checks happen after peak/timing collection.
Raw JSONL also records geometry, every rank/sample, timing, and allocator totals.
Driver SHA-256: `42e9898ce34d566da5b9ebd2d3ef608317ae61b2d51f3ede058216d2ee61e788`.

On a configured GPU machine, after
`INSTALL_VLLM_RUNTIME=false bash src/art/megatron/setup.sh`:

```sh
ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE=4 \
ART_MEGATRON_CONTEXT_PARALLEL_SIZE=1 \
ART_MEGATRON_DATA_PARALLEL_SIZE=1 \
ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE=1 \
uv run --project megatron_runtime --no-sync python -m torch.distributed.run \
  --standalone --nproc-per-node=4 dev/trainer_rank_recompute_memory.py \
  --mode selective --pairs --tokens 2048 4096 8192 --reported-pair \
  --evidence scratch/recompute-memory/tp4-selective.jsonl
```

Use `--mode none`, `ART_DISABLE_MEGATRON_COMPILE=1`, or `--modules core_attn mlp`
for the corresponding controls. Change both process count and TP for TP8.
The [SkyPilot task](trainer_rank_recompute_memory.sky.yaml) defaults to TP4 on
free Kubernetes; pass the synced commit as `ART_CALIBRATION_SOURCE_SHA` and use
`--idle-minutes-to-autostop 15 --down`.

Final cluster jobs: `art-915-sharded-tp4-0916:4` (selective/none), `:5` (eager),
`:6` (MLP recompute), `:7` (MoE), `:8` (MoE recompute), all on free `k8s/cks-wb3`;
`art-915-sharded-tp8-ext-0916:2` on free `k8s/ext-collab2`. TP2 controls ran locally.
The clusters used 15-minute autodown and two-hour pod deadlines. Raw evidence is
retained locally under `scratch/recompute-memory-final/`. No paid clusters were
used; both clusters were explicitly torn down after successful jobs and evidence
retrieval.

The [first unsharded campaign](https://github.com/OpenPipe/ART/blob/be21a2599/dev/trainer_rank_recompute_memory.csv)
remains historical evidence. It also found underestimation in the unchanged full
recompute heuristic (for the long group, 5.894 GiB estimated versus 13.647 GiB
observed at TP2). This PR does not validate or fix that legacy path. Larger LoRA
ranks, pretrained routing imbalance, CP, deeper prefix trees, mixed gradient
groups, and other kernels/hardware are not calibrated by these measurements.
Conservative expert/module pricing remains deliberate; these results support
removing the selective guard for the measured paths, not a universal memory bound.
