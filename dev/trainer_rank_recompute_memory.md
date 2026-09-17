# Recompute memory calibration

The revised estimate is **10–13% above measured 27B peaks** for the paired
2k-and-larger workloads, including MLP recompute. The previous TP4 estimate was
about 60% high. The new floor admits a 12k TP4 pair and, at TP8, the original
19,221 + 19,222-token pair from #913. Both complete forward and backward.

All numbers below are **incremental allocated GiB above the pre-forward
baseline**, not total GPU use or reserved memory. The estimate includes the
existing 10% safety margin. Measurements use native H200 execution with bf16,
rank-1 LoRA, random model/adaptor weights, SP enabled, and DP/CP/PP=1.

## Requested 27B series and admission boundary

Qwen3.8-27B has 48 GDN and 16 full-attention layers. Pairs share 30% of their
prefix; the planner chooses the layout, including TP padding.

| Tokens per sequence | TP | Packed tokens | Previous estimate | New estimate | Compiled forward | Eager forward |
| --: | --: | --: | --: | --: | --: | --: |
| 2,048 | 4 | 4,096 | 32.974 | 22.854 | 20.513 | 20.550 |
| 4,096 | 4 | 6,964 | 56.075 | 38.789 | 35.077 | 35.071 |
| 8,192 | 4 | 13,928 | 112.151 | 77.273 | 69.800 | 69.926 |
| 12,288 | 4 | 20,892 | — | 115.757 | 104.661 | 104.780 |
| 2,048 | 8 | 4,096 | 19.259 | 15.917 | 14.462 | — |
| 4,096 | 8 | 6,968 | 32.775 | 27.027 | 24.613 | — |
| 8,192 | 8 | 13,928 | 65.513 | 53.835 | 48.947 | — |
| 16,384 | 8 | 27,856 | 131.025 | 107.484 | 97.649 | — |
| 19,221 + 19,222 | 8 | 32,712 | 153.866 | 126.188 | 114.665 | — |

The 12k TP4 and original long TP8 shapes were not used to fit the component
coefficients. The latter was previously refused, so its true selective peak
was unknown. TP4 still refuses the original pair, now at 181.075 GiB. The new
8k TP4 estimate fits both the measured budget and #913's 119.289 GiB budget.
The issue's error labels say GB but divide bytes by 1024³.

Adding `mlp` to selective recomputation at TP4 gives:

| Tokens per sequence | Estimate | Forward peak | Forward + backward peak |
| --: | --: | --: | --: |
| 2,048 | 12.567 | 11.150 | 11.209 |
| 4,096 | 21.300 | 19.006 | 19.163 |
| 8,192 | 42.294 | 37.881 | 38.195 |
| 16,384 | 84.283 | 75.761 | 76.393 |

## What is being priced

Component hooks on eager native layers separated MLP, attention, and GDN
retention. They exposed native activation fusion as the main MLP distinction:
Qwen3.8 uses fused SwiGLU even in eager execution, while Qwen3-1.7B uses the
unfused path. Compilation alone is not a reliable discount because it can fall
back to eager.

Let H be hidden width, F dense FFN width, Q the full query width, KV the full
key/value width, and K/V the GDN key/value widths. SP is TP when sequence
parallel is enabled, otherwise 1. The common per-component term is
`2H/SP + gathered`, with `gathered = H` when SP>1 and zero otherwise: without
sequence sharding, the LoRA input aliases norm output.

- Dense MLP storage is `common + cF/TP`: c=3 for native fused activations,
  c=5 for unfused SwiGLU, with an additional allowance for unfused clamping.
  These count gate/up, activation output, and the unfused SiLU/offset tensors.
- Full attention costs `common + (5Q + 3KV)/TP`, with another `2Q/TP` for
  output gating. If KV groups are fewer than TP ranks, ART's replicated-QKV
  LoRA path adds the global QKV storage that survives slicing. Omitting that
  term underprices TP8.
- GDN costs `common + (4K + 8V)/TP`. Attention and GDN terms are multiplied by
  their actual layer counts. These are calibrated envelopes of the measured
  native paths, not an exact liveness proof for every kernel.
- Checkpointed dense MLPs retain their inputs. Native MoE checkpoint flags
  determine the discounted MoE layer count; its external norm is also priced.
  One full MLP workspace remains charged, including worst-case expert dispatch.
- Each gradient-enabled prefix segment gets an allowance for initial/final GDN
  recurrent states (fp32) and convolution history. Exact plans use actual
  segment counts; cheap admission uses the radix-tree bound, and optimistic
  pruning omits this nonnegative term. A separate 64 MiB allowance covers the
  observed roughly 58 MiB cold kernel setup cost.

The existing static heuristic and routed FC2 bound remain floors. Profiles can
only increase the estimate, and output storage and the 10% safety factor are
applied afterward. Full-recompute and no-grad requests keep their existing path.

The floor also applies with recompute disabled (`None`), including the `none`
override and EP-overlap MoE trainers. Previously admitted workloads may now split
or, for indivisible groups, be refused. Balanced MoE routing can be priced about
3× above its measured peak because the allowance protects uneven dispatch.
Caladan's full-recompute default is unaffected.

## Cross-checks and remaining conservatism

The [CSV](trainer_rank_recompute_memory.csv) includes short-input failures that
motivated the fixed costs. Without segment states, a 64-token 27B TP4 pair was
estimated at 0.776 GiB against 0.832 GiB observed. The corrected estimate is
0.934 GiB against a fresh eager peak of 0.842 GiB. Eight unshared 64-token
sequences at TP2 also pass: 6.294 estimated versus 5.262 observed. Larger
8-sequence batches and 4B MLP-checkpointed pairs provide additional checks.

Qwen3.5-4B TP2 with MLP recompute, pairs of 2,048 / 5,120 / 10,240 tokens and a
50% common prefix, yields estimates 7.253 / 17.872 / 26.801 against peaks
6.443 / 16.036 / 24.078 GiB. A 9B TP2 check used previously unmeasured 5k / 10k
lengths with a 60% prefix: 25.002 / 49.936 estimated against 22.832 / 45.405 GiB,
before adding the nonnegative segment-state term.

The unfused attention-only eager control is covered from 64 through 16,384
tokens per sequence. Its larger inputs have about 9% headroom. Compiled-only
Qwen3-1.7B peaks are lower: 12.172 / 24.275 estimated against 8.832 / 17.515 GiB
for 4k / 8k pairs. That remaining 38–39% overestimate preserves the eager
fallback allowance; it does not apply to the natively fused 27B MLP.

MoE cannot assume balanced expert dispatch. Qwen3.5-35B-A3B, TP4/EP4/ETP1:

| Tokens per sequence | Default estimate | Balanced peak | Concentrated peak | With `moe`: estimate | Balanced peak | Concentrated peak |
| --: | --: | --: | --: | --: | --: | --: |
| 1,500 | 18.283 | 6.483 | 13.941 | 4.290 | 3.749 | 4.056 |
| 3,000 | 31.023 | 10.380 | 23.027 | 7.236 | 5.681 | 6.193 |
| 6,000 | 61.878 | 20.297 | 45.639 | 14.305 | 11.291 | 12.343 |

The concentrated eager run routes about 31/32 of tokens to the first expert
rank; the balanced run uses the native random-weight router with compilation.
Their difference includes both routing and execution mode. The routed term
uses actual top-k expert plus shared FFN widths, not the unrelated dense FFN
fallback. It still makes no EP/ETP discount and is deliberately loose for
balanced routing. MoE recomputation removes most of that retained storage.

A fully concentrated stress attempt segfaulted in Transformer Engine's grouped
GEMM on peers receiving zero tokens, before a complete measurement. The revised
stress keeps those peers nonempty. This kernel limitation is not fixed by the
memory estimate and is not counted as a successful calibration cell.

## Evidence and reproduction

The final campaign contains **45 cells, 360 measured rank-samples, and 4 refused
rank-samples**. Every measured forward is covered; every backward completes;
all measured losses and adapter gradients are finite. The smallest observed
forward headroom is 5.4%. The CSV also preserves earlier controls and negative
controls as separate phases, with the actual source/driver hash for each row.
It aggregates maximum peaks across every rank and both repetitions, and minimum
available memory. Refusals have no invented observed peak.

Estimator source: `850e20a2c4c60247430b3d2f4e463eaf860f92a7`. Stress-driver-only
revision: `7760536e7bf10e4054d708383a1333e7b9f0e120`. A later type annotation does
not change the calculation. Native execution uses PyTorch 2.11.0+cu128, CUDA
12.8, and H200s. Component instrumentation was used only for diagnosis; final
measurements have no component hooks.

Every sample clears learned profiles, gradients, and unused cached allocations.
The first workload in each process includes cold kernel setup; later lengths
can reuse initialized kernels. CSV `first_` and `repeat_` columns refer to the
two executions of each shape, not independent fresh processes. The driver tries
the minimum-memory layout before refusing, executes no split or budget bypass,
and backpropagates a mean-square hidden-state loss. Peak collection precedes
the finite-gradient checks. No optimizer step is included.

After `INSTALL_VLLM_RUNTIME=false bash src/art/megatron/setup.sh`:

```sh
ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE=4 \
ART_MEGATRON_CONTEXT_PARALLEL_SIZE=1 \
ART_MEGATRON_DATA_PARALLEL_SIZE=1 \
ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE=1 \
uv run --project megatron_runtime --no-sync python -m torch.distributed.run \
  --standalone --nproc-per-node=4 dev/trainer_rank_recompute_memory.py \
  --mode selective --pairs --tokens 64 256 2048 4096 8192 12288 --reported-pair \
  --evidence scratch/recompute-memory-calibrated/tp4-selective.jsonl
```

Use `ART_DISABLE_MEGATRON_COMPILE=1` for eager, `--modules core_attn mlp` for
dense MLP checkpoints, or `--modules core_attn moe` for MoE checkpoints.
`--sequences 8 --prefix-fraction 0` checks many short unshared sequences.
`--concentrate-routing` enables the expert-imbalance stress. When running four
processes on an eight-GPU host, also set
`ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE=4` and
`ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE=1` for the MoE case.

The [SkyPilot task](trainer_rank_recompute_memory.sky.yaml) uses free Kubernetes.
Final jobs: `art-915-tight-tp4-0917:6,7,8,9` and
`art-915-tight-tp8-0917:5,7`; TP2 controls ran locally. Both clusters had autodown
and two-hour pod deadlines and were explicitly torn down after evidence
retrieval. No paid clusters were used. Raw JSONL is retained locally under
`scratch/recompute-memory-calibrated/`, with diagnostic and negative-control
runs in `scratch/recompute-memory-components/` and `scratch/recompute-memory-tight/`.

The [previous report](https://github.com/OpenPipe/ART/blob/57f9de2f9/dev/trainer_rank_recompute_memory.md)
records the looser estimate and the earlier full-recompute underestimation.
This initial campaign does not validate that legacy path, larger LoRA ranks, CP,
pretrained routing distributions, arbitrary deep prefix trees, or other hardware/kernels.
These measurements support the calibrated native paths, not a universal memory
bound.

## Context-parallel follow-up (#922)

Dense selective and disabled-recompute floors now use the largest actual
attention/GDN token load on any CP rank, summed across forward groups and padded
as execution pads them. Balanced layouts approach a 1/CP token discount. Uneven
layouts must use their actual load: the 27B 4k pair packs 6,964 global tokens but
puts 4,096 on its busiest rank. Dividing by two would underprice its observed
62.918 GiB peak. GDN segment states and cold workspace are not divided by CP;
gathered outputs, the old static floor, and profile floors remain unchanged.

Width selection uses the existing exact-plan fallback for these CP requests,
reusing the native planner cache. This requires more CPU planning than a global
token-count probe. The average CP load is used only as an optimistic split-search
bound. MoE retains the previous estimate because expert dispatch can concentrate
tokens from multiple CP ranks; it needs separate calibration before a discount.

The follow-up measurements use two local H200s, TP1/CP2, eager native execution,
bf16, rank-1 LoRA, random weights, and paired sequences with 30% shared prefixes.
The [CP CSV](trainer_rank_recompute_memory_cp.csv) records maxima across ranks and
both repetitions, including source hashes and forward/backward peaks. Values
below are incremental allocated GiB and include the existing 10% estimate margin.

| Model / tokens per sequence | Recompute | Previous estimate | CP estimate | Forward peak |
| --- | --- | ---: | ---: | ---: |
| 27B / 2,048 | selective | 69.123 | 34.954 | 31.518 |
| 27B / 4,096 | selective | 117.374 | 69.524 | 62.918 |
| 4B / 4,096 | selective | 41.478 | 20.922 | 18.727 |
| 4B / 4,096 | none | 41.478 | 20.922 | 19.356 |
| 4B / 4,096 | selective + mlp | 25.760 | 13.063 | 11.581 |
| 1.7B / 4,096 | selective | 21.002 | 15.477 | 13.932 |

All 13 cells (52 rank-samples) cover the measured forward peak, with at least 8.1%
headroom; every backward completes with finite losses and adapter gradients.
The 64-token 4B pair's cold backward peaks above its forward estimate (0.699 vs
0.602 GiB); admission estimates forward memory, not arbitrary caller backward.
Estimator source is `7d5fb47ce164fce3801cc8df1ce531c3a84832f5`; the initial 4B
selective series uses `7467f35bb179ae11de6dfd960ff1b037ab4b8690`, before a TP-padding
correction that does not change TP1. The CSV records both source and driver hashes.

The 27B 4k pair now fits its 81.868 GiB incremental budget and completes backward;
the previous estimate would refuse it. To reproduce, use the command above with
TP=1, CP=2, `ART_DISABLE_MEGATRON_COMPILE=1`, and `--nproc-per-node=2`.
Use `--model Qwen/Qwen3.5-4B` for the 4B checks and `--mode none` or
`--modules core_attn mlp` for the other modes. Raw JSONL is retained under
`scratch/recompute-memory-cp/`. This extends calibration to these dense CP2
paths; larger CP sizes, combined TP/CP, and MoE CP remain uncalibrated.
