# Default trainer row arithmetic

`build_training_runtime` unconditionally registers `modules.install` after LoRA
construction and before DDP/optimizer construction. There is no runtime switch.
The hook replaces instance forwards and keeps existing Parameters, sharding,
checkpoint keys and module hooks. Quantized, delayed-wgrad and split-parameter
linear modes fail before installation instead of silently using different math.

`fixed-tree-rows-v1` defines one four-warp RMSNorm program per row, FP32
accumulation, a width-only power-of-two reduction, explicit zero-centered gamma,
and one final output cast. The backward computes the channel contraction with
the same row geometry and weight gradients with 128-row partial trees accumulated
in row order. GDN applies this norm before its existing FP32 SiLU gate and output
cast; gradients reach both norm input and gate. Fused norm-linear returns the
same differentiable normalized activation to LoRA. Fused residual norms retain
the residual output.

Dense forward, dgrad and wgrad use 128x64x32 M/N/K geometry, four warps, two stages,
IEEE FP32 input math and a single increasing-K accumulator: no autotuning,
split-K or atomics. Tail operands are explicitly zero. Rank-3 inputs flatten all
leading dimensions and restore them afterward. LoRA, segmented adapters, expert
base weights, shared/per-expert A/B factors, router projections and the native
LM head use these primitives. Expert groups execute independent fixed GEMMs;
this conservatively replaces the unqualified grouped backend and may cost
substantially more launches. The Stark target head lives separately in Caladan
and must use the same primitives. Its logsumexp uses fixed 1024-column trees.
TE cross entropy already uses one program per row, a vocabulary-only block and
fixed vocabulary traversal; its scalar loss reduction depends on the actual loss
normalization as expected. ART resident token scoring uses the fixed logsumexp.

The shared-expert sigmoid gate also uses the fixed GEMM. Its native functional
linear call bypassed module replacement; GPU8 first observed equal inputs and
different outputs at a shared-expert boundary. That observation does not assign
the cause. CPU coverage now includes this gate's prefix, suffix VJP and FP64
parity. Shared-expert overlap is refused before installation because its separate
communication path still contains an unqualified functional gate contraction.

CPU `fixed_sum` and padded matmul are executable ordering contracts and FP64
gradient oracles, not promises of cross-device bitwise equivalence. Shared weight
gradients necessarily depend on all contributing rows. Their reduction schedule
is fixed, but changing the contribution order can change floating-point sums.
Only equal inputs and equal cotangents warrant a row-dgrad equality assertion.

Attention is a separate reduction domain. Existing SM90 block-sparse FLASH uses
head-width-only tiles (128x128 up to D=128), and `_flash_attn_fwd` defaults to
`num_splits=1`; the Torch call supplies no adaptive split. This proves fixed tile
selection, **not arbitrary packed-offset invariance**. Dense canonical execution
isolates/pads roots. Independent unshared packed roots now use root-local native attention subkernels
with canonical contiguous strides in both ART attention entrypoints. This keeps
the whole decoder packed while removing global softmax tile offsets. Shared
trees and CP state composition still need the GPU gates below. FLASH backward now always uses ART's existing Triton
lowering with fixed 32x32 tiles and the required natural-log/log2 LSE and dLSE
conversion, without relying on a process-wide determinism opt-in.

An adapter span may contain several independent roots. Segmented attention
splits each root within its adapter, preserves zero-length adapter slots and
resets local positions; a root crossing adapters remains invalid. GPU8 exposed
the previous one-root-per-adapter restriction. This repair has CPU forward/VJP
and isolation regressions but has not been rerun on a GPU.

FLA GDN chunks are 64 tokens from each segment's origin. ART supplies segment
`cu_seqlens` (or a uniform dense bucket), and CP explicitly refuses non-final
rank boundaries not divisible by 64. Padding/another segment cannot change a
valid segment's chunk origin. CP1 now invokes the same B=1 dense FLA recurrence
per segment for both uniform and varlen buckets: FLA's gate cumsum autotunes on
B and IS_VARLEN, so chunk alignment alone did not fix its reduction tree. The
autograd context retains this geometry in backward. This is a boundary proof
and CPU regression result, not CUDA numerical or backward accuracy qualification: floating-point cross-rank state composition and kernel dispatch
still need matched-input GPU tests.

The retained GPU3 base FLA chunk accuracy failures (1.405% worst relative L2;
232/1400 synthetic gate failures and two retained marked-test failures) remain
unresolved. Chunk prefixes passed; recurrent accuracy was 0.276%. We do not
change the recurrence backend or relax that gate. The undefined-padding and
FlashInfer packed-NaN findings are also retained as separate robustness issues.

Tests: `tests/unit/test_batch_invariant.py` and the grant-only
`tests/integration/megatron/test_batch_invariant.py`. Real-model prefix scores,
step time, CP/packing/fused/elastic behavior and long context are preregistered in
Caladan `docs/design/stark-batch-invariant/GPU-PLAN.md`. CPU checks or offline
compilation do not establish production GPU qualification. Do not merge before
those gates and the source/runtime compatibility checks pass.

GPU8 failed real MoE prefix, backward parity and packing gates. Dense prefix
passed, but dense backward parity and packing failed. Informational T1024 full
steps were about8.3–9.2x slower for MoE and4.0x slower for dense. The subsequent
CPU repairs do not establish GPU qualification or remove these failures.

Design reference: Thinking Machines Lab's MIT
[batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops/tree/f22b1fbe534b1fe045080f44a970a441caaf3c4f)
and [Defeating Nondeterminism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
These are independently written training kernels; no upstream source is vendored.
