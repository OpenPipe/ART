# Recompute memory calibration, 2026-09-16

First GPU campaign for #915, measuring estimator commit
`22f628d6a8e87d18b28eb9d1d61579520d012d23`. The new selective/no-recompute
estimate covered every executed forward. It remains conservative, particularly
at larger TP sizes. The unchanged full-recompute estimate underestimated every
measured shape; these results do not validate that legacy estimate.

[CSV evidence](trainer_rank_recompute_memory.csv) contains 54 model/topology/mode/
shape cells. Each row aggregates both repetitions and all ranks: peaks are maxima,
available memory is the minimum, and refused cells have blank measurement fields.
There were 202 measured rank-samples and 22 refused rank-samples. All measured
losses were finite, and every backward completed without CUDA OOM.

## Method

- NVIDIA H200, PyTorch 2.11.0+cu128, CUDA 12.8, bf16, compiled transformer layers.
- Native Qwen3.8-27B (64 layers) at TP1/2/4, and attention-only Qwen3-1.7B at TP2.
  DP/CP/PP are 1; active LoRA rank is 1. Models and adapters use random weights.
- Each mode runs in a fresh process: `full/uniform/1`, selective `core_attn`, or
  no recompute. Every sample clears the learned memory profile. The first sample
  includes any compilation/autotuning required by that shape; the second is warm.
- Requests retain hidden states with gradients. The driver executes the selected
  unsplit native plan only when its cold admission check passes, then backpropagates
  a mean-square hidden-state loss. No admission bypass, split, or optimizer step.
- Estimates and measured peaks below are **incremental allocated GiB above the
  pre-forward baseline**, not total GPU usage. Forward/backward peaks are separate
  CSV fields; backward includes the diagnostic loss. Allocator reservation is not
  the measured allocation peak.
- GDN lengths: 1,024 / 2,048 / 3,072 / 4,096, plus two synthetic sibling sequences
  of 19,221 and 19,222 tokens sharing 5,733 prefix tokens. The siblings reproduce
  #913's 38,443 logical / 32,710 packed tokens (32,712 after TP4 padding), not its
  original token contents. Attention-only lengths: 1,024 / 4,096 / 16,384.
- TP2 ran locally. SkyPilot jobs `art-915-recompute-tp1-0916:1` and
  `art-915-recompute-tp4-0916:1` both succeeded on free `k8s/cks-wb3` capacity.
  Both clusters had ten-minute autodown and two-hour pod deadlines, and were
  explicitly torn down after evidence retrieval.

## Results

Qwen3.8-27B, 2,048 tokens:

| TP | New estimate | Selective forward peak | No-recompute forward peak |
| -- | --: | --: | --: |
| 1 | 58.321 | 31.440 | 31.443 |
| 2 | 58.321 | 18.199 | 18.201 |
| 4 | 58.321 | 10.368 | 10.367 |

The long sibling group was refused in both non-full modes at all three TP sizes.
Its estimate is 931.552 GiB at TP1/2 and 931.609 GiB at TP4. These are admission
results, not measured non-full peaks. Smaller false refusals remain possible:
TP1 declined 3,072 and 4,096 tokens; TP2 declined 4,096 tokens.

Full recompute completed the long sibling group:

| TP | Legacy estimate | Forward peak |
| -- | --: | --: |
| 1 | 5.894 | 26.965 |
| 2 | 5.894 | 13.647 |
| 4 | 5.894 | 6.988 |

Qwen3-1.7B, TP2:

| Tokens | New estimate | Selective forward peak | No-recompute forward peak |
| --: | --: | --: | --: |
| 1,024 | 2.531 | 1.341 | 1.342 |
| 4,096 | 10.123 | 5.113 | 5.116 |
| 16,384 | 40.494 | 20.451 | 20.465 |

## Remaining work

Keep #915 in draft. This campaign supports the retained-activation floor for the
tested dense attention/GDN shapes; it is not a general memory bound. A universal
division by TP would already underestimate the attention-only cold 1,024-token
case (2.531 / 2 < 1.341 GiB). A tighter model must separate replicated activations
and fixed workspace from sharded storage, then validate on held-out shapes.

Full recompute needs a separate correction for retained checkpoint boundaries and
workspace; preserving its old heuristic does not make it calibrated. MoE/EP/CP,
other full-recompute intervals, optional selective modules, deep prefix trees,
mixed gradient groups, other LoRA ranks, and pretrained correctness are untested.
This commit adds measurement evidence without changing the estimator coefficients.

## Reproduce

After `INSTALL_VLLM_RUNTIME=false bash src/art/megatron/setup.sh`, run on two GPUs:

```sh
ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE=2 \
ART_MEGATRON_CONTEXT_PARALLEL_SIZE=1 \
ART_MEGATRON_DATA_PARALLEL_SIZE=1 \
ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE=1 \
uv run --project megatron_runtime --no-sync python -m torch.distributed.run \
  --standalone --nproc-per-node=2 dev/trainer_rank_recompute_memory.py \
  --mode selective --tokens 1024 2048 3072 4096 --reported-pair \
  --evidence scratch/recompute-memory/tp2-selective.jsonl
```

Repeat with `--mode none` and `--mode full`. For the attention-only control, add
`--model Qwen/Qwen3-1.7B --tokens 1024 4096 16384` and omit `--reported-pair`.
The [SkyPilot task](trainer_rank_recompute_memory.sky.yaml) runs all three modes
at TP4; pass the synced commit as `ART_CALIBRATION_SOURCE_SHA` and launch with
`--idle-minutes-to-autostop 10 --down`. Override both GPU count and TP environment
variable for TP1. The driver emits per-rank JSONL and identical rows in job logs.
Driver SHA-256 for this campaign:
`33a7291041913b60232ac2e287edf26c7766b7ab268af034dbcd71df0eae840e`.
