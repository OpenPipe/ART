# GLM-5.3 Flash adapter contract: shared_outer_v1

M5 owns this contract. M2/M3 provide projection hooks, M4 installs the wrappers,
M6 gathers EP shards, and M8 loads the serving artifact. Rank `r` is an integer
1–32 for **every** adapted projection, including routed experts. It is never
divided by the eight active experts. The three independent flags default to
true: `train_attn`, `train_mlp`, `train_unembed`. False disables precisely its
group; all false is a valid empty adapter. Attention-only includes both layer
families and must not raise or narrow to MLA up-projections.

## Shapes and counts

Source: `workstreams/glm-mvp/hfmeta/shapes.json`, BF16 weight shapes at the
training revision `f9fed3bbc82291073373d67452e9f25317ef7577`. Its representative
layers 0, 3, and 4 supply the dimensions below; the pinned config/index establish
the repetition counts. The shared expert has the same shape as one routed
expert. An ordinary weight is `[out,in]`, disk A is `[r,in]`, disk B is `[out,r]`,
and `delta_W = (alpha/r) B @ A`. ART stores their transposes as `A_T`, `B_T`.
The expert axis is first. Shapes below count **global unique training
parameters**, not physical replicas on CP/EP ranks or expanded export tensors.

| Target (number of layers) | Base weight | A at r=1 | B at r=1 | A at r=32 | B at r=32 | Parameters per rank, all layers |
|---|---|---|---|---|---|---:|
| KDA q, k, v, separately (34) | [8192,4096] | [1,4096] | [8192,1] | [32,4096] | [8192,32] | 3 × 34 × 12288 = 1,253,376 |
| KDA o (34) | [4096,8192] | [1,8192] | [4096,1] | [32,8192] | [4096,32] | 34 × 12288 = 417,792 |
| MLA q_a (11) | [1536,4096] | [1,4096] | [1536,1] | [32,4096] | [1536,32] | 11 × 5632 = 61,952 |
| MLA kv_a (11) | [512,4096] | [1,4096] | [512,1] | [32,4096] | [512,32] | 11 × 4608 = 50,688 |
| MLA o (11) | [4096,16384] | [1,16384] | [4096,1] | [32,16384] | [4096,32] | 11 × 20480 = 225,280 |
| Dense gate, up, separately (3) | [12288,4096] | [1,4096] | [12288,1] | [32,4096] | [12288,32] | 2 × 3 × 16384 = 98,304 |
| Dense down (3) | [4096,12288] | [1,12288] | [4096,1] | [32,12288] | [4096,32] | 3 × 16384 = 49,152 |
| Shared gate, up, separately (42) | [2048,4096] | [1,4096] | [2048,1] | [32,4096] | [2048,32] | 2 × 42 × 6144 = 516,096 |
| Shared down (42) | [4096,2048] | [1,2048] | [4096,1] | [32,2048] | [4096,32] | 42 × 6144 = 258,048 |
| Routed gate, up, separately (42) | [288,2048,4096] | [1,1,4096] | [288,2048,1] | [1,32,4096] | [288,2048,32] | 2 × 42 × (4096 + 288 × 2048) = 49,889,280 |
| Routed down (42) | [288,4096,2048] | [288,1,2048] | [1,4096,1] | [288,32,2048] | [1,4096,32] | 42 × (288 × 2048 + 4096) = 24,944,640 |
| lm_head (1) | [154880,4096] | [1,4096] | [154880,1] | [32,4096] | [154880,32] | 158,976 |

Attention = **2,009,088r**; MLP = **75,755,520r**; unembed = **158,976r**;
all = **77,923,584r**, or **2,493,554,688** at r=32. KDA layers are all
0–44 except `3,7,11,15,19,23,27,31,35,39,43`; those eleven are MLA. Layers
0–2 have dense MLPs; 3–44 have 288 routed experts and one shared expert.

For routed gate/up, one hidden-side A is shared across all 288 experts **per
projection**. Gate and up do not share A with each other. For routed down,
one hidden-side B is shared across all experts. These are ART #1133's
`LoraLayout(expert="shared_outer")` storage semantics. CP/DP replicas do not
multiply the logical count; EP8 keeps 36 expert-specific factors per rank and
replicates the singleton factor, whose gradients sum across EP as in ART.

Excluded: `q_b_proj`, `kv_b_proj`, all indexer weights, KDA `b_proj`, `f_a_proj`,
`f_b_proj`, `g_a_proj`, `g_b_proj`, convs/norms/gate parameters, router, mHC,
embeddings, MTP and vision. Never use suffix matching against the vision tower.

## Names and packing

All listed disk names end in `.lora_A.weight` or `.lora_B.weight`. Trainer
logical exports use `T = base_model.model.model.layers.{L}`. MCore projection
attributes are bound by M2/M3's posted IFACE; logical disk names remain stable.
HF/PEFT uses `P = base_model.model.model.language_model.layers.{L}`.
vLLM artifacts use **runtime** names under
`V = base_model.model.language_model.model.layers.{L}`; stripping
`base_model.model.` gives actual conditional-generation module names. M8 must
not prepend `model.language_model.` a second time. The `language_model_only`
entrypoint needs a separate namespace conversion and is not this artifact.

| Trainer logical projection | vLLM 0.32 runtime module / slot | PEFT projection |
|---|---|---|
| T.self_attn.q_proj | V.self_attn.in_proj_qkvbfg_a / 0 | P.self_attn.q_proj |
| T.self_attn.k_proj | V.self_attn.in_proj_qkvbfg_a / 1 | P.self_attn.k_proj |
| T.self_attn.v_proj | V.self_attn.in_proj_qkvbfg_a / 2 | P.self_attn.v_proj |
| T.self_attn.o_proj | V.self_attn.o_proj | P.self_attn.o_proj |
| T.self_attn.q_a_proj | V.self_attn.fused_qkv_a_proj / 0 | P.self_attn.q_a_proj |
| T.self_attn.kv_a_proj_with_mqa | V.self_attn.fused_qkv_a_proj / 1 | P.self_attn.kv_a_proj_with_mqa |
| T.mlp.gate_proj / up_proj | V.mlp.gate_up_proj / 0,1 | P.mlp.gate_proj / up_proj |
| T.mlp.down_proj | V.mlp.down_proj | P.mlp.down_proj |
| T.mlp.shared_experts.gate_proj / up_proj | V.mlp.shared_experts.gate_up_proj / 0,1 | P.mlp.shared_experts.gate_proj / up_proj |
| T.mlp.shared_experts.down_proj | V.mlp.shared_experts.down_proj | P.mlp.shared_experts.down_proj |
| T.mlp.experts.{expert}.gate_proj | V.mlp.experts.w1 | P.mlp.experts.gate_up_proj parameter, gate half |
| T.mlp.experts.{expert}.down_proj | V.mlp.experts.w2 | P.mlp.experts.down_proj parameter |
| T.mlp.experts.{expert}.up_proj | V.mlp.experts.w3 | P.mlp.experts.gate_up_proj parameter, up half |
| base_model.model.lm_head | base_model.model.language_model.lm_head | base_model.model.lm_head |

vLLM attention/dense/shared disk keys keep the **unpacked** projection names
(for example `V.self_attn.q_proj.lora_A.weight`). SkyRL's packed mapping on
both GLM entrypoint classes assembles those into the runtime modules above:

```
in_proj_qkvbfg_a = [q_proj, k_proj, v_proj, b_proj, f_a_proj, g_a_proj]
fused_qkv_a_proj = [q_a_proj, kv_a_proj_with_mqa]
gate_up_proj = [gate_proj, up_proj]
```

KDA global output offsets: q=0, k=8192, v=16384, b=24576, f_a=24640,
g_a=24768; total 24896. Slots 3–5 are absent adapters, not extra trainable
zero factors. TP slicing keeps f_a/g_a replicated (slots 4,5). MLA global
offsets are q_a=0, kv_a=1536, total 2048. If M8 uses synthetic RoPE padding,
the additional kv_a B rows must be zeros and must not enter training/counts.

The expert artifact uses **pre-stacked shared-outer** keys
`V.mlp.experts.w{1,2,3}.lora_{A,B}.weight`, with the exact 3D shapes in the
table, and `enable_moe_shared_loras=True`. This path exists in vLLM v0.32.0's
`process_packed_modules_mapping` and `PackedLoRALayerWeights.pack_moe_stacked`.
It preserves the independent gate/up rank-r spaces without rank expansion.
Use `lora_target_modules` including `experts`, the attention packed modules,
dense/shared projections and `lm_head` as selected. M8 supplies SkyRL packing
patch pieces 1–3 and the output-head patch. No kv_b decode/prefill patch is
needed by this target set. Keep `max_num_seqs <= 512` per the serving plan.

## Scale: one convention, no top-k normalization

Stark uses its existing alpha default **32**, explicit `alpha/r` (not rsLoRA),
and full rank r on every target. Tinker's unpublished alpha/init are not
claimed to be known. A starts Kaiming, B starts zero, following ART.

For every expert e and projection p:

```
trainer_delta_W[e,p] = (alpha/r) * B[e,p] @ A[e,p]
sampler_delta_W[e,p] = (serving_alpha/serving_r) * B_export[e,p] @ A_export[e,p]
serving_r = r; serving_alpha = alpha; A_export = A; B_export = B
```

vLLM folds its scale into B in `PackedLoRALayerWeights.optimize`, exactly
once. Export does not pre-scale B. Router top-k=8 and routed scaling=2.5 act
on expert outputs in both bases; neither divides LoRA rank or scale. Reject
`rank_pattern`, `alpha_pattern`, rsLoRA and a reduced expert rank on this
interface. SkyRL's reduced expert rank `r/8` gives trainer scale `8alpha/r`;
shipping those factors with global `alpha/r` loses 7/8 of the update. That
layout is incompatible here. A future converter must explicitly multiply B
by `(alpha_expert/r_expert)/(serving_alpha/serving_r)` and record it; do not
silently treat a SkyRL checkpoint as shared_outer_v1.

## PEFT export

HF's routed weights are fused 3D Parameters, not per-expert `nn.Linear`s.
Use PEFT 0.20.0's `target_parameters` for
`model.language_model.layers.{L}.mlp.experts.gate_up_proj` then `down_proj`,
in that order, and an exact regex of the selected ordinary target module
paths. Never `all-linear`. Stock PEFT cannot maintain shared-outer training
ties; this is an inference/merge export and expands singleton factors.

Export global rank `R=2r`, alpha `2alpha`, so alpha/R stays alpha/r. Pad
ordinary and down factors to R with zero rows/columns. Per expert, gate/up
A is `cat(A_gate,A_up)` and B is `block_diag(B_gate,B_up)`. PEFT gate/up
wrapper is nested under `P.mlp.experts.base_layer`, and down under
`P.mlp.experts`; the four disk keys end with the usual A/B suffixes.
For E experts and width I:

```
gate_up A: [E*R,H]        gate_up B: [2I,R*E]
down A:    [E*R,I]        down B:    [H,R*E]
A = stack(per_expert_A).reshape(E*R,in)
B = stack(per_expert_B).permute(1,2,0).reshape(out,R*E)
```

PEFT 0.20.0 handles `[E,out,in]` base weights directly. Older PEFT
orientations are not interchangeable. Metadata records native rank/alpha,
flags, dimensions and format; import verifies zero padding, block diagonals,
and identical shared factors before collapsing. Modified independent factors
must raise rather than be silently averaged.
All-false flags use PEFT's `dummy-target-modules` sentinel for a loadable
identity adapter with no tensors.

## EP gather interface and validation ownership

M6 calls `glm5next.lora_export.gather_and_export_lora` on already-collected
rank entries (ART `lora_publish` manifests). It merges TP shards through
`merge_sharded_adapter_entries`, then groups **logical expert IDs**, verifies
complete 0..E-1 coverage and identical singleton factors, and emits the
vLLM stack or PEFT expansion. Missing shards, conflicting replicas, duplicate
ownership and shape/flag/scale mismatch must fail before publication. This
function does no collective itself: M6 uses ART's existing manifest/tensor
transport; every participating rank must complete transport before rank zero
converts. Ordinary CP/DP replica selection remains ART's existing policy.

The concrete entrypoints are:

```python
from art.megatron.glm5next.lora import (
    LoraDimensions, LoraFlags, apply_glm5next_lora, parameter_counts,
)
from art.megatron.glm5next.lora_export import (
    adapter_config, gather_and_export_lora,
    to_vllm_lora_tensors, from_vllm_lora_tensors,
    to_peft_lora_tensors, from_peft_lora_tensors,
)

flags = LoraFlags(train_attn=True, train_mlp=True, train_unembed=True)
config = adapter_config(8, dimensions=LoraDimensions.from_config(hf_config), flags=flags)
apply_glm5next_lora(model_chunks, provider, rank=8, alpha=32, flags=flags)
serving_tensors, serving_config = gather_and_export_lora(
    entries_by_key, adapter_config=config, format="vllm",
)
```

`adapter_config` records `art_glm5next_lora` with version `shared_outer_v1`,
format (`art`, `vllm`, `peft`), native `rank`, native `alpha`, `dimensions`
and the three `flags`. Preserve this metadata for validated import. The native
revision is the BF16 training revision; serving may normalize identity to its
pinned FP8 sibling through M6/M8's existing model identity checks.

M2's posted IFACE uses separate `q_proj/k_proj/v_proj/o_proj`. M3's posted
IFACE preserves GLM52's `q_a_lora`, `kv_a_lora` and `linear_proj`. KDA/head
residual hooks register ART `LoRA` children while preserving the original base
parameter paths for Bridge. M4's merged-weight exporter calls
`glm5next.lora_layers.build_glm5next_adapter_weights_by_base(model_chunks)`.
All installed LoRA modules carry the shared-outer layout metadata. M4/M6 must
also bind runtime/snapshot layout to shared_outer_v1; ART's generic non-Qwen
explicit-layout fence is outside these lane-owned files.

M1 F4 passes the predeclared rank-8 BF16 merged-versus-low-rank-branch logits
gate (relative L2 **0.00753053**, threshold **0.01**). That CPU reference uses
HF base modules and the same factors exported to vLLM; M4 owns the separate
complete MCore pipeline qualification. This does not claim a GPU sampler load
or full 45-layer parity measurement.

Pinned source references: vLLM tag `v0.32.0`,
`vllm/models/glm5next/common/{model,kda,attention}.py`,
`vllm/lora/{utils,lora_weights}.py`; SkyRL `3ba6f33d`,
`skyrl/backends/skyrl_train/patches/vllm/patch_glm5next_lora_packing.py`;
ART main `057f1ba76`, `lora.py` and `weights/lora_publish.py`.
