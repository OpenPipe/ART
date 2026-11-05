"""M3 single-layer HF/CP parity driver; run with python -m or torchrun --module.

CPU: --fixture R2 --device cpu --lengths 32. GPU: F4/F8, BF16 TileLang,
512,2048,2052,8192,16384 tokens. CP8 is the same driver in M4's node window.
CPU replaces only the CUDA sparse kernel with differentiable torch math;
the attention module, indexer, CP planner, fetches and reductions are unchanged.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig
from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

from art.megatron.context_parallel.runtime import (
    prepare_megatron_context_parallel_state,
)
from art.megatron.context_parallel.types import ContextParallelConfig, ParallelTopology
from art.megatron.glm5next.dsa import (
    Glm5NextSelfAttention,
    Glm5NextSparseCore,
    glm5next_core_builder,
)
from art.megatron.glm5next.dsa_cp import (
    Glm5NextDsaState,
    build_glm5next_dsa_state,
    dsa_pack_alignment,
)
from art.megatron.glm5next.kpool_indexer import KPoolIndexer
from art.megatron.glm52 import attention as glm52_attention
from art.megatron.glm52 import cp_attention
from art.megatron.training.finalize_grads import coalesced_all_reduce

DIST: Any = torch.distributed


class Linear(nn.Linear):
    def __init__(self, n: int, m: int, **kwargs: Any) -> None:
        super().__init__(n, m, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:  # ty: ignore[invalid-method-override]
        return F.linear(x, self.weight), None


class Norm(nn.LayerNorm):
    def __init__(self, *, hidden_size: int, eps: float, **kwargs: Any) -> None:
        super().__init__(hidden_size, eps=eps)


class LowRank(nn.Module):
    def __init__(self, n: int, m: int, rank: int, weight: torch.Tensor) -> None:
        super().__init__()
        self.a = nn.Parameter(weight.new_empty(n, rank).normal_(std=0.01))
        self.b = nn.Parameter(weight.new_empty(rank, m).normal_(std=0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from art.megatron.glm52.lora_projection import glm52_lora_a

            return glm52_lora_a(x.contiguous(), self.a.contiguous()) @ self.b
        return (x @ self.a) @ self.b


class OutputWithLoRA(nn.Module):
    def __init__(self, base: Any, lora: LowRank) -> None:
        super().__init__()
        self.base, self.lora = base, lora

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.base(x)[0] + self.lora(x), None


def create_attention(
    config: Any, device: torch.device, dtype: torch.dtype
) -> Glm5NextSelfAttention:
    mc = MLATransformerConfig(
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_attention_heads,
        kv_channels=config.v_head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_head_dim=config.qk_nope_head_dim,
        qk_pos_emb_head_dim=0,
        v_head_dim=config.v_head_dim,
        layernorm_epsilon=config.rms_norm_eps,
        hidden_dropout=0,
        attention_dropout=0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        dsa_indexer_n_heads=config.index_n_heads,
        dsa_indexer_head_dim=config.index_head_dim,
        dsa_indexer_topk=config.index_topk,
        dsa_indexer_loss_coeff=0.0,
    )
    submodules = MLASelfAttentionSubmodules(
        linear_q_down_proj=Linear,
        linear_q_up_proj=Linear,
        linear_kv_down_proj=Linear,
        linear_kv_up_proj=Linear,
        core_attention=glm5next_core_builder(Linear, Norm),
        linear_proj=Linear,
        q_layernorm=Norm,
        kv_layernorm=Norm,
    )
    return Glm5NextSelfAttention(
        mc,
        submodules,
        1,
        AttnMaskType.causal,
        pg_collection=SimpleNamespace(tp=None, cp=None),
    ).to(device=device, dtype=dtype)


def load_layer(
    root: Path, config: Any, layer: int, device: torch.device, dtype: torch.dtype
) -> Any:
    reference = Glm5NextTextAttention(config, layer).to(device=device, dtype=dtype)
    prefix = f"model.language_model.layers.{layer}.self_attn."
    state = {}
    index = root / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    for name in files:
        with safe_open(root / name, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = source.get_tensor(key)
    reference.load_state_dict(state, strict=True)
    return reference.requires_grad_(False)


def copy_weights(actual: Glm5NextSelfAttention, reference: Any) -> None:
    for src, dst in (
        ("q_a_proj", "linear_q_down_proj"),
        ("q_b_proj", "linear_q_up_proj"),
        ("kv_a_proj_with_mqa", "linear_kv_down_proj"),
        ("kv_b_proj", "linear_kv_up_proj"),
        ("o_proj", "linear_proj"),
        ("q_a_layernorm", "q_layernorm"),
        ("kv_a_layernorm", "kv_layernorm"),
    ):
        getattr(actual, dst).weight.data.copy_(getattr(reference, src).weight)
    assert isinstance(actual.core_attention, Glm5NextSparseCore)
    indexer = actual.core_attention.indexer
    assert isinstance(indexer, KPoolIndexer)
    indexer.load_state_dict(
        {
            key.replace("wq_b.", "linear_wq_b.")
            .replace("wk.", "linear_wk.")
            .replace("weights_proj.", "linear_weights_proj."): value
            for key, value in reference.indexer.state_dict().items()
        }
    )


def add_lora(actual: Glm5NextSelfAttention, rank: int) -> dict[str, LowRank]:
    adapters = {}
    for name, attr in (
        ("q_a_proj", "linear_q_down_proj"),
        ("kv_a_proj_with_mqa", "linear_kv_down_proj"),
        ("o_proj", "linear_proj"),
    ):
        weight = getattr(actual, attr).weight
        adapters[name] = LowRank(weight.shape[1], weight.shape[0], rank, weight)
    actual.q_a_lora = adapters["q_a_proj"]
    actual.kv_a_lora = adapters["kv_a_proj_with_mqa"]
    actual.linear_proj = OutputWithLoRA(actual.linear_proj, adapters["o_proj"])
    return adapters


def _cpu_math(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, *, scale: float, **_: Any
) -> torch.Tensor:
    mask = (
        indices[..., None]
        == torch.arange(kv.shape[1], device=q.device)[None, None, None, :]
    ).any(-2)
    scores = torch.einsum("bqhd,bkd->bhqk", q, kv) * scale
    probs = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1).nan_to_num()
    return torch.einsum("bhqk,bkd->bqhd", probs, kv[..., :512])


def _cpu_forward(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, *, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return _cpu_math(q, kv, indices, scale=scale), q.new_zeros(
        q.shape[:-1], dtype=torch.float32
    )


def _cpu_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q1, kv1 = q.detach().requires_grad_(), kv.detach().requires_grad_()
        result = _cpu_math(q1, kv1, indices, scale=scale)
        dq, dkv = torch.autograd.grad(result, (q1, kv1), grad)
    return dq, dkv


def metadata(
    length: int, world: int, rank: int, device: torch.device
) -> tuple[Glm5NextDsaState, torch.Tensor]:
    alignment = dsa_pack_alignment(world)
    packed_length = (
        ((length + alignment - 1) // alignment) * alignment if world > 1 else length
    )
    groups = torch.full((1, packed_length), -1, dtype=torch.long)
    groups[:, :length] = 0
    parents = torch.full_like(groups, -1)
    positions = torch.arange(packed_length)[None]
    cp = None
    if world > 1:
        cp, _, _, _ = prepare_megatron_context_parallel_state(
            micro=cast(
                Any,
                {
                    "tokens": groups,
                    "group_ids": groups,
                    "parent_ids": parents,
                    "input_pos": positions,
                },
            ),
            topology=ParallelTopology(cp=world),
            config=ContextParallelConfig(
                block_size=8, planner_chunk_size=max(64, dsa_pack_alignment(world))
            ),
            cp_group=DIST.group.WORLD,
            cp_rank=rank,
        )
        positions = torch.zeros(
            1, sum(cp.rank_plan.local_valid_lengths), dtype=torch.long
        )
    state = build_glm5next_dsa_state(
        position_ids=positions,
        group_ids=groups,
        parent_ids=parents,
        device=device,
        context_parallel_state=cp,
    )
    ids = torch.full((positions.shape[1],), -1, dtype=torch.long, device=device)
    for plan in state.pool_plans:
        ids[plan.owner_rows] = plan.query_ids
    if bool((ids < 0).any()):
        raise ValueError("Parity fixture has unassigned local tokens.")
    return state, ids


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("Nonfinite parity tensors.")
    bn = b.norm()
    return {
        "rel_l2": float((a - b).norm() / bn) if bn else float(a.norm()),
        "cosine": float(F.cosine_similarity(a, b, dim=0))
        if bn
        else float(not bool(a.norm())),
    }


def lora_gradients(
    adapters: dict[str, LowRank], *, cp_group: Any | None = None
) -> dict[str, torch.Tensor]:
    """Mirror TrainerRank._reduce_dynamic_grads, TP1 and scale_grads=1.

    Local autograd gradients retain the parameter dtype, including BF16
    rounding. TrainerRank promotes those to FP32 before its DP+CP SUM and
    returns FP32 gradients to the optimizer, without casting back to BF16.
    """
    grads = {
        f"{name}.{factor}": (
            torch.zeros_like(param, dtype=torch.float32)
            if param.grad is None
            else param.grad.detach().float().mul(1.0)
        )
        for name, adapter in adapters.items()
        for factor, param in adapter.named_parameters()
    }
    if cp_group is not None and cp_group.size() > 1:
        coalesced_all_reduce(list(grads.values()), group=cp_group, op=DIST.ReduceOp.SUM)
    return grads


def meta_plan(
    root: Path, config: Any, lengths: list[int], lora_rank: int
) -> dict[str, Any]:
    """Exercise real checkpoint projections at every requested token volume."""
    with torch.device("meta"):
        attention = create_attention(config, torch.device("meta"), torch.bfloat16)
        adapters = add_lora(attention, lora_rank)
        volumes = []
        for length in lengths:
            x = torch.empty(length, 1, config.hidden_size, dtype=torch.bfloat16)
            q_low = attention.linear_q_down_proj(x)[0]
            q = attention.linear_q_up_proj(q_low)[0]
            kv = attention.linear_kv_down_proj(x)[0]
            output = attention.linear_proj(
                torch.empty(
                    length,
                    1,
                    config.num_attention_heads * config.v_head_dim,
                    dtype=torch.bfloat16,
                )
            )[0]
            for cp in (1, 2, 8):
                packed = (
                    (length + dsa_pack_alignment(cp) - 1) // dsa_pack_alignment(cp)
                ) * dsa_pack_alignment(cp)
                local = packed // cp
                volumes.append(
                    {
                        "tokens": length,
                        "packed_tokens": packed,
                        "cp": cp,
                        "packing_alignment": dsa_pack_alignment(cp),
                        "q_a": list(q_low.shape),
                        "q_b": list(q.shape),
                        "kv_a": list(kv.shape),
                        "output": list(output.shape),
                        "absorbed_q_bytes": local
                        * config.num_attention_heads
                        * 576
                        * 2,
                        "full_kv_bytes": length * 576 * 2,
                        "indices_bytes": local * 2112 * 4,
                        "lora_parameter_bytes": sum(
                            p.numel() * p.element_size()
                            for module in adapters.values()
                            for p in module.parameters()
                        ),
                    }
                )
    return {
        "fixture_path": str(root),
        "meta_only": True,
        "volumes": volumes,
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def reference_step(
    reference: Any,
    adapters: dict[str, LowRank],
    x: torch.Tensor,
    dy: torch.Tensor,
    indices: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """HF's unabsorbed arithmetic, bounded query workspace, independent backward."""
    outputs = []
    cfg = reference.config
    for first in range(0, x.shape[1], chunk_size):
        last = min(first + chunk_size, x.shape[1])
        local = x[:, first:last]
        q = reference.q_a_layernorm(
            reference.q_a_proj(local) + adapters["q_a_proj"](local)
        )
        q = (
            reference.q_b_proj(q)
            .unflatten(-1, (cfg.num_attention_heads, cfg.qk_nope_head_dim))
            .transpose(1, 2)
        )
        latent = reference.kv_a_layernorm(
            reference.kv_a_proj_with_mqa(x) + adapters["kv_a_proj_with_mqa"](x)
        )
        expanded = (
            reference.kv_b_proj(latent)
            .unflatten(
                -1, (cfg.num_attention_heads, cfg.qk_nope_head_dim + cfg.v_head_dim)
            )
            .transpose(1, 2)
        )
        k, v = expanded.split((cfg.qk_nope_head_dim, cfg.v_head_dim), dim=-1)
        mask = (
            indices[:, first:last, :, None]
            == torch.arange(x.shape[1], device=x.device)[None, None, None, :]
        ).any(-2)
        scores = (q @ k.transpose(-1, -2)) * reference.scaling
        probs = (
            scores.float()
            .masked_fill(~mask[:, None], -torch.inf)
            .softmax(-1)
            .to(q.dtype)
        )
        values = (probs @ v).transpose(1, 2).flatten(-2)
        out = reference.o_proj(values) + adapters["o_proj"](values)
        out.backward(dy[:, first:last])
        outputs.append(out.detach())
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def boundary_ties(indexer: Any, x: torch.Tensor, latent: torch.Tensor) -> list[int]:
    """Exact HF score ties at the selected/unselected pool boundary."""
    length = x.shape[1]
    q = indexer.wq_b(latent).unflatten(-1, (indexer.n_heads, indexer.head_dim))
    keys = indexer.k_norm(indexer.wk(x))
    gates = F.linear(x, indexer.index_kpool_compress_gate)
    packed = torch.cat((keys, gates, keys.new_ones(1, length, 1)), -1)
    pools, pool_ids, _ = indexer.get_pooled_states(packed)
    budget = indexer.index_topk // indexer.index_kpool
    if pools.shape[1] <= budget:
        return []
    weights = indexer.weights_proj(x).float() * indexer.n_heads**-0.5
    tied = []
    for first in range(0, length, 128):
        last = min(first + 128, length)
        scores = (
            q[:, first:last].float() @ pools.transpose(-1, -2).float().unsqueeze(1)
        ) * indexer.softmax_scale
        scores = (weights[:, first:last, None, :] @ scores.relu()).squeeze(-2)
        queries = torch.arange(first, last, device=x.device)
        visible = pool_ids[:, :, -1, None].transpose(1, 2) <= queries[None, :, None]
        scores.masked_fill_(~visible, torch.finfo(scores.dtype).min)
        best = scores.topk(budget + 1, dim=-1).values
        tie = (best[..., budget - 1] == best[..., budget]) & (visible.sum(-1) > budget)
        tied.extend(queries[tie[0]].cpu().tolist())
    return tied


def run_case(
    root: Path,
    config: Any,
    layer: int,
    length: int,
    device: torch.device,
    world: int,
    rank: int,
    lora_rank: int,
    query_chunk_size: int,
) -> dict[str, Any]:
    import copy

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(20261105 + layer)
    reference = load_layer(root, config, layer, device, dtype)
    actual = create_attention(config, device, dtype).requires_grad_(False)
    copy_weights(actual, reference)
    adapters = add_lora(actual, lora_rank)
    ref_adapters = copy.deepcopy(adapters)
    x = torch.randn(1, length, config.hidden_size, device=device, dtype=dtype)
    dy = torch.randn_like(x) / length
    state, local_ids = metadata(length, world, rank, device)
    # HF chooses using the adapted q_a residual, but the indexer stays detached.
    with torch.no_grad():
        latent = reference.q_a_layernorm(
            reference.q_a_proj(x) + ref_adapters["q_a_proj"](x)
        )
        expected_ids = reference.indexer(
            x, latent, torch.ones(1, length, device=device, dtype=torch.bool), None
        )
        tie_query_ids = boundary_ties(reference.indexer, x, latent)
        core = actual.core_attention
        assert isinstance(core, Glm5NextSparseCore)
        hidden = x[:, local_ids].transpose(0, 1).contiguous()
        q_residual = actual._normalize_latent(
            actual.linear_q_down_proj(hidden)[0] + adapters["q_a_proj"](hidden),
            actual.q_layernorm.weight,
        )
        selected = core.topk(hidden, q_residual, state)
        ids = selected if isinstance(selected, torch.Tensor) else selected.indices
        if world > 1:
            assert state.combined_global_ids is not None
            valid = ids < state.combined_k_rows
            ids = torch.where(
                valid,
                state.combined_global_ids[
                    ids.clamp_max(state.combined_k_rows - 1).long()
                ],
                -1,
            )
        else:
            ids = ids.masked_fill(ids >= length, -1)
        expected_local = expected_ids[:, local_ids]
        width = ids.shape[-1]
        expected_local = F.pad(
            expected_local, (0, width - expected_local.shape[-1]), value=-1
        )
        matching = (ids.sort(-1).values == expected_local.sort(-1).values).all(-1)
        selection = float(matching.float().mean())
    actual_out = actual(
        hidden, attention_bias=SimpleNamespace(model_state={"glm5next_dsa": state})
    )[0]
    actual_out.backward(dy[:, local_ids].transpose(0, 1))
    actual_grads = lora_gradients(
        adapters, cp_group=DIST.group.WORLD if world > 1 else None
    )
    expected_out = reference_step(
        reference, ref_adapters, x, dy, expected_ids, query_chunk_size
    )
    output_metrics = metrics(actual_out.transpose(0, 1), expected_out[:, local_ids])
    ref_grads = lora_gradients(ref_adapters)
    gradients = {
        key: metrics(grad, ref_grads[key]) for key, grad in actual_grads.items()
    }
    cp_metrics = None
    if world > 1:
        cp1 = create_attention(config, device, dtype).requires_grad_(False)
        copy_weights(cp1, reference)
        cp1_adapters = add_lora(cp1, lora_rank)
        for name, adapter in adapters.items():
            cp1_adapters[name].load_state_dict(adapter.state_dict())
        state1, _ = metadata(length, 1, 0, device)
        cp1_out = cp1(
            x.transpose(0, 1).contiguous(),
            attention_bias=SimpleNamespace(model_state={"glm5next_dsa": state1}),
        )[0]
        cp1_out.backward(dy.transpose(0, 1))
        cp1_grads = lora_gradients(cp1_adapters)
        cp_metrics = {"output": metrics(actual_out, cp1_out[local_ids])}
        cp_metrics.update(
            {key: metrics(grad, cp1_grads[key]) for key, grad in actual_grads.items()}
        )
    passed = (
        selection >= 0.999
        and output_metrics["cosine"] >= 0.9999
        and output_metrics["rel_l2"] <= 1e-2
        and all(m["rel_l2"] <= 2e-2 for m in gradients.values())
    )
    if cp_metrics is not None:
        passed = passed and all(m["rel_l2"] <= 1e-3 for m in cp_metrics.values())
    # No indexer parameters or excluded projections received a gradient.
    assert isinstance(actual.core_attention, Glm5NextSparseCore)
    assert isinstance(actual.core_attention.indexer, KPoolIndexer)
    assert all(p.grad is None for p in actual.core_attention.indexer.parameters())
    return {
        "layer": layer,
        "length": length,
        "cp": world,
        "rank": rank,
        "selection_matching_fraction": selection,
        "mismatched_query_ids": local_ids[~matching[0]].cpu().tolist(),
        "hf_boundary_tie_query_ids": tie_query_ids,
        "output": output_metrics,
        "lora_gradients": gradients,
        "gradient_reduction": {
            "path": "TrainerRank._reduce_dynamic_grads",
            "local_dtype": str(dtype),
            "reduction_dtype": "torch.float32",
            "comparison_dtype": "torch.float32",
            "op": "SUM",
            "cp_dkv_accumulation_dtype": str(state.cp_dkv_accumulation_dtype),
        },
        "cp1": cp_metrics,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("R2", "F4", "F8"), required=True)
    parser.add_argument(
        "--root", type=Path, default=Path("/tmp/stark-gpu/glm-fixtures")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--lengths", default="512,2048,2052,8192,16384")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-plan", action="store_true")
    args = parser.parse_args()
    world, rank = (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
    )
    if not 1 <= args.lora_rank <= 32:
        parser.error("LoRA rank must be in [1,32].")
    if transformers.__version__ != "5.20.2":
        raise RuntimeError("HF parity requires Transformers 5.20.2.")
    if args.meta_plan:
        root = args.root / args.fixture
        config = Glm5NextTextConfig.from_pretrained(root, local_files_only=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "meta-plan.json").write_text(
            json.dumps(
                meta_plan(
                    root,
                    config,
                    list(map(int, args.lengths.split(","))),
                    args.lora_rank,
                ),
                indent=2,
            )
            + "\n"
        )
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU validation must set CUDA_VISIBLE_DEVICES empty.")
        setattr(glm52_attention, "sparse_mla", _cpu_math)
        setattr(cp_attention, "sparse_mla_forward", _cpu_forward)
        setattr(cp_attention, "sparse_mla_backward", _cpu_backward)
    if world > 1:
        DIST.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=10)
        )
    try:
        root = args.root / args.fixture
        config = Glm5NextTextConfig.from_pretrained(root, local_files_only=True)
        assert config.layer_types is not None
        layers = [
            i
            for i, kind in enumerate(config.layer_types)
            if kind == "indexed_attention"
        ]
        results = []
        for layer in layers:
            for length in map(int, args.lengths.split(",")):
                result = run_case(
                    root,
                    config,
                    layer,
                    length,
                    device,
                    world,
                    rank,
                    args.lora_rank,
                    args.query_chunk_size,
                )
                results.append(result)
                print(json.dumps(result), flush=True)
        path = args.output / f"cp{world}-rank{rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fixture": args.fixture,
                    "device": str(device),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "transformers": transformers.__version__,
                    "cases": results,
                },
                indent=2,
            )
            + "\n"
        )
        if not all(item["passed"] for item in results):
            raise SystemExit("DSA parity gate failed; see receipt.")
    finally:
        if world > 1:
            DIST.destroy_process_group()


if __name__ == "__main__":
    main()
