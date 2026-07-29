from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, cast

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.pipeline_parallel_layer_layout import (
    PipelineParallelLayerLayout,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from art.megatron.context_parallel.types import (
    ContextParallelStageWorkProfile,
    ContextParallelWorkloadProfile,
)
from art.megatron.glm52.attention import (
    Glm52SelfAttention,
    glm52_core_builder,
)


def build_glm52_pipeline_layout(
    indexer_types: tuple[str, ...], pp_size: int, vp_size: int
) -> list[list[str]]:
    """Balance complete IndexShare groups across virtual and physical stages."""
    starts = [index for index, mode in enumerate(indexer_types) if mode == "full"]
    stages = pp_size * vp_size
    if not indexer_types or not starts or starts[0] != 0:
        raise ValueError("GLM-5.2 indexer_types must start with a full layer.")
    if stages > len(starts):
        raise ValueError(
            f"GLM-5.2 has {len(starts)} complete IndexShare groups but {stages} "
            "PP/VPP stages were requested."
        )

    def score(boundaries: tuple[int, ...]) -> tuple[Any, ...]:
        chunks = [
            end - start
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
        physical = [sum(chunks[pp_rank::pp_size]) for pp_rank in range(pp_size)]
        return (
            max(chunks),
            max(physical),
            max(physical) - min(physical),
            max(chunks) - min(chunks),
            boundaries,
        )

    boundaries = min(
        (
            (0, *selected, len(indexer_types))
            for selected in combinations(starts[1:], stages - 1)
        ),
        key=score,
    )
    layout = [
        ["decoder"] * (end - start)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    layout[0].insert(0, "embedding")
    layout[-1].append("loss")
    return layout


def _glm52_pipeline_stage_ranges(config: Any) -> tuple[tuple[int, int, int], ...]:
    """Return physical PP rank and layer ranges in VPP-major execution order."""
    indexer_types = tuple(config.glm52_indexer_types)
    layout = config.pipeline_model_parallel_layout
    stages = int(config.pipeline_model_parallel_size or 1) * int(
        config.virtual_pipeline_model_parallel_size or 1
    )
    if stages == 1 and layout is None:
        return ((0, 0, len(indexer_types)),)
    if not isinstance(layout, PipelineParallelLayerLayout):
        raise RuntimeError("GLM-5.2 PP/VPP requires a finalized flexible layout.")
    full_groups = indexer_types.count("full")
    if stages > full_groups:
        raise ValueError(
            f"GLM-5.2 has {full_groups} complete IndexShare groups but {stages} "
            "PP/VPP stages were configured."
        )
    offset = 0
    ranges = []
    for vp_rank in range(layout.virtual_pipeline_model_parallel_size):
        for pp_rank in range(layout.pipeline_model_parallel_size):
            count = layout.layout[pp_rank][vp_rank].count(LayerType.decoder)
            if count:
                if indexer_types[offset] != "full":
                    raise ValueError(
                        "GLM-5.2 pipeline chunk starts at shared index layer "
                        f"{offset} (PP={pp_rank}, VPP={vp_rank}); split only at "
                        "full IndexShare layers."
                    )
            ranges.append((pp_rank, offset, offset + count))
            offset += count
    if offset != len(indexer_types):
        raise ValueError(
            f"GLM-5.2 pipeline layout covers {offset} decoder layers, expected "
            f"{len(indexer_types)}."
        )
    return tuple(ranges)


def _validate_glm52_pipeline_layout(config: Any) -> None:
    """Reject a finalized layout that makes shared layers cross process chunks."""
    _glm52_pipeline_stage_ranges(config)


def _train_matmul_flops(
    in_features: int,
    out_features: int,
    *,
    forward_executions: int,
) -> int:
    # Frozen base weights still execute one input-gradient matmul.
    return 2 * int(in_features) * int(out_features) * (forward_executions + 1)


def build_glm52_context_parallel_profile(
    config: Any,
) -> ContextParallelWorkloadProfile:
    """Describe the GLM work that changes with CP token ownership."""
    indexer_types = tuple(config.glm52_indexer_types)
    moe_layers = tuple(bool(value) for value in config.moe_layer_freq)
    if len(moe_layers) != len(indexer_types):
        raise ValueError(
            "GLM-5.2 MLP and indexer layer patterns must have equal length."
        )

    forward_executions = (
        2 if getattr(config, "recompute_granularity", None) == "full" else 1
    )
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    q_rank = int(config.q_lora_rank)
    kv_rank = int(config.kv_lora_rank)
    qk_nope = int(config.qk_head_dim)
    rope = int(config.qk_pos_emb_head_dim)
    value = int(config.v_head_dim)
    combined_dim = kv_rank + rope
    topk = int(config.dsa_indexer_topk)
    index_heads = int(config.dsa_indexer_n_heads)
    index_dim = int(config.dsa_indexer_head_dim)
    dense_intermediate = int(config.ffn_hidden_size)
    shared_intermediate = int(config.moe_shared_expert_intermediate_size or 0)
    experts = int(config.num_moe_experts)

    attention_projection = sum(
        (
            _train_matmul_flops(hidden, q_rank, forward_executions=forward_executions),
            _train_matmul_flops(
                q_rank,
                heads * (qk_nope + rope),
                forward_executions=forward_executions,
            ),
            _train_matmul_flops(
                hidden, combined_dim, forward_executions=forward_executions
            ),
            heads
            * _train_matmul_flops(
                qk_nope, kv_rank, forward_executions=forward_executions
            ),
            heads
            * _train_matmul_flops(
                kv_rank, value, forward_executions=forward_executions
            ),
            _train_matmul_flops(
                heads * value, hidden, forward_executions=forward_executions
            ),
        )
    )
    sparse_attention = (
        2 * (forward_executions + 2) * heads * topk * (combined_dim + kv_rank)
    )
    dense_mlp = _train_matmul_flops(
        hidden, 2 * dense_intermediate, forward_executions=forward_executions
    ) + _train_matmul_flops(
        dense_intermediate, hidden, forward_executions=forward_executions
    )
    local_sparse_mlp = _train_matmul_flops(
        hidden, experts, forward_executions=forward_executions
    )
    if shared_intermediate:
        local_sparse_mlp += _train_matmul_flops(
            hidden,
            2 * shared_intermediate,
            forward_executions=forward_executions,
        ) + _train_matmul_flops(
            shared_intermediate,
            hidden,
            forward_executions=forward_executions,
        )
    indexer_projection = (
        2
        * forward_executions
        * (q_rank * index_heads * index_dim + hidden * index_dim + hidden * index_heads)
    )
    indexer_pair = forward_executions * index_heads * (2 * index_dim + 3)

    pp_size = int(config.pipeline_model_parallel_size or 1)
    stage_layers = [0 for _ in range(pp_size)]
    stage_indexers = [0 for _ in range(pp_size)]
    stage_query_flops = [0 for _ in range(pp_size)]
    for pp_rank, start, end in _glm52_pipeline_stage_ranges(config):
        full_indexers = indexer_types[start:end].count("full")
        layer_count = end - start
        query_flops = layer_count * (attention_projection + sparse_attention)
        query_flops += sum(
            local_sparse_mlp if moe_layers[layer] else dense_mlp
            for layer in range(start, end)
        )
        query_flops += full_indexers * indexer_projection
        stage_layers[pp_rank] += layer_count
        stage_indexers[pp_rank] += full_indexers
        stage_query_flops[pp_rank] += query_flops

    stages = []
    sparse_fetches = forward_executions + 1
    for pp_rank, (layer_count, full_indexers, query_flops) in enumerate(
        zip(stage_layers, stage_indexers, stage_query_flops, strict=True)
    ):
        k_fetch_bytes = layer_count * sparse_fetches * combined_dim * 2
        k_fetch_bytes += full_indexers * forward_executions * index_dim * 2
        dkv_reduce_bytes = layer_count * combined_dim * 2
        # Each fetch concatenates CP stages and adds TileLang's sentinel row.
        # Backward also zeroes four FP32 dKV splits, reduces, and casts to BF16.
        k_hbm_bytes = layer_count * combined_dim * (8 * sparse_fetches + 42)
        checkpoint_bytes = layer_count * hidden * 2
        persistent_topk_bytes = full_indexers * topk * 4
        sparse_query_workspace = (
            2 * heads * combined_dim * 2
            + 2 * heads * kv_rank * 2
            + 2 * heads * 4
            + topk * 4
        )
        # Backward holds original and padded BF16 KV, four FP32 dKV splits,
        # the FP32 reduction result, and the returned BF16 dKV.
        k_memory = combined_dim * (2 + 2 + 4 * 4 + 4 + 2)
        stages.append(
            ContextParallelStageWorkProfile(
                physical_pipeline_rank=pp_rank,
                query_flops_per_token=query_flops,
                tile_pair_flops=full_indexers * indexer_pair,
                k_hbm_bytes_per_token=k_hbm_bytes,
                k_fetch_bytes_per_token=k_fetch_bytes,
                dkv_reduce_bytes_per_token=dkv_reduce_bytes,
                query_memory_bytes_per_token=(
                    checkpoint_bytes + persistent_topk_bytes + sparse_query_workspace
                ),
                k_memory_bytes_per_token=k_memory,
            )
        )
    return ContextParallelWorkloadProfile(
        stages=tuple(stages),
        query_tile_size=128 // index_heads,
        key_tile_size=64,
        indexer_score_workspace_elements=(256 * 1024 * 1024) // 8,
        indexer_max_k_tokens=32 * 1024,
    )


def get_glm52_decoder_block_spec(config: Any, vp_stage: int | None = None) -> Any:
    """Build GLM-5.2 layers without entering MCore's incomplete DSA path."""
    _validate_glm52_pipeline_layout(config)
    block_spec = deepcopy(
        get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=True,
            normalization="RMSNorm",
            vp_stage=vp_stage,
        )
    )
    backend = TESpecProvider()
    attention = ModuleSpec(
        module=Glm52SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_linear(),
            core_attention=glm52_core_builder(
                backend.linear(),
                backend.layer_norm(rms_norm=False, for_qk=True),
            ),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
            kv_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
        ),
        metainfo={"fuse_input_layernorm": False},
    )
    for layer_spec in block_spec.layer_specs or ():
        cast(Any, layer_spec.submodules).self_attention = attention
    return block_spec
