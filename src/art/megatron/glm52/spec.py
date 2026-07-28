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


def _validate_glm52_pipeline_layout(config: Any) -> None:
    """Reject a finalized layout that makes shared layers cross process chunks."""
    indexer_types = tuple(config.glm52_indexer_types)
    layout = config.pipeline_model_parallel_layout
    stages = int(config.pipeline_model_parallel_size or 1) * int(
        config.virtual_pipeline_model_parallel_size or 1
    )
    if stages == 1 and layout is None:
        return
    if not isinstance(layout, PipelineParallelLayerLayout):
        raise RuntimeError("GLM-5.2 PP/VPP requires a finalized flexible layout.")
    full_groups = indexer_types.count("full")
    if stages > full_groups:
        raise ValueError(
            f"GLM-5.2 has {full_groups} complete IndexShare groups but {stages} "
            "PP/VPP stages were configured."
        )
    offset = 0
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
                offset += count
    if offset != len(indexer_types):
        raise ValueError(
            f"GLM-5.2 pipeline layout covers {offset} decoder layers, expected "
            f"{len(indexer_types)}."
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
