from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from art.megatron.glm52.attention import (
    Glm52SelfAttention,
    glm52_core_builder,
)


def get_glm52_decoder_block_spec(config: Any, vp_stage: int | None = None) -> Any:
    """Build GLM-5.2 layers without entering MCore's incomplete DSA path."""
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
