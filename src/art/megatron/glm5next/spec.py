from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_linear_attention_pattern,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from art.megatron.glm5next.layer import (
    Glm5NextFinalNorm,
    Glm5NextRMSNorm,
    Glm5NextRouter,
    Glm5NextTransformerLayer,
)
from art.megatron.glm5next.reference import HfReferenceAttention


def native_attention_specs() -> tuple[ModuleSpec, ModuleSpec]:
    from art.megatron.glm5next.dsa import Glm5NextSelfAttention, glm5next_core_builder
    from art.megatron.glm5next.kda import (
        KimiDeltaAttention,
        KimiDeltaAttentionSubmodules,
    )

    backend = TESpecProvider()
    column, row, linear = (
        backend.column_parallel_linear(),
        backend.row_parallel_linear(),
        backend.linear(),
    )
    kda = ModuleSpec(
        module=KimiDeltaAttention,
        submodules=KimiDeltaAttentionSubmodules(
            q_proj=column,
            k_proj=column,
            v_proj=column,
            o_proj=row,
            b_proj=column,
            f_a_proj=linear,
            f_b_proj=column,
            g_a_proj=linear,
            g_b_proj=column,
        ),
        metainfo={"fuse_input_layernorm": False},
    )
    dsa = ModuleSpec(
        module=Glm5NextSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_down_proj=linear,
            linear_q_up_proj=column,
            linear_kv_down_proj=linear,
            linear_kv_up_proj=column,
            core_attention=glm5next_core_builder(
                linear, backend.layer_norm(rms_norm=False, for_qk=True)
            ),
            linear_proj=row,
            q_layernorm=Glm5NextRMSNorm,
            kv_layernorm=Glm5NextRMSNorm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )
    return kda, dsa


def get_glm5next_decoder_block_spec(config: Any, vp_stage: int | None = None) -> Any:
    """Reuse Qwen3.5's generic 1=linear/0=full pattern on ART's GPT stack."""
    block = deepcopy(
        get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=True,
            normalization="RMSNorm",
            vp_stage=vp_stage,
        )
    )
    pattern = list(get_linear_attention_pattern(config))
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    block.layer_norm = Glm5NextFinalNorm
    backend = TESpecProvider()
    native_kda, native_dsa = native_attention_specs()
    # MCore repeats the same dense/MoE spec objects across layers. Clone each
    # occurrence independently before assigning a per-layer attention family.
    block.layer_specs = [deepcopy(layer) for layer in block.layer_specs or ()]
    for index, layer in enumerate(block.layer_specs or ()):
        layer.module = Glm5NextTransformerLayer
        modules = cast(Any, layer.submodules)
        modules.input_layernorm = Glm5NextRMSNorm
        modules.pre_mlp_layernorm = Glm5NextRMSNorm
        # Explicit per-provider overrides support CPU references and lane tests.
        attention = getattr(
            config,
            "glm5next_kda_spec" if pattern[offset + index] else "glm5next_dsa_spec",
            native_kda if pattern[offset + index] else native_dsa,
        )
        modules.self_attention = (
            deepcopy(attention)
            if attention
            else ModuleSpec(
                module=HfReferenceAttention,
                metainfo={"fuse_input_layernorm": False},
            )
        )
        modules.sharded_state_dict_keys_map = {}
        mlp = modules.mlp
        if mlp.module == MoELayer:
            mlp.submodules.router = Glm5NextRouter
        else:
            # The norm is explicit above; TE's default dense FC1 fuses another.
            mlp.submodules.linear_fc1 = backend.column_parallel_linear()
    return block
