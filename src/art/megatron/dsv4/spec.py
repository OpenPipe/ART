from copy import deepcopy
from typing import Any, cast

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec

from art.megatron.dsv4.deepseek_v4 import DeepSeekV4Attention
from art.megatron.dsv4.layer import (
    Dsv4FinalNorm,
    Dsv4MoELayer,
    Dsv4Router,
    Dsv4TransformerLayer,
)

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


def get_dsv4_decoder_block_spec(config: Any, vp_stage: int | None = None) -> Any:
    config.moe_layer_freq = [1] * int(config.num_layers)
    block_spec = deepcopy(
        get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=HAVE_TE,
            normalization="RMSNorm",
            vp_stage=vp_stage,
        )
    )
    block_spec.layer_norm = Dsv4FinalNorm
    for layer_spec in block_spec.layer_specs or ():
        layer_spec.module = Dsv4TransformerLayer
        submodules = cast(Any, layer_spec.submodules)
        submodules.input_layernorm = submodules.pre_mlp_layernorm
        submodules.self_attention = ModuleSpec(
            module=DeepSeekV4Attention,
            submodules=None,
            metainfo={"fuse_input_layernorm": False},
        )
        mlp = submodules.mlp
        if isinstance(mlp, ModuleSpec) and mlp.module == MoELayer:
            mlp.module = Dsv4MoELayer
            mlp.submodules.router = Dsv4Router
    return block_spec
