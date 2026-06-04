import copy
import inspect
from typing import Any

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.transformer.spec_utils import ModuleSpec
from pydantic import BaseModel, ConfigDict, SkipValidation

from art.megatron.model_support.spec import ModelSupportHandler, ModelSupportSpec


class ProviderBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: SkipValidation[GPTModelProvider]
    bridge: SkipValidation[AutoBridge]
    handler: SkipValidation[ModelSupportHandler]
    spec: ModelSupportSpec


def resolve_layer_spec(
    base_layer_spec: Any,
    config: Any,
    vp_stage: int | None = None,
) -> Any:
    if isinstance(base_layer_spec, ModuleSpec):
        return copy.deepcopy(base_layer_spec)
    kwargs = (
        {"vp_stage": vp_stage}
        if vp_stage in inspect.signature(base_layer_spec).parameters
        else {}
    )
    return base_layer_spec(config, **kwargs)


def patch_core_attention(layer_spec: object, core_attention: object) -> None:
    submodules = getattr(layer_spec, "submodules", None)
    self_attention = getattr(submodules, "self_attention", None)
    attention_submodules = getattr(self_attention, "submodules", None)
    if attention_submodules is None or not hasattr(
        attention_submodules,
        "core_attention",
    ):
        return
    attention_submodules.core_attention = core_attention


def patch_layer_spec_tree(layer_spec: object, core_attention: object) -> None:
    layer_specs = getattr(layer_spec, "layer_specs", None)
    if layer_specs is None:
        patch_core_attention(layer_spec, core_attention)
        return
    for block_layer_spec in layer_specs:
        patch_core_attention(block_layer_spec, core_attention)
