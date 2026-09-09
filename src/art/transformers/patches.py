from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from transformers import masking_utils
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

_preprocess_mask_arguments = masking_utils._preprocess_mask_arguments


def _patched_preprocess_mask_arguments(
    config: PretrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, "BlockMask"]],
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
    encoder_hidden_states: Optional[torch.Tensor] = None,
) -> tuple[Any, ...]:
    if position_ids is not None and len(position_ids.shape) == 3:
        position_ids = position_ids[0]
    return _preprocess_mask_arguments(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids,
        layer_idx,
        encoder_hidden_states,
    )


def patch_preprocess_mask_arguments() -> None:
    masking_utils._preprocess_mask_arguments = _patched_preprocess_mask_arguments  # ty:ignore[invalid-assignment]


def disable_broken_torchvision_for_transformers() -> None:
    try:
        import torchvision  # noqa: F401

        return
    except Exception:
        import sys

        for module_name in list(sys.modules):
            if module_name == "torchvision" or module_name.startswith("torchvision."):
                sys.modules.pop(module_name, None)

    from transformers import utils as transformers_utils
    from transformers.utils import import_utils

    def _torchvision_unavailable() -> bool:
        return False

    for module in (import_utils, transformers_utils):
        for name in ("is_torchvision_available", "is_torchvision_v2_available"):
            original = getattr(module, name, None)
            cache_clear = getattr(original, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
            setattr(module, name, _torchvision_unavailable)
