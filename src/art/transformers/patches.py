from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import masking_utils
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

_preprocess_mask_arguments = masking_utils._preprocess_mask_arguments


def _patched_preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, "BlockMask"]],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[torch.Tensor, "BlockMask"]], int, int]:
    if position_ids is not None and len(position_ids.shape) == 3:
        position_ids = position_ids[0]
    return _preprocess_mask_arguments(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids,
        layer_idx,
    )


def patch_preprocess_mask_arguments() -> None:
    masking_utils._preprocess_mask_arguments = _patched_preprocess_mask_arguments  # ty:ignore[invalid-assignment]
