from typing import TYPE_CHECKING

import torch

from .pack import PackedTensors

if TYPE_CHECKING:
    from .. import dev, types


class TrainInputs(PackedTensors):
    """Training inputs with config attached."""

    config: "types.TrainConfig"
    _config: "dev.TrainConfig"
    return_new_logprobs: bool


def create_train_inputs(
    packed_tensors: PackedTensors,
    offset: int,
    config: "types.TrainConfig",
    _config: "dev.TrainConfig",
    warmup: bool,
) -> TrainInputs:
    """Create TrainInputs for a single batch offset."""
    return TrainInputs(  # ty:ignore[missing-typed-dict-key]
        **{
            k: (
                v[offset : offset + 1, :1024]
                if warmup and v.dim() > 1
                else v[offset : offset + 1]
            )
            for k, v in packed_tensors.items()
            if isinstance(v, torch.Tensor)
        },
        pixel_values=(
            [None] if warmup else packed_tensors["pixel_values"][offset : offset + 1]
        ),
        image_grid_thw=(
            [None] if warmup else packed_tensors["image_grid_thw"][offset : offset + 1]
        ),
        config=(
            config.model_copy(update={"lr": 1e-9, "beta": 0.0, "kl_coef": 0.0})
            if warmup
            else config
        ),
        _config=_config,
        return_new_logprobs=False,
    )
