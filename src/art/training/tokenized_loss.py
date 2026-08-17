from __future__ import annotations

from pydantic import BaseModel, ConfigDict
import torch

from .contracts import LossConfig
from .tokenized import tokenized_clip_bounds


class TokenizedLossOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss_sum: torch.Tensor
    probability_ratio: torch.Tensor | None = None


def tokenized_loss(
    config: LossConfig,
    *,
    target_logprobs: torch.Tensor,
    weights: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> TokenizedLossOutput:
    if config.name == "cross_entropy":
        return TokenizedLossOutput(loss_sum=-(target_logprobs * weights).sum())

    ratio = torch.exp(target_logprobs - sampling_logprobs)
    if config.name == "importance_sampling":
        loss = -(ratio * advantages).sum()
    else:
        if config.name not in {"ppo", "cispo"}:
            raise AssertionError(config.name)
        low, high = tokenized_clip_bounds(config.name, config.values)
        clipped = ratio.clamp(
            min=low,
            max=high,
        )
        loss = -(
            torch.minimum(ratio * advantages, clipped * advantages).sum()
            if config.name == "ppo"
            else (clipped.detach() * target_logprobs * advantages).sum()
        )
    return TokenizedLossOutput(loss_sum=loss, probability_ratio=ratio)
