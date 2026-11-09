import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from art.preprocessing.tokenize import SFTBatch
from art.unsloth.train import UnslothTrainContext, run_unsloth_sft_forward


class _TokenLossModel(torch.nn.Module):
    """Charges each trainable target token its label value, like a summed CE."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.grad_enabled: list[bool] = []
        self.training_modes: list[bool] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        num_items_in_batch: torch.Tensor,
    ) -> SimpleNamespace:
        del input_ids, attention_mask
        self.grad_enabled.append(torch.is_grad_enabled())
        self.training_modes.append(self.training)
        targets = labels[:, 1:]
        token_losses = targets.clamp_min(0).float() * self.scale
        loss = token_losses[targets != -100].sum() / num_items_in_batch
        return SimpleNamespace(loss=loss)


def _trajectory(labels: list[int]) -> dict[str, torch.Tensor]:
    tensor = torch.tensor([labels])
    return {
        "input_ids": torch.ones_like(tensor),
        "attention_mask": torch.ones_like(tensor),
        "labels": tensor,
    }


def _batch(*trajectories: list[int]) -> SFTBatch:
    tensors = [_trajectory(labels) for labels in trajectories]
    return SFTBatch(
        trajectory_tensors=tensors,
        learning_rate=1e-4,
        num_trajectories=len(tensors),
        num_tokens=sum(int(t["attention_mask"].sum()) for t in tensors),
        num_trainable_tokens=sum(
            int((t["labels"][:, 1:] != -100).sum()) for t in tensors
        ),
    )


def test_sft_forward_sums_loss_without_gradients_and_restores_mode() -> None:
    model = _TokenLossModel()
    model.train()
    ctx = cast(UnslothTrainContext, SimpleNamespace(peft_model=model))
    batches = [
        _batch([-100, -100, 2, 3], [-100, 4]),
        _batch([-100, -100, -100]),  # no trainable tokens: skipped
        _batch([-100, 5, -100, 1]),
    ]

    result = asyncio.run(run_unsloth_sft_forward(ctx, batches))

    assert result == {"loss_sum": pytest.approx(15.0), "num_tokens": 5.0}
    assert model.grad_enabled == [False, False, False]
    assert model.training_modes == [False, False, False]
    assert model.training
    assert model.scale.grad is None
