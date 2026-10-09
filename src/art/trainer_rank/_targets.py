"""Target logits the head statistics gather beside the softmax statistics."""

from __future__ import annotations

import torch


def gather_target_logits(
    logits: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
) -> torch.Tensor:
    """FP32 ``logits[rows, columns]``; 0 where the column is -1 (an ignored
    label or one in another tensor-parallel rank's vocabulary shard)."""
    values = logits[rows, columns.clamp(min=0)].float()
    return values.masked_fill(columns < 0, 0.0)
