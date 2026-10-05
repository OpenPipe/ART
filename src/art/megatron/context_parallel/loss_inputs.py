from typing import Any, Literal

import torch

from art.loss import AlignedLossInputs


class ContextParallelLossInputs(AlignedLossInputs):
    loss_all_reduce_group: Any | None = None
    entropies_are_aligned: bool = True

    def group_mean(self, values: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        if self.loss_all_reduce_group is None:
            return super().group_mean(values, by)
        return _distributed_group_mean(
            values,
            by=by,
            group=self.loss_all_reduce_group,
        )

    def masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.loss_all_reduce_group is None:
            return super().masked_mean(values, mask)
        numerator = values.sum()
        denominator = mask.sum()
        torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
            numerator,
            group=self.loss_all_reduce_group,
        )
        torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
            denominator,
            group=self.loss_all_reduce_group,
        )
        return numerator / (denominator + 1e-18)

    def denominator(
        self,
        mask: torch.Tensor,
        reduction: Literal["mean", "sum"],
    ):
        if self.loss_all_reduce_group is None or reduction == "sum":
            return super().denominator(mask, reduction)
        denominator = mask.sum()
        torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
            denominator,
            group=self.loss_all_reduce_group,
        )
        return denominator + 1e-18


def _distributed_group_mean(
    values: torch.Tensor,
    *,
    by: torch.Tensor,
    group: Any,
) -> torch.Tensor:
    flat_values = values.reshape(-1)
    flat_by = by.reshape(-1)
    by_dtype = flat_by.dtype
    unique_local = torch.unique(flat_by, sorted=True)
    world_size = torch.distributed.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    local_count = torch.tensor(
        [unique_local.numel()],
        device=values.device,
        dtype=torch.long,
    )
    gathered_counts = [torch.empty_like(local_count) for _ in range(world_size)]
    torch.distributed.all_gather(  # ty: ignore[possibly-missing-attribute]
        gathered_counts,
        local_count,
        group=group,
    )
    max_count = int(torch.stack(gathered_counts).max().item())
    padded_ids = torch.zeros(max_count, device=values.device, dtype=by_dtype)
    padded_ids[: unique_local.numel()] = unique_local
    gathered_ids = [torch.empty_like(padded_ids) for _ in range(world_size)]
    torch.distributed.all_gather(  # ty: ignore[possibly-missing-attribute]
        gathered_ids,
        padded_ids,
        group=group,
    )
    global_ids = torch.unique(
        torch.cat(
            [
                gathered[: int(count.item())]
                for gathered, count in zip(gathered_ids, gathered_counts, strict=True)
            ]
        ),
        sorted=True,
    )
    group_indices = torch.searchsorted(global_ids, flat_by)
    sums = torch.zeros_like(global_ids)
    counts = torch.zeros_like(global_ids)
    sums.scatter_add_(0, group_indices, flat_values.to(dtype=sums.dtype))
    counts.scatter_add_(
        0,
        group_indices,
        torch.ones_like(flat_values, dtype=sums.dtype),
    )
    torch.distributed.all_reduce(sums, group=group)  # ty: ignore[possibly-missing-attribute]
    torch.distributed.all_reduce(counts, group=group)  # ty: ignore[possibly-missing-attribute]
    return (sums / (counts + 1e-18)).gather(0, group_indices).reshape_as(values)
