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

    def reduce_masked(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        loss_type: Literal["grpo", "bnpo", "dr_grpo"],
        max_completion_length: int | None,
        reduction: Literal["mean", "sum"],
    ) -> torch.Tensor:
        """Reduce GRPO values without bias from context-parallel sharding.

        A completion tail can straddle CP ranks.  Computing its mean from
        each rank's local slice would weight the tail once per rank, so gather
        global token counts and let every rank contribute only its local
        numerator.  M-Core then all-reduces the raw ``sum`` path in the same
        way as BNPO/Dr. GRPO.  Dense and non-GRPO paths retain the base
        implementation.
        """

        if self.loss_all_reduce_group is None:
            return super().reduce_masked(
                values,
                mask,
                loss_type=loss_type,
                max_completion_length=max_completion_length,
                reduction=reduction,
            )

        if (
            loss_type == "dr_grpo"
            and reduction == "mean"
            and max_completion_length is not None
        ):
            # Dr. GRPO keeps a fixed denominator, but the numerator still
            # spans token slices owned by every CP rank.  Aggregate that
            # numerator here for callers using the ordinary ``mean``
            # reduction (Megatron's schedule uses ``sum`` below).
            numerator = (values * mask).sum()
            torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
                numerator,
                group=self.loss_all_reduce_group,
            )
            denominator = values.new_tensor(
                float(self.logical_sequence_count(mask) * max_completion_length)
            )
            return numerator / denominator.clamp_min(1.0)

        if loss_type != "grpo":
            return super().reduce_masked(
                values,
                mask,
                loss_type=loss_type,
                max_completion_length=max_completion_length,
                reduction=reduction,
            )

        local_sum, global_count = self._distributed_grouped_mean_sum(values, mask)
        if reduction == "sum":
            return local_sum
        torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
            local_sum,
            group=self.loss_all_reduce_group,
        )
        return local_sum / max(global_count, 1)

    def _distributed_grouped_mean_sum(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Return this rank's contribution to global per-completion means."""

        if (
            values.ndim != 2
            or mask.shape != values.shape
            or self.group_ids.shape != mask.shape
        ):
            return super()._grouped_mean_sum(values, mask)
        active = mask > 0
        flat_values = values[active]
        flat_mask = mask[active]
        flat_groups = self.group_ids.to(device=mask.device, dtype=torch.long)[active]
        world_size = torch.distributed.get_world_size(self.loss_all_reduce_group)
        local_ids = torch.unique(flat_groups, sorted=True)
        local_count = torch.tensor(
            [local_ids.numel()], device=mask.device, dtype=torch.long
        )
        gathered_counts = [torch.empty_like(local_count) for _ in range(world_size)]
        torch.distributed.all_gather(  # ty: ignore[possibly-missing-attribute]
            gathered_counts,
            local_count,
            group=self.loss_all_reduce_group,
        )
        max_count = int(torch.stack(gathered_counts).max().item())
        if max_count == 0:
            return values.new_zeros(()), self.logical_sequence_count(mask)
        padded_ids = torch.zeros(max_count, device=mask.device, dtype=torch.long)
        padded_ids[: local_ids.numel()] = local_ids
        gathered_ids = [torch.empty_like(padded_ids) for _ in range(world_size)]
        torch.distributed.all_gather(  # ty: ignore[possibly-missing-attribute]
            gathered_ids,
            padded_ids,
            group=self.loss_all_reduce_group,
        )
        global_ids = torch.unique(
            torch.cat(
                [
                    gathered[: int(count.item())]
                    for gathered, count in zip(
                        gathered_ids, gathered_counts, strict=True
                    )
                ]
            ),
            sorted=True,
        )
        group_indices = torch.searchsorted(global_ids, flat_groups)
        # The dispatcher can include empty physical rows (or packed tails
        # whose active tokens are entirely on another rank) in this count.
        # Prefer its complete logical count when available; otherwise the
        # globally observed IDs are the best local fallback.
        global_count = (
            self.logical_sequence_count(mask)
            if self.logical_sequence_count_override is not None
            else int(global_ids.numel())
        )
        global_count = max(global_count, 1)
        local_sums = values.new_zeros((int(global_ids.numel()),))
        local_counts = torch.zeros(
            int(global_ids.numel()),
            device=values.device,
            dtype=torch.float32,
        )
        local_sums.scatter_add_(0, group_indices, flat_values)
        local_counts.scatter_add_(0, group_indices, flat_mask.to(dtype=torch.float32))
        global_counts = local_counts.detach().clone()
        torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
            global_counts,
            group=self.loss_all_reduce_group,
        )
        local_contribution = (local_sums / global_counts.clamp_min(1e-18)).sum()
        return local_contribution, global_count

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
    flat_by = by.reshape(-1).to(dtype=torch.float32)
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
    padded_ids = torch.zeros(max_count, device=values.device, dtype=torch.float32)
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
