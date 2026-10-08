from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from art.trainer_rank._targets import gather_target_logits

# (local max, local sum, top-k values, top-k tokens)
type _KernelStats = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
# (local max, local sum, top-k values, top-k tokens, target logits)
type LocalTopKStats = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
# (local max, local sum, target logits)
type LocalLogSumExpStats = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# (rows, local columns): -1 marks a target this vocabulary shard does not own.
type LocalTargets = tuple[torch.Tensor, torch.Tensor]


@triton.jit
def _stats_stage1_kernel(
    logits_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_values_ptr,
    partial_tokens_ptr,
    stride_row: tl.constexpr,
    vocab_size: tl.constexpr,
    n_blocks: tl.constexpr,
    k: tl.constexpr,
    block_v: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_v + tl.arange(0, block_v)
    mask = offsets < vocab_size
    values = tl.load(
        logits_ptr + row * stride_row + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    block_max = tl.max(values, axis=0)
    block_sum = tl.sum(tl.exp(values - block_max), axis=0)
    partial_offset = row * n_blocks + block
    tl.store(partial_max_ptr + partial_offset, block_max)
    tl.store(partial_sum_ptr + partial_offset, block_sum)

    work = values
    arange = tl.arange(0, block_v)
    for slot in tl.static_range(0, k):
        top_value, top_index = tl.max(
            work,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        output_offset = (partial_offset * k) + slot
        tl.store(partial_values_ptr + output_offset, top_value)
        tl.store(
            partial_tokens_ptr + output_offset,
            (block * block_v + top_index).to(tl.int64),
        )
        work = tl.where(arange == top_index, -float("inf"), work)


@triton.jit
def _stats_stage2_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_values_ptr,
    partial_tokens_ptr,
    local_max_ptr,
    local_sum_ptr,
    values_ptr,
    tokens_ptr,
    n_blocks: tl.constexpr,
    k: tl.constexpr,
    block_b: tl.constexpr,
    block_candidates: tl.constexpr,
):
    row = tl.program_id(0)

    block_offsets = tl.arange(0, block_b)
    block_mask = block_offsets < n_blocks
    partial_base = row * n_blocks
    block_max = tl.load(
        partial_max_ptr + partial_base + block_offsets,
        mask=block_mask,
        other=-float("inf"),
    )
    row_max = tl.max(block_max, axis=0)
    block_sum = tl.load(
        partial_sum_ptr + partial_base + block_offsets,
        mask=block_mask,
        other=0.0,
    )
    row_sum = tl.sum(block_sum * tl.exp(block_max - row_max), axis=0)
    tl.store(local_max_ptr + row, row_max)
    tl.store(local_sum_ptr + row, row_sum)

    if k > 0:
        candidate_offsets = tl.arange(0, block_candidates)
        candidate_mask = candidate_offsets < n_blocks * k
        candidate_base = row * n_blocks * k
        candidates = tl.load(
            partial_values_ptr + candidate_base + candidate_offsets,
            mask=candidate_mask,
            other=-float("inf"),
        )
        work = candidates
        for slot in tl.static_range(0, k):
            top_value, top_index = tl.max(
                work,
                axis=0,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            output_offset = row * k + slot
            tl.store(values_ptr + output_offset, top_value)
            tl.store(
                tokens_ptr + output_offset,
                tl.load(partial_tokens_ptr + candidate_base + top_index),
            )
            work = tl.where(candidate_offsets == top_index, -float("inf"), work)


@triton.jit
def _stats_backward_kernel(
    logits_ptr,
    local_max_ptr,
    tokens_ptr,
    grad_sum_ptr,
    grad_values_ptr,
    target_offsets_ptr,
    target_tokens_ptr,
    target_grads_ptr,
    grad_logits_ptr,
    stride_row: tl.constexpr,
    vocab_size: tl.constexpr,
    n_blocks: tl.constexpr,
    k: tl.constexpr,
    block_v: tl.constexpr,
    has_targets: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_v + tl.arange(0, block_v)
    mask = offsets < vocab_size

    logits = tl.load(
        logits_ptr + row * stride_row + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    local_max = tl.load(local_max_ptr + row)
    grad = tl.load(grad_sum_ptr + row).to(tl.float32) * tl.exp(logits - local_max)

    for slot in tl.static_range(0, k):
        token = tl.load(tokens_ptr + row * k + slot)
        value_grad = tl.load(grad_values_ptr + row * k + slot).to(tl.float32)
        grad += tl.where(offsets == token, value_grad, 0.0)

    # Gathered target logits, grouped by (row, vocab block): their gradients
    # join the softmax gradient in FP32, before the single store in the
    # logits' dtype, and each program reads only its own block's targets.
    if has_targets:
        group = row * n_blocks + block
        target_start = tl.load(target_offsets_ptr + group)
        target_end = tl.load(target_offsets_ptr + group + 1)
        for index in range(target_start, target_end):
            token = tl.load(target_tokens_ptr + index)
            target_grad = tl.load(target_grads_ptr + index)
            grad += tl.where(offsets == token, target_grad, 0.0)

    tl.store(grad_logits_ptr + row * stride_row + offsets, grad, mask=mask)


class _LocalStatsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_logits: torch.Tensor,
        k: int,
        target_rows: torch.Tensor,
        target_columns: torch.Tensor,
    ):
        local_max, local_sum, values, tokens = _local_stats_forward(local_logits, k=k)
        target_logits = gather_target_logits(local_logits, target_rows, target_columns)
        ctx.save_for_backward(
            local_logits, local_max, tokens, target_rows, target_columns
        )
        ctx.k = k
        return local_max, local_sum, values, tokens, target_logits

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_local_max, grad_local_sum, grad_values, grad_tokens, grad_targets = (
            grad_outputs
        )
        del grad_local_max, grad_tokens
        logits, local_max, tokens, target_rows, target_columns = ctx.saved_tensors
        k = int(ctx.k)
        rows = int(logits.shape[0])
        vocab_size = int(logits.shape[1])
        block_v = 4096
        n_blocks = int(triton.cdiv(vocab_size, block_v))

        if grad_local_sum is None:
            grad_local_sum = torch.zeros_like(local_max)
        if grad_values is None:
            grad_values = torch.zeros(
                (rows, k),
                device=logits.device,
                dtype=torch.float32,
            )
        has_targets = grad_targets is not None and bool(target_rows.numel())
        if has_targets:
            # Group the targets by (row, vocab block) (CSR): group g = row *
            # n_blocks + block holds [offsets[g], offsets[g + 1]). Unowned (-1)
            # columns sort after the last group, so no program reads them.
            groups = torch.where(
                target_columns >= 0,
                target_rows * n_blocks
                + target_columns.div(block_v, rounding_mode="floor"),
                rows * n_blocks,
            )
            order = torch.argsort(groups, stable=True)
            target_offsets = torch.searchsorted(
                groups[order],
                torch.arange(
                    rows * n_blocks + 1, device=logits.device, dtype=groups.dtype
                ),
            )
            del groups
            target_tokens = target_columns[order]
            target_grads = grad_targets[order].float()
            del order
        else:
            # The kernel loads no target, but its pointers must be allocated.
            target_offsets = target_tokens = local_max.new_empty((1,), dtype=torch.long)
            target_grads = local_max.new_empty((1,))

        grad_logits = torch.empty_like(logits)
        _stats_backward_kernel[(rows, n_blocks)](
            logits,
            local_max,
            tokens,
            grad_local_sum.contiguous(),
            grad_values.contiguous(),
            target_offsets,
            target_tokens,
            target_grads,
            grad_logits,
            logits.stride(0),
            vocab_size=vocab_size,  # ty: ignore[invalid-argument-type]
            n_blocks=n_blocks,  # ty: ignore[invalid-argument-type]
            k=k,  # ty: ignore[invalid-argument-type]
            block_v=block_v,  # ty: ignore[invalid-argument-type]
            has_targets=has_targets,  # ty: ignore[invalid-argument-type]
            num_warps=8,  # ty: ignore[unknown-argument]
        )
        return grad_logits, None, None, None


def _no_targets(logits: torch.Tensor) -> LocalTargets:
    empty = torch.empty(0, dtype=torch.long, device=logits.device)
    return empty, empty


def _check_local_logits(local_logits: torch.Tensor) -> torch.Tensor:
    if local_logits.ndim != 2:
        raise ValueError(
            f"expected [rows, vocab] logits, got {tuple(local_logits.shape)}"
        )
    if not local_logits.is_cuda:
        raise ValueError("local top-k helpers require CUDA logits")
    return local_logits.contiguous()


def _local_stats_forward(local_logits: torch.Tensor, *, k: int) -> _KernelStats:
    logits = _check_local_logits(local_logits)
    if k < 0 or k > int(local_logits.shape[1]):
        raise ValueError(
            f"k={k} is outside local vocab size {int(local_logits.shape[1])}"
        )

    rows = int(logits.shape[0])
    vocab_size = int(logits.shape[1])
    block_v = 4096
    n_blocks = int(triton.cdiv(vocab_size, block_v))
    block_b = int(triton.next_power_of_2(n_blocks))
    block_candidates = int(triton.next_power_of_2(n_blocks * k)) if k else 1

    partial_shape = (rows, n_blocks)
    partial_max = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    partial_topk_shape = (rows, n_blocks, k) if k else (1,)
    partial_values = torch.empty(
        partial_topk_shape, device=logits.device, dtype=torch.float32
    )
    partial_tokens = torch.empty(
        partial_topk_shape, device=logits.device, dtype=torch.long
    )
    local_max = torch.empty((rows,), device=logits.device, dtype=torch.float32)
    local_sum = torch.empty_like(local_max)
    values = torch.empty((rows, k), device=logits.device, dtype=torch.float32)
    tokens = torch.empty((rows, k), device=logits.device, dtype=torch.long)

    _stats_stage1_kernel[(rows, n_blocks)](
        logits,
        partial_max,
        partial_sum,
        partial_values,
        partial_tokens,
        stride_row=logits.stride(0),  # ty: ignore[invalid-argument-type]
        vocab_size=vocab_size,  # ty: ignore[invalid-argument-type]
        n_blocks=n_blocks,  # ty: ignore[invalid-argument-type]
        k=k,  # ty: ignore[invalid-argument-type]
        block_v=block_v,  # ty: ignore[invalid-argument-type]
        num_warps=8,  # ty: ignore[unknown-argument]
    )
    _stats_stage2_kernel[(rows,)](
        partial_max,
        partial_sum,
        partial_values,
        partial_tokens,
        local_max,
        local_sum,
        values,
        tokens,
        n_blocks=n_blocks,  # ty: ignore[invalid-argument-type]
        k=k,  # ty: ignore[invalid-argument-type]
        block_b=block_b,  # ty: ignore[invalid-argument-type]
        block_candidates=block_candidates,  # ty: ignore[invalid-argument-type]
        num_warps=8,  # ty: ignore[unknown-argument]
    )
    return local_max, local_sum, values, tokens


def local_topk_stats(
    local_logits: torch.Tensor,
    *,
    k: int,
    targets: LocalTargets | None = None,
) -> LocalTopKStats:
    """Local softmax statistics, top-k and FP32 target logits for ``[rows, vocab]``.

    A target logit's gradient is added to the softmax gradient in FP32 inside
    the backward, so the logits receive one gradient store.
    """
    logits = local_logits.contiguous()
    if targets is None:
        targets = _no_targets(logits)
    if not logits.requires_grad:
        stats = _local_stats_forward(logits, k=k)
        return *stats, gather_target_logits(logits, *targets)
    return _LocalStatsFunction.apply(logits, k, *targets)


def local_logsumexp_stats(
    local_logits: torch.Tensor,
    *,
    targets: LocalTargets | None = None,
) -> LocalLogSumExpStats:
    local_max, local_sum, _, _, target_logits = local_topk_stats(
        local_logits, k=0, targets=targets
    )
    return local_max, local_sum, target_logits
