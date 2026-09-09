from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

type LocalTopKStats = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
type LocalLogSumExpStats = tuple[torch.Tensor, torch.Tensor]


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
    grad_logits_ptr,
    stride_row: tl.constexpr,
    vocab_size: tl.constexpr,
    k: tl.constexpr,
    block_v: tl.constexpr,
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

    tl.store(grad_logits_ptr + row * stride_row + offsets, grad, mask=mask)


class _LocalStatsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_logits: torch.Tensor, k: int):
        local_max, local_sum, values, tokens = _local_stats_forward(local_logits, k=k)
        ctx.save_for_backward(local_logits, local_max, tokens)
        ctx.k = k
        return local_max, local_sum, values, tokens

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_local_max, grad_local_sum, grad_values, grad_tokens = grad_outputs
        del grad_local_max, grad_tokens
        logits, local_max, tokens = ctx.saved_tensors
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

        grad_logits = torch.empty_like(logits)
        _stats_backward_kernel[(rows, n_blocks)](
            logits,
            local_max,
            tokens,
            grad_local_sum.contiguous(),
            grad_values.contiguous(),
            grad_logits,
            logits.stride(0),
            vocab_size=vocab_size,  # ty: ignore[invalid-argument-type]
            k=k,  # ty: ignore[invalid-argument-type]
            block_v=block_v,  # ty: ignore[invalid-argument-type]
            num_warps=8,  # ty: ignore[unknown-argument]
        )
        return grad_logits, None


def _check_local_logits(local_logits: torch.Tensor) -> torch.Tensor:
    if local_logits.ndim != 2:
        raise ValueError(
            f"expected [rows, vocab] logits, got {tuple(local_logits.shape)}"
        )
    if not local_logits.is_cuda:
        raise ValueError("local top-k helpers require CUDA logits")
    return local_logits.contiguous()


def _local_stats_forward(local_logits: torch.Tensor, *, k: int) -> LocalTopKStats:
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


def local_topk_stats(local_logits: torch.Tensor, *, k: int) -> LocalTopKStats:
    logits = local_logits.contiguous()
    if not logits.requires_grad:
        return _local_stats_forward(logits, k=k)
    return _LocalStatsFunction.apply(logits, k)


def local_logsumexp_stats(local_logits: torch.Tensor) -> LocalLogSumExpStats:
    logits = local_logits.contiguous()
    if not logits.requires_grad:
        local_max, local_sum, _, _ = _local_stats_forward(logits, k=0)
        return local_max, local_sum
    local_max, local_sum, _, _ = _LocalStatsFunction.apply(logits, 0)
    return local_max, local_sum
