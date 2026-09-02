from __future__ import annotations

from typing import Any

from megatron.core import parallel_state as ps
import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(
    logits,
    row_max,
    row_sum,
    LOCAL_VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    running_max = -float("inf")
    running_sum = 0.0
    for start in tl.range(0, LOCAL_VOCAB_SIZE, BLOCK_V, num_stages=1):
        offsets = start + tl.arange(0, BLOCK_V)
        values = tl.load(
            logits + row * LOCAL_VOCAB_SIZE + offsets,
            mask=offsets < LOCAL_VOCAB_SIZE,
            other=-float("inf"),
        ).to(tl.float32)
        block_max = tl.max(values, axis=0)
        next_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - next_max) + tl.sum(
            tl.exp(values - next_max), axis=0
        )
        running_max = next_max
    tl.store(row_max + row, running_max)
    tl.store(row_sum + row, running_sum)


@triton.jit
def _target_logits_kernel(
    logits,
    targets,
    target_logits,
    value_count,
    LOCAL_VOCAB_SIZE: tl.constexpr,
    target_count,
    VOCAB_START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < value_count
    targets_at_offsets = tl.load(targets + offsets, mask=mask, other=-1)
    local_targets = targets_at_offsets - VOCAB_START
    owned = mask & (local_targets >= 0) & (local_targets < LOCAL_VOCAB_SIZE)
    rows = offsets // target_count
    values = tl.load(
        logits + rows * LOCAL_VOCAB_SIZE + local_targets,
        mask=owned,
        other=0.0,
    ).to(tl.float32)
    tl.store(target_logits + offsets, values, mask=mask)


@triton.jit
def _coefficient_sum_kernel(
    coefficients,
    targets,
    coefficient_sums,
    target_count,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    running_sum = 0.0
    for start in tl.range(0, target_count, BLOCK_K, num_stages=1):
        offsets = start + tl.arange(0, BLOCK_K)
        mask = offsets < target_count
        target_offsets = row * target_count + offsets
        valid = mask & (tl.load(targets + target_offsets, mask=mask, other=-1) >= 0)
        values = tl.load(coefficients + target_offsets, mask=valid, other=0.0).to(
            tl.float32
        )
        running_sum += tl.sum(values, axis=0)
    tl.store(coefficient_sums + row, running_sum)


@triton.jit
def _softmax_workspace_kernel(
    logits,
    row_max,
    row_sum,
    LOCAL_VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offsets < LOCAL_VOCAB_SIZE
    values = tl.load(
        logits + row * LOCAL_VOCAB_SIZE + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    maximum = tl.load(row_max + row)
    denominator = tl.load(row_sum + row)
    probabilities = tl.exp(values - maximum) / denominator
    tl.store(logits + row * LOCAL_VOCAB_SIZE + offsets, probabilities, mask=mask)


@triton.jit
def _scale_softmax_backward_kernel(
    probabilities,
    coefficient_sums,
    LOCAL_VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offsets < LOCAL_VOCAB_SIZE
    values = tl.load(
        probabilities + row * LOCAL_VOCAB_SIZE + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    coefficient_sum = tl.load(coefficient_sums + row)
    tl.store(
        probabilities + row * LOCAL_VOCAB_SIZE + offsets,
        -values * coefficient_sum,
        mask=mask,
    )


@triton.jit
def _target_gradient_kernel(
    coefficients,
    targets,
    grad_logits,
    value_count,
    LOCAL_VOCAB_SIZE: tl.constexpr,
    target_count,
    VOCAB_START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < value_count
    target_values = tl.load(targets + offsets, mask=mask, other=-1)
    local_targets = target_values - VOCAB_START
    owned = mask & (local_targets >= 0) & (local_targets < LOCAL_VOCAB_SIZE)
    rows = offsets // target_count
    coefficients_at_offsets = tl.load(coefficients + offsets, mask=owned, other=0.0)
    tl.atomic_add(
        grad_logits + rows * LOCAL_VOCAB_SIZE + local_targets,
        coefficients_at_offsets,
        mask=owned,
    )


class _VocabParallelMultiTargetLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        local_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        group: Any,
        vocab_start: int,
        tp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_logits.ndim != 2 or target_tokens.ndim != 2:
            raise ValueError(
                "multi-target logprobs require [N,V] logits and [N,K] targets"
            )
        if int(local_logits.shape[0]) != int(target_tokens.shape[0]):
            raise ValueError("multi-target logits and targets do not align")
        if not local_logits.is_cuda or not target_tokens.is_cuda:
            raise ValueError("multi-target logprobs require CUDA tensors")
        if not local_logits.is_contiguous() or not target_tokens.is_contiguous():
            raise ValueError("multi-target logits and targets must be contiguous")
        row_count, local_vocab_size = map(int, local_logits.shape)
        target_count = int(target_tokens.shape[1])
        global_vocab_size = local_vocab_size * tp_size
        valid = target_tokens >= 0
        torch._assert_async(
            torch.all((~valid) | (target_tokens < global_vocab_size)),
            "target token is outside the vocabulary-parallel output",
        )

        local_max = torch.empty(
            row_count, device=local_logits.device, dtype=torch.float32
        )
        local_sum = torch.empty_like(local_max)
        _row_stats_kernel[(row_count,)](
            local_logits,
            local_max,
            local_sum,
            LOCAL_VOCAB_SIZE=local_vocab_size,
            BLOCK_V=1024,
            num_warps=8,
        )
        global_max = local_max.clone()
        if tp_size > 1:
            torch.distributed.all_reduce(
                global_max,
                op=torch.distributed.ReduceOp.MAX,  # ty: ignore[possibly-missing-attribute]
                group=group,
            )
        global_sum = local_sum * torch.exp(local_max - global_max)
        if tp_size > 1:
            torch.distributed.all_reduce(global_sum, group=group)

        target_logits = torch.empty(
            target_tokens.shape, device=local_logits.device, dtype=torch.float32
        )
        value_count = int(target_tokens.numel())
        _target_logits_kernel[(triton.cdiv(value_count, 256),)](
            local_logits,
            target_tokens,
            target_logits,
            value_count,
            LOCAL_VOCAB_SIZE=local_vocab_size,
            target_count=target_count,
            VOCAB_START=vocab_start,
            BLOCK=256,
            num_warps=4,
        )
        if tp_size > 1:
            torch.distributed.all_reduce(target_logits, group=group)
        logprobs = (
            target_logits - (global_max + global_sum.log()).unsqueeze(1)
        ).masked_fill(~valid, 0.0)
        _softmax_workspace_kernel[(row_count, triton.cdiv(local_vocab_size, 256))](
            local_logits,
            global_max,
            global_sum,
            LOCAL_VOCAB_SIZE=local_vocab_size,
            BLOCK_V=256,
            num_warps=4,
        )
        ctx.mark_dirty(local_logits)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(local_logits, target_tokens)
        ctx.vocab_start = vocab_start
        return logprobs, local_logits

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Any
    ) -> tuple[torch.Tensor, None, None, None, None]:
        coefficients = grad_outputs[0].contiguous()
        probabilities, target_tokens = ctx.saved_tensors
        row_count, local_vocab_size = map(int, probabilities.shape)
        target_count = int(target_tokens.shape[1])
        coefficient_sums = torch.empty(
            row_count, device=probabilities.device, dtype=torch.float32
        )
        _coefficient_sum_kernel[(row_count,)](
            coefficients,
            target_tokens,
            coefficient_sums,
            target_count=target_count,
            BLOCK_K=256,
            num_warps=4,
        )
        _scale_softmax_backward_kernel[(row_count, triton.cdiv(local_vocab_size, 256))](
            probabilities,
            coefficient_sums,
            LOCAL_VOCAB_SIZE=local_vocab_size,
            BLOCK_V=256,
            num_warps=4,
        )
        value_count = int(target_tokens.numel())
        _target_gradient_kernel[(triton.cdiv(value_count, 256),)](
            coefficients,
            target_tokens,
            probabilities,
            value_count,
            LOCAL_VOCAB_SIZE=local_vocab_size,
            target_count=target_count,
            VOCAB_START=ctx.vocab_start,
            BLOCK=256,
            num_warps=4,
        )
        return probabilities, None, None, None, None


def vocab_parallel_multi_target_logprobs(
    local_logits: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute K target logprobs from one vocabulary normalization per row."""
    if int(target_tokens.shape[1]) <= 1:
        raise ValueError("multi-target logprobs require at least two targets per row")
    initialized = ps.model_parallel_is_initialized()
    tp_size = int(ps.get_tensor_model_parallel_world_size()) if initialized else 1
    tp_rank = int(ps.get_tensor_model_parallel_rank()) if initialized else 0
    group = (
        ps.get_tensor_model_parallel_group(check_initialized=False)
        if initialized and tp_size > 1
        else None
    )
    logprobs, _ = _VocabParallelMultiTargetLogprobs.apply(
        local_logits,
        target_tokens,
        group,
        tp_rank * int(local_logits.shape[1]),
        tp_size,
    )
    return logprobs
