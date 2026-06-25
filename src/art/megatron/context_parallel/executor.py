from __future__ import annotations

from typing import Any, cast

import torch
from torch._dynamo import config as dynamo_config
import torch.distributed as dist
from torch.nn.attention.flex_attention import AuxOutput, AuxRequest, BlockMask
import triton
import triton.language as tl

from art.megatron.flex_attn.compiled import (
    SparseBlockSize,
    flash_sparse_block_size_for_head_dim,
    flex_backend_for_head_dims,
    get_sparse_compiled_flex_attention,
    normalize_flex_lse,
    normalize_sparse_block_size,
    select_sparse_execution_family,
    sparse_compiled_flex_attention,
)

from .block_mask import build_block_mask
from .comm import A2AVCommunicator
from .range_ops import (
    range_gather_head_major,
    range_reduce_sum_,
    range_reduce_sum_head_major_,
)
from .types import (
    ArtContextParallelState,
    AttnSlice,
    CpBlockMaskVariant,
    DkvReducePlan,
    ExactMaskMetadata,
    FlexMaskSpec,
    StageExecutionSpec,
    StagePlan,
    TokenRange,
)

_COMMUNICATOR = A2AVCommunicator()
_DIST = cast(Any, dist)
_DYNAMO_CONFIG = cast(Any, dynamo_config)

_DYNAMO_CONFIG.recompile_limit = max(int(_DYNAMO_CONFIG.recompile_limit), 256)
_DYNAMO_CONFIG.cache_size_limit = max(int(_DYNAMO_CONFIG.cache_size_limit), 256)
_STAGE_QUERY_GATHER_STREAMS: dict[tuple[str, int | None], torch.cuda.Stream] = {}


def _stage_sparse_block_size(
    q_stage: torch.Tensor,
    v_stage: torch.Tensor,
) -> tuple[int, int]:
    return flash_sparse_block_size_for_head_dim(
        head_dim=int(q_stage.shape[-1]),
        head_dim_v=int(v_stage.shape[-1]),
        device=q_stage.device,
    )


def _pad_exact_indices(indices: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = int(indices.numel())
    target_len = int(target_len)
    if current_len == target_len:
        return indices
    if current_len > target_len:
        raise RuntimeError(
            f"Cannot shrink exact mask metadata from {current_len} to {target_len}"
        )
    pad = torch.full(
        (target_len - current_len,),
        -1,
        dtype=indices.dtype,
        device=indices.device,
    )
    return torch.cat((indices, pad), dim=0)


def _resize_exact_mask_metadata(
    metadata: ExactMaskMetadata | None,
    *,
    q_len: int,
    k_len: int,
) -> ExactMaskMetadata | None:
    if metadata is None:
        return None
    q_indices = _pad_exact_indices(metadata.q_token_indices, int(q_len))
    k_indices = _pad_exact_indices(metadata.k_token_indices, int(k_len))
    if q_indices is metadata.q_token_indices and k_indices is metadata.k_token_indices:
        return metadata
    return ExactMaskMetadata(
        q_token_indices=q_indices,
        k_token_indices=k_indices,
        cache_key=f"{metadata.cache_key}:q{int(q_len)}:k{int(k_len)}",
    )


def _safe_logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.logaddexp(a, b)
    both_neg_inf = torch.isneginf(a) & torch.isneginf(b)
    return torch.where(both_neg_inf, torch.full_like(out, float("-inf")), out)


def _safe_exp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    both_neg_inf = torch.isneginf(a) & torch.isneginf(b)
    diff = torch.where(both_neg_inf, torch.full_like(diff, float("-inf")), diff)
    return torch.exp(diff)


def _softmax_offset_view(
    softmax_offset: torch.Tensor,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    return softmax_offset.to(device=like.device, dtype=like.dtype).view(-1, 1)


def _apply_softmax_offset(
    accum_out: torch.Tensor,
    accum_lse: torch.Tensor,
    softmax_offset: torch.Tensor | None,
) -> torch.Tensor:
    if softmax_offset is None:
        return accum_out
    sink = _softmax_offset_view(softmax_offset, like=accum_lse)
    merged_lse = _safe_logaddexp(accum_lse, sink)
    scale = _safe_exp_diff(accum_lse, merged_lse).unsqueeze(-1)
    return accum_out * scale.to(dtype=accum_out.dtype)


def _softmax_offset_backward_values(
    *,
    grad_output: torch.Tensor,
    accum_out: torch.Tensor,
    accum_lse: torch.Tensor,
    softmax_offset: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if softmax_offset is None:
        return grad_output, None, None
    sink = _softmax_offset_view(softmax_offset, like=accum_lse)
    merged_lse = _safe_logaddexp(accum_lse, sink)
    key_weight = _safe_exp_diff(accum_lse, merged_lse)
    sink_weight = _safe_exp_diff(sink, merged_lse)
    grad_output_accum = grad_output.to(dtype=accum_out.dtype)
    grad_accum_out = grad_output_accum * key_weight.unsqueeze(-1)
    scale_grad = (
        (grad_output_accum * accum_out.to(dtype=grad_output_accum.dtype)).sum(dim=-1)
        * key_weight
        * sink_weight
    )
    return (
        grad_accum_out,
        scale_grad,
        -scale_grad.sum(dim=1).to(dtype=softmax_offset.dtype),
    )


def _accum_output_dtype(input_dtype: torch.dtype) -> torch.dtype:
    if input_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return input_dtype


def _seed_stage_accumulators(
    *,
    stage_out: torch.Tensor,
    stage_lse: torch.Tensor,
    target_dtype: torch.dtype,
    needs_owned_storage: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stage_out.dtype != target_dtype:
        accum_out = stage_out.to(dtype=target_dtype)
    else:
        accum_out = stage_out.clone() if needs_owned_storage else stage_out
    if stage_lse.dtype != target_dtype:
        accum_lse = stage_lse.to(dtype=target_dtype)
    else:
        accum_lse = stage_lse.clone() if needs_owned_storage else stage_lse
    return accum_out, accum_lse


def _stage_merge_values(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    stage_out: torch.Tensor,
    stage_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_lse = _safe_logaddexp(prev_lse, stage_lse)
    prev_weight = _safe_exp_diff(prev_lse, merged_lse).unsqueeze(-1)
    stage_weight = _safe_exp_diff(stage_lse, merged_lse).unsqueeze(-1)
    merged_out = prev_weight * prev_out + stage_weight * stage_out
    return merged_out, merged_lse


def _stage_merge_values_inplace(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    stage_out: torch.Tensor,
    stage_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_lse = _safe_logaddexp(prev_lse, stage_lse)
    prev_weight = _safe_exp_diff(prev_lse, merged_lse).unsqueeze(-1)
    stage_weight = _safe_exp_diff(stage_lse, merged_lse).unsqueeze(-1)
    prev_out.mul_(prev_weight)
    prev_out.add_(stage_out * stage_weight)
    prev_lse.copy_(merged_lse)
    return prev_out, prev_lse


@triton.jit
def _stage_merge_backward_row_kernel(
    prev_out_ptr,
    prev_lse_ptr,
    stage_out_ptr,
    stage_lse_ptr,
    grad_merged_out_ptr,
    grad_merged_lse_ptr,
    grad_prev_out_ptr,
    grad_prev_lse_ptr,
    grad_stage_out_ptr,
    grad_stage_lse_ptr,
    row_stride,
    lse_stride,
    d: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_d)
    mask = cols < d
    out_offsets = row * row_stride + cols
    lse_offset = row * lse_stride

    prev_out = tl.load(prev_out_ptr + out_offsets, mask=mask, other=0.0)
    stage_out = tl.load(stage_out_ptr + out_offsets, mask=mask, other=0.0)
    grad_merged_out = tl.load(grad_merged_out_ptr + out_offsets, mask=mask, other=0.0)

    neg_inf = float("-inf")
    prev_lse = tl.load(prev_lse_ptr + lse_offset)
    stage_lse = tl.load(stage_lse_ptr + lse_offset)
    grad_merged_lse = tl.load(grad_merged_lse_ptr + lse_offset)

    both_neg_inf = (prev_lse == neg_inf) & (stage_lse == neg_inf)
    max_lse = tl.maximum(prev_lse, stage_lse)
    merged_lse = max_lse + tl.log(
        tl.exp(prev_lse - max_lse) + tl.exp(stage_lse - max_lse)
    )
    merged_lse = tl.where(both_neg_inf, neg_inf, merged_lse)

    prev_diff = tl.where(
        (prev_lse == neg_inf) & (merged_lse == neg_inf),
        neg_inf,
        prev_lse - merged_lse,
    )
    stage_diff = tl.where(
        (stage_lse == neg_inf) & (merged_lse == neg_inf),
        neg_inf,
        stage_lse - merged_lse,
    )
    prev_weight = tl.exp(prev_diff)
    stage_weight = tl.exp(stage_diff)

    delta = tl.sum((grad_merged_out * (stage_out - prev_out)).to(tl.float32), axis=0)
    lse_delta = delta * (prev_weight * stage_weight)

    tl.store(
        grad_prev_out_ptr + out_offsets,
        grad_merged_out * prev_weight,
        mask=mask,
    )
    tl.store(
        grad_stage_out_ptr + out_offsets,
        grad_merged_out * stage_weight,
        mask=mask,
    )
    tl.store(grad_prev_lse_ptr + lse_offset, grad_merged_lse * prev_weight - lse_delta)
    tl.store(
        grad_stage_lse_ptr + lse_offset,
        grad_merged_lse * stage_weight + lse_delta,
    )


def _stage_merge_backward_values_triton(
    *,
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    stage_out: torch.Tensor,
    stage_lse: torch.Tensor,
    grad_merged_out: torch.Tensor,
    grad_merged_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not (
        prev_out.is_cuda
        and prev_lse.is_cuda
        and stage_out.is_cuda
        and stage_lse.is_cuda
        and grad_merged_out.is_cuda
        and grad_merged_lse.is_cuda
    ):
        return None
    if not (
        prev_out.is_contiguous()
        and prev_lse.is_contiguous()
        and stage_out.is_contiguous()
        and stage_lse.is_contiguous()
        and grad_merged_out.is_contiguous()
        and grad_merged_lse.is_contiguous()
    ):
        return None
    if prev_out.ndim != 3 or prev_lse.ndim != 2:
        return None
    if prev_out.shape != stage_out.shape or prev_out.shape != grad_merged_out.shape:
        return None
    if prev_lse.shape != stage_lse.shape or prev_lse.shape != grad_merged_lse.shape:
        return None
    if prev_out.shape[:2] != prev_lse.shape:
        return None
    d = int(prev_out.shape[-1])
    if d <= 0 or d > 256:
        return None
    block_d = 1 << max(0, int((d - 1).bit_length()))

    prev_out_rows = prev_out.reshape(-1, d)
    stage_out_rows = stage_out.reshape(-1, d)
    grad_merged_out_rows = grad_merged_out.reshape(-1, d)
    prev_lse_rows = prev_lse.reshape(-1)
    stage_lse_rows = stage_lse.reshape(-1)
    grad_merged_lse_rows = grad_merged_lse.reshape(-1)

    grad_prev_out = torch.empty_like(prev_out_rows)
    grad_stage_out = torch.empty_like(stage_out_rows)
    grad_prev_lse = torch.empty_like(prev_lse_rows)
    grad_stage_lse = torch.empty_like(stage_lse_rows)
    _stage_merge_backward_row_kernel[(prev_out_rows.shape[0],)](
        prev_out_rows,
        prev_lse_rows,
        stage_out_rows,
        stage_lse_rows,
        grad_merged_out_rows,
        grad_merged_lse_rows,
        grad_prev_out,
        grad_prev_lse,
        grad_stage_out,
        grad_stage_lse,
        prev_out_rows.stride(0),
        prev_lse_rows.stride(0),
        d=d,  # ty: ignore[invalid-argument-type]
        block_d=block_d,  # ty: ignore[invalid-argument-type]
        num_warps=4,  # ty: ignore[unknown-argument]
        num_stages=2,  # ty: ignore[unknown-argument]
    )
    return (
        grad_prev_out.view_as(prev_out),
        grad_prev_lse.view_as(prev_lse),
        grad_stage_out.view_as(stage_out),
        grad_stage_lse.view_as(stage_lse),
    )


def _allocate_stage_accumulators(
    *,
    q_flat: torch.Tensor,
    out_dtype: torch.dtype,
    lse_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(q_flat.shape, device=q_flat.device, dtype=out_dtype),
        torch.full(
            (q_flat.shape[0], q_flat.shape[1]),
            float("-inf"),
            device=q_flat.device,
            dtype=lse_dtype,
        ),
    )


def _maybe_promote_accumulators(
    *,
    accum_out: torch.Tensor,
    accum_lse: torch.Tensor,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if accum_out.dtype == target_dtype and accum_lse.dtype == target_dtype:
        return accum_out, accum_lse
    return accum_out.to(dtype=target_dtype), accum_lse.to(dtype=target_dtype)


def _stage_merge_backward_values(
    *,
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    stage_out: torch.Tensor,
    stage_lse: torch.Tensor,
    grad_merged_out: torch.Tensor | None,
    grad_merged_lse: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if grad_merged_out is None:
        grad_merged_out = torch.zeros_like(prev_out)
    if grad_merged_lse is None:
        grad_merged_lse = torch.zeros_like(prev_lse)
    triton_result = _stage_merge_backward_values_triton(
        prev_out=prev_out,
        prev_lse=prev_lse,
        stage_out=stage_out,
        stage_lse=stage_lse,
        grad_merged_out=grad_merged_out,
        grad_merged_lse=grad_merged_lse,
    )
    if triton_result is not None:
        return triton_result
    merged_lse = _safe_logaddexp(prev_lse, stage_lse)
    prev_weight = _safe_exp_diff(prev_lse, merged_lse)
    stage_weight = _safe_exp_diff(stage_lse, merged_lse)
    lse_delta = (grad_merged_out * (stage_out - prev_out)).sum(dim=-1) * (
        prev_weight * stage_weight
    )
    return (
        grad_merged_out * prev_weight.unsqueeze(-1),
        grad_merged_lse * prev_weight - lse_delta,
        grad_merged_out * stage_weight.unsqueeze(-1),
        grad_merged_lse * stage_weight + lse_delta,
    )


class _StageMergeFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prev_out: torch.Tensor,
        prev_lse: torch.Tensor,
        stage_out: torch.Tensor,
        stage_lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(prev_out, prev_lse, stage_out, stage_lse)
        return _stage_merge_values(prev_out, prev_lse, stage_out, stage_lse)

    @staticmethod
    def backward(
        ctx,
        *grad_outputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_merged_out, grad_merged_lse = cast(
            tuple[torch.Tensor | None, torch.Tensor | None],
            grad_outputs,
        )
        prev_out, prev_lse, stage_out, stage_lse = ctx.saved_tensors
        return _stage_merge_backward_values(
            prev_out=prev_out,
            prev_lse=prev_lse,
            stage_out=stage_out,
            stage_lse=stage_lse,
            grad_merged_out=grad_merged_out,
            grad_merged_lse=grad_merged_lse,
        )


class _StageScatterMergeFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        accum_out: torch.Tensor,
        accum_lse: torch.Tensor,
        stage_out: torch.Tensor,
        stage_lse: torch.Tensor,
        q_index: torch.Tensor,
        index_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_out = torch.index_select(accum_out, index_dim, q_index)
        prev_lse = torch.index_select(accum_lse, index_dim, q_index)
        merged_out, merged_lse = _stage_merge_values(
            prev_out,
            prev_lse,
            stage_out,
            stage_lse,
        )
        ctx.save_for_backward(prev_out, prev_lse, stage_out, stage_lse, q_index)
        ctx.index_dim = int(index_dim)
        ctx.accum_out_shape = tuple(accum_out.shape)
        ctx.accum_lse_shape = tuple(accum_lse.shape)
        ctx.accum_out_dtype = accum_out.dtype
        ctx.accum_lse_dtype = accum_lse.dtype
        ctx.accum_device = accum_out.device
        ctx.mark_dirty(accum_out, accum_lse)
        accum_out.index_copy_(ctx.index_dim, q_index, merged_out)
        accum_lse.index_copy_(ctx.index_dim, q_index, merged_lse)
        return accum_out, accum_lse

    @staticmethod
    def backward(
        ctx,
        *grad_outputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        grad_updated_out, grad_updated_lse = cast(
            tuple[torch.Tensor | None, torch.Tensor | None],
            grad_outputs,
        )
        prev_out, prev_lse, stage_out, stage_lse, q_index = ctx.saved_tensors
        if grad_updated_out is None:
            grad_updated_out = torch.zeros(
                ctx.accum_out_shape,
                device=ctx.accum_device,
                dtype=ctx.accum_out_dtype,
            )
        if grad_updated_lse is None:
            grad_updated_lse = torch.zeros(
                ctx.accum_lse_shape,
                device=ctx.accum_device,
                dtype=ctx.accum_lse_dtype,
            )
        grad_merged_out = torch.index_select(grad_updated_out, ctx.index_dim, q_index)
        grad_merged_lse = torch.index_select(grad_updated_lse, ctx.index_dim, q_index)
        grad_prev_out, grad_prev_lse, grad_stage_out, grad_stage_lse = (
            _stage_merge_backward_values(
                prev_out=prev_out,
                prev_lse=prev_lse,
                stage_out=stage_out,
                stage_lse=stage_lse,
                grad_merged_out=grad_merged_out,
                grad_merged_lse=grad_merged_lse,
            )
        )
        grad_accum_out = grad_updated_out.clone()
        grad_accum_out.index_copy_(ctx.index_dim, q_index, grad_prev_out)
        grad_accum_lse = grad_updated_lse.clone()
        grad_accum_lse.index_copy_(ctx.index_dim, q_index, grad_prev_lse)
        return (
            grad_accum_out,
            grad_accum_lse,
            grad_stage_out,
            grad_stage_lse,
            None,
            None,
        )


def flatten_valid_sequence(
    tensor: torch.Tensor,
    valid_lengths: tuple[int, ...],
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise RuntimeError(f"Expected [S, B, H, D] tensor, got {tuple(tensor.shape)}")
    if len(valid_lengths) != 1 or int(tensor.shape[1]) != 1:
        raise RuntimeError(
            "ART context parallel attention only supports exactly one packed sequence in the hot path, "
            f"got valid_lengths={valid_lengths} and batch={int(tensor.shape[1])}."
        )
    valid_len = int(valid_lengths[0])
    if valid_len <= 0:
        return tensor.new_empty((0, tensor.shape[2], tensor.shape[3]))
    return tensor[:valid_len, 0].contiguous()


def flatten_valid_sequence_head_major(
    tensor: torch.Tensor,
    valid_lengths: tuple[int, ...],
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise RuntimeError(f"Expected [S, B, H, D] tensor, got {tuple(tensor.shape)}")
    if len(valid_lengths) != 1 or int(tensor.shape[1]) != 1:
        raise RuntimeError(
            "ART context parallel attention only supports exactly one packed sequence in the hot path, "
            f"got valid_lengths={valid_lengths} and batch={int(tensor.shape[1])}."
        )
    valid_len = int(valid_lengths[0])
    if valid_len <= 0:
        return tensor.new_empty((tensor.shape[2], 0, tensor.shape[3]))
    return tensor[:valid_len, 0].permute(1, 0, 2)


def unflatten_valid_sequence(
    flat: torch.Tensor,
    *,
    valid_lengths: tuple[int, ...],
    seq_len: int,
) -> torch.Tensor:
    if flat.ndim != 3:
        raise RuntimeError(f"Expected [N, H, D] flat tensor, got {tuple(flat.shape)}")
    if len(valid_lengths) != 1:
        raise RuntimeError(
            "ART context parallel attention only supports exactly one packed sequence in the hot path, "
            f"got valid_lengths={valid_lengths}."
        )
    valid_len = int(valid_lengths[0])
    if int(flat.shape[0]) != valid_len:
        raise RuntimeError(
            "unflatten_valid_sequence expected flat rows to match valid length: "
            f"{int(flat.shape[0])} != {valid_len}"
        )
    if valid_len == seq_len:
        return flat.unsqueeze(1).contiguous()
    output = flat.new_zeros((seq_len, 1, flat.shape[1], flat.shape[2]))
    if valid_len > 0:
        output[:valid_len, 0] = flat
    return output


def unflatten_valid_sequence_head_major(
    flat: torch.Tensor,
    *,
    valid_lengths: tuple[int, ...],
    seq_len: int,
) -> torch.Tensor:
    if flat.ndim != 3:
        raise RuntimeError(
            f"Expected [H, N, D] head-major flat tensor, got {tuple(flat.shape)}"
        )
    if len(valid_lengths) != 1:
        raise RuntimeError(
            "ART context parallel attention only supports exactly one packed sequence in the hot path, "
            f"got valid_lengths={valid_lengths}."
        )
    valid_len = int(valid_lengths[0])
    if int(flat.shape[1]) != valid_len:
        raise RuntimeError(
            "unflatten_valid_sequence_head_major expected flat token dim to match valid length: "
            f"{int(flat.shape[1])} != {valid_len}"
        )
    token_major = flat.permute(1, 0, 2)
    if valid_len == seq_len:
        return token_major.unsqueeze(1)
    return unflatten_valid_sequence(
        token_major,
        valid_lengths=valid_lengths,
        seq_len=seq_len,
    )


class FlexAttentionKernel:
    def __init__(
        self,
        *,
        compile_enabled: bool,
        triton_num_stages_2_head_dims: tuple[int, ...] = (),
    ) -> None:
        if not compile_enabled:
            raise RuntimeError(
                "ART context parallel attention requires compiled flex attention."
            )
        self.triton_num_stages_2_head_dims = tuple(
            int(dim) for dim in triton_num_stages_2_head_dims
        )

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_local_stage: bool = True,
        compile_key: str | None = None,
        block_mask: BlockMask | None,
        scale: float,
        enable_gqa: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            q.dtype.is_floating_point
            and k.dtype.is_floating_point
            and v.dtype.is_floating_point
        ):
            raise RuntimeError(
                "ART context parallel attention requires floating-point inputs for compiled flex attention, "
                f"got q={q.dtype}, k={k.dtype}, v={v.dtype}."
            )
        if block_mask is None:
            raise RuntimeError(
                "ART context parallel attention requires a concrete block mask for compiled flex attention."
            )
        backend = flex_backend_for_head_dims(
            head_dim=int(q.shape[-1]),
            head_dim_v=int(v.shape[-1]),
        )
        if compile_key is None:
            _q_len, _k_len, compile_key = select_sparse_execution_family(
                is_local_stage=bool(is_local_stage),
                q_len=int(q.shape[2]),
                k_len=int(k.shape[2]),
                block_size=block_mask.BLOCK_SIZE,
            )
        compiled_flex_attention = (
            sparse_compiled_flex_attention
            if str(compile_key) == "sparse" and backend == "FLASH"
            else get_sparse_compiled_flex_attention(
                family_key=str(compile_key),
                backend=backend,
                head_dim=int(q.shape[-1]),
                head_dim_v=int(v.shape[-1]),
                triton_num_stages_2_head_dims=self.triton_num_stages_2_head_dims,
            )
        )
        out, aux = cast(
            tuple[torch.Tensor, AuxOutput],
            compiled_flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
                return_aux=AuxRequest(lse=True),
            ),
        )
        lse = aux.lse
        if lse is None:
            raise RuntimeError("Compiled flex attention did not return lse.")
        lse = normalize_flex_lse(lse, backend=backend)
        return out, lse


def _build_stage_block_mask(
    *,
    stage_plan: StagePlan,
    state: ArtContextParallelState,
    device: torch.device,
    execution_spec: StageExecutionSpec | None = None,
    block_size: SparseBlockSize | None = None,
    sliding_window: int | None = None,
) -> BlockMask | None:
    if block_size is None:
        block_size = state.config.attention_sparse_block_size or state.config.block_size
    resolved_block_size = normalize_sparse_block_size(block_size)
    execution_spec = (
        _resolve_stage_execution_spec(
            stage_plan=stage_plan,
            state=state,
            block_size=resolved_block_size,
        )
        if execution_spec is None
        else execution_spec
    )
    cache_key = _stage_block_mask_cache_key(
        stage_plan=stage_plan,
        execution_spec=execution_spec,
        block_size=resolved_block_size,
        sliding_window=sliding_window,
        device=device,
    )
    cache = state.execution_cache.block_masks
    cached = cache.get(cache_key)
    if cached is not None or cache_key in cache:
        return cached
    mask_metadata = (
        stage_plan.mask_metadata
        if execution_spec.mask_metadata is None
        else execution_spec.mask_metadata
    )
    if mask_metadata is None:
        raise RuntimeError(
            f"Stage {stage_plan.stage_index} is missing exact mask metadata"
        )
    mask = build_block_mask(
        FlexMaskSpec(
            q_len=int(execution_spec.q_len),
            k_len=int(execution_spec.k_len),
            block_size=resolved_block_size,
            slices=stage_plan.slices,
            exact_mask=mask_metadata.model_dump(mode="python"),
        ),
        group_ids=state.group_ids,
        parent_ids=state.parent_ids,
        input_pos=state.input_pos,
        sliding_window=sliding_window,
        device=device,
    )
    cache[cache_key] = mask
    return mask


def _get_prepared_stage_block_mask(
    *,
    stage_plan: StagePlan,
    state: ArtContextParallelState,
    device: torch.device,
    execution_spec: StageExecutionSpec,
    block_size: SparseBlockSize,
    sliding_window: int | None,
) -> BlockMask:
    cache_key = _stage_block_mask_cache_key(
        stage_plan=stage_plan,
        execution_spec=execution_spec,
        block_size=normalize_sparse_block_size(block_size),
        sliding_window=sliding_window,
        device=device,
    )
    cache = state.execution_cache.block_masks
    if cache_key not in cache:
        raise RuntimeError(
            "ART context parallel forward hit an unprepared stage block-mask cache key. "
            "Mask construction is CPU planning work and must finish before model forward. "
            f"stage={int(stage_plan.stage_index)} q_len={int(execution_spec.q_len)} "
            f"k_len={int(execution_spec.k_len)} "
            f"block_size={normalize_sparse_block_size(block_size)} "
            f"sliding_window={sliding_window} device={device}"
        )
    block_mask = cache[cache_key]
    if block_mask is None:
        raise RuntimeError(
            "ART context parallel forward found an empty prepared block mask for a non-empty stage. "
            f"stage={int(stage_plan.stage_index)} q_len={int(execution_spec.q_len)} "
            f"k_len={int(execution_spec.k_len)} sliding_window={sliding_window}"
        )
    return cast(BlockMask, block_mask)


def _stage_block_mask_cache_key(
    *,
    stage_plan: StagePlan,
    execution_spec: StageExecutionSpec,
    block_size: tuple[int, int],
    sliding_window: int | None,
    device: torch.device,
) -> tuple[int, int, int, tuple[int, int], int | None, str, int | None]:
    return (
        int(stage_plan.stage_index),
        int(execution_spec.q_len),
        int(execution_spec.k_len),
        block_size,
        None if sliding_window is None else int(sliding_window),
        device.type,
        device.index,
    )


def prepare_context_parallel_execution_state(
    *,
    state: ArtContextParallelState,
    device: torch.device,
) -> None:
    variants = state.block_mask_variants or (
        CpBlockMaskVariant(
            sliding_window=None,
            block_size=normalize_sparse_block_size(state.config.block_size),
        ),
    )
    for stage_plan in state.rank_plan.stage_plans:
        if stage_plan.q_len <= 0 or stage_plan.k_len <= 0 or not stage_plan.slices:
            continue
        for variant in variants:
            execution_spec = _resolve_stage_execution_spec(
                stage_plan=stage_plan,
                state=state,
                block_size=variant.block_size,
            )
            _build_stage_block_mask(
                stage_plan=stage_plan,
                state=state,
                device=device,
                execution_spec=execution_spec,
                block_size=variant.block_size,
                sliding_window=variant.sliding_window,
            )


def _causal_slice_pair_count(slice_: AttnSlice) -> int:
    q_start = int(slice_.q_range.start)
    q_end = int(slice_.q_range.end)
    k_start = int(slice_.k_range.start)
    k_end = int(slice_.k_range.end)
    if q_end <= q_start or k_end <= k_start:
        return 0

    k_len = k_end - k_start
    partial_q_start = max(q_start, k_start)
    partial_q_end = min(q_end - 1, k_end - 2)
    partial = 0
    if partial_q_start <= partial_q_end:
        count = partial_q_end - partial_q_start + 1
        partial = count * (partial_q_start + partial_q_end + 2 - 2 * k_start) // 2

    full_q_start = max(q_start, k_end - 1)
    full_q_end = q_end - 1
    full = 0
    if full_q_start <= full_q_end:
        full = (full_q_end - full_q_start + 1) * k_len
    return int(partial + full)


def _validate_stage_block_alignment(
    *,
    q_len: int,
    k_len: int,
    block_mask: BlockMask,
) -> None:
    q_block, k_block = normalize_sparse_block_size(block_mask.BLOCK_SIZE)
    if q_len <= 0 or k_len <= 0:
        return
    if (q_len % q_block) != 0 or (k_len % k_block) != 0:
        raise RuntimeError(
            "ART context parallel attention requires block-aligned stage shapes, "
            f"got q_len={q_len} k_len={k_len} "
            f"with block_size=({q_block}, {k_block})"
        )


def _logical_stage_q_len(stage_plan: StagePlan) -> int:
    return int(sum(range_.size() for range_ in stage_plan.owner_local_q_ranges))


def _logical_stage_k_len(stage_plan: StagePlan) -> int:
    return int(sum(range_.size() for range_ in stage_plan.owner_local_k_ranges))


def _pad_stage_token_tensor(
    tensor: torch.Tensor,
    *,
    target_len: int,
    head_major: bool = False,
) -> torch.Tensor:
    current_len = int(tensor.shape[1] if head_major else tensor.shape[0])
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        raise RuntimeError(
            f"Cannot shrink stage tensor from {current_len} to {target_len} rows"
        )
    pad_shape = list(tensor.shape)
    pad_shape[1 if head_major else 0] = target_len - current_len
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    dim = 1 if head_major else 0
    return torch.cat((tensor, pad), dim=dim)


def _resolve_stage_execution_spec(
    *,
    stage_plan: StagePlan,
    state: ArtContextParallelState,
    block_size: SparseBlockSize | None = None,
) -> StageExecutionSpec:
    resolved_block_size = normalize_sparse_block_size(
        state.config.block_size if block_size is None else block_size
    )
    cache_key = (int(stage_plan.stage_index), resolved_block_size)
    execution_cache = getattr(state, "execution_cache", None)
    if execution_cache is None:
        target_q_len, target_k_len, compile_key = select_sparse_execution_family(
            is_local_stage=bool(stage_plan.is_local_stage),
            q_len=int(stage_plan.q_len),
            k_len=int(stage_plan.k_len),
            block_size=resolved_block_size,
        )
        return StageExecutionSpec(
            q_len=int(target_q_len),
            k_len=int(target_k_len),
            compile_key=str(compile_key),
            mask_metadata=_resize_exact_mask_metadata(
                stage_plan.mask_metadata,
                q_len=int(target_q_len),
                k_len=int(target_k_len),
            ),
        )
    cache = getattr(execution_cache, "stage_execution_specs", None)
    if cache is None:
        target_q_len, target_k_len, compile_key = select_sparse_execution_family(
            is_local_stage=bool(stage_plan.is_local_stage),
            q_len=int(stage_plan.q_len),
            k_len=int(stage_plan.k_len),
            block_size=resolved_block_size,
        )
        return StageExecutionSpec(
            q_len=int(target_q_len),
            k_len=int(target_k_len),
            compile_key=str(compile_key),
            mask_metadata=_resize_exact_mask_metadata(
                stage_plan.mask_metadata,
                q_len=int(target_q_len),
                k_len=int(target_k_len),
            ),
        )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    target_q_len, target_k_len, compile_key = select_sparse_execution_family(
        is_local_stage=bool(stage_plan.is_local_stage),
        q_len=int(stage_plan.q_len),
        k_len=int(stage_plan.k_len),
        block_size=resolved_block_size,
    )
    resolved = StageExecutionSpec(
        q_len=int(target_q_len),
        k_len=int(target_k_len),
        compile_key=str(compile_key),
        mask_metadata=_resize_exact_mask_metadata(
            stage_plan.mask_metadata,
            q_len=int(target_q_len),
            k_len=int(target_k_len),
        ),
    )
    cache[cache_key] = resolved
    return resolved


def _run_stage_attention(
    *,
    q_stage: torch.Tensor,
    k_stage: torch.Tensor,
    v_stage: torch.Tensor,
    stage_plan: StagePlan,
    state: ArtContextParallelState,
    kernel: FlexAttentionKernel,
    scale: float,
    enable_gqa: bool,
    sliding_window: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sparse_block_size = _stage_sparse_block_size(q_stage, v_stage)
    execution_spec = _resolve_stage_execution_spec(
        stage_plan=stage_plan,
        state=state,
        block_size=sparse_block_size,
    )
    block_mask = _get_prepared_stage_block_mask(
        stage_plan=stage_plan,
        state=state,
        device=q_stage.device,
        execution_spec=execution_spec,
        block_size=sparse_block_size,
        sliding_window=sliding_window,
    )
    _validate_stage_block_alignment(
        q_len=int(execution_spec.q_len),
        k_len=int(execution_spec.k_len),
        block_mask=block_mask,
    )
    logical_q_len = _logical_stage_q_len(stage_plan)
    input_head_major = q_stage.ndim == 3 and int(q_stage.shape[1]) == logical_q_len
    q_stage = _pad_stage_token_tensor(
        q_stage,
        target_len=int(execution_spec.q_len),
        head_major=input_head_major,
    )
    k_stage = _pad_stage_token_tensor(
        k_stage,
        target_len=int(execution_spec.k_len),
        head_major=input_head_major,
    )
    v_stage = _pad_stage_token_tensor(
        v_stage,
        target_len=int(execution_spec.k_len),
        head_major=input_head_major,
    )
    if input_head_major:
        q_flex = q_stage.unsqueeze(0)
        k_flex = k_stage.unsqueeze(0)
        v_flex = v_stage.unsqueeze(0)
    else:
        q_flex = q_stage.permute(1, 0, 2).unsqueeze(0).contiguous()
        k_flex = k_stage.permute(1, 0, 2).unsqueeze(0).contiguous()
        v_flex = v_stage.permute(1, 0, 2).unsqueeze(0).contiguous()
    out, lse = kernel.run(
        q_flex,
        k_flex,
        v_flex,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    if input_head_major:
        out_tokens = out.squeeze(0)
        lse_tokens = lse.squeeze(0).to(dtype=torch.float32)
        return (
            (
                out_tokens[:, :logical_q_len]
                if int(execution_spec.q_len) == logical_q_len
                else out_tokens[:, :logical_q_len].contiguous()
            ),
            (
                lse_tokens[:, :logical_q_len]
                if int(execution_spec.q_len) == logical_q_len
                else lse_tokens[:, :logical_q_len].contiguous()
            ),
        )
    out_tokens = out.squeeze(0).permute(1, 0, 2).contiguous()
    lse_tokens = lse.squeeze(0).permute(1, 0).contiguous().to(dtype=torch.float32)
    return (
        out_tokens[:logical_q_len].contiguous(),
        lse_tokens[:logical_q_len].contiguous(),
    )


def _range_index_tensor(
    ranges: tuple,
    *,
    device: torch.device,
    range_index_cache: dict[Any, torch.Tensor] | None = None,
) -> torch.Tensor:
    key = (
        tuple((range_.start, range_.end) for range_ in ranges if range_.size() > 0),
        device.type,
        device.index,
    )
    if range_index_cache is not None:
        cached = range_index_cache.get(key)
        if cached is not None:
            return cached
    parts = [
        torch.arange(range_.start, range_.end, device=device, dtype=torch.int64)
        for range_ in ranges
        if range_.size() > 0
    ]
    if not parts:
        cached = torch.empty((0,), device=device, dtype=torch.int64)
    else:
        cached = torch.cat(parts, dim=0)
    if range_index_cache is not None:
        range_index_cache[key] = cached
    return cached


def _ranges_cover_full_length(
    ranges: tuple,
    *,
    length: int,
) -> bool:
    cursor = 0
    for range_ in ranges:
        if range_.size() <= 0:
            continue
        if range_.start != cursor:
            return False
        cursor = range_.end
    return cursor == length


def _ordered_stage_plans(stage_plans: tuple[StagePlan, ...]) -> list[StagePlan]:
    return sorted(
        stage_plans,
        key=lambda stage_plan: (not stage_plan.is_local_stage, stage_plan.stage_index),
    )


def _stage_q_is_full(
    *,
    stage_plan: StagePlan,
    q_flat_len: int,
) -> bool:
    return _ranges_cover_full_length(
        stage_plan.owner_local_q_ranges,
        length=q_flat_len,
    )


def _stage_requires_reduce(stage_plan: StagePlan) -> bool:
    if stage_plan.dkv_reduce_plan is None:
        return False
    return bool(
        sum(stage_plan.dkv_reduce_plan.send_splits)
        or sum(stage_plan.dkv_reduce_plan.recv_splits)
    )


def _distributed_cp_comm_enabled(state: ArtContextParallelState) -> bool:
    return state.cp_group is not None and _DIST.get_world_size(state.cp_group) > 1


def _remote_comm_launch_enabled(
    *,
    state: ArtContextParallelState,
    remote_stages: list[StagePlan],
) -> bool:
    if not remote_stages:
        return False
    if not _distributed_cp_comm_enabled(state):
        raise RuntimeError(
            "ART context parallel remote stages require distributed async per-stage KV fetch."
        )
    return True


def _ready_remote_stage_batch(
    *,
    pending_stages: list[StagePlan],
    fetch_works_by_stage_index: dict[int, Any],
) -> list[StagePlan]:
    ready_stages: list[StagePlan] = []
    for stage_plan in pending_stages:
        fetch_work = fetch_works_by_stage_index.get(int(stage_plan.stage_index))
        if fetch_work is None or fetch_work.is_completed():
            ready_stages.append(stage_plan)
    if ready_stages:
        return ready_stages
    if not pending_stages:
        return []
    fetch_work = fetch_works_by_stage_index.get(int(pending_stages[0].stage_index))
    if fetch_work is None:
        return [pending_stages[0]]
    fetch_work.wait()
    ready_stages = []
    for stage_plan in pending_stages:
        fetch_work = fetch_works_by_stage_index.get(int(stage_plan.stage_index))
        if fetch_work is None or fetch_work.is_completed():
            ready_stages.append(stage_plan)
    return ready_stages


def _drain_launched_remote_fetch_works(
    *,
    fetch_works_by_stage_index: dict[int, Any],
) -> None:
    for fetch_work in fetch_works_by_stage_index.values():
        if fetch_work is not None:
            fetch_work.wait()


class _StageQueryGatherWork:
    def __init__(
        self,
        *,
        gathered_q: torch.Tensor,
        stream: torch.cuda.Stream | None,
    ) -> None:
        self.gathered_q = gathered_q
        self.stream = stream

    def wait_post_process(self) -> torch.Tensor:
        if self.stream is not None:
            torch.cuda.current_stream(self.gathered_q.device).wait_stream(self.stream)
        return self.gathered_q


def _get_stage_query_gather_stream(tensor: torch.Tensor) -> torch.cuda.Stream | None:
    if not tensor.is_cuda:
        return None
    key = (tensor.device.type, tensor.device.index)
    stream = _STAGE_QUERY_GATHER_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=tensor.device)
        _STAGE_QUERY_GATHER_STREAMS[key] = stream
    return stream


def _launch_stage_query_gather(
    *,
    q_flat: torch.Tensor,
    state: ArtContextParallelState,
    stage_plan: StagePlan,
) -> _StageQueryGatherWork | None:
    if stage_plan.q_len == 0:
        return None
    if _ranges_cover_full_length(
        stage_plan.owner_local_q_ranges,
        length=int(q_flat.shape[1]),
    ):
        return None
    stream = _get_stage_query_gather_stream(q_flat)
    if stream is None:
        return None
    gathered_q = q_flat.new_empty(
        (q_flat.shape[0], _logical_stage_q_len(stage_plan), q_flat.shape[2])
    )
    current_stream = torch.cuda.current_stream(q_flat.device)
    stream.wait_stream(current_stream)
    q_flat.record_stream(stream)
    gathered_q.record_stream(stream)
    with torch.cuda.stream(stream):
        range_gather_head_major(
            q_flat,
            stage_plan.owner_local_q_ranges,
            output=gathered_q,
            range_meta_cache=state.execution_cache.range_meta,
        )
    return _StageQueryGatherWork(gathered_q=gathered_q, stream=stream)


def _stage_remote_kv_tensors(
    *,
    stage_plan: StagePlan,
    fetch_work: Any,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if fetch_work is None:
        raise RuntimeError(
            f"Remote stage {stage_plan.stage_index} is missing async KV fetch work"
        )
    output_layout = str(getattr(fetch_work, "output_layout", "head_major"))
    if output_layout != "head_major":
        raise RuntimeError(
            "Remote stage KV fetch must land in head-major layout for flex attention, "
            f"got output_layout={output_layout!r} for stage={stage_plan.stage_index}"
        )
    k_stage, v_stage = fetch_work.wait_post_process()
    k_rows = int(k_stage.shape[-2])
    v_rows = int(v_stage.shape[-2])
    expected_rows = _logical_stage_k_len(stage_plan)
    if k_rows != expected_rows or v_rows != expected_rows:
        raise RuntimeError(
            "Remote stage fetch returned the wrong number of rows: "
            f"stage={stage_plan.stage_index} expected={expected_rows} "
            f"got_k={k_rows} got_v={v_rows}"
        )
    return k_stage, v_stage, True


def _stage_query_tensor(
    *,
    q_flat: torch.Tensor,
    state: ArtContextParallelState,
    stage_plan: StagePlan,
) -> torch.Tensor:
    if stage_plan.q_len == 0:
        return q_flat.new_empty((q_flat.shape[0], 0, q_flat.shape[2]))
    if _ranges_cover_full_length(
        stage_plan.owner_local_q_ranges,
        length=int(q_flat.shape[1]),
    ):
        return q_flat
    return range_gather_head_major(
        q_flat,
        stage_plan.owner_local_q_ranges,
        range_meta_cache=state.execution_cache.range_meta,
    )


def _stage_local_kv_tensors(
    *,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    state: ArtContextParallelState,
    stage_plan: StagePlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stage_plan.k_len == 0:
        empty = k_flat.new_empty((k_flat.shape[0], 0, k_flat.shape[2]))
        return empty, empty
    kv_is_full = _ranges_cover_full_length(
        stage_plan.owner_local_k_ranges,
        length=int(k_flat.shape[1]),
    )
    if kv_is_full:
        return k_flat, v_flat
    return (
        range_gather_head_major(
            k_flat,
            stage_plan.owner_local_k_ranges,
            range_meta_cache=state.execution_cache.range_meta,
        ),
        range_gather_head_major(
            v_flat,
            stage_plan.owner_local_k_ranges,
            range_meta_cache=state.execution_cache.range_meta,
        ),
    )


def _merge_stage_output(
    *,
    accum_out: torch.Tensor,
    accum_lse: torch.Tensor,
    stage_out: torch.Tensor,
    stage_lse: torch.Tensor,
    state: ArtContextParallelState,
    stage_plan: StagePlan,
    q_is_full: bool | None = None,
    produced_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q_is_full is None:
        q_is_full = _ranges_cover_full_length(
            stage_plan.owner_local_q_ranges,
            length=int(accum_out.shape[1]),
        )
    if not produced_output:
        if q_is_full:
            return stage_out, stage_lse
        cursor = 0
        for range_ in stage_plan.owner_local_q_ranges:
            size = range_.size()
            if size <= 0:
                continue
            next_cursor = cursor + size
            accum_out[:, range_.start : range_.end].copy_(
                stage_out[:, cursor:next_cursor]
            )
            accum_lse[:, range_.start : range_.end].copy_(
                stage_lse[:, cursor:next_cursor]
            )
            cursor = next_cursor
        return accum_out, accum_lse
    if q_is_full:
        if (
            accum_out.requires_grad
            or accum_lse.requires_grad
            or stage_out.requires_grad
            or stage_lse.requires_grad
        ):
            return _StageMergeFn.apply(accum_out, accum_lse, stage_out, stage_lse)
        return _stage_merge_values_inplace(
            accum_out,
            accum_lse,
            stage_out,
            stage_lse,
        )
    if (
        accum_out.requires_grad
        or accum_lse.requires_grad
        or stage_out.requires_grad
        or stage_lse.requires_grad
    ):
        q_index = _range_index_tensor(
            stage_plan.owner_local_q_ranges,
            device=accum_out.device,
            range_index_cache=state.execution_cache.range_indices,
        )
        return _StageScatterMergeFn.apply(
            accum_out,
            accum_lse,
            stage_out,
            stage_lse,
            q_index,
            1,
        )
    cursor = 0
    for range_ in stage_plan.owner_local_q_ranges:
        size = range_.size()
        if size <= 0:
            continue
        next_cursor = cursor + size
        _stage_merge_values_inplace(
            accum_out[:, range_.start : range_.end],
            accum_lse[:, range_.start : range_.end],
            stage_out[:, cursor:next_cursor],
            stage_lse[:, cursor:next_cursor],
        )
        cursor = next_cursor
    return accum_out, accum_lse


def _capture_stage_merge_tape(
    *,
    accum_out: torch.Tensor | None,
    accum_lse: torch.Tensor | None,
    q_flat_len: int,
    device: torch.device,
    state: ArtContextParallelState,
    stage_plan: StagePlan,
    produced_output: bool,
) -> dict[str, Any]:
    q_is_full = _ranges_cover_full_length(
        stage_plan.owner_local_q_ranges,
        length=q_flat_len,
    )
    q_index = None
    if not q_is_full:
        q_index = _range_index_tensor(
            stage_plan.owner_local_q_ranges,
            device=device,
            range_index_cache=state.execution_cache.range_indices,
        )
    if not produced_output:
        return {
            "merge_is_copy": True,
            "merge_q_is_full": q_is_full,
            "merge_q_index": q_index,
        }
    if accum_out is None or accum_lse is None:
        raise RuntimeError("Missing merge accumulators for produced stage output")
    if q_is_full:
        prev_out = accum_out.detach().clone()
        prev_lse = accum_lse.detach().clone()
    else:
        prev_out = torch.index_select(accum_out, 1, cast(torch.Tensor, q_index))
        prev_lse = torch.index_select(accum_lse, 1, cast(torch.Tensor, q_index))
    return {
        "merge_is_copy": False,
        "merge_q_is_full": q_is_full,
        "merge_q_index": q_index,
        "merge_prev_out": prev_out,
        "merge_prev_lse": prev_lse,
    }


def _release_replay_record_merge_tape(record: dict[str, Any]) -> None:
    for key in (
        "merge_is_copy",
        "merge_q_is_full",
        "merge_q_index",
        "merge_prev_out",
        "merge_prev_lse",
    ):
        record.pop(key, None)


def _release_replay_record_tensors(record: dict[str, Any]) -> None:
    _release_replay_record_merge_tape(record)
    for key in (
        "q_input",
        "k_input",
        "v_input",
        "stage_out",
        "stage_lse",
    ):
        record.pop(key, None)


def _merge_stage_output_grads_from_tape(
    *,
    replay_records: list[dict[str, Any]],
    grad_output_flat: torch.Tensor,
    grad_lse_flat: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if not replay_records:
        return [], []
    accum_dtype = _accum_output_dtype(grad_output_flat.dtype)
    grad_accum_out = grad_output_flat.to(
        dtype=accum_dtype,
        memory_format=torch.contiguous_format,
    )
    if grad_lse_flat is None:
        grad_accum_lse = torch.zeros(
            (grad_output_flat.shape[0], grad_output_flat.shape[1]),
            device=grad_output_flat.device,
            dtype=accum_dtype,
        )
    else:
        grad_accum_lse = grad_lse_flat.to(
            dtype=accum_dtype,
            memory_format=torch.contiguous_format,
        )
    stage_out_grads: list[torch.Tensor] = []
    stage_lse_grads: list[torch.Tensor] = []
    for record in replay_records:
        stage_out_grads.append(
            torch.zeros_like(cast(torch.Tensor, record["stage_out"]))
        )
        stage_lse_grads.append(
            torch.zeros_like(cast(torch.Tensor, record["stage_lse"]))
        )
    for record_index in range(len(replay_records) - 1, -1, -1):
        record = replay_records[record_index]
        q_index = cast(torch.Tensor | None, record.get("merge_q_index"))
        if bool(record.get("merge_q_is_full", False)):
            grad_merged_out = grad_accum_out
            grad_merged_lse = grad_accum_lse
        else:
            if q_index is None:
                raise RuntimeError("Missing stage q index for partial merge tape")
            grad_merged_out = torch.index_select(grad_accum_out, 1, q_index)
            grad_merged_lse = torch.index_select(grad_accum_lse, 1, q_index)
        stage_out = cast(torch.Tensor, record["stage_out"])
        stage_lse = cast(torch.Tensor, record["stage_lse"])
        if bool(record.get("merge_is_copy", False)):
            stage_out_grads[record_index] = grad_merged_out.to(dtype=stage_out.dtype)
            stage_lse_grads[record_index] = grad_merged_lse.to(dtype=stage_lse.dtype)
            _release_replay_record_merge_tape(record)
            continue
        prev_out = cast(torch.Tensor, record["merge_prev_out"])
        prev_lse = cast(torch.Tensor, record["merge_prev_lse"])
        grad_prev_out, grad_prev_lse, grad_stage_out, grad_stage_lse = (
            _stage_merge_backward_values(
                prev_out=prev_out,
                prev_lse=prev_lse,
                stage_out=stage_out.detach().to(accum_dtype),
                stage_lse=stage_lse.detach().to(accum_dtype),
                grad_merged_out=grad_merged_out,
                grad_merged_lse=grad_merged_lse,
            )
        )
        stage_out_grads[record_index] = grad_stage_out.to(dtype=stage_out.dtype)
        stage_lse_grads[record_index] = grad_stage_lse.to(dtype=stage_lse.dtype)
        if bool(record.get("merge_q_is_full", False)):
            grad_accum_out = grad_prev_out
            grad_accum_lse = grad_prev_lse
            continue
        if q_index is None:
            raise RuntimeError("Missing stage q index for partial merge tape")
        grad_accum_out.index_copy_(1, q_index, grad_prev_out)
        grad_accum_lse.index_copy_(1, q_index, grad_prev_lse)
        _release_replay_record_merge_tape(record)
    return stage_out_grads, stage_lse_grads


def _forward_stage_records(
    *,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    state: ArtContextParallelState,
    kernel: FlexAttentionKernel,
    scale: float,
    enable_gqa: bool,
    sliding_window: int | None,
    record_for_backward: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    q_source = q_flat.detach() if record_for_backward else q_flat
    k_source = k_flat.detach() if record_for_backward else k_flat
    v_source = v_flat.detach() if record_for_backward else v_flat

    accum_dtype = _accum_output_dtype(q_flat.dtype)
    accum_out: torch.Tensor | None = None
    accum_lse: torch.Tensor | None = None

    ordered_stages = _ordered_stage_plans(state.rank_plan.stage_plans)
    local_stage = next(
        (stage for stage in ordered_stages if stage.is_local_stage), None
    )
    remote_stages = [stage for stage in ordered_stages if not stage.is_local_stage]
    wave_pipeline_enabled = _remote_comm_launch_enabled(
        state=state,
        remote_stages=remote_stages,
    )
    replay_records: list[dict[str, Any]] = []
    produced_output = False

    remote_fetch_works_by_stage_index: dict[int, Any] = {}
    remote_query_works_by_stage_index: dict[int, _StageQueryGatherWork] = {}
    if wave_pipeline_enabled:
        for stage_plan in remote_stages:
            remote_fetch_works_by_stage_index[int(stage_plan.stage_index)] = (
                _COMMUNICATOR.launch_kv_fetch(
                    k_local=k_source,
                    v_local=v_source,
                    plan=cast(Any, stage_plan.kv_fetch_plan),
                    group=state.cp_group,
                    async_op=True,
                    range_meta_cache=state.execution_cache.range_meta,
                    label=(
                        f"kv_fetch.wave{stage_plan.wave_index}."
                        f"stage{stage_plan.stage_index}.src{stage_plan.source_rank}"
                    ),
                    input_layout="head_major",
                    output_layout="head_major",
                )
            )
            query_work = _launch_stage_query_gather(
                q_flat=q_source,
                state=state,
                stage_plan=stage_plan,
            )
            if query_work is not None:
                remote_query_works_by_stage_index[int(stage_plan.stage_index)] = (
                    query_work
                )
    pending_remote_stages = [
        stage_plan
        for stage_plan in remote_stages
        if stage_plan.q_len > 0 and stage_plan.k_len > 0 and stage_plan.slices
    ]

    if (
        local_stage is not None
        and local_stage.q_len > 0
        and local_stage.k_len > 0
        and local_stage.slices
    ):
        local_q_is_full = _stage_q_is_full(
            stage_plan=local_stage,
            q_flat_len=int(q_flat.shape[1]),
        )
        q_stage = _stage_query_tensor(
            q_flat=q_source,
            state=state,
            stage_plan=local_stage,
        )
        k_stage, v_stage = _stage_local_kv_tensors(
            k_flat=k_source,
            v_flat=v_source,
            state=state,
            stage_plan=local_stage,
        )
        if record_for_backward:
            q_leaf = q_stage.detach().requires_grad_(bool(q_flat.requires_grad))
            k_leaf = k_stage.detach().requires_grad_(bool(k_flat.requires_grad))
            v_leaf = v_stage.detach().requires_grad_(bool(v_flat.requires_grad))
        else:
            q_leaf = k_leaf = v_leaf = None
        if record_for_backward:
            stage_out, stage_lse = _run_stage_attention(
                q_stage=cast(torch.Tensor, q_leaf),
                k_stage=cast(torch.Tensor, k_leaf),
                v_stage=cast(torch.Tensor, v_leaf),
                stage_plan=local_stage,
                state=state,
                kernel=kernel,
                scale=scale,
                enable_gqa=enable_gqa,
                sliding_window=sliding_window,
            )
            replay_records.append(
                {
                    "stage_plan": local_stage,
                    "q_input": q_leaf,
                    "k_input": k_leaf,
                    "v_input": v_leaf,
                    "stage_out": stage_out,
                    "stage_lse": stage_lse,
                }
            )
        else:
            stage_out, stage_lse = _run_stage_attention(
                q_stage=q_stage,
                k_stage=k_stage,
                v_stage=v_stage,
                stage_plan=local_stage,
                state=state,
                kernel=kernel,
                scale=scale,
                enable_gqa=enable_gqa,
                sliding_window=sliding_window,
            )
        stage_out_value = stage_out.detach() if record_for_backward else stage_out
        stage_lse_value = stage_lse.detach() if record_for_backward else stage_lse
        if record_for_backward:
            replay_records[-1].update(
                _capture_stage_merge_tape(
                    accum_out=accum_out,
                    accum_lse=accum_lse,
                    q_flat_len=int(q_flat.shape[1]),
                    device=q_flat.device,
                    state=state,
                    stage_plan=local_stage,
                    produced_output=produced_output,
                )
            )
        if not produced_output and local_q_is_full:
            accum_out, accum_lse = _seed_stage_accumulators(
                stage_out=stage_out_value,
                stage_lse=stage_lse_value,
                target_dtype=accum_dtype,
                needs_owned_storage=bool(record_for_backward and pending_remote_stages),
            )
            produced_output = True
        else:
            if not produced_output:
                accum_out, accum_lse = _allocate_stage_accumulators(
                    q_flat=q_flat,
                    out_dtype=stage_out_value.dtype,
                    lse_dtype=stage_lse_value.dtype,
                )
            else:
                if accum_out is None or accum_lse is None:
                    raise RuntimeError("Missing accumulators before merge")
                accum_out, accum_lse = _maybe_promote_accumulators(
                    accum_out=accum_out,
                    accum_lse=accum_lse,
                    target_dtype=accum_dtype,
                )
            if accum_out is None or accum_lse is None:
                raise RuntimeError("Missing accumulators for merge")
            accum_out, accum_lse = _merge_stage_output(
                accum_out=accum_out,
                accum_lse=accum_lse,
                stage_out=stage_out_value,
                stage_lse=stage_lse_value,
                state=state,
                stage_plan=local_stage,
                q_is_full=local_q_is_full,
                produced_output=produced_output,
            )
            produced_output = True

    while pending_remote_stages:
        ready_stages = _ready_remote_stage_batch(
            pending_stages=pending_remote_stages,
            fetch_works_by_stage_index=remote_fetch_works_by_stage_index,
        )
        if not ready_stages:
            raise RuntimeError(
                "Remote stage scheduler failed to produce a ready stage batch"
            )
        ready_stage_indices = {
            int(stage_plan.stage_index) for stage_plan in ready_stages
        }
        pending_remote_stages = [
            stage_plan
            for stage_plan in pending_remote_stages
            if int(stage_plan.stage_index) not in ready_stage_indices
        ]
        for ready_index, stage_plan in enumerate(ready_stages):
            stage_q_is_full = _stage_q_is_full(
                stage_plan=stage_plan,
                q_flat_len=int(q_flat.shape[1]),
            )
            stage_index = int(stage_plan.stage_index)
            query_work = remote_query_works_by_stage_index.get(stage_index)
            if query_work is None:
                q_stage = _stage_query_tensor(
                    q_flat=q_source,
                    state=state,
                    stage_plan=stage_plan,
                )
            else:
                q_stage = query_work.wait_post_process()
                remote_query_works_by_stage_index.pop(stage_index, None)
            fetch_work = remote_fetch_works_by_stage_index.get(stage_index)
            k_stage, v_stage, _kv_head_major = _stage_remote_kv_tensors(
                stage_plan=stage_plan,
                fetch_work=fetch_work,
            )
            remote_fetch_works_by_stage_index.pop(stage_index, None)
            if record_for_backward:
                q_leaf = q_stage.detach().requires_grad_(bool(q_flat.requires_grad))
                k_leaf = k_stage.detach().requires_grad_(bool(k_flat.requires_grad))
                v_leaf = v_stage.detach().requires_grad_(bool(v_flat.requires_grad))
            else:
                q_leaf = k_leaf = v_leaf = None
            del query_work, fetch_work
            if record_for_backward:
                stage_out, stage_lse = _run_stage_attention(
                    q_stage=cast(torch.Tensor, q_leaf),
                    k_stage=cast(torch.Tensor, k_leaf),
                    v_stage=cast(torch.Tensor, v_leaf),
                    stage_plan=stage_plan,
                    state=state,
                    kernel=kernel,
                    scale=scale,
                    enable_gqa=enable_gqa,
                    sliding_window=sliding_window,
                )
                replay_records.append(
                    {
                        "stage_plan": stage_plan,
                        "q_input": q_leaf,
                        "k_input": k_leaf,
                        "v_input": v_leaf,
                        "stage_out": stage_out,
                        "stage_lse": stage_lse,
                    }
                )
            else:
                stage_out, stage_lse = _run_stage_attention(
                    q_stage=q_stage,
                    k_stage=k_stage,
                    v_stage=v_stage,
                    stage_plan=stage_plan,
                    state=state,
                    kernel=kernel,
                    scale=scale,
                    enable_gqa=enable_gqa,
                    sliding_window=sliding_window,
                )
            stage_out_value = stage_out.detach() if record_for_backward else stage_out
            stage_lse_value = stage_lse.detach() if record_for_backward else stage_lse
            del q_stage, k_stage, v_stage
            if produced_output:
                if accum_out is None or accum_lse is None:
                    raise RuntimeError("Missing accumulators before remote merge")
                accum_out, accum_lse = _maybe_promote_accumulators(
                    accum_out=accum_out,
                    accum_lse=accum_lse,
                    target_dtype=accum_dtype,
                )
            if record_for_backward:
                replay_records[-1].update(
                    _capture_stage_merge_tape(
                        accum_out=accum_out,
                        accum_lse=accum_lse,
                        q_flat_len=int(q_flat.shape[1]),
                        device=q_flat.device,
                        state=state,
                        stage_plan=stage_plan,
                        produced_output=produced_output,
                    )
                )
            if not produced_output and stage_q_is_full:
                accum_out, accum_lse = _seed_stage_accumulators(
                    stage_out=stage_out_value,
                    stage_lse=stage_lse_value,
                    target_dtype=accum_dtype,
                    needs_owned_storage=bool(
                        record_for_backward
                        and (
                            pending_remote_stages or ready_index + 1 < len(ready_stages)
                        )
                    ),
                )
                produced_output = True
                continue
            if not produced_output:
                accum_out, accum_lse = _allocate_stage_accumulators(
                    q_flat=q_flat,
                    out_dtype=stage_out_value.dtype,
                    lse_dtype=stage_lse_value.dtype,
                )
            if accum_out is None or accum_lse is None:
                raise RuntimeError("Missing accumulators for remote merge")
            accum_out, accum_lse = _merge_stage_output(
                accum_out=accum_out,
                accum_lse=accum_lse,
                stage_out=stage_out_value,
                stage_lse=stage_lse_value,
                state=state,
                stage_plan=stage_plan,
                q_is_full=stage_q_is_full,
                produced_output=produced_output,
            )
            produced_output = True

    _drain_launched_remote_fetch_works(
        fetch_works_by_stage_index=remote_fetch_works_by_stage_index
    )

    if not produced_output:
        if int(q_flat.shape[1]) == 0:
            return (
                q_flat.new_empty((q_flat.shape[0], 0, q_flat.shape[2])),
                q_flat.new_empty((q_flat.shape[0], 0)),
                replay_records,
            )
        raise RuntimeError("Sparse attention produced no stage outputs")
    if accum_out is None or accum_lse is None:
        raise RuntimeError("Sparse attention produced no accumulated output")
    return accum_out, accum_lse, replay_records


def _flatten_qkv(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: ArtContextParallelState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        flatten_valid_sequence_head_major(query, state.rank_plan.local_valid_lengths),
        flatten_valid_sequence_head_major(key, state.rank_plan.local_valid_lengths),
        flatten_valid_sequence_head_major(value, state.rank_plan.local_valid_lengths),
    )


def _run_context_parallel_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: ArtContextParallelState,
    scale: float,
    enable_gqa: bool,
    compile_enabled: bool,
    sliding_window: int | None,
    triton_num_stages_2_head_dims: tuple[int, ...],
    softmax_offset: torch.Tensor | None,
) -> torch.Tensor:
    kernel = FlexAttentionKernel(
        compile_enabled=compile_enabled,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
    )
    q_flat, k_flat, v_flat = _flatten_qkv(
        query=query,
        key=key,
        value=value,
        state=state,
    )
    accum_out, accum_lse, _ = _forward_stage_records(
        q_flat=q_flat,
        k_flat=k_flat,
        v_flat=v_flat,
        state=state,
        kernel=kernel,
        scale=scale,
        enable_gqa=enable_gqa,
        sliding_window=sliding_window,
        record_for_backward=False,
    )
    accum_out = _apply_softmax_offset(accum_out, accum_lse, softmax_offset)
    return unflatten_valid_sequence_head_major(
        accum_out.to(dtype=query.dtype),
        valid_lengths=state.rank_plan.local_valid_lengths,
        seq_len=query.shape[0],
    )


def _run_context_parallel_forward_recorded(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: ArtContextParallelState,
    scale: float,
    enable_gqa: bool,
    compile_enabled: bool,
    sliding_window: int | None,
    triton_num_stages_2_head_dims: tuple[int, ...],
    softmax_offset: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    kernel = FlexAttentionKernel(
        compile_enabled=compile_enabled,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
    )
    q_flat, k_flat, v_flat = _flatten_qkv(
        query=query,
        key=key,
        value=value,
        state=state,
    )
    accum_out, accum_lse, replay_records = _forward_stage_records(
        q_flat=q_flat,
        k_flat=k_flat,
        v_flat=v_flat,
        state=state,
        kernel=kernel,
        scale=scale,
        enable_gqa=enable_gqa,
        sliding_window=sliding_window,
        record_for_backward=True,
    )
    output_accum = _apply_softmax_offset(accum_out, accum_lse, softmax_offset)
    return (
        unflatten_valid_sequence_head_major(
            output_accum.to(dtype=query.dtype),
            valid_lengths=state.rank_plan.local_valid_lengths,
            seq_len=query.shape[0],
        ),
        accum_out,
        accum_lse,
        replay_records,
    )


def _scatter_stage_grad(
    *,
    target: torch.Tensor,
    grad: torch.Tensor | None,
    ranges: tuple[TokenRange, ...],
    state: ArtContextParallelState | None = None,
    head_major: bool = False,
) -> None:
    if grad is None or grad.numel() == 0:
        return
    grad = grad.contiguous()
    if grad.dtype != target.dtype:
        grad = grad.to(dtype=target.dtype)
    full_length = _ranges_cover_full_length(
        ranges,
        length=int(target.shape[1] if head_major else target.shape[0]),
    )
    if full_length:
        target.add_(grad)
        return
    if head_major:
        range_reduce_sum_head_major_(
            grad,
            output_tensor=target,
            ranges=ranges,
            range_meta_cache=(
                None if state is None else state.execution_cache.range_meta
            ),
        )
        return
    range_reduce_sum_(
        grad,
        output_tensor=target,
        ranges=ranges,
        range_meta_cache=(None if state is None else state.execution_cache.range_meta),
    )


def _sanitize_nested_stage_input_grad(
    grad: torch.Tensor | None,
) -> torch.Tensor | None:
    if grad is None:
        return None
    # Nested autograd.grad can hand back view-backed stage input grads tied to
    # raw compiled flex backward storage. Clone away from that base lineage and
    # synchronize before first downstream use.
    cloned = grad.detach().clone()
    if cloned.is_cuda:
        torch.cuda.current_stream(device=cloned.device).synchronize()
    return cloned


def _zero_stage_grads_like(
    tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros_like(tensor)


def _run_context_parallel_backward(
    *,
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: ArtContextParallelState,
    scale: float,
    enable_gqa: bool,
    compile_enabled: bool,
    sliding_window: int | None,
    triton_num_stages_2_head_dims: tuple[int, ...],
    softmax_offset: torch.Tensor | None,
    replay_records: list[dict[str, Any]] | None = None,
    replay_accum_out: torch.Tensor | None = None,
    replay_accum_lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    kernel = FlexAttentionKernel(
        compile_enabled=compile_enabled,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
    )
    comm_async_enabled = _distributed_cp_comm_enabled(state)
    q_flat, k_flat, v_flat = _flatten_qkv(
        query=query,
        key=key,
        value=value,
        state=state,
    )
    grad_output_flat = flatten_valid_sequence_head_major(
        grad_output,
        state.rank_plan.local_valid_lengths,
    )
    if replay_records is None:
        replay_accum_out, replay_accum_lse, replay_records = _forward_stage_records(
            q_flat=q_flat,
            k_flat=k_flat,
            v_flat=v_flat,
            state=state,
            kernel=kernel,
            scale=scale,
            enable_gqa=enable_gqa,
            sliding_window=sliding_window,
            record_for_backward=True,
        )
    if softmax_offset is None:
        grad_accum_out = grad_output_flat
        grad_accum_lse = None
        grad_softmax_offset = None
    else:
        if replay_accum_out is None or replay_accum_lse is None:
            raise RuntimeError(
                "Missing context-parallel accumulators for softmax_offset backward."
            )
        grad_accum_out, grad_accum_lse, grad_softmax_offset = (
            _softmax_offset_backward_values(
                grad_output=grad_output_flat,
                accum_out=replay_accum_out,
                accum_lse=replay_accum_lse,
                softmax_offset=softmax_offset,
            )
        )
    stage_out_grads, stage_lse_grads = _merge_stage_output_grads_from_tape(
        replay_records=replay_records,
        grad_output_flat=grad_accum_out,
        grad_lse_flat=grad_accum_lse,
    )
    if stage_out_grads and stage_out_grads[0].is_cuda:
        # Nested FLASH flex backward consumes these external grad_outputs on an
        # internal stream; complete the merge-backward producers before the handoff.
        torch.cuda.current_stream(stage_out_grads[0].device).synchronize()
    grad_by_stage_index: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for record, stage_out_grad, stage_lse_grad in zip(
        replay_records,
        stage_out_grads,
        stage_lse_grads,
        strict=True,
    ):
        stage_plan = cast(StagePlan, record["stage_plan"])
        grad_by_stage_index[int(stage_plan.stage_index)] = (
            (
                _zero_stage_grads_like(record["stage_out"])
                if stage_out_grad is None
                else stage_out_grad
            ),
            (
                _zero_stage_grads_like(record["stage_lse"])
                if stage_lse_grad is None
                else stage_lse_grad
            ),
        )
    del stage_out_grads, stage_lse_grads

    grad_accum_dtype = q_flat.dtype
    dq_flat = torch.zeros(q_flat.shape, device=q_flat.device, dtype=grad_accum_dtype)
    dk_flat = torch.zeros(k_flat.shape, device=k_flat.device, dtype=grad_accum_dtype)
    dv_flat = torch.zeros(v_flat.shape, device=v_flat.device, dtype=grad_accum_dtype)
    reduce_works: list[Any] = []
    needs_remote_reduce = any(
        not stage_plan.is_local_stage for stage_plan in state.rank_plan.stage_plans
    )
    if needs_remote_reduce and not comm_async_enabled:
        raise RuntimeError(
            "ART context parallel backward remote reductions require distributed async per-stage "
            "dKV reduce."
        )
    if not any(
        (not stage_plan.is_local_stage) and _stage_requires_reduce(stage_plan)
        for stage_plan in state.rank_plan.stage_plans
    ) and (
        sum(state.rank_plan.remote_dkv_reduce_plan.send_splits) > 0
        or sum(state.rank_plan.remote_dkv_reduce_plan.recv_splits) > 0
    ):
        empty = k_flat.new_empty((k_flat.shape[0], 0, k_flat.shape[2]))
        reduce_works.append(
            _COMMUNICATOR.launch_dkv_reduce(
                dk_remote=empty,
                dv_remote=empty,
                plan=state.rank_plan.remote_dkv_reduce_plan,
                group=state.cp_group,
                async_op=comm_async_enabled,
                dk_local=dk_flat,
                dv_local=dv_flat,
                range_meta_cache=state.execution_cache.range_meta,
                label="dkv_reduce.global",
                input_layout="head_major",
            )
        )
    records_by_stage_index = {
        int(cast(StagePlan, record["stage_plan"]).stage_index): record
        for record in replay_records
    }

    for stage_index in state.rank_plan.backward_stage_indices:
        stage_plan = state.rank_plan.stage_plans[int(stage_index)]
        stage_record = records_by_stage_index.get(int(stage_plan.stage_index))
        if stage_record is None:
            if stage_plan.is_local_stage:
                continue
            empty = k_flat.new_empty((k_flat.shape[0], 0, k_flat.shape[2]))
            reduce_works.append(
                _COMMUNICATOR.launch_dkv_reduce(
                    dk_remote=empty,
                    dv_remote=empty,
                    plan=cast(DkvReducePlan, stage_plan.dkv_reduce_plan),
                    group=state.cp_group,
                    async_op=comm_async_enabled,
                    dk_local=dk_flat,
                    dv_local=dv_flat,
                    range_meta_cache=state.execution_cache.range_meta,
                    label=(
                        f"dkv_reduce.stage{stage_plan.stage_index}."
                        f"src{stage_plan.source_rank}"
                    ),
                    input_layout="head_major",
                )
            )
            continue

        stage_out_grad, stage_lse_grad = grad_by_stage_index[
            int(stage_plan.stage_index)
        ]
        inputs: list[torch.Tensor] = []
        input_names: list[str] = []
        for name in ("q_input", "k_input", "v_input"):
            tensor = cast(torch.Tensor, stage_record[name])
            if tensor.requires_grad:
                inputs.append(tensor)
                input_names.append(name)
        if not inputs:
            grad_by_stage_index.pop(int(stage_plan.stage_index), None)
            _release_replay_record_tensors(stage_record)
            stage_record.clear()
            continue
        stage_outputs: list[torch.Tensor] = []
        stage_output_grads: list[torch.Tensor] = []
        stage_out_tensor = cast(torch.Tensor, stage_record["stage_out"])
        stage_lse_tensor = cast(torch.Tensor, stage_record["stage_lse"])
        if stage_out_tensor.requires_grad:
            stage_outputs.append(stage_out_tensor)
            stage_output_grads.append(stage_out_grad)
        if stage_lse_tensor.requires_grad:
            stage_outputs.append(stage_lse_tensor)
            stage_output_grads.append(stage_lse_grad)
        if not stage_outputs:
            grad_by_stage_index.pop(int(stage_plan.stage_index), None)
            _release_replay_record_tensors(stage_record)
            stage_record.clear()
            continue
        input_grads = torch.autograd.grad(
            outputs=tuple(stage_outputs),
            inputs=inputs,
            grad_outputs=tuple(stage_output_grads),
            allow_unused=True,
        )
        grad_map = {
            name: grad for name, grad in zip(input_names, input_grads, strict=True)
        }
        for grad_name in ("q_input", "k_input", "v_input"):
            grad_map[grad_name] = _sanitize_nested_stage_input_grad(
                cast(torch.Tensor | None, grad_map.get(grad_name)),
            )
        _scatter_stage_grad(
            target=dq_flat,
            grad=cast(torch.Tensor | None, grad_map.get("q_input")),
            ranges=stage_plan.owner_local_q_ranges,
            state=state,
            head_major=True,
        )
        if stage_plan.is_local_stage:
            _scatter_stage_grad(
                target=dk_flat,
                grad=cast(torch.Tensor | None, grad_map.get("k_input")),
                ranges=stage_plan.owner_local_k_ranges,
                state=state,
                head_major=True,
            )
            _scatter_stage_grad(
                target=dv_flat,
                grad=cast(torch.Tensor | None, grad_map.get("v_input")),
                ranges=stage_plan.owner_local_k_ranges,
                state=state,
                head_major=True,
            )
            grad_by_stage_index.pop(int(stage_plan.stage_index), None)
            _release_replay_record_tensors(stage_record)
            stage_record.clear()
            continue
        if not stage_plan.is_local_stage:
            dk_remote = cast(torch.Tensor | None, grad_map.get("k_input"))
            dv_remote = cast(torch.Tensor | None, grad_map.get("v_input"))
            if dk_remote is None:
                dk_remote = k_flat.new_empty((k_flat.shape[0], 0, k_flat.shape[2]))
            if dv_remote is None:
                dv_remote = v_flat.new_empty((v_flat.shape[0], 0, v_flat.shape[2]))
            reduce_works.append(
                _COMMUNICATOR.launch_dkv_reduce(
                    dk_remote=dk_remote.contiguous(),
                    dv_remote=dv_remote.contiguous(),
                    plan=cast(DkvReducePlan, stage_plan.dkv_reduce_plan),
                    group=state.cp_group,
                    async_op=comm_async_enabled,
                    dk_local=dk_flat,
                    dv_local=dv_flat,
                    range_meta_cache=state.execution_cache.range_meta,
                    label=(
                        f"dkv_reduce.stage{stage_plan.stage_index}."
                        f"src{stage_plan.source_rank}"
                    ),
                    input_layout="head_major",
                )
            )
        grad_by_stage_index.pop(int(stage_plan.stage_index), None)
        _release_replay_record_tensors(stage_record)
        stage_record.clear()

    for work in reduce_works:
        work.wait_post_process()
    records_by_stage_index.clear()
    replay_records.clear()

    return (
        unflatten_valid_sequence_head_major(
            dq_flat.to(dtype=query.dtype),
            valid_lengths=state.rank_plan.local_valid_lengths,
            seq_len=query.shape[0],
        ),
        unflatten_valid_sequence_head_major(
            dk_flat.to(dtype=key.dtype),
            valid_lengths=state.rank_plan.local_valid_lengths,
            seq_len=key.shape[0],
        ),
        unflatten_valid_sequence_head_major(
            dv_flat.to(dtype=value.dtype),
            valid_lengths=state.rank_plan.local_valid_lengths,
            seq_len=value.shape[0],
        ),
        grad_softmax_offset,
    )


class ArtContextParallelFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_offset: torch.Tensor | None,
        state: ArtContextParallelState,
        scale: float,
        enable_gqa: bool,
        compile_enabled: bool,
        sliding_window: int | None,
        triton_num_stages_2_head_dims: tuple[int, ...],
    ) -> torch.Tensor:
        ctx.state = state
        ctx.scale = float(scale)
        ctx.enable_gqa = bool(enable_gqa)
        ctx.compile_enabled = bool(compile_enabled)
        ctx.sliding_window = sliding_window
        ctx.triton_num_stages_2_head_dims = tuple(
            int(dim) for dim in triton_num_stages_2_head_dims
        )
        tensors_to_save = [query, key, value]
        if softmax_offset is not None:
            tensors_to_save.append(softmax_offset)
        ctx.has_softmax_offset = softmax_offset is not None
        with torch.enable_grad():
            query_record = query.detach().requires_grad_(bool(query.requires_grad))
            key_record = key.detach().requires_grad_(bool(key.requires_grad))
            value_record = value.detach().requires_grad_(bool(value.requires_grad))
            output, replay_accum_out, replay_accum_lse, replay_records = (
                _run_context_parallel_forward_recorded(
                    query=query_record,
                    key=key_record,
                    value=value_record,
                    state=state,
                    scale=float(scale),
                    enable_gqa=bool(enable_gqa),
                    compile_enabled=bool(compile_enabled),
                    sliding_window=sliding_window,
                    triton_num_stages_2_head_dims=ctx.triton_num_stages_2_head_dims,
                    softmax_offset=softmax_offset,
                )
            )
        if softmax_offset is not None:
            tensors_to_save.extend((replay_accum_out, replay_accum_lse))
        ctx.save_for_backward(*tensors_to_save)
        ctx.replay_records = replay_records
        return output.detach()

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        (grad_output,) = cast(tuple[torch.Tensor], grad_outputs)
        saved_tensors = ctx.saved_tensors
        query, key, value = saved_tensors[:3]
        if bool(ctx.has_softmax_offset):
            softmax_offset = saved_tensors[3]
            replay_accum_out, replay_accum_lse = saved_tensors[4:6]
        else:
            softmax_offset = None
            replay_accum_out = None
            replay_accum_lse = None
        try:
            dq, dk, dv, grad_softmax_offset = _run_context_parallel_backward(
                grad_output=grad_output,
                query=query,
                key=key,
                value=value,
                state=ctx.state,
                scale=ctx.scale,
                enable_gqa=ctx.enable_gqa,
                compile_enabled=ctx.compile_enabled,
                sliding_window=ctx.sliding_window,
                triton_num_stages_2_head_dims=ctx.triton_num_stages_2_head_dims,
                softmax_offset=softmax_offset,
                replay_records=cast(list[dict[str, Any]], ctx.replay_records),
                replay_accum_out=replay_accum_out,
                replay_accum_lse=replay_accum_lse,
            )
        finally:
            ctx.replay_records = None
        return dq, dk, dv, grad_softmax_offset, None, None, None, None, None, None


def run_context_parallel(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: ArtContextParallelState,
    scale: float,
    enable_gqa: bool,
    compile_enabled: bool,
    sliding_window: int | None = None,
    triton_num_stages_2_head_dims: tuple[int, ...] = (),
    softmax_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    if torch.is_grad_enabled() and (
        query.requires_grad
        or key.requires_grad
        or value.requires_grad
        or (softmax_offset is not None and softmax_offset.requires_grad)
    ):
        return ArtContextParallelFn.apply(
            query,
            key,
            value,
            softmax_offset,
            state,
            float(scale),
            bool(enable_gqa),
            bool(compile_enabled),
            None if sliding_window is None else int(sliding_window),
            tuple(int(dim) for dim in triton_num_stages_2_head_dims),
        )
    return _run_context_parallel_forward(
        query=query,
        key=key,
        value=value,
        state=state,
        scale=scale,
        enable_gqa=enable_gqa,
        compile_enabled=compile_enabled,
        sliding_window=sliding_window,
        triton_num_stages_2_head_dims=tuple(
            int(dim) for dim in triton_num_stages_2_head_dims
        ),
        softmax_offset=softmax_offset,
    )
