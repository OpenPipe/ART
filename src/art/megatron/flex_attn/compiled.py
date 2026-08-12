"""Compiled flex attention entrypoints."""

import math
from typing import Any, Literal, TypeAlias, cast

import torch
from torch._higher_order_ops.flex_attention import (
    flex_attention as flex_attention_hop,
)
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)

from art.megatron.flex_attn.flash_dlse_patch import apply_flash_flex_dlse_patch

apply_flash_flex_dlse_patch()


# Integration tests patch this module in-process when they need a non-default
# backend; production ART uses FLASH except for unsupported wide-head cases.
_FORCED_FLEX_BACKEND = "FLASH"
_FLASH_LSE_RESCALE = math.log(2.0)
FlexBackend: TypeAlias = Literal["FLASH", "TRITON"]
SparseBlockSize: TypeAlias = int | tuple[int, int]


def flex_backend_for_head_dims(
    *,
    head_dim: int,
    head_dim_v: int,
    device: torch.device | None = None,
) -> FlexBackend:
    if _FORCED_FLEX_BACKEND != "FLASH":
        return "TRITON"
    if device is not None and device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(device)
        if major in {10, 11}:
            return "TRITON"
    if int(head_dim) > 256 or int(head_dim_v) > 256:
        return "TRITON"
    return "FLASH"


def normalize_flex_lse(
    lse: torch.Tensor,
    *,
    backend: FlexBackend | None = None,
) -> torch.Tensor:
    if (_FORCED_FLEX_BACKEND if backend is None else backend) != "FLASH":
        return lse
    return lse / _FLASH_LSE_RESCALE


_FLASH_FLEX_KERNEL_OPTIONS = cast(FlexKernelOptions, {"BACKEND": "FLASH"})
_TRITON_FLEX_KERNEL_OPTIONS = cast(FlexKernelOptions, {"BACKEND": "TRITON"})
_TRITON_NUM_STAGES_2_FLEX_KERNEL_OPTIONS = cast(
    FlexKernelOptions,
    {"BACKEND": "TRITON", "num_stages": 2},
)
_BLACKWELL_WIDE_HEAD_FLEX_KERNEL_OPTIONS = cast(
    FlexKernelOptions,
    {
        "BACKEND": "TRITON",
        "num_stages": 2,
        "BLOCK_M": 32,
        "BLOCK_M1": 32,
    },
)
_FORCED_FLEX_KERNEL_OPTIONS = cast(
    FlexKernelOptions,
    {"BACKEND": _FORCED_FLEX_BACKEND},
)


def normalize_sparse_block_size(block_size: SparseBlockSize) -> tuple[int, int]:
    if isinstance(block_size, tuple):
        if len(block_size) != 2:
            raise RuntimeError(f"Expected 2D sparse block size, got {block_size!r}")
        return int(block_size[0]), int(block_size[1])
    value = int(block_size)
    return value, value


def flash_sparse_block_size_for_head_dim(
    *,
    head_dim: int,
    head_dim_v: int,
    device: torch.device,
) -> tuple[int, int]:
    if (
        flex_backend_for_head_dims(
            head_dim=head_dim, head_dim_v=head_dim_v, device=device
        )
        != "FLASH"
    ):
        return (128, 128)
    if device.type != "cuda":
        return (128, 128)
    major, _minor = torch.cuda.get_device_capability(device)
    if major != 9:
        return (128, 128)
    del head_dim_v
    if int(head_dim) <= 128:
        return (128, 128)
    if int(head_dim) <= 192:
        return (128, 96)
    return (128, 64)


def _forced_flex_attention_dense(
    q,
    k,
    v,
    *,
    block_mask,
    scale,
    enable_gqa,
    return_aux: AuxRequest | None = None,
):
    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        kernel_options=_FORCED_FLEX_KERNEL_OPTIONS,
        return_aux=return_aux,
    )


def _identity_score(score, _batch, _head, _query, _key):
    return score


def _sparse_flex_attention_with_options(kernel_options: FlexKernelOptions) -> Any:
    resolved_options = dict(kernel_options)
    resolved_options.setdefault("PRESCALE_QK", False)
    resolved_options.setdefault("ROWS_GUARANTEED_SAFE", False)
    resolved_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
    resolved_options.setdefault("WRITE_DQ", True)
    resolved_options["OUTPUT_LOGSUMEXP"] = True
    resolved_options["OUTPUT_MAX"] = False

    def _flex_attention(q, k, v, *, block_mask, scale, enable_gqa):
        del enable_gqa
        for tensor in (q, k, v):
            torch._dynamo.mark_static(tensor, -3)
            torch._dynamo.mark_static(tensor, -1)
        out, lse, _max_scores = flex_attention_hop(
            q,
            k,
            v,
            _identity_score,
            (q.shape[-2], k.shape[-2], *block_mask.as_tuple()[2:]),
            scale,
            resolved_options,
        )
        return out, lse * _FLASH_LSE_RESCALE

    return _flex_attention


def prepare_sparse_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: BlockMask,
    enable_gqa: bool,
) -> None:
    """Validate ART's sparse boundary and keep stage lengths symbolic."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError("Sparse flex attention requires rank-4 q, k, and v.")
    if int(q.shape[0]) != int(k.shape[0]) or k.shape[:-1] != v.shape[:-1]:
        raise RuntimeError("Sparse flex attention received incompatible q, k, and v.")
    if int(q.shape[-1]) != int(k.shape[-1]):
        raise RuntimeError("Sparse flex attention requires equal q/k head dimensions.")
    q_heads, kv_heads = int(q.shape[1]), int(k.shape[1])
    if (q_heads != kv_heads and not enable_gqa) or q_heads % kv_heads != 0:
        raise RuntimeError(
            f"Sparse flex attention received q_heads={q_heads}, kv_heads={kv_heads}, "
            f"enable_gqa={enable_gqa}."
        )
    if tuple(block_mask.shape[-2:]) != (int(q.shape[2]), int(k.shape[2])):
        raise RuntimeError("Sparse flex attention block-mask lengths do not match q/k.")
    for tensor in (q, k, v):
        torch._dynamo.mark_dynamic(tensor, 2)
    for value in block_mask.as_tuple()[2:]:
        if isinstance(value, torch.Tensor):
            for dim in range(2, value.ndim):
                torch._dynamo.mark_dynamic(value, dim)
    for cell in getattr(block_mask.mask_mod, "__closure__", None) or ():
        value = cell.cell_contents
        if isinstance(value, torch.Tensor):
            for dim in range(value.ndim):
                torch._dynamo.mark_dynamic(value, dim)


def _flex_attention_with_options(kernel_options: FlexKernelOptions) -> Any:
    def _flex_attention(
        q,
        k,
        v,
        *,
        block_mask,
        scale,
        enable_gqa,
        return_aux: AuxRequest | None = None,
    ):
        return flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
            return_aux=return_aux,
        )

    return _flex_attention


def select_sparse_execution_family(
    *,
    is_local_stage: bool,
    q_len: int,
    k_len: int,
    block_size: SparseBlockSize,
) -> tuple[int, int, str]:
    del is_local_stage
    q_block, k_block = normalize_sparse_block_size(block_size)
    # Avoid Flex's separate zero/one-block Dynamo specialization.
    target_q_len = (
        0
        if int(q_len) <= 0
        else max(2, (int(q_len) + q_block - 1) // q_block) * q_block
    )
    target_k_len = (
        0
        if int(k_len) <= 0
        else max(2, (int(k_len) + k_block - 1) // k_block) * k_block
    )
    return int(target_q_len), int(target_k_len), "sparse"


def _needs_triton_num_stages_2(
    *,
    backend: FlexBackend,
    head_dim: int,
    head_dim_v: int,
    triton_num_stages_2_head_dims: tuple[int, ...],
) -> bool:
    if backend != "TRITON":
        return False
    return (
        int(head_dim) in triton_num_stages_2_head_dims
        or int(head_dim_v) in triton_num_stages_2_head_dims
    )


def _needs_blackwell_wide_head_tile(
    *,
    backend: FlexBackend,
    head_dim: int,
    head_dim_v: int,
    triton_num_stages_2_head_dims: tuple[int, ...],
    device: torch.device | None,
) -> bool:
    if device is None or device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major in {10, 11} and _needs_triton_num_stages_2(
        backend=backend,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
    )


def get_dense_compiled_flex_attention(
    *,
    backend: FlexBackend,
    head_dim: int,
    head_dim_v: int,
    triton_num_stages_2_head_dims: tuple[int, ...] = (),
    device: torch.device | None = None,
) -> Any:
    if _needs_blackwell_wide_head_tile(
        backend=backend,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
        device=device,
    ):
        return blackwell_wide_head_dense_compiled_flex_attention
    if _needs_triton_num_stages_2(
        backend=backend,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
    ):
        return triton_num_stages_2_dense_compiled_flex_attention
    if backend == _FORCED_FLEX_BACKEND:
        return dense_compiled_flex_attention
    if backend == "FLASH":
        return flash_dense_compiled_flex_attention
    return triton_dense_compiled_flex_attention


def get_sparse_compiled_flex_attention(
    *,
    family_key: str,
    backend: FlexBackend,
    head_dim: int,
    head_dim_v: int,
    triton_num_stages_2_head_dims: tuple[int, ...] = (),
    device: torch.device | None = None,
) -> Any:
    del family_key
    if _needs_blackwell_wide_head_tile(
        backend=backend,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
        device=device,
    ):
        return blackwell_wide_head_sparse_compiled_flex_attention
    if _needs_triton_num_stages_2(
        backend=backend,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
    ):
        return triton_num_stages_2_sparse_compiled_flex_attention
    if backend == _FORCED_FLEX_BACKEND:
        return sparse_compiled_flex_attention
    if backend == "FLASH":
        return flash_sparse_compiled_flex_attention
    return triton_sparse_compiled_flex_attention


dense_compiled_flex_attention = torch.compile(
    _forced_flex_attention_dense,
)
flash_dense_compiled_flex_attention = torch.compile(
    _flex_attention_with_options(_FLASH_FLEX_KERNEL_OPTIONS),
)
triton_dense_compiled_flex_attention = torch.compile(
    _flex_attention_with_options(_TRITON_FLEX_KERNEL_OPTIONS),
)
triton_num_stages_2_dense_compiled_flex_attention = torch.compile(
    _flex_attention_with_options(_TRITON_NUM_STAGES_2_FLEX_KERNEL_OPTIONS),
)
blackwell_wide_head_dense_compiled_flex_attention = torch.compile(
    _flex_attention_with_options(_BLACKWELL_WIDE_HEAD_FLEX_KERNEL_OPTIONS),
)

sparse_compiled_flex_attention = torch.compile(
    _sparse_flex_attention_with_options(_FORCED_FLEX_KERNEL_OPTIONS),
)
flash_sparse_compiled_flex_attention = torch.compile(
    _sparse_flex_attention_with_options(_FLASH_FLEX_KERNEL_OPTIONS),
)
triton_sparse_compiled_flex_attention = torch.compile(
    _sparse_flex_attention_with_options(_TRITON_FLEX_KERNEL_OPTIONS),
)
triton_num_stages_2_sparse_compiled_flex_attention = torch.compile(
    _sparse_flex_attention_with_options(_TRITON_NUM_STAGES_2_FLEX_KERNEL_OPTIONS),
)
blackwell_wide_head_sparse_compiled_flex_attention = torch.compile(
    _sparse_flex_attention_with_options(_BLACKWELL_WIDE_HEAD_FLEX_KERNEL_OPTIONS),
)
