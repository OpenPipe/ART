"""Compiled flex attention entrypoints."""

import math
from typing import Any, Literal, TypeAlias, cast

import torch
from torch.nn.attention.flex_attention import (
    AuxRequest,
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


def flex_backend_for_head_dims(*, head_dim: int, head_dim_v: int) -> FlexBackend:
    if _FORCED_FLEX_BACKEND != "FLASH":
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
    if flex_backend_for_head_dims(head_dim=head_dim, head_dim_v=head_dim_v) != "FLASH":
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


def _forced_flex_attention_sparse(
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
    target_q_len = (
        0 if int(q_len) <= 0 else ((int(q_len) + q_block - 1) // q_block) * q_block
    )
    target_k_len = (
        0 if int(k_len) <= 0 else ((int(k_len) + k_block - 1) // k_block) * k_block
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


def get_dense_compiled_flex_attention(
    *,
    backend: FlexBackend,
    head_dim: int,
    head_dim_v: int,
    triton_num_stages_2_head_dims: tuple[int, ...] = (),
) -> Any:
    if backend == _FORCED_FLEX_BACKEND:
        return dense_compiled_flex_attention
    if backend == "FLASH":
        return flash_dense_compiled_flex_attention
    if _needs_triton_num_stages_2(
        backend=backend,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
    ):
        return triton_num_stages_2_dense_compiled_flex_attention
    return triton_dense_compiled_flex_attention


def get_sparse_compiled_flex_attention(
    *,
    family_key: str,
    backend: FlexBackend,
    head_dim: int,
    head_dim_v: int,
    triton_num_stages_2_head_dims: tuple[int, ...] = (),
) -> Any:
    del family_key
    if backend == _FORCED_FLEX_BACKEND:
        return sparse_compiled_flex_attention
    if backend == "FLASH":
        return flash_sparse_compiled_flex_attention
    if _needs_triton_num_stages_2(
        backend=backend,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        triton_num_stages_2_head_dims=triton_num_stages_2_head_dims,
    ):
        return triton_num_stages_2_sparse_compiled_flex_attention
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

sparse_compiled_flex_attention = torch.compile(
    _forced_flex_attention_sparse,
)
flash_sparse_compiled_flex_attention = torch.compile(
    _flex_attention_with_options(_FLASH_FLEX_KERNEL_OPTIONS),
)
triton_sparse_compiled_flex_attention = torch.compile(
    _flex_attention_with_options(_TRITON_FLEX_KERNEL_OPTIONS),
)
triton_num_stages_2_sparse_compiled_flex_attention = torch.compile(
    _flex_attention_with_options(_TRITON_NUM_STAGES_2_FLEX_KERNEL_OPTIONS),
)
