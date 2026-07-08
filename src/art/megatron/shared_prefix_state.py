"""Shared-prefix packed-sequence state for ART attention and GDN integration."""

from __future__ import annotations

import gc
from typing import Any

from pydantic import Field
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.context_parallel.block_mask import build_block_mask
from art.megatron.context_parallel.builder import build_shared_prefix_attention_spec
from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.context_parallel.types import (
    AttnMaskKind,
    AttnSlice,
    ExactMaskMetadata,
    FlexMaskSpec,
    TokenRange,
)
from art.megatron.flex_attn.attention import (
    SharedPrefixAttentionState as FlexSharedPrefixAttentionState,
)
from art.megatron.flex_attn.compiled import flash_sparse_block_size_for_head_dim
from art.megatron.gdn.gdn_shared_prefix import (
    GdnPackedExecutionSpec,
    GdnRankExecutionPlan,
    build_gdn_rank_execution_plan,
    move_gdn_rank_execution_plan_to_device,
    parse_gdn_shared_prefix_segments,
)
from art.megatron.model_support.spec import SharedPrefixModelStateContext


class SharedPrefixAttentionState(FlexSharedPrefixAttentionState):
    """Shared-prefix sparsity and optional GDN execution metadata."""

    group_ids: Tensor
    parent_ids: Tensor
    model_state: dict[str, Any] = Field(default_factory=dict)
    gdn_execution_spec: GdnPackedExecutionSpec | None = None
    gdn_execution_plan: GdnRankExecutionPlan | None = None
    gdn_hidden_layout: str = "attention"
    gdn_input_layout: str | None = None
    gdn_output_layout: str | None = None
    gdn_attention_original_shape: tuple[int, int, int] | None = None
    gdn_attention_original_shapes: dict[int, tuple[int, int, int]] = Field(
        default_factory=dict
    )
    gdn_attention_token_uids: Tensor | None = None
    gdn_active_module: Any | None = None


def create_shared_prefix_state(
    group_ids: Tensor,
    parent_ids: Tensor,
    *,
    target_device: torch.device | None = None,
    input_pos: Tensor | None = None,
    sliding_windows: tuple[int, ...] = (),
    build_gdn_execution_spec: bool = False,
    model_support_handler: Any | None = None,
    attention_token_layout_index: TokenLayoutIndex | None = None,
    attention_head_dim: int | None = None,
    attention_value_head_dim: int | None = None,
) -> SharedPrefixAttentionState:
    """Build shared-prefix attention mask state plus optional reusable GDN plan."""
    device = group_ids.device if target_device is None else torch.device(target_device)
    group_ids_cpu = _metadata_cpu(group_ids)
    parent_ids_cpu = _metadata_cpu(parent_ids)
    input_pos_cpu = _metadata_cpu(input_pos) if input_pos is not None else None
    block_size = _shared_prefix_block_size(
        device,
        attention_head_dim=attention_head_dim,
        attention_value_head_dim=attention_value_head_dim,
    )
    block_mask = _build_sparse_shared_prefix_block_mask(
        group_ids_cpu=group_ids_cpu,
        parent_ids_cpu=parent_ids_cpu,
        input_pos_cpu=input_pos_cpu,
        sliding_window=None,
        device=device,
        block_size=block_size,
    )
    sliding_block_masks = {
        window: _build_sparse_shared_prefix_block_mask(
            group_ids_cpu=group_ids_cpu,
            parent_ids_cpu=parent_ids_cpu,
            input_pos_cpu=input_pos_cpu,
            sliding_window=window,
            device=device,
            block_size=block_size,
        )
        for window in tuple(dict.fromkeys(int(window) for window in sliding_windows))
    }
    cp_rank, cp_size, cp_group = _gdn_cp_rank_size_group()
    gdn_execution_spec = _build_gdn_execution_spec_once(
        group_ids_cpu,
        parent_ids_cpu,
        build=build_gdn_execution_spec,
        cp_rank=cp_rank,
        cp_size=cp_size,
        cp_group=cp_group,
    )
    return SharedPrefixAttentionState(
        block_mask=block_mask,
        sliding_block_masks=sliding_block_masks,
        group_ids=group_ids_cpu,
        parent_ids=parent_ids_cpu,
        model_state=_build_model_state_once(
            model_support_handler,
            input_pos=input_pos_cpu,
            group_ids=group_ids_cpu,
            parent_ids=parent_ids_cpu,
            device=device,
            attention_token_layout_index=attention_token_layout_index,
            attention_head_dim=attention_head_dim,
            attention_value_head_dim=attention_value_head_dim,
        ),
        gdn_execution_spec=gdn_execution_spec,
        gdn_execution_plan=_build_gdn_execution_plan_once(
            gdn_execution_spec,
            device=device,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_group=cp_group,
            attention_token_layout_index=attention_token_layout_index,
        ),
    )


def _build_model_state_once(
    model_support_handler: Any | None,
    *,
    input_pos: Tensor | None,
    group_ids: Tensor,
    parent_ids: Tensor,
    device: torch.device,
    attention_token_layout_index: TokenLayoutIndex | None,
    attention_head_dim: int | None,
    attention_value_head_dim: int | None,
) -> dict[str, Any]:
    if model_support_handler is None:
        return {}
    return dict(
        model_support_handler.build_shared_prefix_model_state(
            SharedPrefixModelStateContext(
                input_pos=input_pos,
                group_ids=group_ids,
                parent_ids=parent_ids,
                device=device,
                attention_token_layout_index=attention_token_layout_index,
                attention_head_dim=attention_head_dim,
                attention_value_head_dim=attention_value_head_dim,
            )
        )
    )


def _metadata_cpu(tensor: Tensor) -> Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu" or tensor.dtype != torch.int64:
        tensor = tensor.to(device="cpu", dtype=torch.int64)
    return tensor.contiguous()


def _build_sparse_shared_prefix_block_mask(
    *,
    group_ids_cpu: Tensor,
    parent_ids_cpu: Tensor,
    input_pos_cpu: Tensor | None,
    sliding_window: int | None,
    device: torch.device,
    block_size: tuple[int, int],
):
    batch_spec = build_shared_prefix_attention_spec(
        group_ids=group_ids_cpu,
        parent_ids=parent_ids_cpu,
    )
    row_spec = batch_spec.rows[0]
    seq_len = int(group_ids_cpu.shape[1])
    slices = _full_row_slices_with_padding(
        row_slices=row_spec.slices,
        valid_tokens=int(row_spec.valid_tokens),
        seq_len=seq_len,
    )
    if not slices:
        return _empty_block_mask(seq_len=seq_len, block_size=block_size, device=device)
    return build_block_mask(
        FlexMaskSpec(
            q_len=seq_len,
            k_len=seq_len,
            block_size=block_size,
            slices=slices,
            exact_mask=ExactMaskMetadata(
                q_token_indices=torch.arange(seq_len, dtype=torch.int64),
                k_token_indices=torch.arange(seq_len, dtype=torch.int64),
                cache_key=(
                    f"identity:{seq_len}"
                    if sliding_window is None
                    else f"identity:{seq_len}:sliding:{int(sliding_window)}"
                ),
            ),
        ),
        group_ids=group_ids_cpu[0],
        parent_ids=parent_ids_cpu[0],
        input_pos=None if input_pos_cpu is None else input_pos_cpu[0],
        sliding_window=sliding_window,
        device=device,
    )


def _full_row_slices_with_padding(
    *,
    row_slices: tuple[AttnSlice, ...],
    valid_tokens: int,
    seq_len: int,
) -> tuple[AttnSlice, ...]:
    if valid_tokens >= seq_len:
        return row_slices
    padding_range = TokenRange(start=int(valid_tokens), end=int(seq_len))
    if padding_range.is_empty():
        return row_slices
    return (
        *row_slices,
        AttnSlice(
            q_range=padding_range,
            k_range=padding_range,
            mask_kind=AttnMaskKind.CAUSAL,
            row_index=0,
            family_index=None,
        ),
    )


def _empty_block_mask(
    *,
    seq_len: int,
    block_size: tuple[int, int],
    device: torch.device,
) -> BlockMask:
    q_block, k_block = block_size
    q_blocks = (int(seq_len) + q_block - 1) // q_block
    k_blocks = max((int(seq_len) + k_block - 1) // k_block, 1)
    kv_num_blocks = torch.zeros((1, 1, q_blocks), dtype=torch.int32, device=device)
    kv_indices = torch.zeros(
        (1, 1, q_blocks, k_blocks),
        dtype=torch.int32,
        device=device,
    )
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=_false_mask,
        seq_lengths=(int(seq_len), int(seq_len)),
    )


def _false_mask(
    batch_idx: Tensor,
    head_idx: Tensor,
    query_idx: Tensor,
    kv_idx: Tensor,
) -> Tensor:
    del batch_idx, head_idx, kv_idx
    return torch.zeros_like(query_idx, dtype=torch.bool)


def _shared_prefix_block_size(
    device: torch.device,
    *,
    attention_head_dim: int | None,
    attention_value_head_dim: int | None,
) -> tuple[int, int]:
    if attention_head_dim is None:
        return (128, 128)
    return flash_sparse_block_size_for_head_dim(
        head_dim=int(attention_head_dim),
        head_dim_v=int(
            attention_head_dim
            if attention_value_head_dim is None
            else attention_value_head_dim
        ),
        device=device,
    )


def _build_gdn_execution_spec_once(
    group_ids: Tensor,
    parent_ids: Tensor,
    *,
    build: bool,
    cp_rank: int,
    cp_size: int,
    cp_group: Any | None,
) -> GdnPackedExecutionSpec | None:
    if not build:
        return None
    if cp_size == 1:
        return parse_gdn_shared_prefix_segments(
            group_ids, parent_ids, min_completions_per_family=0
        )
    if (
        not torch.distributed.is_available() or not torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
    ):
        return parse_gdn_shared_prefix_segments(
            group_ids, parent_ids, min_completions_per_family=0
        )
    return parse_gdn_shared_prefix_segments(
        group_ids, parent_ids, min_completions_per_family=0
    )


def _build_gdn_execution_plan_once(
    spec: GdnPackedExecutionSpec | None,
    *,
    device: torch.device,
    cp_rank: int,
    cp_size: int,
    cp_group: Any | None,
    attention_token_layout_index: TokenLayoutIndex | None,
) -> GdnRankExecutionPlan | None:
    if spec is None:
        return None
    planner_device = torch.device("cpu") if device.type == "cuda" else device
    del cp_group
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        plan = build_gdn_rank_execution_plan(
            spec,
            device=planner_device,
            cp_rank=cp_rank,
            cp_size=cp_size,
            attention_token_layout_index=attention_token_layout_index,
        )
    finally:
        if gc_was_enabled:
            gc.enable()
    return move_gdn_rank_execution_plan_to_device(plan, device)


def _gdn_cp_rank_size_group() -> tuple[int, int, Any | None]:
    try:
        from megatron.core import parallel_state as ps

        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            return (
                int(ps.get_context_parallel_rank()),
                int(ps.get_context_parallel_world_size()),
                ps.get_context_parallel_group(),
            )
    except Exception:
        pass
    return 0, 1, None
