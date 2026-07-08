"""Prefix-tree packed-sequence state for ART attention and GDN integration."""

from __future__ import annotations

from dataclasses import replace
import gc
from typing import Any

from pydantic import Field
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.context_parallel.block_mask import (
    build_block_mask_from_context,
    prepare_block_mask_context,
)
from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.context_parallel.types import (
    AttnMaskKind,
    AttnSlice,
    ExactMaskMetadata,
    FlexMaskSpec,
    TokenRange,
)
from art.megatron.flex_attn.attention import (
    PrefixTreeAttentionState as FlexPrefixTreeAttentionState,
)
from art.megatron.flex_attn.compiled import flash_sparse_block_size_for_head_dim
from art.megatron.gdn.gdn_prefix_tree import (
    GdnPackedExecutionSpec,
    GdnPlannerConfig,
    GdnRankExecutionPlan,
    build_gdn_rank_execution_plan,
    move_gdn_rank_execution_plan_to_device,
    parse_gdn_prefix_tree_segments,
)
from art.megatron.model_support.spec import PrefixTreeModelStateContext


class PrefixTreeAttentionState(FlexPrefixTreeAttentionState):
    """Prefix-tree sparsity and optional GDN execution metadata."""

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


def create_prefix_tree_state(
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
    gdn_planner_config: GdnPlannerConfig | None = None,
) -> PrefixTreeAttentionState:
    """Build prefix-tree attention mask state plus optional reusable GDN plan."""
    device = group_ids.device if target_device is None else torch.device(target_device)
    group_ids_cpu = _metadata_cpu(group_ids)
    parent_ids_cpu = _metadata_cpu(parent_ids)
    input_pos_cpu = None if input_pos is None else _metadata_cpu(input_pos)
    block_size = _prefix_tree_block_size(
        device,
        attention_head_dim=attention_head_dim,
        attention_value_head_dim=attention_value_head_dim,
    )
    block_mask = _build_sparse_prefix_tree_block_mask(
        group_ids_cpu=group_ids_cpu,
        parent_ids_cpu=parent_ids_cpu,
        input_pos_cpu=input_pos_cpu,
        sliding_window=None,
        device=device,
        block_size=block_size,
    )
    sliding_block_masks = {
        window: _build_sparse_prefix_tree_block_mask(
            group_ids_cpu=group_ids_cpu,
            parent_ids_cpu=parent_ids_cpu,
            input_pos_cpu=input_pos_cpu,
            sliding_window=window,
            device=device,
            block_size=block_size,
        )
        for window in tuple(dict.fromkeys(int(window) for window in sliding_windows))
    }
    cp_rank, cp_size = _gdn_cp_rank_size()
    gdn_execution_spec = (
        parse_gdn_prefix_tree_segments(
            group_ids_cpu,
            parent_ids_cpu,
        )
        if build_gdn_execution_spec
        else None
    )
    return PrefixTreeAttentionState(
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
            attention_token_layout_index=attention_token_layout_index,
            planner_config=gdn_planner_config,
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
        model_support_handler.build_prefix_tree_model_state(
            PrefixTreeModelStateContext(
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


def _build_sparse_prefix_tree_block_mask(
    *,
    group_ids_cpu: Tensor,
    parent_ids_cpu: Tensor,
    input_pos_cpu: Tensor | None,
    sliding_window: int | None,
    device: torch.device,
    block_size: tuple[int, int],
):
    batch_spec = build_prefix_tree_attention_spec(
        group_ids=group_ids_cpu,
        parent_ids=parent_ids_cpu,
    )
    seq_len = int(group_ids_cpu.shape[1])
    row_masks = []
    token_indices = torch.arange(seq_len, dtype=torch.int64)
    for row_spec in batch_spec.rows:
        row_index = int(row_spec.row_index)
        slices = tuple(replace(slice_, row_index=0) for slice_ in row_spec.slices)
        if int(row_spec.valid_tokens) < seq_len:
            padding_range = TokenRange(start=int(row_spec.valid_tokens), end=seq_len)
            slices = (
                *slices,
                AttnSlice(
                    q_range=padding_range,
                    k_range=padding_range,
                    mask_kind=AttnMaskKind.CAUSAL,
                    row_index=0,
                    family_index=None,
                ),
            )
        if not slices:
            row_masks.append(
                _empty_block_mask(seq_len=seq_len, block_size=block_size, device=device)
            )
            continue
        row_masks.append(
            build_block_mask_from_context(
                FlexMaskSpec(
                    q_len=seq_len,
                    k_len=seq_len,
                    block_size=block_size,
                    slices=slices,
                    exact_mask=ExactMaskMetadata(
                        q_token_indices=token_indices,
                        k_token_indices=token_indices,
                        cache_key=(
                            f"identity:{seq_len}"
                            if sliding_window is None
                            else f"identity:{seq_len}:sliding:{int(sliding_window)}"
                        ),
                    ),
                ),
                context=prepare_block_mask_context(
                    group_ids=group_ids_cpu[row_index],
                    parent_ids=parent_ids_cpu[row_index],
                    input_pos=None
                    if input_pos_cpu is None
                    else input_pos_cpu[row_index],
                ),
                sliding_window=sliding_window,
                device=device,
            )
        )
    if not row_masks:
        return _empty_block_mask(seq_len=seq_len, block_size=block_size, device=device)
    return _stack_row_block_masks(
        row_masks,
        seq_len=seq_len,
        block_size=block_size,
    )


def _stack_optional_block_tensors(
    masks: list[BlockMask],
    name: str,
) -> Tensor | None:
    tensors = [getattr(mask, name) for mask in masks]
    if any(tensor is None for tensor in tensors):
        return None
    return torch.cat(tensors, dim=0)


def _stack_row_block_masks(
    masks: list[BlockMask],
    *,
    seq_len: int,
    block_size: tuple[int, int],
) -> BlockMask:
    if len(masks) == 1:
        return masks[0]
    row_mask_mods = tuple(mask.mask_mod for mask in masks)

    def mask_mod(
        batch_idx: Tensor,
        head_idx: Tensor,
        query_idx: Tensor,
        kv_idx: Tensor,
    ) -> Tensor:
        result = torch.zeros_like(query_idx, dtype=torch.bool)
        for row_index, row_mask_mod in enumerate(row_mask_mods):
            result = torch.where(
                batch_idx == row_index,
                row_mask_mod(batch_idx, head_idx, query_idx, kv_idx),
                result,
            )
        return result

    return BlockMask(
        seq_lengths=(int(seq_len), int(seq_len)),
        kv_num_blocks=torch.cat([mask.kv_num_blocks for mask in masks], dim=0),
        kv_indices=torch.cat([mask.kv_indices for mask in masks], dim=0),
        full_kv_num_blocks=_stack_optional_block_tensors(masks, "full_kv_num_blocks"),
        full_kv_indices=_stack_optional_block_tensors(masks, "full_kv_indices"),
        q_num_blocks=_stack_optional_block_tensors(masks, "q_num_blocks"),
        q_indices=_stack_optional_block_tensors(masks, "q_indices"),
        full_q_num_blocks=_stack_optional_block_tensors(masks, "full_q_num_blocks"),
        full_q_indices=_stack_optional_block_tensors(masks, "full_q_indices"),
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
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


def _prefix_tree_block_size(
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


def _build_gdn_execution_plan_once(
    spec: GdnPackedExecutionSpec | None,
    *,
    device: torch.device,
    cp_rank: int,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None,
    planner_config: GdnPlannerConfig | None,
) -> GdnRankExecutionPlan | None:
    if spec is None:
        return None
    planner_device = torch.device("cpu") if device.type == "cuda" else device
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
            planner_config=planner_config,
        )
    finally:
        if gc_was_enabled:
            gc.enable()
    return move_gdn_rank_execution_plan_to_device(plan, device)


def _gdn_cp_rank_size() -> tuple[int, int]:
    try:
        from megatron.core import parallel_state as ps

        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            return (
                int(ps.get_context_parallel_rank()),
                int(ps.get_context_parallel_world_size()),
            )
    except Exception:
        pass
    return 0, 1
