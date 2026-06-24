"""Shared-prefix packed-sequence state for ART attention and GDN integration."""

from __future__ import annotations

import gc
from typing import Any

from pydantic import Field
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
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


class SharedPrefixAttentionState(FlexSharedPrefixAttentionState):
    """Shared-prefix sparsity and optional GDN execution metadata."""

    group_ids: Tensor
    parent_ids: Tensor
    dsv4_compression_layouts: dict[int, Any] = Field(default_factory=dict)
    dsv4_topk_idx_cache: dict[Any, Any] = Field(default_factory=dict)
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


_compiled_create_block_mask = torch.compile(create_block_mask, backend="aot_eager")


def create_shared_prefix_state(
    group_ids: Tensor,
    parent_ids: Tensor,
    *,
    build_gdn_execution_spec: bool = False,
    build_dsv4_compression_layouts: bool = False,
    dsv4_position_ids: Tensor | None = None,
    dsv4_group_ids: Tensor | None = None,
    dsv4_parent_ids: Tensor | None = None,
    attention_token_layout_index: TokenLayoutIndex | None = None,
    attention_head_dim: int | None = None,
    attention_value_head_dim: int | None = None,
) -> SharedPrefixAttentionState:
    """Build shared-prefix attention mask state plus optional reusable GDN plan."""

    def _shared_prefix_mask(
        batch_idx: Tensor,
        head_idx: Tensor,
        query_idx: Tensor,
        kv_idx: Tensor,
    ) -> Tensor:
        del head_idx
        same_group = group_ids[batch_idx, query_idx] == group_ids[batch_idx, kv_idx]
        parent_prefix = parent_ids[batch_idx, query_idx] == group_ids[batch_idx, kv_idx]
        return (query_idx >= kv_idx) & (same_group | parent_prefix)

    block_mask = _compiled_create_block_mask(
        _shared_prefix_mask,
        group_ids.shape[0],
        None,
        group_ids.shape[1],
        group_ids.shape[1],
        device=group_ids.device,
        BLOCK_SIZE=_shared_prefix_block_size(
            group_ids.device,
            attention_head_dim=attention_head_dim,
            attention_value_head_dim=attention_value_head_dim,
        ),
    )
    cp_rank, cp_size, cp_group = _gdn_cp_rank_size_group()
    gdn_execution_spec = _build_gdn_execution_spec_once(
        group_ids,
        parent_ids,
        build=build_gdn_execution_spec,
        cp_rank=cp_rank,
        cp_size=cp_size,
        cp_group=cp_group,
    )
    return SharedPrefixAttentionState(
        block_mask=block_mask,
        group_ids=group_ids,
        parent_ids=parent_ids,
        dsv4_compression_layouts=_build_dsv4_compression_layouts_once(
            position_ids=dsv4_position_ids,
            group_ids=group_ids if dsv4_group_ids is None else dsv4_group_ids,
            parent_ids=parent_ids if dsv4_parent_ids is None else dsv4_parent_ids,
            build=build_dsv4_compression_layouts,
            device=group_ids.device,
        ),
        gdn_execution_spec=gdn_execution_spec,
        gdn_execution_plan=_build_gdn_execution_plan_once(
            gdn_execution_spec,
            device=group_ids.device,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_group=cp_group,
            attention_token_layout_index=attention_token_layout_index,
        ),
    )


def _build_dsv4_compression_layouts_once(
    *,
    position_ids: Tensor | None,
    group_ids: Tensor,
    parent_ids: Tensor,
    build: bool,
    device: torch.device,
) -> dict[int, Any]:
    if not build:
        return {}
    if position_ids is None:
        raise RuntimeError(
            "DSV4 shared-prefix compression layouts require position ids."
        )
    from art.megatron.dsv4.compressor import build_shared_prefix_compression_layouts

    return build_shared_prefix_compression_layouts(
        position_ids=position_ids,
        group_ids=group_ids,
        parent_ids=parent_ids,
        device=device,
    )


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
