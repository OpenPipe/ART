from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
from art.megatron.context_parallel.types import AttnMaskKind

_ROUTE_STAGE_BITS = 3
_MAX_ROUTE_STAGES = 1 << _ROUTE_STAGE_BITS
_MAX_ROUTE_ROWS = 1 << (31 - _ROUTE_STAGE_BITS)


class Glm52IndexerSlice(BaseModel):
    model_config = ConfigDict(frozen=True)

    k_start: int
    k_end: int
    causal: bool


class Glm52IndexerQueryPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    q_start: int
    q_end: int
    slices: tuple[Glm52IndexerSlice, ...]


class Glm52IndexerRowPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    row_index: int
    valid_tokens: int
    queries: tuple[Glm52IndexerQueryPlan, ...]


class Glm52StageState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    stage_index: int
    global_q_ids: torch.Tensor
    global_k_ids: torch.Tensor


class Glm52PrefixTreeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    position_ids: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    indexer_rows: tuple[Glm52IndexerRowPlan, ...] = ()
    stages: tuple[Glm52StageState, ...] = ()
    context_parallel_state: Any | None = None
    topk_by_full_layer: dict[int, Any] = Field(default_factory=dict)


def _rope_state(
    position_ids: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_ids_device = position_ids.to(
        device=device,
        dtype=torch.int64,
        non_blocking=True,
    ).contiguous()
    inv_freq = 1.0 / (
        8_000_000 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
    )
    frequencies = position_ids_device.float().unsqueeze(-1) * inv_freq
    return (
        position_ids_device,
        frequencies.cos().to(torch.bfloat16),
        frequencies.sin().to(torch.bfloat16),
    )


def build_glm52_prefix_tree_state(
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    device: torch.device,
) -> Glm52PrefixTreeState:
    """Precompute immutable tree rectangles once for every GLM-5.2 layer."""
    if position_ids.ndim != 2:
        raise ValueError(
            f"GLM-5.2 position_ids must be 2D, got {tuple(position_ids.shape)}."
        )
    batch_spec = build_prefix_tree_attention_spec(
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    rows: list[Glm52IndexerRowPlan] = []
    for row in batch_spec.rows:
        slices_by_query: dict[tuple[int, int], list[Glm52IndexerSlice]] = defaultdict(
            list
        )
        for slice_ in row.slices:
            slices_by_query[(slice_.q_range.start, slice_.q_range.end)].append(
                Glm52IndexerSlice(
                    k_start=slice_.k_range.start,
                    k_end=slice_.k_range.end,
                    causal=slice_.mask_kind is AttnMaskKind.CAUSAL,
                )
            )
        queries = tuple(
            Glm52IndexerQueryPlan(
                q_start=q_start,
                q_end=q_end,
                slices=tuple(slices),
            )
            for (q_start, q_end), slices in sorted(slices_by_query.items())
        )
        rows.append(
            Glm52IndexerRowPlan(
                row_index=row.row_index,
                valid_tokens=row.valid_tokens,
                queries=queries,
            )
        )
    position_ids_device, rope_cos, rope_sin = _rope_state(
        position_ids,
        device=device,
    )
    return Glm52PrefixTreeState(
        position_ids=position_ids_device,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        indexer_rows=tuple(rows),
    )


def build_glm52_context_parallel_state(
    *,
    position_ids: torch.Tensor,
    context_parallel_state: Any,
    device: torch.device,
) -> Glm52PrefixTreeState:
    """Materialize GLM stage ids once without reading CUDA data on the host."""
    rank_plan = context_parallel_state.rank_plan
    stages = []
    for stage in rank_plan.stage_plans:
        q_len = sum(range_.size() for range_ in stage.owner_local_q_ranges)
        k_len = sum(range_.size() for range_ in stage.owner_local_k_ranges)
        if int(stage.stage_index) >= _MAX_ROUTE_STAGES:
            raise RuntimeError(
                f"GLM-5.2 route encoding supports {_MAX_ROUTE_STAGES} stages."
            )
        if k_len >= _MAX_ROUTE_ROWS:
            raise RuntimeError(
                f"GLM-5.2 stage {stage.stage_index} has too many KV rows: {k_len}."
            )
        metadata = stage.mask_metadata
        if metadata is None and (q_len or k_len):
            raise RuntimeError(
                f"GLM-5.2 stage {stage.stage_index} is missing exact token ids."
            )
        if metadata is None:
            q_ids = k_ids = torch.empty(0, dtype=torch.int32, device=device)
        else:
            q_ids = metadata.q_token_indices[:q_len].to(
                device=device, dtype=torch.int32, non_blocking=True
            )
            k_ids = metadata.k_token_indices[:k_len].to(
                device=device, dtype=torch.int32, non_blocking=True
            )
        stages.append(
            Glm52StageState(
                stage_index=int(stage.stage_index),
                global_q_ids=q_ids.contiguous(),
                global_k_ids=k_ids.contiguous(),
            )
        )
    position_ids_device, rope_cos, rope_sin = _rope_state(
        position_ids,
        device=device,
    )
    return Glm52PrefixTreeState(
        position_ids=position_ids_device,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        stages=tuple(stages),
        context_parallel_state=context_parallel_state,
    )


def require_glm52_state(attention_bias: Any) -> Glm52PrefixTreeState:
    model_state = getattr(attention_bias, "model_state", None)
    state = model_state.get("glm52") if isinstance(model_state, dict) else None
    if not isinstance(state, Glm52PrefixTreeState):
        raise RuntimeError(
            "GLM-5.2 prefix-tree state is missing; build it once per packed "
            "sequence through the model-support handler."
        )
    return state
