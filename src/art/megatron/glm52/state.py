from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
from art.megatron.context_parallel.types import AttnMaskKind


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


class Glm52PrefixTreeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    position_ids: torch.Tensor
    indexer_rows: tuple[Glm52IndexerRowPlan, ...]
    topk_by_full_layer: dict[int, torch.Tensor] = Field(default_factory=dict)


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
    return Glm52PrefixTreeState(
        position_ids=position_ids.to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        ).contiguous(),
        indexer_rows=tuple(rows),
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
