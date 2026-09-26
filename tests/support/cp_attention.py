"""Shared CP2 layout setup for attention and graph-residency tests."""

from typing import cast

import torch
import torch.distributed as dist

from art.megatron.context_parallel import executor
from art.megatron.context_parallel.runtime import (
    prepare_megatron_context_parallel_state,
)
from art.megatron.context_parallel.types import (
    ArtContextParallelState,
    ContextParallelConfig,
    ParallelTopology,
    RankRuntimePlan,
)
from art.preprocessing.pack import PackedTensors


def prepare_cp2_attention(
    rank: int, device: torch.device, length: int
) -> tuple[PackedTensors, ArtContextParallelState, RankRuntimePlan, torch.Tensor]:
    micro = cast(
        PackedTensors,
        {
            "tokens": torch.arange(length)[None],
            "group_ids": torch.ones((1, length), dtype=torch.long),
            "parent_ids": torch.ones((1, length), dtype=torch.long),
            "input_pos": torch.arange(length)[None],
        },
    )
    state, plan, _, _ = prepare_megatron_context_parallel_state(
        micro=micro,
        topology=ParallelTopology(cp=2),
        config=ContextParallelConfig(
            planner_chunk_size=128, planner_owned_token_ms=1.0
        ),
        cp_group=dist.group.WORLD,
        cp_rank=rank,
        target_device=device,
    )
    indices = torch.tensor(
        [
            index
            for start, end, _ in plan.token_layout_index.ownership_ranges_by_rank[rank]
            for index in range(start, end)
        ],
        device=device,
    )
    assert indices.numel() > 0
    assert indices.numel() == sum(plan.local_valid_lengths)
    executor.prepare_context_parallel_execution_state(state=state, device=device)
    return micro, state, plan, indices
