"""Shared CP2 runtime and layout setup for attention and graph-residency tests."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from typing import cast
from unittest.mock import patch

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
from art.megatron.flex_attn import compiled
from art.megatron.runtime.compile_cache import configure_reusable_backward
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


@contextmanager
def cp2_runtime(
    rank: int, init_method: str, device: torch.device, *, backend: str, timeout: int
) -> Iterator[None]:
    torch.cuda.set_device(device)
    configure_reusable_backward()
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=timeout),
        device_id=device,
    )
    try:
        with ExitStack() as stack:
            if backend == "TRITON":
                stack.enter_context(
                    patch.object(compiled, "_FORCED_FLEX_BACKEND", "TRITON")
                )
                stack.enter_context(
                    patch.object(
                        compiled,
                        "sparse_compiled_flex_attention",
                        compiled.triton_sparse_compiled_flex_attention,
                    )
                )
            yield
    finally:
        dist.destroy_process_group()
