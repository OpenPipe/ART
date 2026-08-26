from __future__ import annotations

from typing import Any, cast

import torch

from art.megatron.context_parallel.comm import A2AVCommunicator
from art.megatron.context_parallel.range_ops import range_gather, range_reduce_sum_
from art.megatron.context_parallel.types import (
    ArtContextParallelState,
    DkvReducePlan,
    StagePlan,
)

_COMMUNICATOR = A2AVCommunicator()


def stage_query_rows(
    tensor: torch.Tensor,
    stage: StagePlan,
    state: ArtContextParallelState,
) -> torch.Tensor:
    ranges = stage.owner_local_q_ranges
    if len(ranges) == 1 and ranges[0].start == 0 and ranges[0].end == tensor.shape[0]:
        return tensor
    return range_gather(
        tensor,
        ranges,
        range_meta_cache=state.execution_cache.range_meta,
    )


def stage_local_kv_rows(
    tensor: torch.Tensor,
    stage: StagePlan,
    state: ArtContextParallelState,
) -> torch.Tensor:
    ranges = stage.owner_local_k_ranges
    if len(ranges) == 1 and ranges[0].start == 0 and ranges[0].end == tensor.shape[0]:
        return tensor
    return range_gather(
        tensor,
        ranges,
        range_meta_cache=state.execution_cache.range_meta,
    )


def launch_remote_stage_fetches(
    tensor: torch.Tensor,
    state: ArtContextParallelState,
) -> dict[int, Any]:
    return {
        int(stage.stage_index): _COMMUNICATOR.launch_tensor_fetch(
            tensor_local=tensor,
            plan=cast(Any, stage.kv_fetch_plan),
            group=state.cp_group,
            async_op=True,
            range_meta_cache=state.execution_cache.range_meta,
        )
        for stage in state.rank_plan.stage_plans
        if not stage.is_local_stage
    }


def stage_kv_rows(
    tensor: torch.Tensor,
    stage: StagePlan,
    state: ArtContextParallelState,
    fetches: dict[int, Any],
) -> torch.Tensor:
    return (
        stage_local_kv_rows(tensor, stage, state)
        if stage.is_local_stage
        else fetches.pop(int(stage.stage_index)).wait_post_process()
    )


def drain_stage_fetches(fetches: dict[int, Any]) -> None:
    for work in fetches.values():
        work.wait_post_process()
    fetches.clear()


def reduce_local_stage_rows_(
    target: torch.Tensor,
    stage_grad: torch.Tensor,
    stage: StagePlan,
    state: ArtContextParallelState,
) -> None:
    range_reduce_sum_(
        stage_grad,
        output_tensor=target,
        ranges=stage.owner_local_k_ranges,
        range_meta_cache=state.execution_cache.range_meta,
    )


def launch_remote_stage_reduce(
    stage_grad: torch.Tensor,
    stage: StagePlan,
    state: ArtContextParallelState,
    output: torch.Tensor,
) -> Any:
    return _COMMUNICATOR.launch_tensor_reduce(
        remote=stage_grad.contiguous(),
        plan=cast(DkvReducePlan, stage.dkv_reduce_plan),
        group=state.cp_group,
        async_op=True,
        output=output,
        range_meta_cache=state.execution_cache.range_meta,
    )
