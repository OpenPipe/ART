from types import SimpleNamespace
from unittest.mock import AsyncMock

from art_vllm_runtime import lora_state as state_module
from art_vllm_runtime.lora_state import (
    ExpectedLoraState,
    ExpectedLoraWorker,
    LoraWorkerStateRequest,
    WorkerLoraState,
    query_lora_worker_state,
)
import pytest

LORA = WorkerLoraState(
    lora_id=1,
    lora_name="model@3",
    lora_path="/step/3",
    policy_version=3,
    update_seq=1,
)


def _report(rank: int, *, dp: int, executor_world_size: int) -> dict[str, object]:
    dp_rank, executor_rank = divmod(rank, executor_world_size)
    local_world_size = executor_world_size
    return {
        "rank": rank,
        "executor_rank": executor_rank,
        "executor_world_size": executor_world_size,
        "data_parallel_rank": dp_rank,
        "data_parallel_size": dp,
        "local_rank": executor_rank,
        "node_rank": dp_rank,
        "process_uuid": f"process-{dp_rank}",
        "generation": 4,
        "worker_pid": 100 + rank,
        "hostname": f"host-{dp_rank}",
        "engine_instance_id": f"engine-{dp_rank}",
        "world_size": dp * executor_world_size,
        "local_world_size": local_world_size,
        "loaded_loras": [LORA.model_dump()],
    }


def _request(world_size: int, executor_world_size: int) -> LoraWorkerStateRequest:
    return LoraWorkerStateRequest(
        expected_workers=tuple(
            ExpectedLoraWorker(
                rank=rank,
                local_rank=rank % executor_world_size,
                node_rank=rank // executor_world_size,
                process_uuid=f"process-{rank // executor_world_size}",
                generation=4,
            )
            for rank in range(world_size)
        ),
        expected_lora=ExpectedLoraState(
            lora_name=LORA.lora_name,
            lora_path=LORA.lora_path,
            policy_version=LORA.policy_version,
        ),
    )


@pytest.mark.asyncio
async def test_query_lora_worker_state_validates_non_dp_workers() -> None:
    replies = [_report(rank, dp=1, executor_world_size=2) for rank in range(2)]
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        ),
        collective_rpc=AsyncMock(return_value=replies),
    )

    response = await query_lora_worker_state(client, _request(2, 2))

    assert response.expected_worker_ranks == (0, 1)
    assert response.lora == LORA


@pytest.mark.asyncio
async def test_query_lora_worker_state_validates_every_dp_executor(
    monkeypatch,
) -> None:
    groups = [
        [_report(rank, dp=2, executor_world_size=2) for rank in range(start, start + 2)]
        for start in (0, 2)
    ]
    query = AsyncMock(return_value=groups)
    monkeypatch.setattr(state_module, "query_engine_cores", query)
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=2)
        )
    )

    response = await query_lora_worker_state(client, _request(4, 2))

    assert response.expected_worker_ranks == (0, 1, 2, 3)
    assert response.lora == LORA
    query.assert_awaited_once()
