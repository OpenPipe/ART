from typing import Any, cast

import pytest

from art.distributed.specs import (
    EndpointSpec,
    ModelServiceMemberSpec,
    ModelServiceSpec,
    VllmParallelSpec,
)
from art.distributed.vllm_replica import (
    HostMemberState,
    ReplicaFailure,
    ReplicaLaunchTemplate,
    ReplicaManager,
)


class Launcher:
    def __init__(self, host_id: str, events: list[str]) -> None:
        self.host_id = host_id
        self.events = events
        self.requests = []
        self.states = {}
        self.failed = False
        self.stops = []

    async def start_member(self, request):
        self.requests.append(request)
        state = HostMemberState(
            replica_id=request.replica_id,
            member_id=request.member.member_id,
            generation=request.generation,
            generation_digest=request.generation_digest,
            process_uuid=request.process_uuid,
            phase="ready",
        )
        self.states[
            (request.replica_id, request.member.member_id, request.generation)
        ] = state
        return state

    async def member_state(self, replica_id, member_id, generation):
        state = self.states[(replica_id, member_id, generation)]
        return state.model_copy(update={"phase": "failed"}) if self.failed else state

    async def stop_member(self, replica_id, member_id, generation):
        self.events.append(f"stop:{self.host_id}")
        self.stops.append((replica_id, member_id, generation))


def _spec() -> ModelServiceSpec:
    return ModelServiceSpec(
        name="model",
        members=tuple(
            ModelServiceMemberSpec(
                member_id=f"node{rank}",
                host_id=f"host{rank}",
                node_rank=rank,
                gpu_ids=(0, 1),
            )
            for rank in range(2)
        ),
        leader_endpoint=EndpointSpec(host="10.0.0.1", port=8000),
        rendezvous=EndpointSpec(host="10.0.0.1", port=29500),
        base_model="base",
        model_revision="revision",
        runtime_fingerprint="runtime",
        parallel=VllmParallelSpec(tp=2, pp=2),
    )


@pytest.mark.asyncio
async def test_failure_stops_whole_gang_before_callback_and_restarts_generation() -> (
    None
):
    events: list[str] = []
    launchers = {f"host{rank}": Launcher(f"host{rank}", events) for rank in range(2)}
    failures: list[ReplicaFailure] = []

    async def failed(event: ReplicaFailure) -> None:
        events.append("callback")
        failures.append(event)

    manager = ReplicaManager(
        _spec(),
        cast(Any, launchers),
        ReplicaLaunchTemplate(served_model_name="model@0", lora_path="/step/0000"),
        on_failure=failed,
        monitor_interval_s=60,
    )
    await manager.start()
    launchers["host1"].failed = True

    await manager.poll()

    assert manager.state.phase == "quarantined"
    assert set(events[:2]) == {"stop:host0", "stop:host1"}
    assert events[2:] == ["callback"]
    assert [(event.replica_id, event.generation) for event in failures] == [
        ("model", 0)
    ]

    launchers["host1"].failed = False
    restarted = await manager.restart(
        served_model_name="model@1",
        lora_path="/step/0001",
        initial_policy_version=1,
    )

    assert restarted.phase == "ready"
    assert restarted.generation == 1
    assert restarted.generation_digest != failures[0].generation_digest
    for launcher in launchers.values():
        assert [request.launch_config.port for request in launcher.requests] == [
            8000,
            8000,
        ]
        assert [request.launch_config.master_port for request in launcher.requests] == [
            29500,
            29500,
        ]
    await manager.stop()
