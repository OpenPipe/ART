import pytest

from art.distributed import vllm_gateway
from art.distributed.specs import (
    EndpointSpec,
    ModelServiceMemberSpec,
    ModelServiceReplicaSpec,
    VllmParallelSpec,
)
from art.distributed.vllm_kv_events import KvEventSource
from art.distributed.vllm_replica import (
    HostMemberState,
    ReplicaFailure,
    ReplicaLaunchTemplate,
    ReplicaManager,
)
from art.distributed.vllm_router import (
    PolicyGenerationCommitError,
    ReplicaRouter,
    ReplicaTelemetry,
    RoutableReplica,
    RoutingTable,
)


class Launcher:
    def __init__(self) -> None:
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
        self.stops.append((replica_id, member_id, generation))


def _spec() -> ModelServiceReplicaSpec:
    members = tuple(
        ModelServiceMemberSpec(
            member_id=f"node{rank}",
            host_id=f"host{rank}",
            node_rank=rank,
            gpu_ids=(0, 1),
            leader=rank == 0,
        )
        for rank in range(2)
    )
    return ModelServiceReplicaSpec(
        service_name="model",
        replica_id="replica",
        members=members,
        leader_endpoint=EndpointSpec(host="10.0.0.1", port=8000),
        rendezvous=EndpointSpec(host="10.0.0.1", port=29500),
        base_model="base",
        model_revision="revision",
        runtime_fingerprint="runtime",
        parallel=VllmParallelSpec(tp=2, pp=2),
        update_mode="lora",
    )


@pytest.mark.asyncio
async def test_replica_restart_reuses_ports_and_fences_failed_generation() -> None:
    launchers = {f"host{rank}": Launcher() for rank in range(2)}
    failures: list[ReplicaFailure] = []

    async def failed(event: ReplicaFailure) -> None:
        failures.append(event)

    manager = ReplicaManager(
        _spec(),
        launchers,
        ReplicaLaunchTemplate(served_model_name="model@0", lora_path="/step/0000"),
        on_failure=failed,
        monitor_interval_s=60,
    )
    await manager.start()
    launchers["host1"].failed = True

    await manager.poll()

    assert manager.state.phase == "quarantined"
    assert [(event.replica_id, event.generation) for event in failures] == [
        ("replica", 0)
    ]

    launchers["host1"].failed = False
    restarted = await manager.restart(
        served_model_name="model@1", lora_path="/step/0001"
    )

    assert restarted.phase == "ready"
    assert restarted.generation == 1
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


def _table() -> RoutingTable:
    replica = RoutableReplica(
        replica_id="replica",
        endpoint=EndpointSpec(host="127.0.0.1", port=8000),
        phase="ready",
        generation=0,
        generation_digest="generation-0",
        committed_version="0",
        policy_digest="policy-0",
        update_identity="update-0",
        telemetry=ReplicaTelemetry(observed_at=0, in_flight=0, capacity=1),
    )
    return RoutingTable(
        policy_generation=3,
        policy_version="0",
        policy_digest="policy-0",
        update_identity="update-0",
        replicas=(replica,),
    )


@pytest.mark.asyncio
async def test_failure_transition_invalidates_prepared_commit_and_ignores_late_event() -> (
    None
):
    router = ReplicaRouter(_table(), clock=lambda: 0)
    candidate = router.table.model_copy(update={"policy_generation": 4})
    prepared = router.prepare(candidate)

    quarantined = await router.quarantine(
        ("replica",),
        "failed",
        expected_generations={"replica": 0},
    )

    assert quarantined.policy_generation == 4
    with pytest.raises(PolicyGenerationCommitError, match="changed after"):
        await router.commit(prepared)

    old = quarantined.replicas[0]
    replacement = old.model_copy(
        update={
            "phase": "ready",
            "generation": 1,
            "generation_digest": "generation-1",
            "quarantine_reason": None,
        }
    )
    replaced = await router.replace_replica(replacement, expected_generation=0)
    late = await router.quarantine(
        ("replica",),
        "late",
        expected_generations={"replica": 0},
    )

    assert replaced.policy_generation == 5
    assert late == replaced
    assert late.replicas[0].phase == "ready"


@pytest.mark.asyncio
async def test_gateway_replaces_generation_in_current_and_pinned_policy_views(
    monkeypatch,
) -> None:
    subscribers = []

    class Subscriber:
        def __init__(self, source, *_callbacks) -> None:
            self.source = source
            self.started = False
            self.closed = False
            subscribers.append(self)

        def start(self) -> None:
            self.started = True

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(vllm_gateway, "VllmKvEventSubscriber", Subscriber)
    source0 = KvEventSource(
        replica_id="replica",
        generation=0,
        publisher_rank=0,
        endpoint="tcp://127.0.0.1:9000",
        replay_endpoint="tcp://127.0.0.1:9001",
        topic="generation-0",
    )
    gateway = vllm_gateway.VllmGateway(_table(), kv_event_sources=(source0,))
    exact = _table().model_copy(
        update={
            "policy_version": "exact",
            "policy_digest": "policy-exact",
            "update_identity": "update-exact",
            "replicas": (
                _table()
                .replicas[0]
                .model_copy(
                    update={
                        "committed_version": "exact",
                        "policy_digest": "policy-exact",
                        "update_identity": "update-exact",
                    }
                ),
            ),
        }
    )
    await gateway.add_policy(exact)
    await gateway.quarantine_replica("replica", 0, "failed")
    source1 = source0.model_copy(update={"generation": 1, "topic": "generation-1"})

    generation = await gateway.replace_replica(
        replica_id="replica",
        previous_generation=0,
        generation=1,
        generation_digest="generation-1",
        endpoint=EndpointSpec(host="127.0.0.1", port=8000),
        kv_event_sources=(source1,),
    )

    assert generation == gateway.router.table.policy_generation
    for router in set(gateway._policies.values()):
        assert router.table.replicas[0].phase == "ready"
        assert router.table.replicas[0].generation == 1
    assert subscribers[0].closed
    assert subscribers[1].started
    await gateway.quarantine_replica("replica", 0, "late")
    assert gateway.router.table.replicas[0].phase == "ready"
    await gateway.close()
