from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from art.distributed.specs import (
    EndpointSpec,
    ModelServiceMemberSpec,
    ModelServiceReplicaSpec,
    VllmParallelSpec,
)
from art.distributed.vllm_replica import (
    HostMemberState,
    ReplicaLaunchTemplate,
    ReplicaManager,
    ReplicaState,
)


def manager() -> ReplicaManager:
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
    spec = ModelServiceReplicaSpec(
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
    value = ReplicaManager(
        spec,
        {"host0": SimpleNamespace(), "host1": SimpleNamespace()},
        ReplicaLaunchTemplate(served_model_name="model@1"),
    )
    value._state = ReplicaState(
        replica_id="replica",
        generation=0,
        generation_digest=value.state.generation_digest,
        phase="ready",
        members=tuple(
            HostMemberState(
                replica_id="replica",
                member_id=member.member_id,
                generation=0,
                generation_digest=value.state.generation_digest,
                process_uuid=f"process-{member.node_rank}",
                phase="ready",
            )
            for member in reversed(members)
        ),
    )
    return value


def test_expected_worker_identities_follow_physical_rank_order() -> None:
    assert manager().expected_worker_identities() == (
        {
            "rank": 0,
            "local_rank": 0,
            "node_rank": 0,
            "process_uuid": "process-0",
            "generation": 0,
        },
        {
            "rank": 1,
            "local_rank": 1,
            "node_rank": 0,
            "process_uuid": "process-0",
            "generation": 0,
        },
        {
            "rank": 2,
            "local_rank": 0,
            "node_rank": 1,
            "process_uuid": "process-1",
            "generation": 0,
        },
        {
            "rank": 3,
            "local_rank": 1,
            "node_rank": 1,
            "process_uuid": "process-1",
            "generation": 0,
        },
    )


def test_expected_worker_identities_reject_incomplete_membership() -> None:
    value = manager()
    value._state = value.state.model_copy(update={"members": value.state.members[:1]})
    with pytest.raises(RuntimeError, match="membership is incomplete"):
        value.expected_worker_identities()


def test_launch_pins_model_and_tokenizer_revisions() -> None:
    value = manager()
    request = value._launch_request(value.spec.members[0])
    assert request.launch_config.engine_args["revision"] == "revision"
    assert request.launch_config.engine_args["tokenizer_revision"] == "revision"


def test_effective_hash_block_size_requires_runtime_confirmation() -> None:
    value = manager()
    with pytest.raises(RuntimeError, match="not confirmed"):
        _ = value.prefix_hash_block_size

    value.confirm_prefix_hash_block_size(64)
    assert value.prefix_hash_block_size == 64
    value.confirm_prefix_hash_block_size(64)
    with pytest.raises(ValueError, match="positive"):
        value.confirm_prefix_hash_block_size(0)
    with pytest.raises(RuntimeError, match="changed"):
        value.confirm_prefix_hash_block_size(32)


@pytest.mark.asyncio
async def test_stop_invalidates_effective_hash_block_size() -> None:
    value = manager()
    value.confirm_prefix_hash_block_size(64)
    value._stop_current_members = AsyncMock()

    await value.stop()

    with pytest.raises(RuntimeError, match="not confirmed"):
        _ = value.prefix_hash_block_size


def test_conflicting_untyped_revision_is_rejected() -> None:
    value = manager()
    with pytest.raises(ValueError, match="revision conflicts"):
        ReplicaManager(
            value.spec,
            {"host0": SimpleNamespace(), "host1": SimpleNamespace()},
            ReplicaLaunchTemplate(
                served_model_name="model@1", engine_args={"revision": "other"}
            ),
        )
