from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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
    ReplicaLaunchTemplate,
    ReplicaManager,
    ReplicaState,
)
from art.megatron import distributed_service as service_module
from art.megatron.distributed_service import DistributedMegatronService


def manager(*, engine_args: dict[str, object] | None = None) -> ReplicaManager:
    members = tuple(
        ModelServiceMemberSpec(
            member_id=f"node{rank}",
            host_id=f"host{rank}",
            node_rank=rank,
            gpu_ids=(0, 1),
        )
        for rank in range(2)
    )
    spec = ModelServiceSpec(
        name="model",
        members=members,
        leader_endpoint=EndpointSpec(host="10.0.0.1", port=8000),
        rendezvous=EndpointSpec(host="10.0.0.1", port=29500),
        base_model="base",
        model_revision="revision",
        runtime_fingerprint="runtime",
        parallel=VllmParallelSpec(tp=1, pp=2, dp=2, enable_expert_parallel=True),
    )
    value = ReplicaManager(
        spec,
        cast(
            Any,
            {"host0": SimpleNamespace(), "host1": SimpleNamespace()},
        ),
        ReplicaLaunchTemplate(
            served_model_name="model@1",
            engine_args=engine_args or {},
            runtime_source_epoch_base=7,
        ),
    )
    value._state = ReplicaState(
        replica_id="model",
        generation=0,
        generation_digest=value.state.generation_digest,
        phase="ready",
        members=tuple(
            HostMemberState(
                replica_id="model",
                member_id=member.member_id,
                generation=0,
                generation_digest=value.state.generation_digest,
                process_uuid=f"process-{member.node_rank}",
                phase="ready",
            )
            for member in reversed(members)
        ),
    )
    value._private_dispatch_token = "private-dispatch-token" * 2
    return value


@pytest.mark.asyncio
async def test_in_flight_update_is_the_only_acknowledgement_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = manager()
    runtime = SimpleNamespace(
        topology=SimpleNamespace(cluster=SimpleNamespace(rpc_timeout_s=5)),
        model_service=lambda name: value if name == "model" else None,
    )
    service = DistributedMegatronService(
        model_name="model",
        base_model="base",
        config={"rollout_weight_update_mode": "in_flight_lora"},
        output_dir=str(tmp_path),
        runtime=cast(Any, runtime),
        enable_expert_replay=False,
    )
    calls: list[tuple[str, dict[str, Any], dict[str, str] | None]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def post(self, url: str, *, json: dict[str, Any], headers):
            calls.append((url, json, headers))
            return Response()

    monkeypatch.setattr(service_module.httpx, "AsyncClient", Client)
    service._latest_step = 1
    service._serving_step = 0
    service._published_adapters = {
        0: SimpleNamespace(generation_id="generation-0"),
        1: SimpleNamespace(generation_id="generation-1"),
    }
    service._base_url = "http://leader.test:8000"
    service._api_key_value = "secret"
    name, path = await service._load_adapter("/step/0001", 1)

    assert (name, path) == ("model:active", "/step/0001")
    assert len(calls) == 1
    assert calls[0][0] == "http://leader.test:8000/art/in_flight_lora_update"
    assert calls[0][1]["generation_id"] == "generation-1"
    assert calls[0][1]["expected_generation_id"] == "generation-0"
    assert calls[0][1]["policy_version"] == 1
    assert calls[0][2] == {"Authorization": "Bearer secret"}


def test_launch_preserves_user_args_and_owns_native_gang_topology() -> None:
    value = manager(
        engine_args={
            "enable_prefix_caching": False,
            "block_size": 32,
            "prefill_context_parallel_size": 2,
        }
    )
    leader = value._launch_request(value.spec.members[0]).launch_config
    follower = value._launch_request(value.spec.members[1]).launch_config

    assert leader.engine_args == {
        "enable_prefix_caching": False,
        "block_size": 32,
        "prefill_context_parallel_size": 2,
        "revision": "revision",
        "tokenizer_revision": "revision",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 2,
        "enable_expert_parallel": True,
    }
    assert leader.host == "10.0.0.1" and not leader.headless
    assert follower.host == "127.0.0.1" and follower.headless
    assert leader.nnodes == follower.nnodes == 2
    assert leader.runtime_target_id == follower.runtime_target_id
    assert leader.runtime_source_id == follower.runtime_source_id == value.spec.name
    assert leader.runtime_source_epoch == follower.runtime_source_epoch == 7
    assert leader.private_dispatch_token == follower.private_dispatch_token
    assert "kv_events_config" not in leader.engine_args


def test_disabled_custom_all_reduce_disables_fused_compiler_pass() -> None:
    value = manager(
        engine_args={
            "disable_custom_all_reduce": True,
            "compilation_config": {"pass_config": {"fuse_norm_quant": True}},
        }
    )
    launch = value._launch_request(value.spec.members[0]).launch_config

    assert launch.engine_args["compilation_config"] == {
        "pass_config": {
            "fuse_norm_quant": True,
            "fuse_allreduce_rms": False,
        }
    }


def test_conflicting_untyped_revision_is_rejected() -> None:
    value = manager()
    with pytest.raises(ValueError, match="revision conflicts"):
        ReplicaManager(
            value.spec,
            cast(
                Any,
                {"host0": SimpleNamespace(), "host1": SimpleNamespace()},
            ),
            ReplicaLaunchTemplate(
                served_model_name="model@1", engine_args={"revision": "other"}
            ),
        )
