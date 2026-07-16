from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from art.distributed import art_runtime as runtime_module
from art.distributed.art_runtime import ArtRuntime
from art.distributed.specs import (
    EndpointSpec,
    ModelServiceMemberSpec,
    ModelServiceSpec,
    VllmParallelSpec,
)
from art.distributed.vllm_replica import ReplicaFailure, ReplicaState
from art.megatron import distributed_service as service_module
from art.megatron.distributed_service import DistributedMegatronService
from art.serving_capabilities import ART_SERVING_PROTOCOL_VERSION, ServingCapabilities


def _spec() -> ModelServiceSpec:
    return ModelServiceSpec(
        name="model",
        members=(
            ModelServiceMemberSpec(
                member_id="node0", host_id="host0", node_rank=0, gpu_ids=(0,)
            ),
        ),
        leader_endpoint=EndpointSpec(host="10.0.0.1", port=8000),
        rendezvous=EndpointSpec(host="10.0.0.1", port=29500),
        base_model="base",
        model_revision="revision",
        runtime_fingerprint="runtime",
        parallel=VllmParallelSpec(),
        update_mode="lora",
    )


def _service(tmp_path, runtime) -> DistributedMegatronService:
    return DistributedMegatronService(
        model_name="model",
        base_model="base",
        config={},
        output_dir=str(tmp_path),
        runtime=runtime,
        enable_expert_replay=False,
    )


@pytest.mark.asyncio
async def test_runtime_retains_model_service_until_stop_succeeds(monkeypatch) -> None:
    spec = _spec()
    manager = SimpleNamespace(
        start=AsyncMock(side_effect=RuntimeError("start failed")),
        stop=AsyncMock(side_effect=RuntimeError("stop failed")),
    )
    runtime = ArtRuntime.__new__(ArtRuntime)
    runtime.topology = SimpleNamespace(
        model_services=(spec,),
        cluster=SimpleNamespace(startup_timeout_s=1, rpc_timeout_s=1),
    )
    runtime._host_actors = {"host0": object()}
    runtime._model_services = {}
    runtime._started, runtime._closed = True, False
    runtime._preflight_launch = AsyncMock()
    monkeypatch.setattr(runtime_module, "MonarchVllmHostLauncher", lambda _: object())
    monkeypatch.setattr(runtime_module, "ReplicaManager", lambda *_a, **_kw: manager)

    with pytest.raises(RuntimeError, match="start failed"):
        await runtime.start_model_service(spec, SimpleNamespace())
    assert runtime.model_service("model") is manager

    with pytest.raises(RuntimeError, match="stop failed"):
        await runtime.stop_model_service("model")
    assert runtime.model_service("model") is manager

    manager.stop = AsyncMock(return_value="stopped")
    assert await runtime.stop_model_service("model") == "stopped"
    with pytest.raises(RuntimeError, match="not managed"):
        runtime.model_service("model")


@pytest.mark.asyncio
async def test_failed_recovery_unpublishes_dead_endpoint(tmp_path) -> None:
    failure = ReplicaFailure(
        replica_id="model", generation=2, generation_digest="digest", reason="dead"
    )
    manager = SimpleNamespace(
        state=ReplicaState(
            replica_id="model",
            generation=2,
            generation_digest="digest",
            phase="quarantined",
        )
    )
    service = _service(
        tmp_path,
        SimpleNamespace(model_service=lambda _name: manager),
    )
    service._managed_service_name = "model"
    service._base_url = "http://10.0.0.1:8000"
    service._loaded_adapter_steps = {1, 2}
    service._loaded_exact_adapter_steps = {1}
    service._recover_replica_locked = AsyncMock(
        side_effect=RuntimeError("restart failed")
    )

    await service._recover_failed_replica(failure)

    assert service._managed_service_name == "model"
    assert service._base_url is None
    assert not service._loaded_adapter_steps
    assert not service._loaded_exact_adapter_steps
    with pytest.raises(RuntimeError, match="unavailable"):
        await service.start_openai_server(None)


@pytest.mark.asyncio
async def test_recovery_rebuilds_loaded_adapter_index(monkeypatch, tmp_path) -> None:
    spec = _spec()
    ready = ReplicaState(
        replica_id="model",
        generation=3,
        generation_digest="generation",
        phase="ready",
    )
    manager = SimpleNamespace(
        restart=AsyncMock(return_value=ready),
        prepare_update=Mock(return_value=ready),
        verify_update=Mock(return_value=ready),
        quarantine=Mock(),
        stop=AsyncMock(),
    )
    runtime = SimpleNamespace(
        topology=SimpleNamespace(model_services=(spec,)),
        model_service=lambda _name: manager,
    )
    service = _service(tmp_path, runtime)
    capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        exact_lora_worker_state=True,
    )
    service._latest_step = 5
    service._managed_service_name = "model"
    service._base_url = spec.leader_endpoint.url
    service._serving_capabilities = capabilities
    service._current_lora_name = "model@5"
    service._loaded_adapter_steps = {1, 3, 5}
    service._loaded_exact_adapter_steps = {2}
    service._exact_adapter_refcounts = {2: 1}
    service._checkpoint_digest = AsyncMock(return_value="policy")
    service._acknowledge_lora_workers = AsyncMock()
    service._load_adapter_at = AsyncMock(return_value=("model@2", "/step/2"))
    monkeypatch.setattr(
        service_module,
        "discover_serving_capabilities",
        AsyncMock(return_value=capabilities),
    )

    await service._recover_replica_locked(
        ReplicaFailure(
            replica_id="model",
            generation=2,
            generation_digest="old",
            reason="dead",
        )
    )

    assert service._loaded_adapter_steps == {5}
    assert service._loaded_exact_adapter_steps == {2}
    assert service._exact_adapter_refcounts == {2: 1}
    service._load_adapter_at.assert_awaited_once()
