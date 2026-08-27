from types import MethodType, SimpleNamespace
from typing import Any, cast
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
from art.local import checkpoints as checkpoints_module
from art.megatron import distributed_service as service_module
from art.megatron.backend import MegatronBackend
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


def test_endpoint_url_brackets_ipv6_literals() -> None:
    assert EndpointSpec(host="2001:db8::1", port=8000).url == (
        "http://[2001:db8::1]:8000"
    )


def test_multihost_model_service_requires_routable_leader() -> None:
    with pytest.raises(ValueError, match="leader endpoint must be routable"):
        ModelServiceSpec(
            name="model",
            members=(
                ModelServiceMemberSpec(
                    member_id="node0", host_id="host0", node_rank=0, gpu_ids=(0,)
                ),
                ModelServiceMemberSpec(
                    member_id="node1", host_id="host1", node_rank=1, gpu_ids=(0,)
                ),
            ),
            leader_endpoint=EndpointSpec(host="127.0.0.1", port=8000),
            rendezvous=EndpointSpec(host="10.0.0.1", port=29500),
            base_model="base",
            runtime_fingerprint="runtime",
            parallel=VllmParallelSpec(tp=2),
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
    runtime._host_services = {"host0": object()}
    runtime._adapter_services = {"host0": object()}
    runtime._model_services = {}
    runtime._started, runtime._closed = True, False
    runtime._preflight_launch = AsyncMock()
    monkeypatch.setattr(
        runtime_module, "MonarchVllmHostLauncher", lambda *_args: object()
    )
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
    )
    service._latest_step = 5
    service._serving_step = 5
    service._managed_service_name = "model"
    service._base_url = spec.leader_endpoint.url
    service._serving_capabilities = capabilities
    service._current_lora_name = "model@5"
    service._loaded_adapter_steps = {1, 3, 5}
    service._loaded_exact_adapter_steps = {2}
    service._exact_adapter_refcounts = {2: 1}
    service._published_adapters[5] = cast(
        Any,
        SimpleNamespace(
            generation_id="policy",
            identity=str(tmp_path / "checkpoints" / "0005"),
        ),
    )
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


@pytest.mark.asyncio
async def test_recovery_uses_serving_generation_while_learner_is_ahead(
    monkeypatch, tmp_path
) -> None:
    spec = _spec()
    ready = ReplicaState(
        replica_id="model",
        generation=4,
        generation_digest="restarted",
        phase="ready",
    )
    manager = SimpleNamespace(
        restart=AsyncMock(return_value=ready),
        prepare_update=Mock(return_value=ready),
        verify_update=Mock(return_value=ready),
        quarantine=Mock(),
        stop=AsyncMock(),
    )
    service = _service(
        tmp_path,
        SimpleNamespace(
            topology=SimpleNamespace(model_services=(spec,)),
            model_service=lambda _name: manager,
        ),
    )
    capabilities = ServingCapabilities(
        runtime="art_vllm", protocol_version=ART_SERVING_PROTOCOL_VERSION
    )
    serving_path = str(tmp_path / "checkpoints" / "0005")
    service._latest_step = 6
    service._serving_step = 5
    service._managed_service_name = "model"
    service._base_url = spec.leader_endpoint.url
    service._serving_capabilities = capabilities
    service._current_lora_name = "model@5"
    service._published_adapters[5] = cast(
        Any,
        SimpleNamespace(
            generation_id="serving-generation",
            identity=serving_path,
        ),
    )
    monkeypatch.setattr(
        service_module,
        "discover_serving_capabilities",
        AsyncMock(return_value=capabilities),
    )

    await service._recover_replica_locked(
        ReplicaFailure(
            replica_id="model",
            generation=3,
            generation_digest="failed",
            reason="dead",
        )
    )

    manager.restart.assert_awaited_once_with(
        served_model_name="model@5",
        lora_path=serving_path,
        initial_policy_version=5,
        initial_generation_id="serving-generation",
    )
    report = manager.verify_update.call_args.args[0]
    assert report.policy_version == "5"
    assert report.policy_digest == "serving-generation"
    assert service._latest_step == 6
    assert service._serving_step == 5
    assert service._loaded_adapter_steps == {5}


@pytest.mark.asyncio
async def test_retention_protects_absent_learner_and_serving_steps(
    monkeypatch, tmp_path
) -> None:
    model = SimpleNamespace(
        project="project", name="model", _storage_name=lambda: "model"
    )
    output_dir = tmp_path / "project" / "models" / "model"
    service = _service(output_dir, SimpleNamespace())
    service._latest_step = 3
    service._serving_step = 2
    service._loaded_adapter_steps = {1, 2, 3}
    service._unload_adapter = AsyncMock()

    await service.prune_loaded_adapters(retain_steps={3})
    assert service._loaded_adapter_steps == {2, 3}
    service._unload_adapter.assert_awaited_once_with("model@1")

    checkpoints = output_dir / "checkpoints"
    for step in (1, 2, 4):
        (checkpoints / f"{step:04d}").mkdir(parents=True)
    staging = output_dir / "staging-0003"
    staging.mkdir()
    original_delete = checkpoints_module.delete_checkpoints

    def publish_during_retention(path: str, excluding: list[int]) -> None:
        staging.rename(checkpoints / "0003")
        original_delete(path, excluding)

    monkeypatch.setattr(
        checkpoints_module, "delete_checkpoints", publish_during_retention
    )
    backend = object.__new__(MegatronBackend)
    backend._runtime = object()
    backend._path = str(tmp_path)
    backend._training_clients = {}

    async def get_service(_self, _model):
        return service

    backend._get_service = MethodType(get_service, backend)
    await backend._delete_checkpoint_files(model, [1])
    assert (checkpoints / "0001").is_dir()
    assert (checkpoints / "0002").is_dir()
    assert (checkpoints / "0003").is_dir()
    assert not (checkpoints / "0004").exists()
