import asyncio
from collections import Counter
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from art.distributed.adapter_transport import (
    AdapterReceiveResult,
    ExternalAdapterObjectSource,
    ExternalAdapterShard,
    ExternalAdapterShardedSource,
)
from art.distributed.specs import (
    ClusterSpec,
    HostSpec,
    LocalTransferEndpoint,
    PairedTransferIdentity,
)
from art.megatron.operation_handler import (
    MegatronInferenceUpdateUsage,
    MegatronPolicyActivationTiming,
    MegatronRetainedState,
    MegatronSamplerPublicationReceipt,
)
from art.megatron.paired_inference import (
    MegatronPairedInferencePublisher,
    _adapter_transport_metrics,
    _engine_args,
    _paired_lora_transport,
    _paired_transfer_identity,
)
from art.training import CheckpointRef, SamplerWeightsResult


def _publisher() -> MegatronPairedInferencePublisher:
    publisher = object.__new__(MegatronPairedInferencePublisher)
    publisher.service = SimpleNamespace(
        leader_endpoint=SimpleNamespace(url="http://holder.test:8000")
    )
    publisher.api_key = "secret"
    return publisher


def _activated_publisher() -> MegatronPairedInferencePublisher:
    publisher = _publisher()
    publisher._lock = asyncio.Lock()
    publisher._publication_locks = {}
    publisher._active_publications = 0
    publisher._publications_idle = asyncio.Event()
    publisher._publications_idle.set()
    publisher._active_generations = {}
    publisher._active_update_sequences = {}
    publisher._retained_transfers = {}
    publisher._retained_adapter_sources = {}
    publisher._activated_publications = {}
    publisher._latest_activated_publications = {}
    publisher._exact_evaluation_publications = {}
    publisher._activated_publication_leases = Counter()
    publisher._closed = False
    return publisher


@pytest.mark.asyncio
async def test_holder_update_retries_ambiguous_transport_response(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object], dict[str, str]]] = []
    receipt = {"update_identity": "update-1"}

    class _Client:
        def __init__(self, *, timeout: float) -> None:
            assert timeout == 330.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, url, *, json, headers):
            calls.append((url, json, headers))
            if len(calls) == 1:
                raise httpx.ReadError("holder response was lost")
            return httpx.Response(
                200,
                json=receipt,
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(
        "art.megatron.paired_inference.httpx.AsyncClient",
        _Client,
    )
    payload: dict[str, object] = {"operation_id": "operation-1"}

    assert await _publisher()._post_update(payload) == receipt
    assert calls == [
        (
            "http://holder.test:8000/art/in_flight_lora_update",
            payload,
            {"Authorization": "Bearer secret"},
        ),
        (
            "http://holder.test:8000/art/in_flight_lora_update",
            payload,
            {"Authorization": "Bearer secret"},
        ),
    ]


@pytest.mark.asyncio
async def test_publication_scope_serializes_one_alias() -> None:
    publisher = _activated_publisher()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        async with publisher._publication_scope("model"):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        async with publisher._publication_scope("model"):
            second_entered.set()

    first_task = asyncio.create_task(first())
    await first_entered.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)
    assert not second_entered.is_set()
    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_publication_scope_overlaps_distinct_aliases() -> None:
    publisher = _activated_publisher()
    entered = {"first": asyncio.Event(), "second": asyncio.Event()}
    release = asyncio.Event()

    async def publish(alias: str) -> None:
        async with publisher._publication_scope(alias):
            entered[alias].set()
            await release.wait()

    tasks = tuple(asyncio.create_task(publish(alias)) for alias in ("first", "second"))
    await asyncio.gather(*(event.wait() for event in entered.values()))
    assert publisher._active_publications == 2
    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_close_waits_for_active_publication() -> None:
    publisher = _activated_publisher()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def publish() -> None:
        async with publisher._publication_scope("model"):
            entered.set()
            await release.wait()

    publication = asyncio.create_task(publish())
    await entered.wait()
    close = asyncio.create_task(publisher.aclose())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    await asyncio.gather(publication, close)
    assert publisher._closed


def _publication(
    step: int, *, mode: str = "versioned_lora"
) -> MegatronSamplerPublicationReceipt:
    operation_id = f"publication-{step}"
    generation_id = f"generation-{step}"
    runtime_lora_name = "model:active" if mode == "in_flight_lora" else f"model@{step}"
    return MegatronSamplerPublicationReceipt(
        operation_id=operation_id,
        request_id=operation_id,
        publication_mode=mode,
        requested_public_alias="model",
        runtime_model_name="base",
        runtime_lora_name=runtime_lora_name,
        serving_generation_id=generation_id,
        learner_version=step,
        policy_activation_timing=MegatronPolicyActivationTiming(
            trainer_completed_monotonic_s=float(step),
            serving_activated_monotonic_s=float(step) + 0.25,
        ),
        inference_update_usage=MegatronInferenceUpdateUsage(
            staging_s=0.1, apply_s=0.15
        ),
        paired_transfer=(
            PairedTransferIdentity(
                lora_backend="local",
                trainer_endpoints=(
                    LocalTransferEndpoint(
                        host_id="slot", domain="slot", root="/dev/shm"
                    ),
                ),
                inference_endpoints=(
                    LocalTransferEndpoint(
                        host_id="slot", domain="slot", root="/dev/shm"
                    ),
                ),
                lora_source_host_id="slot",
                route_source=LocalTransferEndpoint(
                    host_id="slot", domain="slot", root="/dev/shm"
                ),
                route_delivery="local",
            )
            if mode in {"versioned_lora", "in_flight_lora"}
            else None
        ),
        holder_update_sequence=step,
        holder_update_id=f"update-{step}",
        retained=(
            MegatronRetainedState(
                owner_id=f"lora-{step}",
                resource="lora",
                bytes=1,
                work_fingerprint=generation_id,
            ),
        ),
        result=SamplerWeightsResult(
            operation_id=operation_id,
            checkpoint=CheckpointRef(
                run_id="run",
                learner_version=step,
                checkpoint_id=f"checkpoint-{step}",
            ),
            lora=runtime_lora_name,
            metrics={"publication/policy_activation_lag_s": 0.25},
        ),
    )


def test_paired_lora_transport_follows_resolved_placement() -> None:
    cluster = ClusterSpec(
        hosts=(
            HostSpec(
                host_id="trainer",
                node_rank=0,
                worker_address="tcp://trainer:1",
                cpu_slots=1,
                local_transfer_domain="slot",
                local_transfer_root="/dev/shm/routes",
            ),
            HostSpec(
                host_id="inference",
                node_rank=1,
                worker_address="tcp://inference:1",
                cpu_slots=1,
                local_transfer_domain="slot",
                local_transfer_root="/dev/shm/routes",
            ),
        ),
        controller_host_id="trainer",
    )
    route_source = LocalTransferEndpoint(
        host_id="inference", domain="slot", root="/dev/shm/routes"
    )
    runtime = SimpleNamespace(
        topology=SimpleNamespace(cluster=cluster),
        retained_route_source=route_source,
    )
    spec = SimpleNamespace(
        trainer_mesh=SimpleNamespace(
            ranks=(SimpleNamespace(host_id="trainer"),), coordinator_rank=0
        )
    )
    service = SimpleNamespace(members=(SimpleNamespace(host_id="inference"),))

    identity = _paired_transfer_identity(runtime, spec, service)
    assert (
        _paired_lora_transport(runtime, spec, service)
        == identity.lora_backend
        == "local"
    )
    assert identity.lora_source_host_id == "trainer"
    assert identity.route_source == route_source
    assert identity.route_delivery == "local"
    split_cluster = ClusterSpec(
        hosts=(
            cluster.hosts[0],
            HostSpec(
                **cluster.hosts[1].model_dump(exclude={"local_transfer_domain"}),
                local_transfer_domain="inference-pod",
            ),
        ),
        controller_host_id="trainer",
    )
    identity = _paired_transfer_identity(
        SimpleNamespace(
            topology=SimpleNamespace(cluster=split_cluster),
            retained_route_source=LocalTransferEndpoint(
                host_id="inference",
                domain="inference-pod",
                root="/dev/shm/routes",
            ),
        ),
        spec,
        service,
    )
    assert identity.lora_backend == "nixl"
    assert identity.route_delivery == "nixl"
    assert identity.route_backend("trainer") == "nixl"


def test_paired_transport_metrics_preserve_authoritative_receive_evidence() -> None:
    received = tuple(
        AdapterReceiveResult(
            host_id=f"inference-{index}",
            generation_id="generation-1",
            path="/adapter/generation-1",
            tensor_bytes=100,
            config_bytes=10,
            materialization_s=0.2 + index / 10,
            used_bytes=100,
            capacity_bytes=128 * (index + 1),
            prepare_s=0.01 + index / 100,
            pool_wait_s=0.02 + index / 100,
            registration_s=0.03 + index / 100,
            sender_staging_s=0.04 + index / 100,
            sender_registration_s=0.05 + index / 100,
        )
        for index in range(2)
    )

    metrics = _adapter_transport_metrics(received, wait_s=0.06)

    assert metrics == pytest.approx(
        {
            "publication/adapter_transport_bytes": 200.0,
            "publication/adapter_transport_capacity_bytes": 384.0,
            "publication/adapter_transport_capacity_utilization": 200 / 384,
            "publication/adapter_transport_wait_s": 0.06,
            "publication/adapter_transport_pool_wait_s": 0.03,
            "publication/adapter_transport_prepare_s": 0.02,
            "publication/adapter_transport_registration_s": 0.04,
            "publication/adapter_transport_sender_staging_s": 0.05,
            "publication/adapter_transport_sender_registration_s": 0.06,
            "publication/adapter_materialization_s": 0.3,
        }
    )


def test_paired_engine_args_match_serving_profile_lora_rank() -> None:
    profile = SimpleNamespace(lora_rank=32, route_replay=True)
    args = _engine_args(
        {"engine_args": {"max_lora_rank": 8}},
        False,
        "Qwen/Qwen3.5-35B-A3B",
        profile=profile,
    )

    assert args["max_lora_rank"] == profile.lora_rank == 32
    assert args["generation_config"] == "vllm"
    assert args["logprobs_mode"] == "raw_logprobs"
    with pytest.raises(ValueError, match="raw model logprobs"):
        _engine_args(
            {"engine_args": {"logprobs_mode": "processed_logprobs"}},
            False,
            "Qwen/Qwen3.5-35B-A3B",
            profile=profile,
        )
    with pytest.raises(ValueError, match="vLLM generation defaults"):
        _engine_args(
            {"engine_args": {"generation_config": "auto"}},
            False,
            "Qwen/Qwen3.5-35B-A3B",
            profile=profile,
        )


@pytest.mark.asyncio
async def test_exact_publication_lease_pins_version_until_prune() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.released = []

        async def release_adapter_transfer(self, generation_id):
            self.released.append(generation_id)

    publisher = _activated_publisher()
    manager = _Manager()
    unloaded = []

    async def unload(runtime_lora_name):
        unloaded.append(runtime_lora_name)

    publisher._post_unload = unload  # type: ignore[method-assign]
    for step in (1, 2):
        receipt = _publication(step)
        assert receipt.runtime_lora_name is not None
        publisher._active_generations[receipt.runtime_lora_name] = (
            receipt.serving_generation_id
        )
        publisher._retained_transfers[receipt.runtime_lora_name] = (
            manager,
            receipt.serving_generation_id,
        )
        publisher._record_activated_publication(receipt)

    assert publisher.activated_publication("model") == _publication(2)
    async with publisher.exact_publication_lease("model", 1) as leased:
        assert leased == _publication(1)
        await publisher.prune_versioned_adapters("model", retain_steps=set())
        assert publisher.activated_publication("model", 1) == leased

    await publisher.prune_versioned_adapters("model", retain_steps=set())
    assert publisher.activated_publication("model", 1) is None
    assert publisher.activated_publication("model", 2) == _publication(2)
    assert unloaded == ["model@1"]
    assert manager.released == ["generation-1"]


def _activate_in_flight(
    publisher: MegatronPairedInferencePublisher,
    receipt: MegatronSamplerPublicationReceipt,
    manager: object,
) -> None:
    assert receipt.runtime_lora_name == "model:active"
    publisher._active_generations[receipt.runtime_lora_name] = (
        receipt.serving_generation_id
    )
    publisher._retained_transfers[receipt.runtime_lora_name] = (
        manager,
        receipt.serving_generation_id,
    )
    publisher._retained_adapter_sources[receipt.runtime_lora_name] = {
        "path": f"/adapter/{receipt.learner_version}",
        "source_identity": receipt.serving_generation_id,
        "layout": "peft_safetensors_v1",
        "model_bytes": 4096,
        "config_bytes": 512,
    }
    publisher._record_activated_publication(receipt)


@pytest.mark.asyncio
async def test_in_flight_exact_lease_does_not_block_next_activation() -> None:
    class _Manager:
        def quarantine(self, reason):
            raise AssertionError(reason)

    publisher = _activated_publisher()
    manager = _Manager()
    updates = []
    unloaded = []

    async def update(payload):
        updates.append(payload)
        return {
            "generation_id": payload["generation_id"],
            "lora_slot": payload["lora_slot"],
            "policy_version": payload["policy_version"],
            "update_seq": 7,
            "update_identity": "eval-update-1",
        }

    async def unload(runtime_lora_name):
        unloaded.append(runtime_lora_name)

    publisher._post_update = update  # type: ignore[method-assign]
    publisher._post_unload = unload  # type: ignore[method-assign]
    first = _publication(1, mode="in_flight_lora")
    _activate_in_flight(publisher, first, manager)

    async with publisher.exact_publication_lease("model", 1) as exact:
        assert exact.runtime_lora_name == "model:eval@1"
        assert exact.result.lora == "model:eval@1"
        assert exact.serving_generation_id == "generation-1"
        async with publisher.exact_publication_lease("model", 1) as shared:
            assert shared is exact
            assert publisher._activated_publication_leases[("model", 1)] == 2
        assert unloaded == []

        second = _publication(2, mode="in_flight_lora")
        _activate_in_flight(publisher, second, manager)
        assert publisher.activated_publication("model") == second
        assert publisher.activated_publication("model", 1) == first

    assert updates[0]["source"]["path"] == "/adapter/1"
    assert updates[0]["expected_generation_id"] is None
    assert unloaded == ["model:eval@1"]
    assert publisher.activated_publication("model", 1) is None


@pytest.mark.parametrize("sharded", [False, True], ids=["object", "shards"])
@pytest.mark.asyncio
async def test_external_source_uses_the_ordinary_holder_update_path(
    sharded: bool,
) -> None:
    config = json.dumps(
        {"art_lora_format": "vllm", "r": 1, "target_modules": ["q_proj"]},
        separators=(",", ":"),
    )
    model_bytes = 4096
    config_bytes = len(config.encode())
    if sharded:
        source = ExternalAdapterShardedSource(
            generation_id="generation-3",
            source_identity="manifest:" + "a" * 64,
            model_bytes=model_bytes,
            config_bytes=config_bytes,
            shards=(
                ExternalAdapterShard(
                    index=0,
                    relative_path="adapter_config.json",
                    file_offset=0,
                    object_url="https://objects.example/config?signature=secret",
                    object_bytes=config_bytes,
                    object_sha256="b" * 64,
                ),
                ExternalAdapterShard(
                    index=1,
                    relative_path="adapter_model.safetensors",
                    file_offset=0,
                    object_url="https://objects.example/model?signature=secret",
                    object_bytes=model_bytes,
                    object_sha256="c" * 64,
                ),
            ),
        )
    else:
        source = ExternalAdapterObjectSource(
            generation_id="generation-3",
            source_identity="sha256:" + "a" * 64,
            object_url="https://objects.example/adapter.safetensors?signature=secret",
            object_bytes=model_bytes,
            object_sha256="a" * 64,
            adapter_config_json=config,
            adapter_config_sha256=hashlib.sha256(config.encode()).hexdigest(),
            lora_rank=1,
            target_modules=("q_proj",),
        )

    class Manager:
        def __init__(self) -> None:
            self.released: list[str] = []

        async def materialize_external_adapter(self, value, *, timeout_s):
            assert value is source
            assert timeout_s == 300
            return (
                AdapterReceiveResult(
                    host_id="inference-0",
                    generation_id=source.generation_id,
                    path="/adapter/generation-3",
                    tensor_bytes=model_bytes,
                    config_bytes=config_bytes,
                    materialization_s=0.25,
                    used_bytes=model_bytes + config_bytes,
                    capacity_bytes=model_bytes + config_bytes,
                    source_identity=source.source_identity,
                ),
            )

        async def release_adapter_transfer(self, generation_id):
            self.released.append(generation_id)

        def quarantine(self, reason):
            raise AssertionError(reason)

    manager = Manager()
    publisher = _activated_publisher()
    publisher.runtime = SimpleNamespace(model_service=lambda _name: manager)
    publisher.service = SimpleNamespace(
        name="inference",
        members=(SimpleNamespace(host_id="inference-0"),),
        leader_endpoint=SimpleNamespace(url="http://holder.test:8000"),
    )
    publisher.config = {}
    updates = []

    async def update(payload):
        updates.append(payload)
        return {
            "generation_id": payload["generation_id"],
            "lora_slot": payload["lora_slot"],
            "policy_version": payload["policy_version"],
            "update_seq": 3,
            "update_identity": "update-3",
            "source_identity": payload["source"]["source_identity"],
            "apply_s": 0.05,
        }

    publisher._post_update = update  # type: ignore[method-assign]
    result = await publisher.apply_external_adapter(
        operation_id="operation-3",
        public_alias="model",
        generation_id=source.generation_id,
        expected_generation_id=None,
        policy_version=3,
        source=source,
    )
    replayed = await publisher.apply_external_adapter(
        operation_id="operation-3",
        public_alias="model",
        generation_id=source.generation_id,
        expected_generation_id=None,
        policy_version=3,
        source=source,
    )

    assert updates[0]["source"] == {
        "path": "/adapter/generation-3",
        "source_identity": source.source_identity,
        "layout": "peft_safetensors_v1",
        "model_bytes": model_bytes,
        "config_bytes": config_bytes,
    }
    assert result.staging_s == 0.25
    assert result.apply_s == 0.05
    assert result.transfer_bytes == model_bytes + config_bytes
    assert replayed == result
    assert len(updates) == 2
    assert publisher._active_generations["model:active"] == "generation-3"
    assert manager.released == []
