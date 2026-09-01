import asyncio
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest

from art.megatron.operation_handler import (
    MegatronInferenceUpdateUsage,
    MegatronPolicyActivationTiming,
    MegatronRetainedState,
    MegatronSamplerPublicationReceipt,
)
from art.megatron.paired_inference import (
    MegatronPairedInferencePublisher,
    _engine_args,
    _paired_lora_transport,
)
from art.training import CheckpointRef, SamplerWeightsResult


class _ReceiptClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[tuple[str, dict[str, object]]] = []

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str] | None,
        timeout: float,
    ) -> httpx.Response:
        del headers, timeout
        self.requests.append((url, json))
        return self.response


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
    publisher._active_generations = {}
    publisher._retained_transfers = {}
    publisher._retained_adapter_paths = {}
    publisher._activated_publications = {}
    publisher._latest_activated_publications = {}
    publisher._exact_evaluation_publications = {}
    publisher._activated_publication_leases = Counter()
    publisher._closed = False
    return publisher


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
    spec = SimpleNamespace(
        trainer_mesh=SimpleNamespace(ranks=(SimpleNamespace(host_id="trainer"),))
    )

    assert (
        _paired_lora_transport(
            spec,
            SimpleNamespace(members=(SimpleNamespace(host_id="trainer"),)),
        )
        == "local"
    )
    assert (
        _paired_lora_transport(
            spec,
            SimpleNamespace(members=(SimpleNamespace(host_id="inference"),)),
        )
        == "nixl"
    )


def test_paired_engine_args_require_raw_vllm_sampling_defaults() -> None:
    args = _engine_args(
        {},
        False,
        "Qwen/Qwen3.5-35B-A3B",
        enable_moe_routing_replay=True,
    )

    assert args["generation_config"] == "vllm"
    assert args["logprobs_mode"] == "raw_logprobs"
    with pytest.raises(ValueError, match="raw model logprobs"):
        _engine_args(
            {"engine_args": {"logprobs_mode": "processed_logprobs"}},
            False,
            "Qwen/Qwen3.5-35B-A3B",
            enable_moe_routing_replay=True,
        )
    with pytest.raises(ValueError, match="vLLM generation defaults"):
        _engine_args(
            {"engine_args": {"generation_config": "auto"}},
            False,
            "Qwen/Qwen3.5-35B-A3B",
            enable_moe_routing_replay=True,
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
    publisher._retained_adapter_paths[receipt.runtime_lora_name] = (
        f"/adapter/{receipt.learner_version}"
    )
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

    assert updates[0]["lora_path"] == "/adapter/1"
    assert updates[0]["expected_generation_id"] is None
    assert unloaded == ["model:eval@1"]
    assert publisher.activated_publication("model", 1) is None


@pytest.mark.asyncio
async def test_lost_update_response_recovers_exact_holder_receipt() -> None:
    holder_result = {
        "status": "updated",
        "generation_id": "generation-2",
        "update_seq": 2,
        "update_identity": "update-2",
        "apply_s": 0.125,
    }
    request = httpx.Request(
        "POST", "http://holder.test:8000/art/in_flight_lora_update/receipt"
    )
    client = _ReceiptClient(
        httpx.Response(
            200,
            request=request,
            json={
                "operation_id": "operation-2",
                "state": "settled",
                "response_status": 200,
                "response": holder_result,
            },
        )
    )
    payload: dict[str, object] = {
        "operation_id": "operation-2",
        "generation_id": "generation-2",
    }

    recovered = await _publisher()._recover_update_receipt(
        client,
        payload,
        httpx.ReadError("response lost", request=request),
    )

    assert recovered == holder_result
    assert client.requests == [
        (
            "http://holder.test:8000/art/in_flight_lora_update/receipt",
            payload,
        )
    ]


@pytest.mark.asyncio
async def test_ambiguous_holder_receipt_does_not_reexecute_update() -> None:
    request = httpx.Request(
        "POST", "http://holder.test:8000/art/in_flight_lora_update/receipt"
    )
    client = _ReceiptClient(
        httpx.Response(
            200,
            request=request,
            json={"operation_id": "operation-2", "state": "ambiguous"},
        )
    )

    with pytest.raises(RuntimeError, match="outcome is ambiguous"):
        await _publisher()._recover_update_receipt(
            client,
            {"operation_id": "operation-2"},
            httpx.ReadError("response lost", request=request),
        )

    assert len(client.requests) == 1
