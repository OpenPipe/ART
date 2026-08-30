from __future__ import annotations

import asyncio
import hashlib
from http import HTTPStatus
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, cast

import httpx

from art import dev
from art.adapter_leases import in_flight_lora_name
from art.distributed.adapter_transport import (
    AdapterReceiveResult,
    AdapterTransferTarget,
)
from art.distributed.art_runtime import ArtRuntime
from art.distributed.specs import ModelServiceSpec
from art.distributed.vllm_replica import ReplicaLaunchTemplate
from art.serving_capabilities import (
    PairedInferenceEndpoint,
    ServingCapabilities,
    ServingProfileIdentity,
    discover_serving_capabilities,
)
from art.training import (
    CheckpointRef,
    OperationRef,
    SamplerWeightsResult,
    SaveWeightsForSamplerRequest,
)
from art.vllm_runtime import map_checkpoint_path_for_vllm

from .model_support import get_model_support_handler
from .operation_handler import (
    POLICY_ACTIVATION_LAG_METRIC,
    MegatronArtifactResourcePlan,
    MegatronInferenceUpdateUsage,
    MegatronPolicyActivationTiming,
    MegatronRetainedState,
    MegatronSamplerPublicationReceipt,
)
from .runtime.publication import TrainerRankPublication
from .runtime.specs import (
    CommandPublicationSpec,
    TrainerGeneration,
    TrainerRuntimeSpec,
)


class MegatronPairedInferencePublisher:
    """Publish shared-trainer snapshots into the paired vLLM holder."""

    def __init__(
        self,
        runtime: ArtRuntime,
        trainer: Any,
        service: ModelServiceSpec,
        config: dev.BackendModelConfig,
        endpoint: PairedInferenceEndpoint,
        *,
        api_key: str | None,
    ) -> None:
        self.runtime = runtime
        self.trainer = trainer
        self.service = service
        self.config = config
        self.endpoint = endpoint
        self.api_key = api_key
        self._lock = asyncio.Lock()
        self._active_generations: dict[str, str] = {}
        self._retained_transfers: dict[str, tuple[Any, str]] = {}
        self._closed = False

    @classmethod
    async def start(
        cls,
        runtime: ArtRuntime,
        trainer: Any,
        *,
        base_model: str,
        config: dev.BackendModelConfig,
        runtime_spec: TrainerRuntimeSpec,
    ) -> "MegatronPairedInferencePublisher":
        services = tuple(
            service
            for service in runtime.topology.model_services
            if service.base_model == base_model
        )
        if len(services) != 1:
            raise RuntimeError("paired slot requires one matching inference service")
        service = services[0]
        profile = _serving_profile_identity(runtime, runtime_spec, base_model, service)
        template = ReplicaLaunchTemplate(
            served_model_name=service.name,
            engine_args=_engine_args(
                config,
                service.temporal_gpu_sharing,
                base_model,
                enable_moe_routing_replay=runtime_spec.enable_moe_routing_replay,
            ),
            server_args=_server_args(config, base_model),
            serving_profile_identity=profile,
        )
        started = False
        try:
            await runtime.start_model_service(service, template)
            started = True
            capabilities = await discover_serving_capabilities(
                base_url=service.leader_endpoint.url,
                headers=_headers(_api_key(config)),
                allow_openai_compatible=False,
            )
            _validate_serving_profile(capabilities, profile, service)
            manager = runtime.model_service(service.name)
            credentials = manager.dispatch_credentials
            endpoint = PairedInferenceEndpoint(
                url=(
                    f"{service.leader_endpoint.url.rstrip('/')}"
                    "/art/internal/v1/chat/completions"
                ),
                target_id=credentials.target_id,
                runtime_generation=credentials.runtime_generation,
                runtime_source_id=credentials.runtime_source_id,
                runtime_source_epoch=credentials.runtime_source_epoch,
                authorization_token=credentials.authorization_token,
                profile=cast(Any, capabilities.profile),
                fast_metrics=capabilities.fast_metrics,
            )
        except BaseException as error:
            if started:
                try:
                    await runtime.stop_model_service(service.name)
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "paired inference startup and rollback failed",
                        [error, cleanup_error],
                    ) from None
            raise
        return cls(
            runtime,
            trainer,
            service,
            config,
            endpoint,
            api_key=_api_key(config),
        )

    async def save_weights_for_sampler(
        self,
        request: SaveWeightsForSamplerRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
        *,
        template_adapter_path: str,
        optimizer_state_path: str,
        staging_adapter_path: str,
    ) -> MegatronSamplerPublicationReceipt:
        if request.publication.mode not in {"versioned_lora", "in_flight_lora"}:
            raise ValueError("paired publisher received another publication mode")
        alias = request.publication.model_alias
        if alias is None:
            raise ValueError("paired publication requires a model alias")
        async with self._lock:
            self._require_open()
            manager = self.runtime.model_service(self.service.name)
            await self._release_pending_transfers()
            targets = await manager.prepare_adapter_transfer(
                generation.generation_id,
                template_adapter_path,
                transport=_paired_lora_transport(
                    self.trainer.runtime_spec, self.service
                ),
            )
            if not targets:
                raise RuntimeError("paired inference returned no transfer targets")
            rank_task = asyncio.create_task(
                self.trainer.publish_command_generation(
                    CommandPublicationSpec(
                        run_id=operation.run_id,
                        generation=generation,
                        optimizer_state_path=optimizer_state_path,
                        staging_adapter_path=staging_adapter_path,
                        publication_targets=targets,
                    )
                )
            )
            receive_task = asyncio.create_task(
                manager.wait_adapter_transfer(generation.generation_id)
            )
            try:
                (records, publication_metrics), received = await asyncio.gather(
                    rank_task, receive_task
                )
                _validate_rank_publications(records, generation, self.trainer)
                checkpoint, tensor_bytes, config_bytes = _validate_received(
                    received, generation, targets
                )
            except BaseException as error:
                for task in (rank_task, receive_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(rank_task, receive_task, return_exceptions=True)
                try:
                    await asyncio.shield(
                        manager.release_adapter_transfer(generation.generation_id)
                    )
                except BaseException as cleanup_error:
                    error.add_note(
                        "paired adapter transfer cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise

            trainer_completed = time.monotonic()
            runtime_lora_name = (
                in_flight_lora_name(alias)
                if request.publication.mode == "in_flight_lora"
                else f"{alias}@{generation.policy_step}"
            )
            expected_generation = (
                self._active_generations.get(runtime_lora_name)
                if request.publication.mode == "in_flight_lora"
                else None
            )
            payload: dict[str, object] = {
                "operation_id": operation.operation_id,
                "model_name": runtime_lora_name,
                "lora_slot": runtime_lora_name,
                "lora_path": map_checkpoint_path_for_vllm(self.config, checkpoint),
                "generation_id": generation.generation_id,
                "expected_generation_id": expected_generation,
                "policy_version": generation.policy_step,
            }
            try:
                response = await self._post_update(payload)
                update_apply_s = _response_float(response, "apply_s")
                update_sequence = _response_int(response, "update_seq")
                update_identity = str(response["update_identity"])
                if (
                    response.get("generation_id") != generation.generation_id
                    or response.get("lora_slot") != runtime_lora_name
                    or _response_int(response, "policy_version")
                    != generation.policy_step
                ):
                    manager.quarantine("paired holder returned another policy identity")
                    raise RuntimeError("paired holder changed policy identity")
            except BaseException as error:
                try:
                    await asyncio.shield(
                        manager.release_adapter_transfer(generation.generation_id)
                    )
                except BaseException as cleanup_error:
                    error.add_note(
                        "paired adapter transfer cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise
            self._active_generations[runtime_lora_name] = generation.generation_id
            activated = time.monotonic()
            if request.publication.mode == "versioned_lora":
                self._retained_transfers[runtime_lora_name] = (
                    manager,
                    generation.generation_id,
                )
            else:
                try:
                    await manager.release_adapter_transfer(generation.generation_id)
                except BaseException:
                    self._retained_transfers[
                        f"release-pending:{generation.generation_id}"
                    ] = (manager, generation.generation_id)

            lag = MegatronPolicyActivationTiming(
                trainer_completed_monotonic_s=trainer_completed,
                serving_activated_monotonic_s=activated,
            )
            logical_bytes = tensor_bytes + config_bytes
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "generation": generation.generation_id,
                        "lora": runtime_lora_name,
                        "tensor_bytes": tensor_bytes,
                        "config_bytes": config_bytes,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            return MegatronSamplerPublicationReceipt(
                operation_id=operation.operation_id,
                request_id=request.request_id,
                publication_mode=cast(
                    Literal["versioned_lora", "in_flight_lora"],
                    request.publication.mode,
                ),
                requested_public_alias=alias,
                runtime_model_name=self.service.base_model,
                runtime_lora_name=runtime_lora_name,
                serving_generation_id=generation.generation_id,
                learner_version=generation.policy_step,
                policy_activation_timing=lag,
                inference_update_usage=MegatronInferenceUpdateUsage(
                    staging_s=max(float(item.materialization_s) for item in received),
                    apply_s=update_apply_s,
                ),
                holder_update_sequence=update_sequence,
                holder_update_id=update_identity,
                retained=(
                    MegatronRetainedState(
                        owner_id=f"paired-lora:{fingerprint}",
                        resource="lora",
                        bytes=logical_bytes,
                        work_fingerprint=fingerprint,
                    ),
                ),
                result=SamplerWeightsResult(
                    operation_id=operation.operation_id,
                    checkpoint=CheckpointRef(
                        run_id=operation.run_id,
                        learner_version=generation.policy_step,
                        checkpoint_id=request.checkpoint_name,
                    ),
                    lora=runtime_lora_name,
                    metrics={
                        POLICY_ACTIVATION_LAG_METRIC: lag.activation_lag_s,
                        "publication/adapter_transport_bytes": float(
                            tensor_bytes * len(received)
                        ),
                        "publication/adapter_materialization_s": max(
                            float(item.materialization_s) for item in received
                        ),
                        **publication_metrics,
                    },
                ),
            )

    async def plan_artifacts(
        self,
        request: SaveWeightsForSamplerRequest,
        generation: TrainerGeneration,
        *,
        template_adapter_path: str,
    ) -> MegatronArtifactResourcePlan:
        if request.publication.mode not in {"versioned_lora", "in_flight_lora"}:
            raise ValueError("paired publisher received another publication mode")
        del generation
        root = Path(template_adapter_path)
        tensor_bytes = (root / "adapter_model.safetensors").stat().st_size
        config_bytes = (root / "adapter_config.json").stat().st_size
        logical_bytes = tensor_bytes + config_bytes
        return MegatronArtifactResourcePlan(
            basis="bounded",
            checkpoint_objects=1,
            lora_bytes=logical_bytes,
            transfer_bytes=logical_bytes * len(self.service.members),
            storage_bytes=0,
        )

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            retained, self._retained_transfers = self._retained_transfers, {}
            for manager, generation_id in retained.values():
                await manager.release_adapter_transfer(generation_id)

    async def _post_update(self, payload: dict[str, object]) -> dict[str, object]:
        url = f"{self.service.leader_endpoint.url}/art/in_flight_lora_update"
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=_headers(self.api_key),
                )
            except httpx.TransportError as error:
                return await self._recover_update_receipt(client, payload, error)
        response.raise_for_status()
        return _response_object(response)

    async def _recover_update_receipt(
        self,
        client: Any,
        payload: dict[str, object],
        original_error: httpx.TransportError,
    ) -> dict[str, object]:
        url = f"{self.service.leader_endpoint.url}/art/in_flight_lora_update/receipt"
        deadline = asyncio.get_running_loop().time() + 5.0
        last_error: BaseException = original_error
        while True:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=_headers(self.api_key),
                    timeout=1.0,
                )
                if response.status_code == HTTPStatus.NOT_FOUND.value:
                    state = "missing"
                else:
                    response.raise_for_status()
                    receipt = _response_object(response)
                    state = str(receipt.get("state"))
                    if state == "settled":
                        status = receipt.get("response_status")
                        value = receipt.get("response")
                        if (
                            isinstance(status, bool)
                            or not isinstance(status, int)
                            or not isinstance(value, dict)
                        ):
                            raise RuntimeError(
                                "paired holder returned an invalid update receipt"
                            )
                        if not 200 <= status < 300:
                            raise RuntimeError(
                                "paired holder update failed before its response "
                                f"was received: {value!r}"
                            ) from original_error
                        return cast(dict[str, object], value)
                    if state == "ambiguous":
                        raise RuntimeError(
                            "paired holder update outcome is ambiguous"
                        ) from original_error
            except httpx.TransportError as error:
                last_error = error
                state = "unreachable"
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"paired holder update outcome could not be reconciled ({state})"
                ) from last_error
            await asyncio.sleep(0.05)

    async def _release_pending_transfers(self) -> None:
        for name, (manager, generation_id) in tuple(self._retained_transfers.items()):
            if not name.startswith("release-pending:"):
                continue
            await manager.release_adapter_transfer(generation_id)
            self._retained_transfers.pop(name)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("paired inference publisher is closed")


def _validate_rank_publications(
    records: tuple[TrainerRankPublication, ...],
    generation: TrainerGeneration,
    trainer: Any,
) -> None:
    ordered = tuple(sorted(records, key=lambda item: item.rank))
    expected_ranks = tuple(range(len(trainer.runtime_spec.trainer_mesh.ranks)))
    if tuple(item.rank for item in ordered) != expected_ranks or {
        item.generation for item in ordered
    } != {generation}:
        raise RuntimeError("trainer did not publish every paired rank")


def _validate_received(
    received: tuple[AdapterReceiveResult, ...],
    generation: TrainerGeneration,
    targets: tuple[AdapterTransferTarget, ...],
) -> tuple[str, int, int]:
    if len(received) != len(targets) or not received:
        raise RuntimeError("not every inference host received the adapter")
    if {item.generation_id for item in received} != {generation.generation_id}:
        raise RuntimeError("inference hosts received another generation")
    paths = {str(item.path) for item in received}
    sizes = {(int(item.tensor_bytes), int(item.config_bytes)) for item in received}
    if len(paths) != 1 or len(sizes) != 1:
        raise RuntimeError("inference hosts materialized different adapters")
    tensor_bytes, config_bytes = sizes.pop()
    return paths.pop(), tensor_bytes, config_bytes


def _serving_profile_identity(
    runtime: ArtRuntime,
    spec: TrainerRuntimeSpec,
    base_model: str,
    service: ModelServiceSpec,
) -> ServingProfileIdentity:
    retained_routes = runtime.retained_route_prefetch_enabled
    return ServingProfileIdentity(
        base_model=base_model,
        model_identifier=spec.model_source,
        model_revision=spec.model_revision,
        model_support_key=spec.model_support_key,
        handler_name=spec.handler_name,
        lora_rank=spec.lora_rank,
        lora_alpha=spec.lora_alpha,
        lora_target_modules=spec.lora_target_modules,
        trainer_dtype=spec.dtype,
        route_replay=spec.enable_moe_routing_replay,
        lora_transport=_paired_lora_transport(spec, service),
        retained_route_transport=runtime.retained_route_transport,
        retained_route_max_bytes=(
            runtime.config.route_bundle_prefetch_capacity_bytes
            if retained_routes
            else 0
        ),
        retained_route_max_bundles=(
            runtime.config.route_bundle_prefetch_max_bundles if retained_routes else 0
        ),
    )


def _paired_lora_transport(
    spec: TrainerRuntimeSpec, service: ModelServiceSpec
) -> Literal["local", "nixl"]:
    trainer_host = spec.trainer_mesh.ranks[0].host_id
    inference_hosts = {member.host_id for member in service.members}
    return "local" if inference_hosts == {trainer_host} else "nixl"


def _validate_serving_profile(
    capabilities: ServingCapabilities,
    identity: ServingProfileIdentity,
    service: ModelServiceSpec,
) -> None:
    profile = capabilities.profile
    if profile is None or profile.identity != identity:
        raise RuntimeError("vLLM returned the wrong serving profile identity")
    actual_parallel = (
        profile.tensor_parallel_size,
        profile.pipeline_parallel_size,
        profile.data_parallel_size,
        profile.enable_expert_parallel,
    )
    expected_parallel = (
        service.parallel.tp,
        service.parallel.pp,
        service.parallel.dp,
        service.parallel.enable_expert_parallel,
    )
    exact_features = (
        capabilities.inplace_lora_load,
        capabilities.in_flight_lora_updates,
        capabilities.policy_token_spans,
        capabilities.fast_metrics is not None,
        capabilities.binary_routed_experts == identity.route_replay,
    )
    if actual_parallel != expected_parallel or not all(exact_features):
        raise RuntimeError("vLLM returned an incompatible paired serving profile")


def _engine_args(
    config: dev.BackendModelConfig,
    temporal_gpu_sharing: bool,
    base_model: str,
    *,
    enable_moe_routing_replay: bool,
) -> dict[str, object]:
    allow_unvalidated = bool(config.get("allow_unvalidated_arch", False))
    handler = get_model_support_handler(
        base_model, allow_unvalidated_arch=allow_unvalidated
    )
    values = dict(config.get("engine_args", {}))
    for key, value in handler.vllm_engine_args().items():
        values.setdefault(key, value)
    values["enable_sleep_mode"] = temporal_gpu_sharing
    values["enable_lora"] = True
    values["enable_return_routed_experts"] = enable_moe_routing_replay
    values.setdefault("max_loras", 2)
    values.setdefault("generation_config", "vllm")
    for key in ("model", "served_model_name"):
        values.pop(key, None)
    return values


def _server_args(
    config: dev.BackendModelConfig,
    base_model: str,
) -> dict[str, object]:
    allow_unvalidated = bool(config.get("allow_unvalidated_arch", False))
    handler = get_model_support_handler(
        base_model, allow_unvalidated_arch=allow_unvalidated
    )
    values: dict[str, object] = {
        "return_tokens_as_token_ids": True,
        "enable_auto_tool_choice": True,
        "tool_call_parser": "hermes",
        **handler.vllm_server_args(),
        **dict(config.get("server_args", {})),
    }
    for key in ("port", "host", "lora_modules"):
        values.pop(key, None)
    return values


def _api_key(config: dev.BackendModelConfig) -> str | None:
    value = dict(config.get("server_args", {})).get("api_key")
    return None if value is None else str(value)


def _headers(api_key: str | None) -> dict[str, str] | None:
    return None if api_key is None else {"Authorization": f"Bearer {api_key}"}


def _response_object(response: httpx.Response) -> dict[str, object]:
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("paired holder returned a non-object receipt")
    return cast(dict[str, object], value)


def _response_float(response: dict[str, object], field: str) -> float:
    value = response.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RuntimeError(f"paired holder returned invalid {field}")
    return float(value)


def _response_int(response: dict[str, object], field: str) -> int:
    value = response.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"paired holder returned invalid {field}")
    return value
