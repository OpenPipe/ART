from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
from http import HTTPStatus
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, AsyncIterator, Literal, cast
import uuid

import httpx

from art import dev
from art.adapter_leases import in_flight_lora_name
from art.distributed.adapter_transport import (
    AdapterReceiveResult,
    AdapterTransferTarget,
    ExternalAdapterObjectSource,
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

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MegatronExternalLoraApplyResult:
    runtime_lora_name: str
    generation_id: str
    policy_version: int
    holder_update_sequence: int
    holder_update_id: str
    source_identity: str
    tensor_bytes: int
    config_bytes: int
    staging_s: float
    apply_s: float


class MegatronPairedInferencePublisher:
    """Publish shared-trainer snapshots into the paired vLLM holder."""

    def __init__(
        self,
        runtime: ArtRuntime,
        trainer: Any,
        service: ModelServiceSpec,
        config: dev.BackendModelConfig,
        endpoint: PairedInferenceEndpoint,
        capabilities: ServingCapabilities,
        *,
        api_key: str | None,
    ) -> None:
        self.runtime = runtime
        self.trainer = trainer
        self.service = service
        self.config = config
        self.endpoint = endpoint
        self.capabilities = capabilities
        self.architecture_attestation = endpoint.profile.architecture
        self.api_key = api_key
        self._lock = asyncio.Lock()
        self._publication_locks: dict[str, asyncio.Lock] = {}
        self._active_publications = 0
        self._publications_idle = asyncio.Event()
        self._publications_idle.set()
        self._active_generations: dict[str, str] = {}
        self._active_update_sequences: dict[str, int] = {}
        self._retained_transfers: dict[str, tuple[Any, str]] = {}
        self._retained_adapter_sources: dict[str, dict[str, object]] = {}
        self._activated_publications: dict[
            tuple[str, int], MegatronSamplerPublicationReceipt
        ] = {}
        self._latest_activated_publications: dict[
            str, MegatronSamplerPublicationReceipt
        ] = {}
        self._exact_evaluation_publications: dict[
            tuple[str, int], MegatronSamplerPublicationReceipt
        ] = {}
        self._activated_publication_leases: Counter[tuple[str, int]] = Counter()
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
            capabilities,
            api_key=_api_key(config),
        )

    def activated_publication(
        self,
        public_alias: str,
        learner_version: int | None = None,
    ) -> MegatronSamplerPublicationReceipt | None:
        """Return the exact retained receipt for an activated public alias."""

        self._require_open()
        if learner_version is None:
            return self._latest_activated_publications.get(public_alias)
        return self._activated_publications.get((public_alias, learner_version))

    @asynccontextmanager
    async def _publication_scope(self, public_alias: str) -> AsyncIterator[None]:
        alias_lock = self._publication_locks.setdefault(public_alias, asyncio.Lock())
        async with alias_lock:
            async with self._lock:
                self._require_open()
                self._active_publications += 1
                self._publications_idle.clear()
            try:
                yield
            finally:
                async with self._lock:
                    self._active_publications -= 1
                    if self._active_publications == 0:
                        self._publications_idle.set()

    @asynccontextmanager
    async def exact_publication_lease(
        self,
        public_alias: str,
        learner_version: int,
    ) -> AsyncIterator[MegatronSamplerPublicationReceipt]:
        """Lease the exact immutable holder alias for one activated learner."""

        key = (public_alias, learner_version)
        async with self._lock:
            self._require_open()
            receipt = self._activated_publications.get(key)
            if receipt is None:
                raise RuntimeError(
                    "paired inference has no activated publication for "
                    f"{public_alias!r} at learner version {learner_version}"
                )
            receipt = await self._acquire_exact_publication(key, receipt)
        try:
            yield receipt
        finally:
            async with self._lock:
                await self._release_exact_publication(key, receipt)

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
        publication_mode = cast(
            Literal["versioned_lora", "in_flight_lora"], request.publication.mode
        )
        runtime_lora_name = (
            in_flight_lora_name(alias)
            if publication_mode == "in_flight_lora"
            else f"{alias}@{generation.policy_step}"
        )
        publication_key = (alias, generation.policy_step)
        async with self._publication_scope(alias):
            self._require_open()
            prior = self._activated_publications.get(publication_key)
            if prior is not None:
                if (
                    prior.operation_id != operation.operation_id
                    or prior.request_id != request.request_id
                ):
                    raise RuntimeError("learner publication identity changed")
                prior.validate_command(request, operation, generation)
                if prior.result.checkpoint.checkpoint_id != request.checkpoint_name:
                    raise RuntimeError(
                        "sampler publication receipt changed checkpoint identity"
                    )
                return prior
            latest = self._latest_activated_publications.get(alias)
            if latest is not None and generation.policy_step <= latest.learner_version:
                raise RuntimeError(
                    "paired publication learner lineage is not monotonic"
                )
            manager = self.runtime.model_service(self.service.name)
            async with self._lock:
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
                    self._retain_pending_transfer(manager, generation.generation_id)
                    error.add_note(
                        "paired adapter transfer cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise

            trainer_completed = time.monotonic()
            expected_generation = (
                self._active_generations.get(runtime_lora_name)
                if publication_mode == "in_flight_lora"
                else None
            )
            payload: dict[str, object] = {
                "operation_id": operation.operation_id,
                "model_name": runtime_lora_name,
                "lora_slot": runtime_lora_name,
                "source": {
                    "path": map_checkpoint_path_for_vllm(self.config, checkpoint),
                    "source_identity": generation.generation_id,
                    "layout": "peft_safetensors_v1",
                    "model_bytes": tensor_bytes,
                    "config_bytes": config_bytes,
                },
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
                    self._retain_pending_transfer(manager, generation.generation_id)
                    error.add_note(
                        "paired adapter transfer cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise
            activated = time.monotonic()
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
            receipt = MegatronSamplerPublicationReceipt(
                operation_id=operation.operation_id,
                request_id=request.request_id,
                publication_mode=publication_mode,
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
            async with self._lock:
                prior_transfer = self._retained_transfers.get(runtime_lora_name)
                self._active_generations[runtime_lora_name] = generation.generation_id
                self._active_update_sequences[runtime_lora_name] = update_sequence
                self._retained_transfers[runtime_lora_name] = (
                    manager,
                    generation.generation_id,
                )
                self._retained_adapter_sources[runtime_lora_name] = cast(
                    dict[str, object], payload["source"]
                )
                self._record_activated_publication(receipt)

            transfer_to_release = (
                prior_transfer
                if prior_transfer is not None
                and prior_transfer[1] != generation.generation_id
                else None
            )
            if transfer_to_release is not None:
                release_manager, release_generation = transfer_to_release
                try:
                    await release_manager.release_adapter_transfer(release_generation)
                except BaseException:
                    self._retain_pending_transfer(release_manager, release_generation)
            return receipt

    async def apply_external_adapter(
        self,
        *,
        operation_id: str,
        public_alias: str,
        generation_id: str,
        expected_generation_id: str | None,
        policy_version: int,
        source: ExternalAdapterObjectSource,
        timeout_s: float = 300.0,
    ) -> MegatronExternalLoraApplyResult:
        """Materialize one exact object and apply it through the normal holder."""

        async with self._publication_scope(public_alias):
            return await self._apply_external_adapter(
                operation_id=operation_id,
                public_alias=public_alias,
                generation_id=generation_id,
                expected_generation_id=expected_generation_id,
                policy_version=policy_version,
                source=source,
                timeout_s=timeout_s,
            )

    async def _apply_external_adapter(
        self,
        *,
        operation_id: str,
        public_alias: str,
        generation_id: str,
        expected_generation_id: str | None,
        policy_version: int,
        source: ExternalAdapterObjectSource,
        timeout_s: float,
    ) -> MegatronExternalLoraApplyResult:

        if source.generation_id != generation_id:
            raise ValueError("external adapter generation identity changed")
        if not operation_id or not public_alias or policy_version < 0:
            raise ValueError("external adapter apply identity is invalid")
        self._require_open()
        manager = self.runtime.model_service(self.service.name)
        received = await manager.materialize_external_adapter(
            source, timeout_s=timeout_s
        )
        runtime_lora_name = in_flight_lora_name(public_alias)
        try:
            checkpoint, tensor_bytes, config_bytes = _validate_external_received(
                received, source, self.service
            )
            staged = {
                "path": map_checkpoint_path_for_vllm(self.config, checkpoint),
                "source_identity": source.source_identity,
                "layout": "peft_safetensors_v1",
                "model_bytes": tensor_bytes,
                "config_bytes": config_bytes,
            }
            response = await self._post_update(
                {
                    "operation_id": operation_id,
                    "model_name": runtime_lora_name,
                    "lora_slot": runtime_lora_name,
                    "source": staged,
                    "generation_id": generation_id,
                    "expected_generation_id": expected_generation_id,
                    "policy_version": policy_version,
                }
            )
            update_sequence = _response_int(response, "update_seq")
            apply_s = _response_float(response, "apply_s")
            update_identity = str(response["update_identity"])
            if (
                response.get("generation_id") != generation_id
                or response.get("lora_slot") != runtime_lora_name
                or response.get("source_identity") != source.source_identity
                or _response_int(response, "policy_version") != policy_version
            ):
                manager.quarantine("paired holder returned another external policy")
                raise RuntimeError("paired holder changed external policy identity")
        except BaseException as error:
            try:
                await asyncio.shield(
                    manager.release_adapter_transfer(source.generation_id)
                )
            except BaseException as cleanup_error:
                self._retain_pending_transfer(manager, source.generation_id)
                error.add_note(
                    "external adapter cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

        async with self._lock:
            previous_sequence = self._active_update_sequences.get(runtime_lora_name, -1)
            if update_sequence <= previous_sequence:
                manager.quarantine("paired external update sequence regressed")
                raise RuntimeError("paired external update sequence is not monotonic")
            prior_transfer = self._retained_transfers.get(runtime_lora_name)
            self._active_generations[runtime_lora_name] = generation_id
            self._active_update_sequences[runtime_lora_name] = update_sequence
            self._retained_transfers[runtime_lora_name] = (
                manager,
                source.generation_id,
            )
            self._retained_adapter_sources[runtime_lora_name] = staged

        if prior_transfer is not None and prior_transfer[1] != source.generation_id:
            release_manager, release_generation = prior_transfer
            try:
                await release_manager.release_adapter_transfer(release_generation)
            except BaseException:
                self._retain_pending_transfer(release_manager, release_generation)
        return MegatronExternalLoraApplyResult(
            runtime_lora_name=runtime_lora_name,
            generation_id=generation_id,
            policy_version=policy_version,
            holder_update_sequence=update_sequence,
            holder_update_id=update_identity,
            source_identity=source.source_identity,
            tensor_bytes=tensor_bytes,
            config_bytes=config_bytes,
            staging_s=max(item.materialization_s for item in received),
            apply_s=apply_s,
        )

    async def prune_versioned_adapters(
        self,
        public_alias: str,
        *,
        retain_steps: set[int],
    ) -> None:
        """Unload unprotected publisher-owned versioned adapters."""

        async with self._lock:
            self._require_open()
            await self._release_pending_transfers()
            protected = set(retain_steps)
            latest = self._latest_activated_publications.get(public_alias)
            if latest is not None and latest.publication_mode == "versioned_lora":
                protected.add(latest.learner_version)
            protected.update(
                step
                for (alias, step), count in self._activated_publication_leases.items()
                if alias == public_alias and count > 0
            )
            candidates = sorted(
                (
                    (key, receipt)
                    for key, receipt in self._activated_publications.items()
                    if key[0] == public_alias
                    and receipt.publication_mode == "versioned_lora"
                    and key[1] not in protected
                ),
                key=lambda item: item[0][1],
            )
            for key, receipt in candidates:
                runtime_lora_name = self._validate_prunable_publication(key, receipt)
                manager, generation_id = self._retained_transfers[runtime_lora_name]
                await self._post_unload(runtime_lora_name)
                self._activated_publications.pop(key)
                if (
                    self._active_generations.get(runtime_lora_name)
                    == receipt.serving_generation_id
                ):
                    self._active_generations.pop(runtime_lora_name)
                    self._active_update_sequences.pop(runtime_lora_name, None)
                self._retained_transfers.pop(runtime_lora_name)
                self._retained_adapter_sources.pop(runtime_lora_name, None)
                try:
                    await manager.release_adapter_transfer(generation_id)
                except BaseException:
                    self._retain_pending_transfer(manager, generation_id)
                    raise

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
            if self._closed and not self._retained_transfers:
                return
            if self._activated_publication_leases:
                raise RuntimeError(
                    "cannot close paired inference with activated publication leases"
                )
            if self._exact_evaluation_publications:
                raise RuntimeError(
                    "cannot close paired inference with materialized exact adapters"
                )
            self._closed = True
        await self._publications_idle.wait()
        async with self._lock:
            await self._release_transfers(tuple(self._retained_transfers.items()))
            self._active_generations.clear()
            self._active_update_sequences.clear()
            self._retained_adapter_sources.clear()
            self._activated_publications.clear()
            self._latest_activated_publications.clear()
            self._publication_locks.clear()

    async def _post_unload(self, runtime_lora_name: str) -> None:
        url = f"{self.service.leader_endpoint.url.rstrip('/')}/v1/unload_lora_adapter"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json={"lora_name": runtime_lora_name},
                headers=_headers(self.api_key),
            )
        if response.status_code != HTTPStatus.NOT_FOUND.value:
            response.raise_for_status()

    async def _post_update(self, payload: dict[str, object]) -> dict[str, object]:
        url = f"{self.service.leader_endpoint.url}/art/in_flight_lora_update"
        async with httpx.AsyncClient(timeout=330.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=_headers(self.api_key),
            )
        response.raise_for_status()
        return _response_object(response)

    async def _release_pending_transfers(self) -> None:
        pending = tuple(
            item
            for item in self._retained_transfers.items()
            if item[0].startswith("release-pending:")
        )
        await self._release_transfers(pending)

    async def _release_transfers(
        self,
        retained: tuple[tuple[str, tuple[Any, str]], ...],
    ) -> None:
        failures: list[BaseException] = []
        for name, transfer in retained:
            manager, generation_id = transfer
            try:
                await manager.release_adapter_transfer(generation_id)
            except BaseException as error:
                failures.append(error)
            else:
                if self._retained_transfers.get(name) == transfer:
                    self._retained_transfers.pop(name)
        if failures:
            raise BaseExceptionGroup("paired adapter transfer cleanup failed", failures)

    def _retain_pending_transfer(self, manager: Any, generation_id: str) -> None:
        key = f"release-pending:{generation_id}"
        pending = self._retained_transfers.get(key)
        transfer = (manager, generation_id)
        if pending is not None and pending != transfer:
            raise RuntimeError("paired pending transfer ownership changed")
        self._retained_transfers[key] = transfer

    def _record_activated_publication(
        self, receipt: MegatronSamplerPublicationReceipt
    ) -> None:
        alias = receipt.requested_public_alias
        key = (alias, receipt.learner_version)
        runtime_lora_name = receipt.runtime_lora_name
        if (
            runtime_lora_name is None
            or self._active_generations.get(runtime_lora_name)
            != receipt.serving_generation_id
        ):
            raise RuntimeError("paired publication activation lineage changed")
        if receipt.publication_mode == "versioned_lora":
            transfer = self._retained_transfers.get(runtime_lora_name)
            if transfer is None or transfer[1] != receipt.serving_generation_id:
                raise RuntimeError("paired publication transfer ownership changed")

        previous = self._latest_activated_publications.get(alias)
        if (
            previous is not None
            and previous.publication_mode == "in_flight_lora"
            and (previous.requested_public_alias, previous.learner_version) != key
        ):
            previous_key = (
                previous.requested_public_alias,
                previous.learner_version,
            )
            if not self._activated_publication_leases.get(previous_key, 0):
                self._activated_publications.pop(previous_key, None)
        self._activated_publications[key] = receipt
        self._latest_activated_publications[alias] = receipt

    async def _acquire_exact_publication(
        self,
        key: tuple[str, int],
        receipt: MegatronSamplerPublicationReceipt,
    ) -> MegatronSamplerPublicationReceipt:
        if receipt.publication_mode == "versioned_lora":
            self._validate_retained_versioned_publication(key, receipt)
            self._activated_publication_leases[key] += 1
            return receipt

        existing = self._exact_evaluation_publications.get(key)
        if existing is not None:
            self._validate_exact_evaluation_publication(key, receipt, existing)
            self._activated_publication_leases[key] += 1
            return existing

        await self._release_pending_transfers()
        runtime_lora_name = receipt.runtime_lora_name
        if (
            receipt.publication_mode != "in_flight_lora"
            or self._latest_activated_publications.get(key[0]) is not receipt
            or runtime_lora_name is None
            or receipt.requested_public_alias != key[0]
            or receipt.learner_version != key[1]
            or self._active_generations.get(runtime_lora_name)
            != receipt.serving_generation_id
        ):
            raise RuntimeError("in-flight publication is no longer active")

        transfer = self._retained_transfers.get(runtime_lora_name)
        source = self._retained_adapter_sources.get(runtime_lora_name)
        if (
            transfer is None
            or transfer[1] != receipt.serving_generation_id
            or source is None
        ):
            raise RuntimeError("in-flight publication generation is not retained")

        exact_name = f"{key[0]}:eval@{key[1]}"
        if exact_name in self._active_generations:
            raise RuntimeError("paired exact adapter slot ownership is ambiguous")
        payload: dict[str, object] = {
            "operation_id": uuid.uuid4().hex,
            "model_name": exact_name,
            "lora_slot": exact_name,
            "source": source,
            "generation_id": receipt.serving_generation_id,
            "expected_generation_id": None,
            "policy_version": receipt.learner_version,
        }
        try:
            response = await self._post_update(payload)
            if (
                response.get("generation_id") != receipt.serving_generation_id
                or response.get("lora_slot") != exact_name
                or _response_int(response, "policy_version") != receipt.learner_version
            ):
                transfer[0].quarantine(
                    "paired exact holder returned another generation"
                )
                raise RuntimeError("paired exact holder changed policy identity")
            exact_receipt = receipt.model_copy(
                update={
                    "runtime_lora_name": exact_name,
                    "holder_update_sequence": _response_int(response, "update_seq"),
                    "holder_update_id": str(response["update_identity"]),
                    "result": receipt.result.model_copy(update={"lora": exact_name}),
                }
            )
        except BaseException as error:
            try:
                await asyncio.shield(self._post_unload(exact_name))
            except BaseException as cleanup_error:
                transfer[0].quarantine("paired exact adapter cleanup failed")
                error.add_note(
                    "paired exact adapter cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        self._active_generations[exact_name] = receipt.serving_generation_id
        self._exact_evaluation_publications[key] = exact_receipt
        self._activated_publication_leases[key] = 1
        return exact_receipt

    async def _release_exact_publication(
        self,
        key: tuple[str, int],
        receipt: MegatronSamplerPublicationReceipt,
    ) -> None:
        count = self._activated_publication_leases.get(key, 0)
        if count <= 0:
            raise RuntimeError("paired exact adapter lease ownership changed")
        exact = self._exact_evaluation_publications.get(key)
        if exact is None:
            if receipt.publication_mode != "versioned_lora":
                raise RuntimeError("paired exact adapter lease ownership changed")
            self._validate_retained_versioned_publication(key, receipt)
            if count == 1:
                self._activated_publication_leases.pop(key)
            else:
                self._activated_publication_leases[key] = count - 1
            return
        self._validate_exact_evaluation_publication(
            key, self._activated_publications.get(key), exact
        )
        if receipt != exact:
            raise RuntimeError("paired exact adapter receipt changed")
        if count > 1:
            self._activated_publication_leases[key] = count - 1
            return

        exact_name = cast(str, exact.runtime_lora_name)
        await self._post_unload(exact_name)
        self._activated_publication_leases.pop(key)
        self._exact_evaluation_publications.pop(key)
        if self._active_generations.get(exact_name) == exact.serving_generation_id:
            self._active_generations.pop(exact_name)
        publication = self._activated_publications.get(key)
        if self._latest_activated_publications.get(key[0]) is not publication:
            self._activated_publications.pop(key, None)

    def _validate_exact_evaluation_publication(
        self,
        key: tuple[str, int],
        publication: MegatronSamplerPublicationReceipt | None,
        exact: MegatronSamplerPublicationReceipt,
    ) -> None:
        alias, learner_version = key
        exact_name = f"{alias}:eval@{learner_version}"
        if (
            publication is None
            or publication.publication_mode != "in_flight_lora"
            or publication.requested_public_alias != alias
            or publication.learner_version != learner_version
            or exact.runtime_lora_name != exact_name
            or exact.serving_generation_id != publication.serving_generation_id
            or exact.learner_version != learner_version
            or self._active_generations.get(exact_name) != exact.serving_generation_id
        ):
            raise RuntimeError("paired exact adapter generation changed")

    def _validate_prunable_publication(
        self,
        key: tuple[str, int],
        receipt: MegatronSamplerPublicationReceipt,
    ) -> str:
        if self._activated_publication_leases.get(key, 0):
            raise RuntimeError("cannot prune a leased paired publication")
        if self._latest_activated_publications.get(key[0]) is receipt:
            raise RuntimeError("cannot prune the current paired publication")
        return self._validate_retained_versioned_publication(key, receipt)

    def _validate_retained_versioned_publication(
        self,
        key: tuple[str, int],
        receipt: MegatronSamplerPublicationReceipt,
    ) -> str:
        alias, learner_version = key
        runtime_lora_name = receipt.runtime_lora_name
        if (
            receipt.publication_mode != "versioned_lora"
            or receipt.requested_public_alias != alias
            or receipt.learner_version != learner_version
            or runtime_lora_name != f"{alias}@{learner_version}"
            or self._active_generations.get(runtime_lora_name)
            != receipt.serving_generation_id
        ):
            raise RuntimeError("paired versioned publication lineage changed")
        transfer = self._retained_transfers.get(runtime_lora_name)
        if transfer is None or transfer[1] != receipt.serving_generation_id:
            raise RuntimeError("paired versioned publication ownership changed")
        return cast(str, runtime_lora_name)

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


def _validate_external_received(
    received: tuple[AdapterReceiveResult, ...],
    source: ExternalAdapterObjectSource,
    service: ModelServiceSpec,
) -> tuple[str, int, int]:
    expected_hosts = len({member.host_id for member in service.members})
    if len(received) != expected_hosts or not received:
        raise RuntimeError("not every inference host materialized the external adapter")
    if {item.generation_id for item in received} != {source.generation_id}:
        raise RuntimeError("inference hosts materialized another external generation")
    if {item.source_identity for item in received} != {source.source_identity}:
        raise RuntimeError("inference hosts changed the external source identity")
    paths = {str(item.path) for item in received}
    config_bytes = len(source.adapter_config_json.encode())
    sizes = {(int(item.tensor_bytes), int(item.config_bytes)) for item in received}
    if len(paths) != 1 or sizes != {(source.object_bytes, config_bytes)}:
        raise RuntimeError("inference hosts materialized different external adapters")
    return paths.pop(), source.object_bytes, config_bytes


def _serving_profile_identity(
    runtime: ArtRuntime,
    spec: TrainerRuntimeSpec,
    base_model: str,
    service: ModelServiceSpec,
) -> ServingProfileIdentity:
    retained_routes = runtime.retained_route_prefetch_enabled
    return ServingProfileIdentity(
        base_model=base_model,
        model_identifier=service.runtime_model,
        model_revision=service.model_revision or "default",
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
    if values.get("generation_config", "vllm") != "vllm":
        raise ValueError("paired inference requires vLLM generation defaults")
    if values.get("logprobs_mode", "raw_logprobs") != "raw_logprobs":
        raise ValueError("paired inference requires raw model logprobs")
    values["enable_sleep_mode"] = temporal_gpu_sharing
    values["enable_lora"] = True
    values["enable_return_routed_experts"] = enable_moe_routing_replay
    values.setdefault("max_loras", 2)
    values["generation_config"] = "vllm"
    values["logprobs_mode"] = "raw_logprobs"
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
