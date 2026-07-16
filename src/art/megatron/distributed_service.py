from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Literal, cast
import uuid

import httpx

from art import dev, types
from art.dev.get_model_config import default_target_modules
from art.distributed.art_runtime import ArtRuntime, DistributedPackedBatch
from art.distributed.specs import ModelServiceSpec
from art.distributed.vllm_replica import (
    ReplicaFailure,
    ReplicaLaunchTemplate,
    ReplicaUpdateReport,
)
from art.serving_capabilities import (
    ServingCapabilities,
    discover_serving_capabilities,
)
from art.utils.lifecycle import complete_task, complete_to_thread
from art.utils.output_dirs import get_step_checkpoint_dir
from art.vllm_runtime import (
    get_external_vllm_runtime_config,
    map_checkpoint_path_for_vllm,
    normalize_vllm_server_url,
    wait_for_vllm_http_runtime,
)

from .identity_lora import create_identity_lora
from .lora_config import LORA_ALPHA, default_lora_rank_for_handler
from .model_support import (
    get_model_support_handler,
    get_model_support_handler_for_spec,
    get_model_support_spec,
    model_uses_expert_parallel,
)
from .optimizer_state import (
    OptimizerAdapter,
    async_optimizer_model_lease,
    commit_optimizer_policy_advance,
    format_megatron_resume_message,
    hash_adapter_checkpoint,
    prepare_megatron_resume_state,
    publish_adapter_checkpoint,
    read_adapter_publication,
    read_committed_optimizer_pointer,
    resolve_committed_optimizer_policy,
)
from .runtime.specs import (
    AdapterReady,
    CurrentTrainConfig,
    DurableTrainOutput,
    ExperimentalTrainConfig,
    TrainAccepted,
    TrainCancelled,
    TrainCompleted,
    TrainerRuntimeSpec,
    TrainFailed,
    TrainingRunSpec,
    TrainJobSpec,
    TrainProgress,
)
from .runtime_config import get_megatron_runtime_config

logger = logging.getLogger(__name__)


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


class DistributedMegatronService:
    """One model's durable checkpoints and run-scoped distributed runtimes."""

    propagate_close_errors = True
    close_timeout_s = 60.0

    def __init__(
        self,
        *,
        model_name: str,
        base_model: str,
        config: dev.BackendModelConfig,
        output_dir: str,
        runtime: ArtRuntime,
        enable_expert_replay: bool,
    ) -> None:
        self.model_name = model_name
        self.base_model = base_model
        self.config = config
        self.output_dir = output_dir
        self.runtime = runtime
        self.enable_expert_replay = enable_expert_replay
        self._latest_step = 0
        self._resume_prepared = False
        self._training_session_id = uuid.uuid4().hex
        self._trainer: Any = None
        self._mutation_lock = asyncio.Lock()
        self._managed_service_name: str | None = None
        self._base_url: str | None = None
        self._serving_capabilities: ServingCapabilities | None = None
        self._api_key_value: str | None = None
        self._current_lora_name: str | None = None
        self._published_adapters: dict[int, OptimizerAdapter] = {}
        self._loaded_adapter_steps: set[int] = set()
        self._loaded_exact_adapter_steps: set[int] = set()
        self._exact_adapter_refcounts: dict[int, int] = {}
        self._recovery_tasks: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def rollout_weights_mode(self) -> str:
        return self.config.get("rollout_weights_mode", "lora")

    @property
    def rollout_weight_update_mode(self) -> str:
        return self.config.get("rollout_weight_update_mode", "step_lora")

    @property
    def _allow_unvalidated_arch(self) -> bool:
        return bool(self.config.get("allow_unvalidated_arch", False))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("distributed model service is closed")

    def _trainer_is_current(self) -> bool:
        return (
            self._trainer is not None
            and self._trainer.valid
            and self._trainer.learner_version == self._latest_step
        )

    @property
    def _optimizer_state_path(self) -> str:
        path = f"{self.output_dir}/optimizer_states_rl"
        os.makedirs(path, exist_ok=True)
        return path

    def _lora_config(self) -> dev.LoRAConfig:
        return cast(dev.LoRAConfig, self.config.get("lora_config") or {})

    def _random_state(self) -> int | None:
        for key in ("lora_config", "init_args"):
            value = self.config.get(key, {}).get("random_state")
            if value is not None:
                return int(value)
        return None

    @property
    def _model_identifier(self) -> str:
        value = self.config.get("init_args", {}).get("model_name", self.base_model)
        if not isinstance(value, str) or not value:
            raise ValueError("init_args.model_name must be a non-empty string")
        return value

    def _resolve_current_lora_path(self) -> str:
        if self._trainer_is_current():
            path = get_step_checkpoint_dir(self.output_dir, self._latest_step)
            if not (Path(path) / "adapter_model.safetensors").is_file():
                raise RuntimeError(
                    f"resident trainer adapter is missing for step {self._latest_step}"
                )
            self._resume_prepared = True
            return path
        resume = prepare_megatron_resume_state(
            output_dir=self.output_dir,
            optimizer_state_path=self._optimizer_state_path,
        )
        print(format_megatron_resume_message(resume))
        self._latest_step = resume.step
        self._published_adapters = {
            step: adapter
            for step, adapter in self._published_adapters.items()
            if step <= resume.step
        }
        path = get_step_checkpoint_dir(self.output_dir, self._latest_step)
        if not (Path(path) / "adapter_model.safetensors").is_file():
            if self._latest_step != 0:
                raise RuntimeError(
                    f"committed adapter is missing for step {self._latest_step}"
                )
            lora = self._lora_config()
            handler = get_model_support_handler(
                self.base_model,
                allow_unvalidated_arch=self._allow_unvalidated_arch,
            )
            create_identity_lora(
                self._model_identifier,
                path,
                rank=lora.get("rank"),
                target_modules=lora.get("target_modules"),
                random_state=self._random_state(),
                allow_unvalidated_arch=self._allow_unvalidated_arch,
                handler=handler,
            )
        self._resume_prepared = True
        return path

    def _runtime_spec(self) -> TrainerRuntimeSpec:
        mesh = self.runtime.topology.trainer
        if mesh is None:
            raise RuntimeError("ART runtime has no trainer mesh")
        runtime_config = get_megatron_runtime_config()
        if runtime_config.topology != mesh.topology:
            raise ValueError(
                "Megatron runtime topology does not match the ART trainer mesh"
            )
        lora = self._lora_config()
        support_spec = get_model_support_spec(
            self.base_model,
            allow_unvalidated_arch=self._allow_unvalidated_arch,
        )
        handler = get_model_support_handler_for_spec(support_spec)
        targets = lora.get("target_modules") or default_target_modules(self.base_model)
        revision = str(self.config.get("init_args", {}).get("revision") or "default")
        compile_enabled = os.environ.get(
            "ART_DISABLE_MEGATRON_COMPILE", "0"
        ).lower() not in {"1", "true", "yes", "on"}
        identity = {
            "art": _art_source_revision(),
            "model": self._model_identifier,
            "support_model": self.base_model,
            "revision": revision,
            "handler": handler.key,
            "mesh": mesh.model_dump(mode="json"),
        }
        return TrainerRuntimeSpec(
            art_revision=identity["art"],
            model_identifier=self._model_identifier,
            model_revision=revision,
            model_support_key=support_spec.key,
            handler_name=handler.key,
            lora_rank=int(lora.get("rank") or default_lora_rank_for_handler(handler)),
            lora_alpha=float(lora.get("alpha", LORA_ALPHA)),
            lora_target_modules=tuple(targets),
            dtype=_trainer_dtype(self.config),
            trainer_mesh=mesh,
            packed_sequence_length=runtime_config.packed_sequence_length,
            compile_enabled=compile_enabled,
            compile_fingerprint=_digest({**identity, "compile": compile_enabled}),
            optimizer_layout_fingerprint=_digest(
                {"mesh": mesh.model_dump(mode="json")}
            ),
            allow_unvalidated_arch=self._allow_unvalidated_arch,
            enable_moe_routing_replay=self.enable_expert_replay
            and model_uses_expert_parallel(
                self.base_model,
                allow_unvalidated_arch=self._allow_unvalidated_arch,
            ),
            streaming_weight_offload=runtime_config.streaming_weight_offload,
            random_state=self._random_state(),
        )

    async def _ensure_trainer_locked(self) -> Any:
        if self._trainer_is_current():
            return self._trainer
        current = await self._prepare_for_packing_locked()
        assert self._trainer is None
        runtime_spec = self._runtime_spec()
        run_spec = TrainingRunSpec(
            run_id=uuid.uuid4().hex,
            runtime_fingerprint=runtime_spec.fingerprint,
            training_session_id=self._training_session_id,
            initial_learner_version=self._latest_step,
            initial_adapter_path=current,
            optimizer_state_path=self._optimizer_state_path,
        )
        self._trainer = await self.runtime.start_trainer(runtime_spec, run_spec)
        return self._trainer

    async def prepare_for_packing(self) -> int:
        async with self._mutation_lock:
            self._require_open()
            await self._prepare_for_packing_locked()
            return self._latest_step

    async def _prepare_for_packing_locked(self) -> str:
        if self._trainer_is_current():
            return self._resolve_current_lora_path()
        if self._trainer is not None:
            _, cancelled = await complete_task(
                asyncio.create_task(self.runtime.stop_trainer(self._trainer))
            )
            self._trainer = None
            if cancelled is not None:
                raise cancelled
        previous_step = self._latest_step
        current, cancelled = await complete_to_thread(self._resolve_current_lora_path)
        if self._base_url and self._latest_step != previous_step:
            await self._reconcile_serving_locked(current)
        if cancelled is not None:
            raise cancelled
        return current

    async def _reconcile_serving_locked(self, checkpoint: str) -> None:
        previous_name = self._current_lora_name
        await self._register_lora_for_step_locked(self._latest_step, checkpoint)
        invalid_exact = {
            step
            for step in self._loaded_exact_adapter_steps
            if step > self._latest_step
        }
        for step in sorted(invalid_exact):
            name = (
                f"{self.model_name}:eval@{step}"
                if self.rollout_weight_update_mode == "in_flight_lora"
                else f"{self.model_name}@{step}"
            )
            await self._unload_adapter(name)
            self._loaded_exact_adapter_steps.discard(step)
            self._exact_adapter_refcounts.pop(step, None)
        for step in sorted(
            step for step in self._loaded_adapter_steps if step > self._latest_step
        ):
            await self._unload_adapter(f"{self.model_name}@{step}")
            self._loaded_adapter_steps.discard(step)
        if previous_name == f"{self.model_name}:active" and previous_name != (
            self._current_lora_name
        ):
            assert previous_name is not None
            await self._unload_adapter(previous_name)

    @asynccontextmanager
    async def _trainer_transaction(
        self,
        trainer: Any,
        job: TrainJobSpec,
        leases: Any,
        *,
        source: str,
        staging: str,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        async with async_optimizer_model_lease(job.output.optimizer_state_path):
            _, cancelled = await complete_to_thread(
                lambda: _copy_checkpoint(source, staging)
            )
            if cancelled is not None:
                raise cancelled
            events = trainer.train(job, leases)
            try:
                yield events
            finally:
                await events.aclose()

    async def train_packed(
        self,
        batch: DistributedPackedBatch,
        config: types.TrainConfig,
        experimental_config: dev.TrainConfig,
    ) -> AsyncIterator[dict[str, float]]:
        async with self._mutation_lock:
            self._require_open()
            trainer = await self._ensure_trainer_locked()
            next_step = self._latest_step + 1
            source = get_step_checkpoint_dir(self.output_dir, self._latest_step)
            staging = f"{self.output_dir}/megatron_runtime/staging/{next_step:04d}"
            values = {
                key: value
                for key, value in experimental_config.items()
                if key in ExperimentalTrainConfig.model_fields and value is not None
            }
            job = TrainJobSpec(
                job_id=uuid.uuid4().hex,
                run_id=trainer.run_spec.run_id,
                training_session_id=self._training_session_id,
                expected_learner_version=self._latest_step,
                learner_version=next_step,
                batch=batch.leases.ref,
                config=CurrentTrainConfig.model_validate(config.model_dump()),
                experimental_config=ExperimentalTrainConfig.model_validate(values),
                output=DurableTrainOutput(
                    adapter_path=staging,
                    optimizer_state_path=self._optimizer_state_path,
                    optimizer_lease_owner="controller",
                ),
            )
            adapter_ready = False
            completed = False
            checkpoint: str | None = None
            async with self._trainer_transaction(
                trainer,
                job,
                batch.leases,
                source=source,
                staging=staging,
            ) as events:
                async for event in events:
                    if event.job_id != job.job_id or event.run_id != job.run_id:
                        raise RuntimeError(
                            "trainer returned an event for a different job"
                        )
                    if isinstance(event, TrainAccepted):
                        continue
                    if isinstance(event, TrainProgress):
                        yield event.metrics
                        continue
                    if isinstance(event, AdapterReady):
                        if adapter_ready:
                            raise RuntimeError(
                                "trainer returned duplicate AdapterReady events"
                            )
                        if (
                            event.learner_version != next_step
                            or event.adapter_path != staging
                        ):
                            raise RuntimeError("trainer prepared the wrong adapter")
                        published = _publish_checkpoint(
                            staging, self.output_dir, next_step
                        )
                        checkpoint = published.identity
                        self._published_adapters[next_step] = published
                        adapter_ready = True
                        continue
                    if isinstance(event, TrainCompleted):
                        if completed:
                            raise RuntimeError(
                                "trainer returned duplicate TrainCompleted events"
                            )
                        if not adapter_ready:
                            raise RuntimeError(
                                "trainer completed before preparing the adapter"
                            )
                        if event.learner_version != next_step:
                            raise RuntimeError(
                                "trainer completed the wrong learner version"
                            )
                        completed = True
                        continue
                    if isinstance(event, TrainFailed):
                        raise RuntimeError(
                            f"distributed Megatron job failed ({event.error_type}): "
                            f"{event.message}"
                        )
                    if isinstance(event, TrainCancelled):
                        raise asyncio.CancelledError(event.reason)
            if not adapter_ready or not completed:
                raise RuntimeError(
                    "trainer ended without preparing and durably completing the adapter"
                )
            assert checkpoint is not None
            try:
                await self._register_lora_for_step_locked(next_step, checkpoint)
            except BaseException:
                # Training is durable before serving publication. Preserve that
                # lineage even when the serving generation fails closed.
                self._latest_step = next_step
                raise
            self._latest_step = next_step

    async def resolve_global_grad_accumulation_sequences(
        self, config: types.TrainConfig
    ) -> int:
        if config.grad_accumulation_sequences is not None:
            return int(config.grad_accumulation_sequences)
        mesh = self.runtime.topology.trainer
        assert mesh is not None
        topology = mesh.topology
        return len(mesh.ranks) // (topology.tp * topology.cp * topology.pp)

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        async with self._mutation_lock:
            self._require_open()
            if self._base_url:
                return _host_port(self._base_url)
            if self._managed_service_name is not None:
                raise RuntimeError("managed model service is unavailable")
            return await self._start_openai_server_locked(config)

    async def _start_openai_server_locked(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        api_key = self._api_key(config)
        lora_path = await asyncio.to_thread(self._resolve_current_lora_path)
        external = get_external_vllm_runtime_config(self.config)
        if external is not None:
            base_url = normalize_vllm_server_url(external.server_url)
            headers = _headers(external.api_key)
            await wait_for_vllm_http_runtime(
                base_url=base_url,
                timeout=external.health_timeout_s,
                headers=headers,
            )
            capabilities = await discover_serving_capabilities(
                base_url=base_url,
                headers=headers,
                allow_openai_compatible=True,
            )
            lora_name, _ = await self._load_adapter_at(
                lora_path,
                self._latest_step,
                base_url=base_url,
                api_key=api_key,
                latest_step=self._latest_step,
            )
            self._publish_serving_state(
                managed_service_name=None,
                base_url=base_url,
                capabilities=capabilities,
                api_key=api_key,
                current_lora_name=lora_name,
            )
            return _host_port(base_url)

        services = tuple(
            service
            for service in self.runtime.topology.model_services
            if service.name == self.model_name
        )
        if len(services) != 1:
            raise ValueError(
                f"runtime topology must define one model service named "
                f"{self.model_name!r}"
            )
        service = services[0]
        if self.rollout_weights_mode != "lora":
            raise RuntimeError(
                "distributed Megatron currently requires LoRA rollout serving"
            )
        template = ReplicaLaunchTemplate(
            served_model_name=f"{self.model_name}@{self._latest_step}",
            lora_path=lora_path,
            engine_args=self._engine_args(config),
            server_args=self._server_args(config),
        )
        await self.runtime.start_model_service(
            service, template, on_failure=self._replica_failed
        )
        base_url = service.leader_endpoint.url
        try:
            capabilities = await discover_serving_capabilities(
                base_url=base_url,
                headers=_headers(api_key),
                allow_openai_compatible=False,
            )
            capabilities.require(
                "exact_lora_worker_state", operation="distributed LoRA publication"
            )
            digest = await self._checkpoint_digest(lora_path, self._latest_step)
            update_identity = uuid.uuid4().hex
            manager = self.runtime.model_service(service.name)
            state = manager.prepare_update(update_identity=update_identity)
            await self._acknowledge_lora_workers(
                lora_name=template.served_model_name,
                lora_path=lora_path,
                step=self._latest_step,
                service_name=service.name,
                base_url=base_url,
                api_key=api_key,
            )
            report = ReplicaUpdateReport(
                replica_id=service.name,
                generation=state.generation,
                generation_digest=state.generation_digest,
                policy_version=str(self._latest_step),
                policy_digest=digest,
                update_identity=update_identity,
            )
            if manager.verify_update(report).phase != "ready":
                raise RuntimeError("model service rejected its initial policy")
        except BaseException as error:
            cleanup = await self._rollback_server_start_safely(service.name)
            if cleanup:
                raise BaseExceptionGroup(
                    "vLLM startup validation and rollback failed", [error, *cleanup]
                ) from None
            raise
        self._publish_serving_state(
            managed_service_name=service.name,
            base_url=base_url,
            capabilities=capabilities,
            api_key=api_key,
            current_lora_name=template.served_model_name,
        )
        return _host_port(base_url)

    def _publish_serving_state(
        self,
        *,
        managed_service_name: str | None,
        base_url: str,
        capabilities: ServingCapabilities,
        api_key: str | None,
        current_lora_name: str,
    ) -> None:
        self._managed_service_name = managed_service_name
        self._base_url = base_url
        self._serving_capabilities = capabilities
        self._api_key_value = api_key
        self._current_lora_name = current_lora_name
        self._loaded_adapter_steps.add(self._latest_step)

    def _clear_serving_state(self) -> None:
        self._managed_service_name = None
        self._unpublish_serving_state()

    def _unpublish_serving_state(self) -> None:
        self._base_url = None
        self._serving_capabilities = None
        self._api_key_value = None
        self._current_lora_name = None
        self._loaded_adapter_steps.clear()
        self._loaded_exact_adapter_steps.clear()
        self._exact_adapter_refcounts.clear()

    async def _replica_failed(self, failure: ReplicaFailure) -> None:
        if self._closed or failure.replica_id != self._managed_service_name:
            return
        task = asyncio.create_task(self._recover_failed_replica(failure))
        self._recovery_tasks.add(task)
        task.add_done_callback(self._recovery_tasks.discard)
        task.add_done_callback(_consume_task_result)

    async def _recover_failed_replica(self, failure: ReplicaFailure) -> None:
        try:
            async with self._mutation_lock:
                if self._closed or failure.replica_id != self._managed_service_name:
                    return
                manager = self.runtime.model_service(failure.replica_id)
                state = manager.state
                if (
                    state.generation != failure.generation
                    or state.generation_digest != failure.generation_digest
                    or state.phase != "quarantined"
                ):
                    return
                await self._recover_replica_locked(failure)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._unpublish_serving_state()
            logger.exception(
                "vLLM replica %s generation %d recovery failed",
                failure.replica_id,
                failure.generation,
            )

    async def _recover_replica_locked(self, failure: ReplicaFailure) -> None:
        service = self._model_service_spec()
        manager = self.runtime.model_service(failure.replica_id)
        checkpoint = get_step_checkpoint_dir(self.output_dir, self._latest_step)
        digest = await self._checkpoint_digest(checkpoint, self._latest_step)
        current_lora_name = self._current_lora_name or (
            f"{self.model_name}@{self._latest_step}"
        )
        bootstrap_name = f"{self.model_name}@{self._latest_step}"
        base_url = service.leader_endpoint.url
        exact_steps = tuple(sorted(self._loaded_exact_adapter_steps))
        try:
            state = await manager.restart(
                served_model_name=bootstrap_name, lora_path=checkpoint
            )
            capability = await discover_serving_capabilities(
                base_url=base_url,
                headers=_headers(self._api_key()),
                allow_openai_compatible=False,
            )
            if capability != self._serving_capabilities:
                raise RuntimeError("restarted vLLM replica capabilities changed")
            update_identity = uuid.uuid4().hex
            manager.prepare_update(update_identity=update_identity)
            lora_name = bootstrap_name
            lora_path = checkpoint
            if current_lora_name != bootstrap_name:
                lora_name, lora_path = await self._load_adapter_at(
                    checkpoint,
                    self._latest_step,
                    base_url=base_url,
                    api_key=self._api_key(),
                    latest_step=self._latest_step - 1,
                )
            await self._acknowledge_lora_workers(
                lora_name=lora_name,
                lora_path=lora_path,
                step=self._latest_step,
                service_name=failure.replica_id,
                base_url=base_url,
                api_key=self._api_key(),
            )
            report = ReplicaUpdateReport(
                replica_id=failure.replica_id,
                generation=state.generation,
                generation_digest=state.generation_digest,
                policy_version=str(self._latest_step),
                policy_digest=digest,
                update_identity=update_identity,
            )
            if manager.verify_update(report).phase != "ready":
                raise RuntimeError("restarted vLLM replica rejected current policy")
            if current_lora_name != bootstrap_name:
                await self._unload_adapter_at(bootstrap_name, base_url)
            for step in exact_steps:
                if step == self._latest_step and self.rollout_weight_update_mode != (
                    "in_flight_lora"
                ):
                    continue
                exact_name, exact_path = await self._load_adapter_at(
                    get_step_checkpoint_dir(self.output_dir, step),
                    step,
                    exact=True,
                    base_url=base_url,
                    api_key=self._api_key(),
                    latest_step=self._latest_step,
                )
                await self._acknowledge_lora_workers(
                    lora_name=exact_name,
                    lora_path=exact_path,
                    step=step,
                    service_name=failure.replica_id,
                    base_url=base_url,
                    api_key=self._api_key(),
                )
            self._current_lora_name = lora_name
            self._loaded_adapter_steps = {self._latest_step}
            self._loaded_exact_adapter_steps = set(exact_steps)
        except BaseException as error:
            manager.quarantine(f"replica recovery failed: {error}")
            try:
                await manager.stop()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "replica recovery and teardown failed", [error, cleanup_error]
                ) from None
            raise

    def _model_service_spec(self) -> ModelServiceSpec:
        services = tuple(
            service
            for service in self.runtime.topology.model_services
            if service.name == self.model_name
        )
        if len(services) != 1:
            raise RuntimeError(
                f"runtime topology has no unique service {self.model_name!r}"
            )
        return services[0]

    async def _rollback_server_start(
        self, service_name: str | None
    ) -> list[BaseException]:
        if service_name is None:
            return []
        try:
            await self.runtime.stop_model_service(service_name)
        except BaseException as error:
            return [error]
        return []

    async def _rollback_server_start_safely(
        self, service_name: str | None
    ) -> list[BaseException]:
        failures, cancelled = await complete_task(
            asyncio.create_task(self._rollback_server_start(service_name))
        )
        if cancelled is not None:
            failures.append(cancelled)
        return failures

    def _engine_args(self, server: dev.OpenAIServerConfig | None) -> dict[str, object]:
        handler = get_model_support_handler(
            self.base_model,
            allow_unvalidated_arch=self._allow_unvalidated_arch,
        )
        values = dict(self.config.get("engine_args", {}))
        values.update(dict((server or {}).get("engine_args", {})))
        for key, value in handler.vllm_engine_args(rollout_weights_mode="lora").items():
            values.setdefault(key, value)
        values["enable_sleep_mode"] = False
        values.pop("enable_lora", None)
        values.setdefault("max_loras", 2)
        values.setdefault("generation_config", "vllm")
        for key in ("model", "served_model_name"):
            values.pop(key, None)
        return values

    def _server_args(self, server: dev.OpenAIServerConfig | None) -> dict[str, object]:
        handler = get_model_support_handler(
            self.base_model,
            allow_unvalidated_arch=self._allow_unvalidated_arch,
        )
        values: dict[str, object] = {
            "return_tokens_as_token_ids": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
            **handler.vllm_server_args(),
            **dict((server or {}).get("server_args", {})),
        }
        for key in ("port", "host", "lora_modules"):
            values.pop(key, None)
        return values

    def _api_key(self, server: dev.OpenAIServerConfig | None = None) -> str | None:
        value = dict((server or {}).get("server_args", {})).get("api_key")
        return cast(str | None, value) if value is not None else self._api_key_value

    async def _load_adapter(
        self, checkpoint: str, step: int, *, exact: bool = False
    ) -> tuple[str, str]:
        if self._base_url is None:
            raise RuntimeError("vLLM serving has not started")
        return await self._load_adapter_at(
            checkpoint,
            step,
            exact=exact,
            base_url=self._base_url,
            api_key=self._api_key(),
            latest_step=self._latest_step,
        )

    async def _load_adapter_at(
        self,
        checkpoint: str,
        step: int,
        *,
        base_url: str,
        api_key: str | None,
        latest_step: int,
        exact: bool = False,
    ) -> tuple[str, str]:
        name = (
            f"{self.model_name}:eval@{step}"
            if exact and self.rollout_weight_update_mode == "in_flight_lora"
            else f"{self.model_name}@{step}"
        )
        path = map_checkpoint_path_for_vllm(self.config, checkpoint)
        in_flight = (
            not exact
            and self.rollout_weight_update_mode == "in_flight_lora"
            and step != latest_step
        )
        endpoint = (
            "/art/in_flight_lora_update" if in_flight else "/v1/load_lora_adapter"
        )
        payload = (
            {
                "model_name": f"{self.model_name}@{step}",
                "lora_slot": f"{self.model_name}:active",
                "lora_path": path,
                "policy_version": step,
            }
            if in_flight
            else {"lora_name": name, "lora_path": path}
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}{endpoint}",
                json=payload,
                headers=_headers(api_key),
            )
        response.raise_for_status()
        return str(payload.get("lora_slot", name)), path

    async def _acknowledge_lora_workers(
        self,
        *,
        lora_name: str,
        lora_path: str,
        step: int,
        service_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        service_name = service_name or self._managed_service_name
        base_url = base_url or self._base_url
        if service_name is None:
            return
        if base_url is None:
            raise RuntimeError("managed model service has no leader endpoint")
        payload = {
            "expected_workers": self.runtime.model_service(
                service_name
            ).expected_worker_identities(),
            "expected_lora": {
                "lora_name": lora_name,
                "lora_path": lora_path,
                "policy_version": step,
            },
        }
        timeout_s = self.runtime.topology.cluster.rpc_timeout_s
        async with asyncio.timeout(timeout_s):
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(
                    f"{base_url}/art/lora_worker_state",
                    json=payload,
                    headers=_headers(
                        api_key if api_key is not None else self._api_key()
                    ),
                )
        response.raise_for_status()

    async def register_lora_for_step(self, step: int, checkpoint: str) -> None:
        async with self._mutation_lock:
            self._require_open()
            policy = await asyncio.to_thread(
                resolve_committed_optimizer_policy,
                self._optimizer_state_path,
                initial_adapter_path=get_step_checkpoint_dir(self.output_dir, 0),
            )
            if policy.policy_adapter.step != step or policy.policy_adapter.identity != (
                str(Path(checkpoint).absolute())
            ):
                raise RuntimeError(
                    "distributed LoRA registration requires a committed policy step"
                )
            await self._register_lora_for_step_locked(step, checkpoint)
            self._latest_step = step

    async def advance_without_training(
        self,
        *,
        expected_step: int,
        learner_version: int,
    ) -> None:
        async with self._mutation_lock:
            self._require_open()
            if self._trainer is not None and not self._trainer_is_current():
                await self.runtime.stop_trainer(self._trainer)
                self._trainer = None
            if not self._resume_prepared and self._trainer is None:
                _, cancelled = await complete_to_thread(self._resolve_current_lora_path)
                if cancelled is not None:
                    raise cancelled
            if expected_step != self._latest_step:
                raise ValueError(
                    "no-op policy transition expected the wrong service step: "
                    f"expected={expected_step}, current={self._latest_step}"
                )
            if learner_version != expected_step + 1:
                raise ValueError("a no-op policy transition must advance one step")
            trainer = self._trainer
            source = get_step_checkpoint_dir(self.output_dir, expected_step)
            staging = (
                f"{self.output_dir}/megatron_runtime/staging/{learner_version:04d}"
            )
            initial = get_step_checkpoint_dir(self.output_dir, 0)
            published: OptimizerAdapter | None = None
            durable = False
            transition_started = False
            try:
                async with async_optimizer_model_lease(self._optimizer_state_path):
                    current = await asyncio.to_thread(
                        resolve_committed_optimizer_policy,
                        self._optimizer_state_path,
                        initial_adapter_path=initial,
                    )
                    if current.policy_adapter.step > expected_step:
                        raise RuntimeError(
                            "durable policy is newer than the model service"
                        )
                    flush_optimizer = current.policy_adapter.step < expected_step
                    if flush_optimizer and trainer is None:
                        raise RuntimeError(
                            "no resident trainer can checkpoint newer optimizer state"
                        )
                    _, cancelled = await complete_to_thread(
                        lambda: _copy_checkpoint(source, staging)
                    )
                    if cancelled is not None:
                        raise cancelled
                    published, cancelled = await complete_to_thread(
                        lambda: _publish_checkpoint(
                            staging, self.output_dir, learner_version
                        )
                    )
                    self._published_adapters[learner_version] = published
                    assert published is not None
                    if cancelled is not None:
                        raise cancelled
                    if not flush_optimizer:
                        _, cancelled = await complete_to_thread(
                            lambda: commit_optimizer_policy_advance(
                                self._optimizer_state_path,
                                initial_adapter_path=initial,
                                expected_step=expected_step,
                                adapter=cast(OptimizerAdapter, published),
                            )
                        )
                        durable = True
                        if cancelled is not None:
                            raise cancelled
                    if trainer is not None:
                        transition_started = True
                        await trainer.advance_without_training(
                            expected_learner_version=expected_step,
                            learner_version=learner_version,
                            optimizer_state_path=self._optimizer_state_path,
                            adapter=published if flush_optimizer else None,
                        )
                    if flush_optimizer:
                        pointer = await asyncio.to_thread(
                            read_committed_optimizer_pointer,
                            self._optimizer_state_path,
                        )
                        durable = (
                            pointer is not None
                            and pointer.step == learner_version
                            and pointer.adapter == published
                        )
                        if not durable:
                            raise RuntimeError(
                                "trainer did not commit the no-op optimizer checkpoint"
                            )
            except BaseException as error:
                failures: list[BaseException] = [error]
                if published is None:
                    try:
                        published, discovery_cancelled = await complete_to_thread(
                            lambda: read_adapter_publication(
                                get_step_checkpoint_dir(
                                    self.output_dir, learner_version
                                ),
                                step=learner_version,
                                verify_files=True,
                            )
                        )
                        if discovery_cancelled is not None:
                            failures.append(discovery_cancelled)
                    except BaseException as discovery_error:
                        failures.append(discovery_error)
                reconcile_checkpoint: str | None = None
                if published is not None and not durable:
                    try:
                        resume, recovery_cancelled = await complete_to_thread(
                            lambda: prepare_megatron_resume_state(
                                output_dir=self.output_dir,
                                optimizer_state_path=self._optimizer_state_path,
                            )
                        )
                        self._latest_step = resume.step
                        durable = resume.step == learner_version
                        self._published_adapters = {
                            step: adapter
                            for step, adapter in self._published_adapters.items()
                            if step <= resume.step
                        }
                        reconcile_checkpoint = get_step_checkpoint_dir(
                            self.output_dir, resume.step
                        )
                        if recovery_cancelled is not None:
                            failures.append(recovery_cancelled)
                    except BaseException as recovery_error:
                        failures.append(recovery_error)
                if durable:
                    assert published is not None
                    self._latest_step = learner_version
                    reconcile_checkpoint = published.identity
                if trainer is not None and (
                    transition_started or durable or not self._trainer_is_current()
                ):
                    try:
                        await self.runtime.stop_trainer(trainer)
                    except BaseException as cleanup_error:
                        failures.append(cleanup_error)
                    self._trainer = None
                if reconcile_checkpoint is not None:
                    try:
                        await self._reconcile_serving_locked(reconcile_checkpoint)
                    except BaseException as serving_error:
                        failures.append(serving_error)
                if len(failures) > 1:
                    raise BaseExceptionGroup(
                        "no-op policy transition and recovery failed", failures
                    ) from None
                raise
            assert published is not None and durable
            self._latest_step = learner_version
            await self._register_lora_for_step_locked(
                learner_version, published.identity
            )

    async def _register_lora_for_step_locked(self, step: int, checkpoint: str) -> None:
        if self._base_url is None:
            return
        digest = await self._checkpoint_digest(checkpoint, step)
        update_identity = uuid.uuid4().hex
        manager = (
            self.runtime.model_service(self._managed_service_name)
            if self._managed_service_name is not None
            else None
        )
        try:
            state = (
                manager.prepare_update(update_identity=update_identity)
                if manager is not None
                else None
            )
            lora_name, lora_path = await self._load_adapter(checkpoint, step)
            await self._acknowledge_lora_workers(
                lora_name=lora_name,
                lora_path=lora_path,
                step=step,
            )
            if manager is not None and state is not None:
                report = ReplicaUpdateReport(
                    replica_id=manager.spec.name,
                    generation=state.generation,
                    generation_digest=state.generation_digest,
                    policy_version=str(step),
                    policy_digest=digest,
                    update_identity=update_identity,
                )
                if manager.verify_update(report).phase != "ready":
                    raise RuntimeError("model service rejected its LoRA update")
        except BaseException as error:
            if manager is not None:
                manager.quarantine("partial or failed LoRA update")
            try:
                cleanup = await self._rollback_server_start_safely(
                    self._managed_service_name
                )
            finally:
                self._clear_serving_state()
            if cleanup:
                raise BaseExceptionGroup(
                    "LoRA publication and serving rollback failed", [error, *cleanup]
                ) from None
            raise
        self._loaded_adapter_steps.add(step)
        self._current_lora_name = lora_name

    async def _checkpoint_digest(self, checkpoint: str, step: int) -> str:
        published = self._published_adapters.get(step)
        if published is not None and published.identity == str(
            Path(checkpoint).absolute()
        ):
            return published.sha256
        return await asyncio.to_thread(hash_adapter_checkpoint, checkpoint)

    async def acquire_exact_adapter(self, step: int, checkpoint: str) -> str:
        async with self._mutation_lock:
            self._require_open()
            lora_name = (
                f"{self.model_name}:eval@{step}"
                if self.rollout_weight_update_mode == "in_flight_lora"
                else f"{self.model_name}@{step}"
            )
            if step not in self._loaded_exact_adapter_steps:
                if (
                    self.rollout_weight_update_mode == "in_flight_lora"
                    or step not in self._loaded_adapter_steps
                ):
                    lora_name, lora_path = await self._load_adapter(
                        checkpoint, step, exact=True
                    )
                    await self._acknowledge_lora_workers(
                        lora_name=lora_name,
                        lora_path=lora_path,
                        step=step,
                    )
                self._loaded_exact_adapter_steps.add(step)
                self._exact_adapter_refcounts[step] = 0
            self._exact_adapter_refcounts[step] += 1
        return (
            f"{self.model_name}:eval@{step}"
            if self.rollout_weight_update_mode == "in_flight_lora"
            else f"{self.model_name}@{step}"
        )

    async def release_exact_adapter(self, step: int) -> None:
        async with self._mutation_lock:
            self._require_open()
            count = self._exact_adapter_refcounts.get(step, 0)
            if count <= 1:
                self._exact_adapter_refcounts.pop(step, None)
                if self.rollout_weight_update_mode == "in_flight_lora":
                    await self._unload_adapter(f"{self.model_name}:eval@{step}")
                self._loaded_exact_adapter_steps.discard(step)
            else:
                self._exact_adapter_refcounts[step] = count - 1

    async def prune_loaded_adapters(self, *, retain_steps: set[int]) -> None:
        async with self._mutation_lock:
            self._require_open()
            for step in sorted(
                self._loaded_adapter_steps - retain_steps - {self._latest_step}
            ):
                await self._unload_adapter(f"{self.model_name}@{step}")
                self._loaded_adapter_steps.discard(step)

    async def _unload_adapter(self, name: str) -> None:
        if self._base_url is None:
            raise RuntimeError("vLLM serving has not started")
        await self._unload_adapter_at(name, self._base_url)

    async def _unload_adapter_at(self, name: str, base_url: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/v1/unload_lora_adapter",
                json={"lora_name": name},
                headers=_headers(self._api_key()),
            )
        if response.status_code != 404:
            response.raise_for_status()

    async def get_serving_capabilities(self) -> ServingCapabilities:
        if self._serving_capabilities is None:
            raise RuntimeError("vLLM serving capabilities have not been discovered")
        return self._serving_capabilities

    async def vllm_engine_is_sleeping(self) -> bool:
        return False

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(_consume_task_result)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        async with self._mutation_lock:
            failures = []
            recovery_tasks = tuple(self._recovery_tasks)
            for task in recovery_tasks:
                task.cancel()
            if recovery_tasks:
                await asyncio.gather(*recovery_tasks, return_exceptions=True)
            self._recovery_tasks.clear()
            operations = []
            if self._trainer is not None:
                operations.append(self.runtime.stop_trainer(self._trainer))
            if self._managed_service_name is not None:
                operations.append(
                    self.runtime.stop_model_service(self._managed_service_name)
                )
            results = await asyncio.gather(*operations, return_exceptions=True)
            failures.extend(
                result for result in results if isinstance(result, BaseException)
            )
            if failures:
                raise BaseExceptionGroup(
                    "distributed model service close failed", failures
                )


def _copy_checkpoint(source: str, destination: str) -> None:
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _publish_checkpoint(staging: str, output_dir: str, step: int) -> OptimizerAdapter:
    adapter = publish_adapter_checkpoint(staging, step=step)
    expected = str(Path(get_step_checkpoint_dir(output_dir, step)).absolute())
    if adapter.identity != expected:
        raise RuntimeError(
            "Published adapter path does not match the service output: "
            f"published={adapter.identity}, expected={expected}"
        )
    return adapter


def _trainer_dtype(
    config: dev.BackendModelConfig,
) -> Literal["bfloat16", "float16", "float32"]:
    value = str(config.get("init_args", {}).get("dtype") or "bfloat16").lower()
    value = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "torch.bfloat16": "bfloat16",
        "torch.float16": "float16",
        "torch.float32": "float32",
    }.get(value, value)
    if value not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported Megatron trainer dtype {value!r}")
    return cast(
        Literal["bfloat16", "float16", "float32"],
        value,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _art_source_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _headers(api_key: str | None) -> dict[str, str] | None:
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def _host_port(base_url: str) -> tuple[str, int]:
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    assert parsed.hostname is not None
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
