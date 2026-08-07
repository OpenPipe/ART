from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
import hashlib
from itertools import groupby
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Literal, cast
import uuid

import httpx

from art import dev, types
from art.dev.get_model_config import default_target_modules
from art.distributed.art_runtime import ArtRuntime, DistributedPackedBatch
from art.distributed.specs import ModelServiceSpec, NixlTransportSpec, TrainerMeshSpec
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
    CheckpointFile,
    OptimizerAdapter,
    adapter_generation_lease,
    commit_optimizer_policy_advance,
    format_megatron_resume_message,
    new_optimizer_generation,
    optimizer_adapter,
    prepare_megatron_resume_state,
    publish_adapter_checkpoint,
    read_adapter_publication,
    read_committed_optimizer_pointer,
    resolve_committed_optimizer_policy,
    trainer_publication_path,
)
from .runtime.specs import (
    AdapterReady,
    CurrentTrainConfig,
    DurableTrainOutput,
    ExperimentalTrainConfig,
    HybridEpRuntimeSpec,
    TrainAccepted,
    TrainCancelled,
    TrainCompleted,
    TrainerGeneration,
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


def _retire_completed(
    records: dict[int, Any], step: int, completed: asyncio.Future[Any]
) -> None:
    if records.get(step) is completed:
        records.pop(step)


def _hybrid_ep_runtime_spec(
    mesh: TrainerMeshSpec,
    *,
    run_id: str,
    transport: NixlTransportSpec | None,
) -> HybridEpRuntimeSpec | None:
    if mesh.topology.ep <= 1:
        return None
    group_size = mesh.topology.etp * mesh.topology.ep
    domain_sizes: set[int] = set()
    multinode = False
    for offset in range(0, len(mesh.ranks), group_size):
        group = mesh.ranks[offset : offset + group_size]
        domains = [
            (host_id, tuple(ranks))
            for host_id, ranks in groupby(group, key=lambda rank: rank.host_id)
        ]
        if len({host_id for host_id, _ in domains}) != len(domains):
            raise ValueError("HybridEP ranks for each host must be contiguous")
        domain_sizes.update(len(ranks) for _, ranks in domains)
        multinode |= len(domains) > 1
    if len(domain_sizes) != 1:
        raise ValueError(
            "HybridEP TP x EP groups require equal ranks per NVLink domain"
        )
    if multinode and transport is None:
        raise ValueError("cross-host expert parallelism requires NIXL transport")
    return HybridEpRuntimeSpec(
        ranks_per_nvlink_domain=domain_sizes.pop(),
        run_id=run_id,
        nixl_transport=transport if multinode else None,
    )


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
        self._serving_step = 0
        self._durable_step = 0
        self._durable_optimizer_step = 0
        self._resume_prepared = False
        self._training_session_id = uuid.uuid4().hex
        self._learner_generation: TrainerGeneration | None = None
        self._trainer_resident_generation: TrainerGeneration | None = None
        self._trainer: Any = None
        self._train_lock = asyncio.Lock()
        self._mutation_lock = asyncio.Lock()
        self._serving_lock = asyncio.Lock()
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
        self._publication_tasks: dict[int, asyncio.Task[None]] = {}
        self._serving_futures: dict[int, asyncio.Future[None]] = {}
        self._publication_failure: BaseException | None = None
        self._publication_metrics: dict[int, dict[str, float]] = {}
        self._emitted_publication_metrics: dict[int, set[str]] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def rollout_weights_mode(self) -> str:
        return self.config.get("rollout_weights_mode", "lora")

    @property
    def active_learner_step(self) -> int:
        return self._latest_step

    @property
    def serving_step(self) -> int:
        return self._serving_step

    @property
    def durable_step(self) -> int:
        return self._durable_step

    @property
    def durable_optimizer_step(self) -> int:
        return self._durable_optimizer_step

    def drain_publication_metrics(self) -> dict[str, float]:
        metrics = {
            "publication/active_learner_serving_lag_steps": float(
                self._latest_step - self._serving_step
            ),
            "publication/durable_optimizer_lag_steps": float(
                self._latest_step - self._durable_optimizer_step
            ),
            "publication/queue_depth": float(
                sum(not task.done() for task in self._publication_tasks.values())
            ),
        }
        for step in sorted(self._publication_metrics):
            values = self._publication_metrics[step]
            emitted = self._emitted_publication_metrics.setdefault(step, set())
            for name, value in values.items():
                if name not in emitted:
                    metrics[f"publication/{name}"] = value
            emitted.update(values)
            task = self._publication_tasks.get(step)
            if task is None or task.done():
                self._publication_metrics.pop(step, None)
                self._emitted_publication_metrics.pop(step, None)
        return metrics

    async def finalize_publication_metrics(self, step: int) -> dict[str, float]:
        async with self._mutation_lock:
            self._require_open()
            if step != self._latest_step:
                raise ValueError(
                    f"final publication step {step} != learner step {self._latest_step}"
                )
            publication = self._publication_tasks.get(step)
        if publication is not None:
            await asyncio.shield(publication)
        self._raise_publication_failure()
        return self.drain_publication_metrics()

    async def wait_for_serving(self, step: int) -> None:
        async with self._mutation_lock:
            self._require_open()
            self._raise_publication_failure()
            if step < 0 or step > self._latest_step:
                raise ValueError(
                    f"serving step {step} is outside learner lineage 0..{self._latest_step}"
                )
            serving = self._serving_futures.get(step)
        async with self._serving_lock:
            if step <= self._serving_step:
                return
            if serving is None:
                raise RuntimeError(f"learner step {step} has no serving publication")
        await asyncio.shield(serving)
        self._raise_publication_failure()

    def checkpoint_materialization(self, step: int) -> asyncio.Task[None]:
        self._require_open()
        self._raise_publication_failure()
        generation = self._learner_generation
        if generation is None or generation.policy_step != step:
            raise RuntimeError(
                f"learner generation {step} is unavailable for materialization"
            )

        async def wait() -> None:
            await asyncio.to_thread(
                _wait_for_adapter_generation,
                self._optimizer_state_path,
                generation,
            )

        task = asyncio.create_task(wait())
        task.add_done_callback(_consume_task_result)
        return task

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
            if self._learner_generation is None:
                raise RuntimeError("resident trainer has no learner generation")
            self._resume_prepared = True
            return self._learner_generation.adapter_path
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
        if self._latest_step == 0:
            adapter = optimizer_adapter(
                path,
                0,
                training_session_id=self._training_session_id,
            )
        else:
            policy = resolve_committed_optimizer_policy(
                self._optimizer_state_path,
                initial_adapter_path=get_step_checkpoint_dir(self.output_dir, 0),
            )
            adapter = policy.policy_adapter
            if adapter.step != self._latest_step:
                raise RuntimeError("resume policy and checkpoint step disagree")
            self._training_session_id = adapter.training_session_id
        self._published_adapters[self._latest_step] = adapter
        self._learner_generation = TrainerGeneration(
            training_session_id=adapter.training_session_id,
            policy_step=adapter.step,
            generation_id=adapter.generation_id,
            adapter_path=adapter.identity,
        )
        self._durable_step = resume.step
        self._durable_optimizer_step = resume.optimizer_step or 0
        self._resume_prepared = True
        return adapter.identity

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
        hybrid_ep = _hybrid_ep_runtime_spec(
            mesh,
            run_id=self.runtime.runtime_id,
            transport=self.runtime.topology.cluster.nixl_transport,
        )
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
            cache_root=self.runtime.topology.cluster.cache_root,
            model_support_key=support_spec.key,
            handler_name=handler.key,
            lora_rank=int(lora.get("rank") or default_lora_rank_for_handler(handler)),
            lora_alpha=float(lora.get("alpha", LORA_ALPHA)),
            lora_target_modules=tuple(targets),
            dtype=_trainer_dtype(self.config),
            trainer_mesh=mesh,
            packed_sequence_length=runtime_config.packed_sequence_length,
            snapshot_pool_capacity=runtime_config.snapshot_pool_capacity,
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
            hybrid_ep=hybrid_ep,
        )

    async def _ensure_trainer_locked(self) -> tuple[Any, tuple[int, str] | None]:
        if self._trainer_is_current():
            return self._trainer, None
        current, reconcile_step = await self._prepare_for_packing_locked()
        assert self._trainer is None
        runtime_spec = self._runtime_spec()
        run_spec = TrainingRunSpec(
            run_id=uuid.uuid4().hex,
            runtime_fingerprint=runtime_spec.fingerprint,
            training_session_id=self._training_session_id,
            initial_learner_version=self._latest_step,
            initial_adapter_path=current,
            optimizer_state_path=self._optimizer_state_path,
            initial_event_timeout_s=self.runtime.topology.cluster.startup_timeout_s,
        )
        self._trainer = await self.runtime.start_trainer(runtime_spec, run_spec)
        self._trainer_resident_generation = None
        reconcile = None if reconcile_step is None else (reconcile_step, current)
        return self._trainer, reconcile

    async def prepare_for_packing(self) -> int:
        async with self._train_lock:
            async with self._mutation_lock:
                self._require_open()
                self._raise_publication_failure()
                _current, reconcile_step = await self._prepare_for_packing_locked()
                step = self._latest_step
            if reconcile_step is not None:
                async with self._serving_lock:
                    await self._reconcile_serving_locked(step, _current)
            return step

    async def _prepare_for_packing_locked(self) -> tuple[str, int | None]:
        if self._trainer_is_current():
            return self._resolve_current_lora_path(), None
        if self._trainer is not None:
            _, cancelled = await complete_task(
                asyncio.create_task(self.runtime.stop_trainer(self._trainer))
            )
            self._trainer = None
            self._trainer_resident_generation = None
            if cancelled is not None:
                raise cancelled
        previous_step = self._latest_step
        current, cancelled = await complete_to_thread(self._resolve_current_lora_path)
        if cancelled is not None:
            raise cancelled
        reconcile_step = (
            self._latest_step if self._latest_step != previous_step else None
        )
        return current, reconcile_step

    async def _reconcile_serving_locked(self, step: int, checkpoint: str) -> None:
        if self._base_url is None:
            self._serving_step = step
            return
        previous_name = self._current_lora_name
        await self._register_lora_for_step_locked(step, checkpoint)
        invalid_exact = {
            step
            for step in self._loaded_exact_adapter_steps
            if step > self._serving_step
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
            step for step in self._loaded_adapter_steps if step > self._serving_step
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
    ) -> AsyncIterator[AsyncIterator[Any]]:
        cold = self._trainer_resident_generation != job.source
        source = self._published_adapters.get(job.source.policy_step) if cold else None
        if cold and (
            source is None
            or source.training_session_id != job.source.training_session_id
            or source.generation_id != job.source.generation_id
            or source.identity != str(Path(job.source.adapter_path).absolute())
        ):
            raise RuntimeError("cold trainer source generation is not registered")
        with adapter_generation_lease(source) if source is not None else nullcontext():
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
        async with self._train_lock:
            async with self._mutation_lock:
                self._require_open()
                self._raise_publication_failure()
                trainer, reconcile = await self._ensure_trainer_locked()
                source = self._learner_generation
                if source is None:
                    raise RuntimeError("trainer has no source generation")
                next_step = self._latest_step + 1
                generation_id = new_optimizer_generation(next_step)
                final_path = get_step_checkpoint_dir(self.output_dir, next_step)
                output_generation = TrainerGeneration(
                    training_session_id=self._training_session_id,
                    policy_step=next_step,
                    generation_id=generation_id,
                    adapter_path=final_path,
                )
                publication_targets = ()
                if (
                    self._managed_service_name is not None
                    and self.rollout_weight_update_mode == "in_flight_lora"
                ):
                    publication_targets = await self.runtime.model_service(
                        self._managed_service_name
                    ).prepare_adapter_transfer(
                        generation_id,
                        get_step_checkpoint_dir(self.output_dir, 0),
                        trainer.runtime_spec.dtype,
                    )
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
                    source=source,
                    batch=batch.leases.ref,
                    config=CurrentTrainConfig.model_validate(config.model_dump()),
                    experimental_config=ExperimentalTrainConfig.model_validate(values),
                    output=DurableTrainOutput(
                        generation=output_generation,
                        staging_adapter_path=(
                            f"{self.output_dir}/megatron_runtime/staging/"
                            f"{generation_id}"
                        ),
                        optimizer_state_path=self._optimizer_state_path,
                    ),
                    publication_targets=publication_targets,
                )

            if reconcile is not None:
                reconcile_step, checkpoint = reconcile
                async with self._serving_lock:
                    await self._reconcile_serving_locked(reconcile_step, checkpoint)

            snapshot_prepared = False
            completed = False
            final_metrics: dict[str, float] | None = None
            async with self._trainer_transaction(trainer, job, batch.leases) as events:
                async for event in events:
                    if event.job_id != job.job_id or event.run_id != job.run_id:
                        raise RuntimeError(
                            "trainer returned an event for a different job"
                        )
                    if isinstance(event, TrainAccepted):
                        continue
                    if isinstance(event, TrainProgress):
                        if event.step_index + 1 == event.num_steps:
                            final_metrics = dict(event.metrics)
                        else:
                            yield event.metrics
                        continue
                    if isinstance(event, AdapterReady):
                        if snapshot_prepared:
                            raise RuntimeError(
                                "trainer returned duplicate snapshot events"
                            )
                        if (
                            event.learner_version != next_step
                            or event.adapter_path != final_path
                        ):
                            raise RuntimeError("trainer prepared the wrong generation")
                        snapshot_prepared = True
                        continue
                    if isinstance(event, TrainCompleted):
                        if completed or not snapshot_prepared:
                            raise RuntimeError("trainer completed without one snapshot")
                        if event.learner_version != next_step:
                            raise RuntimeError("trainer completed the wrong learner")
                        snapshot_metrics = {
                            name: value
                            for name, value in event.metrics.items()
                            if name.startswith("snapshot_")
                        }
                        self._publication_metrics[next_step] = snapshot_metrics
                        self._emitted_publication_metrics[next_step] = set(
                            snapshot_metrics
                        )
                        final_metrics = dict(event.metrics)
                        completed = True
                        continue
                    if isinstance(event, TrainFailed):
                        raise RuntimeError(
                            f"distributed Megatron job failed ({event.error_type}): "
                            f"{event.message}"
                        )
                    if isinstance(event, TrainCancelled):
                        raise asyncio.CancelledError(event.reason)
            if not snapshot_prepared or not completed or final_metrics is None:
                raise RuntimeError("trainer ended without preparing a generation")

            async with self._mutation_lock:
                if self._latest_step != job.expected_learner_version:
                    raise RuntimeError("learner lineage changed during training")
                self._latest_step = next_step
                self._learner_generation = output_generation
                self._trainer_resident_generation = output_generation
                self._schedule_publication(
                    output_generation,
                    publication_targets=job.publication_targets,
                )
            yield final_metrics

    def _schedule_publication(
        self,
        generation: TrainerGeneration,
        *,
        publication_targets: tuple[Any, ...] = (),
    ) -> None:
        step = generation.policy_step
        if step in self._publication_tasks:
            raise RuntimeError(f"generation publication already exists for step {step}")
        loop = asyncio.get_running_loop()
        previous = self._serving_futures.get(step - 1)
        if previous is None:
            previous = loop.create_future()
            previous.set_result(None)
        serving = loop.create_future()
        serving.add_done_callback(_consume_task_result)
        self._serving_futures[step] = serving
        serving.add_done_callback(
            lambda done: _retire_completed(self._serving_futures, step, done)
        )
        publication = (
            self._publish_generation(
                generation,
                publication_targets=publication_targets,
                previous_serving=previous,
                serving=serving,
            )
            if publication_targets
            else self._publish_generation(
                generation,
                previous_serving=previous,
                serving=serving,
            )
        )
        task = asyncio.create_task(publication)
        self._publication_tasks[step] = task
        task.add_done_callback(_consume_task_result)
        task.add_done_callback(
            lambda done: _retire_completed(self._publication_tasks, step, done)
        )

    async def _publish_generation(
        self,
        generation: TrainerGeneration,
        *,
        publication_targets: tuple[Any, ...] = (),
        previous_serving: asyncio.Future[None],
        serving: asyncio.Future[None],
    ) -> None:
        metrics = self._publication_metrics.setdefault(generation.policy_step, {})
        try:
            materialization_started = time.monotonic()
            manager = (
                self.runtime.model_service(self._managed_service_name)
                if publication_targets and self._managed_service_name is not None
                else None
            )
            if manager is None:
                adapter = await asyncio.to_thread(
                    _wait_for_adapter_generation,
                    self._optimizer_state_path,
                    generation,
                )
                checkpoint = generation.adapter_path
                metrics["adapter_materialization_s"] = (
                    time.monotonic() - materialization_started
                )
            else:
                received = await manager.wait_adapter_transfer(generation.generation_id)
                if len(received) != len(publication_targets):
                    raise RuntimeError("Not every inference host received the adapter")
                paths = {result.path for result in received}
                sizes = {
                    (result.tensor_bytes, result.config_bytes) for result in received
                }
                if len(paths) != 1 or len(sizes) != 1:
                    raise RuntimeError(
                        "Inference hosts materialized different adapters"
                    )
                tensor_bytes, config_bytes = sizes.pop()
                checkpoint = paths.pop()
                adapter = OptimizerAdapter(
                    identity=str(Path(generation.adapter_path).absolute()),
                    training_session_id=generation.training_session_id,
                    step=generation.policy_step,
                    generation_id=generation.generation_id,
                    files=(
                        CheckpointFile(
                            name="adapter_config.json", size_bytes=config_bytes
                        ),
                        CheckpointFile(
                            name="adapter_model.safetensors", size_bytes=tensor_bytes
                        ),
                    ),
                )
                metrics["adapter_transport_wait_s"] = (
                    time.monotonic() - materialization_started
                )
                metrics["adapter_transport_bytes"] = float(tensor_bytes * len(received))
                metrics["adapter_materialization_s"] = max(
                    result.materialization_s for result in received
                )
            await previous_serving
            self._raise_publication_failure()
            activation_started = time.monotonic()
            async with self._mutation_lock:
                self._published_adapters[generation.policy_step] = adapter
            async with self._serving_lock:
                try:
                    await self._register_lora_for_step_locked(
                        generation.policy_step,
                        checkpoint,
                    )
                finally:
                    if manager is not None:
                        await manager.release_adapter_transfer(generation.generation_id)
            metrics["serving_activation_s"] = time.monotonic() - activation_started
            if not serving.done():
                serving.set_result(None)

            durable_started = time.monotonic()
            durable_step, optimizer_step = await asyncio.to_thread(
                _wait_for_durable_generation,
                self._optimizer_state_path,
                generation,
            )
            durable_adapter = await asyncio.to_thread(
                _wait_for_adapter_generation,
                self._optimizer_state_path,
                generation,
            )
            if durable_adapter != adapter:
                raise RuntimeError("Durable and serving adapter manifests differ")
            async with self._mutation_lock:
                self._durable_step = max(self._durable_step, durable_step)
                self._durable_optimizer_step = max(
                    self._durable_optimizer_step, optimizer_step
                )
                metrics["durable_checkpoint_s"] = time.monotonic() - durable_started
                metrics["durable_checkpoint_lag_steps"] = float(
                    self._latest_step - self._durable_optimizer_step
                )
            logger.info(
                "Published trainer generation session=%s step=%d generation=%s "
                "prepare=%.3fs activate=%.3fs durable=%.3fs durable_lag=%d",
                generation.training_session_id,
                generation.policy_step,
                generation.generation_id,
                metrics.get("snapshot_prepare_s", 0.0),
                metrics["serving_activation_s"],
                metrics["durable_checkpoint_s"],
                self._latest_step - self._durable_optimizer_step,
            )
        except BaseException as error:
            if not serving.done():
                serving.set_exception(error)
            self._publication_failure = error
            async with self._serving_lock:
                cleanup = await self._rollback_server_start_safely(
                    self._managed_service_name
                )
                self._clear_serving_state()
            logger.exception(
                "Trainer generation publication failed session=%s step=%d generation=%s",
                generation.training_session_id,
                generation.policy_step,
                generation.generation_id,
            )
            if cleanup:
                raise BaseExceptionGroup(
                    "generation publication and serving teardown failed",
                    [error, *cleanup],
                ) from None
            raise

    def _raise_publication_failure(self) -> None:
        if self._publication_failure is not None:
            raise RuntimeError("trainer generation publication failed") from (
                self._publication_failure
            )

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
        async with self._train_lock:
            self._require_open()
            if serving := self._serving_futures.get(self._latest_step):
                await serving
            async with self._serving_lock:
                if self._base_url:
                    return _host_port(self._base_url)
                if self._managed_service_name is not None:
                    raise RuntimeError("managed model service is unavailable")
            async with self._mutation_lock:
                lora_path = await asyncio.to_thread(self._resolve_current_lora_path)
                step = self._latest_step
            async with self._serving_lock:
                if self._base_url:
                    return _host_port(self._base_url)
                if self._managed_service_name is not None:
                    raise RuntimeError("managed model service is unavailable")
                return await self._start_openai_server_locked(
                    config, lora_path=lora_path, step=step
                )

    async def _start_openai_server_locked(
        self,
        config: dev.OpenAIServerConfig | None,
        *,
        lora_path: str,
        step: int,
    ) -> tuple[str, int]:
        api_key = self._api_key(config)
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
                step,
                base_url=base_url,
                api_key=api_key,
                active_step=step,
            )
            self._publish_serving_state(
                managed_service_name=None,
                base_url=base_url,
                capabilities=capabilities,
                api_key=api_key,
                current_lora_name=lora_name,
                serving_step=step,
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
            served_model_name=f"{self.model_name}@{step}",
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
            generation_id = self._generation_id_for_step(step)
            update_identity = uuid.uuid4().hex
            manager = self.runtime.model_service(service.name)
            state = manager.prepare_update(update_identity=update_identity)
            report = ReplicaUpdateReport(
                replica_id=service.name,
                generation=state.generation,
                generation_digest=state.generation_digest,
                policy_version=str(step),
                policy_digest=generation_id,
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
            serving_step=step,
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
        serving_step: int,
    ) -> None:
        self._managed_service_name = managed_service_name
        self._base_url = base_url
        self._serving_capabilities = capabilities
        self._api_key_value = api_key
        self._current_lora_name = current_lora_name
        self._serving_step = serving_step
        self._loaded_adapter_steps.add(serving_step)

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
        async with self._serving_lock:
            try:
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
        serving_step = self._serving_step
        serving_adapter = self._published_adapters.get(serving_step)
        if serving_adapter is None:
            raise RuntimeError(
                f"serving generation {serving_step} is not registered for recovery"
            )
        checkpoint = serving_adapter.identity
        generation_id = serving_adapter.generation_id
        current_lora_name = self._current_lora_name or (
            f"{self.model_name}@{serving_step}"
        )
        bootstrap_name = f"{self.model_name}@{serving_step}"
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
                    serving_step,
                    base_url=base_url,
                    api_key=self._api_key(),
                    active_step=serving_step - 1,
                )
            report = ReplicaUpdateReport(
                replica_id=failure.replica_id,
                generation=state.generation,
                generation_digest=state.generation_digest,
                policy_version=str(serving_step),
                policy_digest=generation_id,
                update_identity=update_identity,
            )
            if manager.verify_update(report).phase != "ready":
                raise RuntimeError("restarted vLLM replica rejected current policy")
            if current_lora_name != bootstrap_name:
                await self._unload_adapter_at(bootstrap_name, base_url)
            for step in exact_steps:
                if step == serving_step and self.rollout_weight_update_mode != (
                    "in_flight_lora"
                ):
                    continue
                exact_name, exact_path = await self._load_adapter_at(
                    get_step_checkpoint_dir(self.output_dir, step),
                    step,
                    exact=True,
                    base_url=base_url,
                    api_key=self._api_key(),
                    active_step=serving_step,
                )
            self._current_lora_name = lora_name
            self._loaded_adapter_steps = {serving_step}
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
            active_step=self._serving_step,
        )

    async def _load_adapter_at(
        self,
        checkpoint: str,
        step: int,
        *,
        base_url: str,
        api_key: str | None,
        active_step: int,
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
            and step != active_step
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

    async def register_lora_for_step(self, step: int, checkpoint: str) -> None:
        async with self._train_lock:
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
            adapter = policy.policy_adapter
            generation = TrainerGeneration(
                training_session_id=adapter.training_session_id,
                policy_step=adapter.step,
                generation_id=adapter.generation_id,
                adapter_path=adapter.identity,
            )
            async with self._mutation_lock:
                self._published_adapters[step] = adapter
                self._training_session_id = adapter.training_session_id
                self._learner_generation = generation
                self._latest_step = step
                self._durable_step = step
                self._durable_optimizer_step = (
                    0
                    if policy.optimizer_anchor is None
                    else policy.optimizer_anchor.step
                )
            async with self._serving_lock:
                await self._register_lora_for_step_locked(step, checkpoint)

    async def advance_without_training(
        self,
        *,
        expected_step: int,
        learner_version: int,
    ) -> dict[str, float]:
        async with self._train_lock:
            metrics = self.drain_publication_metrics()
            async with self._mutation_lock:
                self._require_open()
                self._raise_publication_failure()
                if expected_step != self._latest_step:
                    raise ValueError(
                        "no-op policy transition expected the wrong learner step"
                    )
                if learner_version != expected_step + 1:
                    raise ValueError("a no-op policy transition must advance one step")
                trainer = self._trainer if self._trainer_is_current() else None
                previous = self._publication_tasks.get(expected_step)
            if trainer is not None:
                await self.wait_for_serving(expected_step)
            elif previous is not None:
                await previous

            async with self._mutation_lock:
                source = self._learner_generation
                if source is None or source.policy_step != expected_step:
                    raise RuntimeError("no-op transition has no immutable source")
                generation = TrainerGeneration(
                    training_session_id=self._training_session_id,
                    policy_step=learner_version,
                    generation_id=new_optimizer_generation(learner_version),
                    adapter_path=get_step_checkpoint_dir(
                        self.output_dir, learner_version
                    ),
                )
            prepare_started = time.monotonic()
            published = await asyncio.to_thread(
                _publish_adapter_alias,
                self._published_adapters[expected_step],
                generation,
                f"{self.output_dir}/megatron_runtime/staging/"
                f"{generation.generation_id}",
            )
            snapshot_metrics = {
                "snapshot_prepare_s": time.monotonic() - prepare_started
            }
            self._published_adapters[learner_version] = published
            if trainer is None:
                await asyncio.to_thread(
                    commit_optimizer_policy_advance,
                    self._optimizer_state_path,
                    initial_adapter_path=get_step_checkpoint_dir(self.output_dir, 0),
                    expected_step=expected_step,
                    adapter=published,
                )
                await asyncio.to_thread(
                    _write_generation_complete,
                    self._optimizer_state_path,
                    generation,
                )
            else:
                snapshot_metrics.update(
                    await trainer.advance_without_training(
                        expected_learner_version=expected_step,
                        learner_version=learner_version,
                        optimizer_state_path=self._optimizer_state_path,
                        adapter=published,
                    )
                )

            async with self._mutation_lock:
                self._latest_step = learner_version
                self._learner_generation = generation
                self._trainer_resident_generation = (
                    generation if trainer is not None else None
                )
                self._publication_metrics[learner_version] = snapshot_metrics
                self._schedule_publication(generation)
            metrics.update(self.drain_publication_metrics())
            return metrics

    async def _register_lora_for_step_locked(self, step: int, checkpoint: str) -> None:
        if self._base_url is None:
            self._serving_step = step
            return
        generation_id = self._generation_id_for_step(step)
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
            lora_name, _lora_path = await self._load_adapter(checkpoint, step)
            if manager is not None and state is not None:
                report = ReplicaUpdateReport(
                    replica_id=manager.spec.name,
                    generation=state.generation,
                    generation_digest=state.generation_digest,
                    policy_version=str(step),
                    policy_digest=generation_id,
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
        if self.rollout_weight_update_mode != "in_flight_lora":
            self._loaded_adapter_steps.add(step)
        self._current_lora_name = lora_name
        self._serving_step = step

    def _generation_id_for_step(self, step: int) -> str:
        published = self._published_adapters.get(step)
        if published is None:
            raise RuntimeError(f"No immutable generation is registered for step {step}")
        return published.generation_id

    async def acquire_exact_adapter(self, step: int, checkpoint: str) -> str:
        self._require_open()
        async with self._mutation_lock:
            published = step in self._published_adapters
        if not published:
            adapter = await asyncio.to_thread(
                read_adapter_publication,
                checkpoint,
                step=step,
                verify_files=True,
            )
            if adapter is None:
                if step != 0:
                    raise RuntimeError("exact adapter is not an immutable generation")
                adapter = optimizer_adapter(
                    checkpoint,
                    0,
                    training_session_id=self._training_session_id,
                )
            async with self._mutation_lock:
                self._require_open()
                self._published_adapters.setdefault(step, adapter)
        async with self._serving_lock:
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
                    lora_name, _lora_path = await self._load_adapter(
                        checkpoint, step, exact=True
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
        async with self._serving_lock:
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
        if self.rollout_weight_update_mode == "in_flight_lora":
            return
        async with self._serving_lock:
            self._require_open()
            for step in sorted(
                self._loaded_adapter_steps - retain_steps - {self._serving_step}
            ):
                await self._unload_adapter(f"{self.model_name}@{step}")
                self._loaded_adapter_steps.discard(step)

    @asynccontextmanager
    async def checkpoint_retention_lease(self) -> AsyncIterator[frozenset[int]]:
        async with self._mutation_lock:
            self._require_open()
            async with self._serving_lock:
                protected = frozenset((self._latest_step, self._serving_step))
            yield protected

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
        async with self._train_lock:
            failures: list[BaseException] = []
            publications = tuple(self._publication_tasks.values())
            if publications:
                results = await asyncio.gather(*publications, return_exceptions=True)
                publication_failures = [
                    result for result in results if isinstance(result, BaseException)
                ]
                failures.extend(publication_failures)
            else:
                publication_failures = []
            if self._publication_failure is not None and not publication_failures:
                failures.append(self._publication_failure)
            recovery_tasks = tuple(self._recovery_tasks)
            for task in recovery_tasks:
                task.cancel()
            if recovery_tasks:
                await asyncio.gather(*recovery_tasks, return_exceptions=True)
            self._recovery_tasks.clear()
            async with self._mutation_lock:
                trainer = self._trainer
                self._trainer = None
                self._publication_tasks.clear()
                self._serving_futures.clear()
                self._publication_metrics.clear()
                self._emitted_publication_metrics.clear()
            if trainer is not None:
                result = await asyncio.gather(
                    self.runtime.stop_trainer(trainer), return_exceptions=True
                )
                failures.extend(
                    value for value in result if isinstance(value, BaseException)
                )
            async with self._serving_lock:
                if self._managed_service_name is not None:
                    result = await asyncio.gather(
                        self.runtime.stop_model_service(self._managed_service_name),
                        return_exceptions=True,
                    )
                    failures.extend(
                        value for value in result if isinstance(value, BaseException)
                    )
                self._clear_serving_state()
            _, cancelled = await complete_to_thread(
                lambda: _remove_staging_root(self.output_dir)
            )
            if cancelled is not None:
                failures.append(cancelled)
            if failures:
                raise BaseExceptionGroup(
                    "distributed model service close failed", failures
                )


def _remove_staging_checkpoint(staging: str) -> None:
    if os.path.exists(staging):
        shutil.rmtree(staging)


def _remove_staging_root(output_dir: str) -> None:
    _remove_staging_checkpoint(f"{output_dir}/megatron_runtime/staging")


def _publish_adapter_alias(
    source: OptimizerAdapter,
    generation: TrainerGeneration,
    staging_path: str,
) -> OptimizerAdapter:
    staging = Path(staging_path)
    if staging.exists() or Path(generation.adapter_path).exists():
        raise RuntimeError("no-op adapter generation path already exists")
    with adapter_generation_lease(source):
        staging.mkdir(parents=True)
        try:
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                os.link(Path(source.identity) / name, staging / name)
            return publish_adapter_checkpoint(
                staging,
                step=generation.policy_step,
                training_session_id=generation.training_session_id,
                generation_id=generation.generation_id,
            )
        except BaseException:
            _remove_staging_checkpoint(str(staging))
            raise


def _write_generation_complete(
    optimizer_state_path: str,
    generation: TrainerGeneration,
) -> None:
    pointer = read_committed_optimizer_pointer(optimizer_state_path)
    path = (
        trainer_publication_path(optimizer_state_path, generation.generation_id)
        / "complete.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(
                {
                    "generation": generation.model_dump(mode="json"),
                    "resume_step": generation.policy_step,
                    "optimizer_step": 0 if pointer is None else pointer.step,
                },
                output,
                sort_keys=True,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _wait_for_adapter_generation(
    optimizer_state_path: str,
    generation: TrainerGeneration,
    *,
    timeout_s: float = 300.0,
) -> OptimizerAdapter:
    deadline = time.monotonic() + timeout_s
    failed = (
        trainer_publication_path(optimizer_state_path, generation.generation_id)
        / "failed.json"
    )
    while True:
        adapter = read_adapter_publication(
            generation.adapter_path,
            step=generation.policy_step,
            verify_files=True,
        )
        if adapter is not None:
            if (
                adapter.training_session_id,
                adapter.step,
                adapter.generation_id,
                adapter.identity,
            ) != (
                generation.training_session_id,
                generation.policy_step,
                generation.generation_id,
                str(Path(generation.adapter_path).absolute()),
            ):
                raise RuntimeError(
                    "adapter publication has the wrong generation identity"
                )
            return adapter
        if error := _publication_marker_failure(failed):
            raise error
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for adapter generation {generation.generation_id}"
            )
        time.sleep(0.05)


def _wait_for_durable_generation(
    optimizer_state_path: str,
    generation: TrainerGeneration,
    *,
    timeout_s: float = 300.0,
) -> tuple[int, int]:
    deadline = time.monotonic() + timeout_s
    complete = (
        trainer_publication_path(optimizer_state_path, generation.generation_id)
        / "complete.json"
    )
    failed = complete.with_name("failed.json")
    while True:
        if complete.is_file():
            payload = json.loads(complete.read_text("utf-8"))
            committed = TrainerGeneration.model_validate(payload["generation"])
            if committed != generation:
                raise RuntimeError("durability marker identifies another generation")
            return int(payload["resume_step"]), int(payload["optimizer_step"])
        if error := _publication_marker_failure(failed):
            raise error
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for durable generation {generation.generation_id}"
            )
        time.sleep(0.05)


def _publication_marker_failure(path: Path) -> RuntimeError | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text("utf-8"))
        error_type = payload["error_type"]
        message = payload["message"]
        if not isinstance(error_type, str) or not isinstance(message, str):
            raise TypeError
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid trainer publication failure marker: {path}"
        ) from error
    return RuntimeError(f"trainer publisher failed ({error_type}): {message}")


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
