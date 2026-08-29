from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Literal, TypedDict, cast
import uuid

import httpx

from art import dev, types
from art.adapter_leases import in_flight_lora_name
from art.distributed.art_runtime import ArtRuntime, DistributedPackedBatch
from art.distributed.specs import ModelServiceSpec
from art.distributed.vllm_replica import (
    ReplicaFailure,
    ReplicaLaunchTemplate,
    ReplicaUpdateReport,
)
from art.serving_capabilities import (
    ServingCapabilities,
    ServingProfileIdentity,
    discover_serving_capabilities,
)
from art.utils.lifecycle import (
    complete_task,
    complete_to_thread,
    consume_future_exception,
)
from art.utils.output_dirs import get_step_checkpoint_dir
from art.vllm_runtime import (
    get_external_vllm_runtime_config,
    map_checkpoint_path_for_vllm,
    normalize_vllm_server_url,
    wait_for_vllm_http_runtime,
)

from .identity_lora import create_identity_lora
from .migrations import optimizer_state_path
from .model_support import get_model_support_handler
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
)
from .runtime.build import build_trainer_runtime_spec
from .runtime.data_plane import SFTBatchData
from .runtime.publication import (
    DurableTrainerPublication,
    TrainerRankPublication,
    commit_trainer_publication,
)
from .runtime.specs import (
    AdapterReady,
    CurrentSFTConfig,
    CurrentTrainConfig,
    DurableTrainOutput,
    ExperimentalTrainConfig,
    ResidentLoraInspectionResult,
    ResidentLoraInspectionSpec,
    ResidentScoreJobSpec,
    ResidentScoreResult,
    SFTJobSpec,
    TrainAccepted,
    TrainCancelled,
    TrainCompleted,
    TrainerGeneration,
    TrainerJobSpec,
    TrainerRuntimeSpec,
    TrainFailed,
    TrainingRunSpec,
    TrainJobSpec,
    TrainProgress,
)

logger = logging.getLogger(__name__)
_POLICY_TIMING_HISTORY = 64


class _TrainerJobFields(TypedDict):
    job_id: str
    run_id: str
    training_session_id: str
    expected_learner_version: int
    learner_version: int
    source: TrainerGeneration
    output: DurableTrainOutput
    publication_targets: tuple[Any, ...]


async def _post_vllm(
    url: str,
    *,
    api_key: str | None,
    timeout_s: float = 30.0,
    **kwargs: Any,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        return await client.post(url, headers=_headers(api_key), **kwargs)


def _retire_completed(
    records: dict[int, Any], step: int, completed: asyncio.Future[Any]
) -> None:
    if records.get(step) is completed:
        records.pop(step)


class DistributedMegatronService:
    """One model's durable checkpoints and run-scoped distributed runtimes."""

    propagate_close_errors = True
    close_timeout_s = 300.0

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
        self._trainer_preparation_task: asyncio.Task[None] | None = None
        self._trainer_preparation_step: asyncio.Future[int] | None = None
        self._trainer_preparation_s = 0.0
        # Nested acquisitions must follow train -> serving -> mutation.
        self._train_lock = asyncio.Lock()
        self._mutation_lock = asyncio.Lock()
        self._serving_lock = asyncio.Lock()
        self._durability_lock = asyncio.Lock()
        self._pipeline_train_dispatch: asyncio.Event | None = None
        self._managed_service_name: str | None = None
        self._base_url: str | None = None
        self._serving_capabilities: ServingCapabilities | None = None
        self._api_key_value: str | None = None
        self._current_lora_name: str | None = None
        self._vllm_sleeping = False
        self._published_adapters: dict[int, OptimizerAdapter] = {}
        self._loaded_adapter_steps: set[int] = set()
        self._loaded_exact_adapter_steps: set[int] = set()
        self._exact_adapter_refcounts: dict[int, int] = {}
        self._recovery_tasks: set[asyncio.Task[None]] = set()
        self._publication_tasks: dict[int, asyncio.Task[None]] = {}
        self._durability_tasks: set[asyncio.Task[Any]] = set()
        self._prepared_adapter_transfers: dict[str, Any] = {}
        self._loaded_adapter_transfers: dict[int, tuple[Any, str]] = {}
        self._next_publication_preparation: (
            tuple[
                Any,
                TrainerGeneration,
                asyncio.Task[tuple[Any, ...]],
            ]
            | None
        ) = None
        self._serving_futures: dict[int, asyncio.Future[None]] = {}
        self._publication_failure: BaseException | None = None
        self._publication_metrics: dict[int, dict[str, float]] = {}
        self._emitted_publication_metrics: dict[int, set[str]] = {}
        self._trainer_completion_times: dict[int, float] = {}
        self._serving_activation_times: dict[int, float] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def openai_server_port(self) -> int:
        return self._model_service_spec().leader_endpoint.port

    def arm_pipeline_train_dispatch(self, event: asyncio.Event) -> None:
        if self._pipeline_train_dispatch is not None:
            raise RuntimeError("pipeline trainer dispatch fence is already armed")
        self._pipeline_train_dispatch = event

    def cancel_pipeline_train_dispatch(self, event: asyncio.Event) -> None:
        if self._pipeline_train_dispatch is event:
            self._pipeline_train_dispatch = None

    def _take_pipeline_train_dispatch(self) -> asyncio.Event | None:
        event = self._pipeline_train_dispatch
        self._pipeline_train_dispatch = None
        return event

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

    def policy_activation_timing(self, step: int) -> tuple[float, float]:
        try:
            return (
                self._trainer_completion_times[step],
                self._serving_activation_times[step],
            )
        except KeyError as error:
            raise RuntimeError(
                f"policy {step} lacks an authoritative trainer/serving timestamp"
            ) from error

    @staticmethod
    def _record_policy_timestamp(history: dict[int, float], step: int) -> None:
        history[step] = time.monotonic()
        while len(history) > _POLICY_TIMING_HISTORY:
            history.pop(next(iter(history)))

    def _record_serving_activation(self, step: int) -> None:
        if (
            step in self._trainer_completion_times
            and step not in self._serving_activation_times
        ):
            self._record_policy_timestamp(self._serving_activation_times, step)

    def checkpoint_materialization(self, step: int) -> asyncio.Task[None]:
        self._require_open()
        self._raise_publication_failure()
        generation = self._learner_generation
        if generation is None or generation.policy_step != step:
            raise RuntimeError(
                f"learner generation {step} is unavailable for materialization"
            )

        async def wait() -> None:
            publication = self._publication_tasks.get(step)
            if publication is not None:
                await asyncio.shield(publication)
            if step not in self._published_adapters:
                raise RuntimeError(f"learner generation {step} is not materialized")

        task = asyncio.create_task(wait())
        task.add_done_callback(consume_future_exception)
        return task

    @property
    def rollout_weight_update_mode(self) -> str:
        return self.config.get("rollout_weight_update_mode", "step_lora")

    @property
    def _temporal_gpu_sharing(self) -> bool:
        return (
            get_external_vllm_runtime_config(self.config) is None
            and self._model_service_spec().temporal_gpu_sharing
        )

    def _serving_lora_name(self, step: int) -> str:
        if self.rollout_weight_update_mode == "in_flight_lora":
            return in_flight_lora_name(self.model_name)
        return f"{self.model_name}@{step}"

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

    def _resident_trainer_for_generation(
        self, generation: TrainerGeneration
    ) -> Any | None:
        if (
            self._trainer_is_current()
            and self._trainer_resident_generation == generation
        ):
            return self._trainer
        return None

    @property
    def _optimizer_state_path(self) -> str:
        path = optimizer_state_path(self.output_dir)
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
        return build_trainer_runtime_spec(
            self.runtime,
            base_model=self.base_model,
            config=self.config,
            enable_expert_replay=self.enable_expert_replay,
            offload_between_jobs=self._temporal_gpu_sharing,
        )

    async def _ensure_trainer_locked(self) -> tuple[Any, tuple[int, str] | None]:
        if self._trainer_is_current():
            return self._trainer, None
        current, reconcile_step = await self._prepare_for_packing_locked()
        assert self._trainer is None
        self._trainer = await self._launch_trainer(current)
        self._trainer_resident_generation = None
        reconcile = None if reconcile_step is None else (reconcile_step, current)
        return self._trainer, reconcile

    async def _launch_trainer(self, current: str) -> Any:
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
        return await self.runtime.start_trainer(runtime_spec, run_spec)

    def prefetch_trainer(self) -> None:
        if (
            self._trainer_is_current()
            or self._trainer_preparation_task is not None
            or self._temporal_gpu_sharing
        ):
            return
        self._require_open()
        source_step = asyncio.get_running_loop().create_future()
        source_step.add_done_callback(consume_future_exception)
        task = asyncio.create_task(self._prepare_trainer(source_step))
        task.add_done_callback(consume_future_exception)
        self._trainer_preparation_step = source_step
        self._trainer_preparation_task = task

    async def _prepare_trainer(self, source_step: asyncio.Future[int]) -> None:
        started = time.perf_counter()
        trainer: Any = None
        assigned = False
        try:
            async with self._mutation_lock:
                self._raise_publication_failure()
                current, reconcile_step = await self._prepare_for_packing_locked()
                step = self._latest_step
                source_step.set_result(step)
            trainer = await self._launch_trainer(current)
            async with self._mutation_lock:
                if self._trainer is not None:
                    raise RuntimeError("trainer appeared during background preparation")
                self._trainer = trainer
                trainer = None
                assigned = True
                self._trainer_resident_generation = None
                if self._latest_step != step:
                    raise RuntimeError("learner changed during trainer preparation")
            if reconcile_step is not None and self._base_url is not None:
                async with self._serving_lock:
                    await self._reconcile_serving_locked(reconcile_step, current)
        except BaseException as error:
            if not source_step.done():
                source_step.set_exception(error)
            if trainer is not None:
                try:
                    _, interrupted = await complete_task(
                        asyncio.create_task(self.runtime.stop_trainer(trainer))
                    )
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "trainer preparation and cleanup failed",
                        [error, cleanup_error],
                    ) from None
                if interrupted is not None:
                    error.add_note("trainer preparation cleanup observed cancellation")
            elif assigned:
                await self._cleanup_failed_trainer_transaction(
                    self._trainer, None, error
                )
            raise
        finally:
            self._trainer_preparation_s = time.perf_counter() - started

    async def _await_trainer_preparation(self) -> None:
        task = self._trainer_preparation_task
        if task is None:
            return
        await asyncio.shield(task)
        if self._trainer_preparation_task is task:
            self._trainer_preparation_task = None
            self._trainer_preparation_step = None

    async def prepare_for_packing(self) -> int:
        self._require_open()
        self._raise_publication_failure()
        if (source_step := self._trainer_preparation_step) is not None:
            return await asyncio.shield(source_step)
        async with self._train_lock:
            async with self._mutation_lock:
                _current, reconcile_step = await self._prepare_for_packing_locked()
                step = self._latest_step
            if reconcile_step is not None:
                async with self._serving_lock:
                    await self._reconcile_serving_locked(step, _current)
            return step

    async def prepare_cp_lookahead(
        self,
        batch: DistributedPackedBatch,
        *,
        global_grad_accumulation_sequences: int | None,
    ) -> dict[str, float]:
        mesh = self.runtime.topology.trainer
        if mesh is None or mesh.topology.cp <= 1:
            return {}
        await self._await_trainer_preparation()
        async with self._mutation_lock:
            self._require_open()
            self._raise_publication_failure()
            trainer = self._trainer
            if trainer is None or not self._trainer_is_current():
                raise RuntimeError("CP lookahead requires the current resident trainer")
        return await trainer.prepare_cp_lookahead(
            batch.leases,
            global_grad_accumulation_sequences=(global_grad_accumulation_sequences),
        )

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
            await self._release_loaded_adapter_transfer(step)
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
        job: TrainerJobSpec,
        start: Callable[[], AsyncIterator[Any]],
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
            events: AsyncIterator[Any] | None = None
            try:
                events = start()
                yield events
                close = getattr(events, "aclose", None)
                if close is not None:
                    await close()
            except BaseException as error:
                await self._cleanup_failed_trainer_transaction(trainer, events, error)
                raise

    async def _cleanup_failed_trainer_transaction(
        self,
        trainer: Any,
        events: AsyncIterator[Any] | None,
        primary: BaseException,
    ) -> None:
        async def cleanup() -> None:
            failures: list[BaseException] = []
            try:
                await self._discard_next_publication_preparation()
            except BaseException as error:
                failures.append(error)
            try:
                await self._release_prepared_adapter_transfers()
            except BaseException as error:
                failures.append(error)
            close = None if events is None else getattr(events, "aclose", None)
            if close is not None:
                try:
                    await close()
                except BaseException as error:
                    failures.append(error)
            try:
                await self._invalidate_trainer_and_restore_serving(trainer)
            except BaseException as error:
                failures.append(error)
            if failures:
                raise BaseExceptionGroup("failed trainer job cleanup failed", failures)

        try:
            _, interrupted = await complete_task(asyncio.create_task(cleanup()))
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "trainer job and cancellation-safe cleanup failed",
                [primary, cleanup_error],
            ) from None
        if interrupted is not None:
            primary.add_note("trainer cleanup completed after another cancellation")

    async def _release_prepared_adapter_transfers(self) -> None:
        prepared = tuple(self._prepared_adapter_transfers.items())
        results = await asyncio.gather(
            *(
                manager.release_adapter_transfer(generation_id)
                for generation_id, manager in prepared
            ),
            return_exceptions=True,
        )
        failures = []
        for (generation_id, manager), result in zip(prepared, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(result)
            elif self._prepared_adapter_transfers.get(generation_id) is manager:
                self._prepared_adapter_transfers.pop(generation_id)
        if failures:
            raise BaseExceptionGroup(
                "prepared adapter transfer cleanup failed", failures
            )

    async def _discard_next_publication_preparation(self) -> None:
        prepared, self._next_publication_preparation = (
            self._next_publication_preparation,
            None,
        )
        if prepared is None:
            return
        _, generation, task = prepared
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        manager = self._prepared_adapter_transfers.pop(generation.generation_id, None)
        if manager is not None:
            await manager.release_adapter_transfer(generation.generation_id)

    async def _release_adapter_transfer(
        self,
        manager: Any,
        generation_id: str,
        primary: BaseException | None = None,
    ) -> asyncio.CancelledError | None:
        try:
            _, interrupted = await complete_task(
                asyncio.create_task(manager.release_adapter_transfer(generation_id))
            )
        except BaseException as cleanup_error:
            if primary is not None:
                raise BaseExceptionGroup(
                    "adapter transfer and cleanup failed", [primary, cleanup_error]
                ) from None
            raise
        return interrupted

    async def _release_loaded_adapter_transfer(self, step: int) -> None:
        transfer = self._loaded_adapter_transfers.get(step)
        if transfer is None:
            return
        manager, generation_id = transfer
        interrupted = await self._release_adapter_transfer(manager, generation_id)
        if self._loaded_adapter_transfers.get(step) == transfer:
            self._loaded_adapter_transfers.pop(step)
        if interrupted is not None:
            raise interrupted

    async def _release_loaded_adapter_transfers(self) -> None:
        results = await asyncio.gather(
            *(
                self._release_loaded_adapter_transfer(step)
                for step in tuple(self._loaded_adapter_transfers)
            ),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("loaded adapter transfer cleanup failed", failures)

    @asynccontextmanager
    async def _trainer_failure_boundary(self) -> AsyncIterator[None]:
        try:
            yield
        except BaseException as error:
            await self._cleanup_failed_trainer_transaction(self._trainer, None, error)
            raise

    async def _invalidate_trainer_and_restore_serving(self, trainer: Any) -> None:
        failures: list[BaseException] = []
        async with self._mutation_lock:
            owned = trainer is not None and self._trainer is trainer
            if owned:
                self._trainer = None
                self._trainer_resident_generation = None
        if owned:
            try:
                await self.runtime.stop_trainer(trainer)
            except BaseException as error:
                failures.append(error)
        if self._temporal_gpu_sharing and self._vllm_sleeping:
            async with self._serving_lock:
                if self._vllm_sleeping:
                    try:
                        await self._wake_for_serving_locked()
                    except BaseException as error:
                        failures.append(error)
                        service_name = self._managed_service_name
                        if service_name is not None:
                            failures.extend(
                                await self._rollback_server_start_safely(service_name)
                            )
                        self._clear_serving_state()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("failed trainer invalidation failed", failures)

    async def _trainer_metrics(
        self,
        job: TrainerJobSpec,
        events: AsyncIterator[Any],
    ) -> AsyncIterator[tuple[bool, dict[str, float]]]:
        snapshot_prepared = False
        completed = False
        final_metrics: dict[str, float] | None = None
        async for event in events:
            if event.job_id != job.job_id or event.run_id != job.run_id:
                raise RuntimeError("trainer returned an event for a different job")
            if isinstance(event, TrainAccepted):
                continue
            if isinstance(event, TrainProgress):
                if event.step_index + 1 == event.num_steps:
                    final_metrics = dict(event.metrics)
                    self._record_policy_timestamp(
                        self._trainer_completion_times, job.learner_version
                    )
                else:
                    yield False, dict(event.metrics)
                continue
            if isinstance(event, AdapterReady):
                if snapshot_prepared:
                    raise RuntimeError("trainer returned duplicate snapshot events")
                if (
                    event.learner_version != job.learner_version
                    or event.adapter_path != job.output_adapter_path
                ):
                    raise RuntimeError("trainer prepared the wrong generation")
                snapshot_prepared = True
                continue
            if isinstance(event, TrainCompleted):
                if completed or not snapshot_prepared:
                    raise RuntimeError("trainer completed without one snapshot")
                if event.learner_version != job.learner_version:
                    raise RuntimeError("trainer completed the wrong learner")
                snapshot_metrics = {
                    name: value
                    for name, value in event.metrics.items()
                    if name.startswith("snapshot_")
                }
                self._publication_metrics[job.learner_version] = snapshot_metrics
                self._emitted_publication_metrics[job.learner_version] = set(
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
        yield True, final_metrics

    async def _prepare_serving_publication(
        self,
        trainer: Any,
        generation_id: str,
    ) -> tuple[Any, ...]:
        if self._managed_service_name is None:
            return ()
        manager = self.runtime.model_service(self._managed_service_name)
        trainer_host = trainer.runtime_spec.trainer_mesh.ranks[0].host_id
        inference_hosts = {member.host_id for member in manager.spec.members}
        try:
            targets = await manager.prepare_adapter_transfer(
                generation_id,
                get_step_checkpoint_dir(self.output_dir, 0),
                transport="local" if inference_hosts == {trainer_host} else "nixl",
            )
            if not targets:
                raise RuntimeError("model service returned no adapter transfer targets")
        except BaseException as error:
            interrupted = await self._release_adapter_transfer(
                manager, generation_id, error
            )
            if interrupted is not None:
                error.add_note("adapter transfer cleanup observed cancellation")
            raise
        self._prepared_adapter_transfers[generation_id] = manager
        return targets

    def _training_generation(self, step: int) -> TrainerGeneration:
        return TrainerGeneration(
            training_session_id=self._training_session_id,
            policy_step=step,
            generation_id=new_optimizer_generation(step),
            adapter_path=get_step_checkpoint_dir(self.output_dir, step),
        )

    async def _take_publication_preparation(
        self, trainer: Any, step: int
    ) -> tuple[TrainerGeneration, tuple[Any, ...]]:
        prepared, self._next_publication_preparation = (
            self._next_publication_preparation,
            None,
        )
        if prepared is not None:
            prepared_trainer, generation, task = prepared
            if prepared_trainer is trainer and generation.policy_step == step:
                return generation, await task
            self._next_publication_preparation = prepared
            await self._discard_next_publication_preparation()
        generation = self._training_generation(step)
        targets = await self._prepare_serving_publication(
            trainer, generation.generation_id
        )
        return generation, targets

    def _prefetch_publication_preparation(self, trainer: Any, step: int) -> None:
        if self._managed_service_name is None:
            return
        if self._next_publication_preparation is not None:
            raise RuntimeError("next publication preparation already exists")
        generation = self._training_generation(step)
        previous_serving = self._serving_futures.get(step - 2)

        async def prepare() -> tuple[Any, ...]:
            if previous_serving is not None:
                await asyncio.shield(previous_serving)
            return await self._prepare_serving_publication(
                trainer, generation.generation_id
            )

        task = asyncio.create_task(prepare())
        task.add_done_callback(consume_future_exception)
        self._next_publication_preparation = trainer, generation, task

    async def _run_train_job(
        self,
        build_job: Callable[[_TrainerJobFields], TrainerJobSpec],
        start_job: Callable[[Any, TrainerJobSpec], AsyncIterator[Any]],
        *,
        lineage_error: str,
        wait_for_serving: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        trainer_prepare_started = time.perf_counter()
        await self._await_trainer_preparation()
        trainer_prepare_wait_s = time.perf_counter() - trainer_prepare_started
        lock_started = time.perf_counter()
        async with self._train_lock:
            lock_wait_s = time.perf_counter() - lock_started
            setup_started = time.perf_counter()
            async with self._trainer_failure_boundary():
                if self._temporal_gpu_sharing and self._base_url is not None:
                    previous = self._serving_futures.get(self._latest_step)
                    if previous is not None:
                        await asyncio.shield(previous)
                    async with self._serving_lock:
                        await self._sleep_for_training_locked()
                async with self._mutation_lock:
                    self._require_open()
                    self._raise_publication_failure()
                    trainer, reconcile = await self._ensure_trainer_locked()
                    source = self._learner_generation
                    if source is None:
                        raise RuntimeError("trainer has no source generation")
                    next_step = self._latest_step + 1

                preparation_started = time.perf_counter()
                (
                    output_generation,
                    publication_targets,
                ) = await self._take_publication_preparation(trainer, next_step)
                preparation_wait_s = time.perf_counter() - preparation_started

                async with self._mutation_lock:
                    self._require_open()
                    self._raise_publication_failure()
                    if self._trainer is not trainer or not self._trainer_is_current():
                        raise RuntimeError(
                            "trainer changed while preparing serving publication"
                        )
                    if (
                        self._learner_generation != source
                        or self._latest_step != next_step - 1
                    ):
                        raise RuntimeError(lineage_error)
                    output = DurableTrainOutput(
                        generation=output_generation,
                        staging_adapter_path=(
                            f"{self.output_dir}/megatron_runtime/staging/"
                            f"{output_generation.generation_id}"
                        ),
                        optimizer_state_path=self._optimizer_state_path,
                    )
                    job = build_job(
                        _TrainerJobFields(
                            job_id=uuid.uuid4().hex,
                            run_id=trainer.run_spec.run_id,
                            training_session_id=self._training_session_id,
                            expected_learner_version=self._latest_step,
                            learner_version=next_step,
                            source=source,
                            output=output,
                            publication_targets=publication_targets,
                        )
                    )

                if reconcile is not None:
                    reconcile_step, checkpoint = reconcile
                    async with self._serving_lock:
                        await self._reconcile_serving_locked(reconcile_step, checkpoint)
                self._prefetch_publication_preparation(trainer, next_step + 1)
            setup_s = time.perf_counter() - setup_started

            final_metrics: dict[str, float] | None = None
            async with self._trainer_transaction(
                trainer, job, lambda: start_job(trainer, job)
            ) as events:
                async for final, metrics in self._trainer_metrics(job, events):
                    if final:
                        final_metrics = metrics
                    else:
                        yield metrics
            assert final_metrics is not None

            commit_started = time.perf_counter()
            async with self._trainer_failure_boundary():
                async with self._mutation_lock:
                    if self._latest_step != job.expected_learner_version:
                        raise RuntimeError(lineage_error)
                    self._latest_step = next_step
                    self._learner_generation = output_generation
                    self._trainer_resident_generation = output_generation
                    self._schedule_publication(
                        output_generation,
                        trainer=trainer,
                        publication_targets=job.publication_targets,
                    )
            commit_s = time.perf_counter() - commit_started
            if wait_for_serving:
                await self.wait_for_serving(next_step)
            final_metrics.update(
                {
                    "time/step_service_lock_wait_s": lock_wait_s,
                    "time/step_service_trainer_prepare_s": (
                        self._trainer_preparation_s
                    ),
                    "time/step_service_trainer_prepare_wait_s": (
                        trainer_prepare_wait_s
                    ),
                    "time/step_service_job_setup_s": setup_s,
                    "time/step_service_publication_prepare_wait_s": (
                        preparation_wait_s
                    ),
                    "time/step_service_generation_commit_s": commit_s,
                }
            )
            yield final_metrics

    async def train_packed(
        self,
        batch: DistributedPackedBatch,
        config: types.TrainConfig,
        experimental_config: dev.TrainConfig,
    ) -> AsyncIterator[dict[str, float]]:
        def build_job(fields: _TrainerJobFields) -> TrainerJobSpec:
            values = {
                key: value
                for key, value in experimental_config.items()
                if key in ExperimentalTrainConfig.model_fields and value is not None
            }
            return TrainJobSpec(
                **fields,
                batch=batch.leases.ref,
                config=CurrentTrainConfig.model_validate(config.model_dump()),
                experimental_config=ExperimentalTrainConfig.model_validate(values),
            )

        dispatch_event = self._take_pipeline_train_dispatch()
        async for metrics in self._run_train_job(
            build_job,
            lambda trainer, job: trainer.train(
                job,
                batch.leases,
                on_dispatched=dispatch_event.set
                if dispatch_event is not None
                else None,
            ),
            lineage_error="learner lineage changed during training",
        ):
            yield metrics

    def _require_resident_score_locked(
        self,
        expected_learner_version: int,
    ) -> tuple[Any, TrainerGeneration]:
        self._require_open()
        self._raise_publication_failure()
        source = self._learner_generation
        trainer = self._resident_trainer_for_generation(source) if source else None
        if expected_learner_version != self._latest_step:
            raise ValueError(
                "resident diagnostic learner version mismatch: "
                f"request={expected_learner_version}, current={self._latest_step}"
            )
        if (
            source is None
            or source.policy_step != expected_learner_version
            or trainer is None
        ):
            raise RuntimeError(
                "resident scoring requires the exact hydrated warm trainer generation"
            )
        return trainer, source

    def _require_resident_inspection_locked(
        self,
        expected_learner_version: int,
    ) -> tuple[Any, TrainerGeneration]:
        self._require_open()
        self._raise_publication_failure()
        source = self._learner_generation
        trainer = self._trainer
        if expected_learner_version != self._latest_step:
            raise ValueError(
                "resident inspection learner version mismatch: "
                f"request={expected_learner_version}, current={self._latest_step}"
            )
        if (
            source is None
            or source.policy_step != expected_learner_version
            or trainer is None
            or not self._trainer_is_current()
            or trainer.run_spec.training_session_id != source.training_session_id
            or self._trainer_resident_generation not in (None, source)
        ):
            raise RuntimeError(
                "resident inspection requires the exact current warm trainer run"
            )
        return trainer, source

    async def score_resident_packed(
        self,
        batch: DistributedPackedBatch,
        *,
        expected_learner_version: int,
        global_grad_accumulation_sequences: int,
        top_k: int = 20,
    ) -> ResidentScoreResult:
        await self._await_trainer_preparation()
        async with self._train_lock:
            async with self._mutation_lock:
                trainer, source = self._require_resident_score_locked(
                    expected_learner_version
                )
                job = ResidentScoreJobSpec(
                    job_id=uuid.uuid4().hex,
                    run_id=trainer.run_spec.run_id,
                    learner=source,
                    batch=batch.leases.ref,
                    global_grad_accumulation_sequences=(
                        global_grad_accumulation_sequences
                    ),
                    top_k=top_k,
                )

            try:
                if self._temporal_gpu_sharing and self._base_url is not None:
                    serving = self._serving_futures.get(expected_learner_version)
                    if serving is not None:
                        await asyncio.shield(serving)
                    async with self._serving_lock:
                        await self._sleep_for_training_locked()
                async with self._trainer_failure_boundary():
                    result = await trainer.score(job, batch.leases)
                    async with self._mutation_lock:
                        current_trainer, current_source = (
                            self._require_resident_score_locked(
                                expected_learner_version
                            )
                        )
                        if current_trainer is not trainer or current_source != source:
                            raise RuntimeError(
                                "resident learner generation changed during scoring"
                            )
                    if result.learner != source:
                        raise RuntimeError(
                            "resident score returned a different learner generation"
                        )
                    if result.expected_score_count != batch.loss_bearing_tokens:
                        raise RuntimeError(
                            "resident score target coverage differs from packed data"
                        )
                    return result
            finally:
                if self._temporal_gpu_sharing and self._vllm_sleeping:
                    async with self._serving_lock:
                        await self._wake_for_serving_locked()

    async def inspect_resident_lora(
        self,
        *,
        expected_learner_version: int,
    ) -> ResidentLoraInspectionResult:
        await self._await_trainer_preparation()
        async with self._train_lock:
            trainer: Any = None
            try:
                if self._temporal_gpu_sharing and self._base_url is not None:
                    serving = self._serving_futures.get(expected_learner_version)
                    if serving is not None:
                        await asyncio.shield(serving)
                    async with self._serving_lock:
                        await self._sleep_for_training_locked()
                async with self._mutation_lock:
                    trainer, reconcile = await self._ensure_trainer_locked()
                    source_trainer, source = self._require_resident_inspection_locked(
                        expected_learner_version
                    )
                    if source_trainer is not trainer:
                        raise RuntimeError(
                            "resident inspection selected another trainer"
                        )
                    request = ResidentLoraInspectionSpec(
                        request_id=uuid.uuid4().hex,
                        run_id=trainer.run_spec.run_id,
                        learner=source,
                        target_modules=trainer.runtime_spec.lora_target_modules,
                    )
                if reconcile is not None:
                    reconcile_step, checkpoint = reconcile
                    async with self._serving_lock:
                        await self._reconcile_serving_locked(reconcile_step, checkpoint)
                result = await trainer.inspect_resident_lora(request)
                async with self._mutation_lock:
                    current_trainer, current_source = (
                        self._require_resident_inspection_locked(
                            expected_learner_version
                        )
                    )
                    if current_trainer is not trainer or current_source != source:
                        raise RuntimeError(
                            "resident learner generation changed during LoRA inspection"
                        )
                if result.learner != source:
                    raise RuntimeError(
                        "resident LoRA inspection returned another learner generation"
                    )
                return result
            except BaseException as error:
                if trainer is not None and not trainer.valid:
                    await self._cleanup_failed_trainer_transaction(trainer, None, error)
                raise
            finally:
                if self._temporal_gpu_sharing and self._vllm_sleeping:
                    async with self._serving_lock:
                        await self._wake_for_serving_locked()

    def _schedule_publication(
        self,
        generation: TrainerGeneration,
        *,
        trainer: Any = None,
        durable: DurableTrainerPublication | None = None,
        publication_targets: tuple[Any, ...] = (),
    ) -> None:
        if (trainer is None) == (durable is None):
            raise ValueError(
                "publication requires exactly one trainer stream or durable result"
            )
        step = generation.policy_step
        if step in self._publication_tasks:
            raise RuntimeError(f"generation publication already exists for step {step}")
        transfer_manager = self._prepared_adapter_transfers.get(
            generation.generation_id
        )
        if publication_targets and transfer_manager is None:
            raise RuntimeError("adapter transfer publication is not prepared")
        publication_waiter = (
            trainer.wait_for_publication(generation.generation_id)
            if trainer is not None
            else None
        )
        loop = asyncio.get_running_loop()
        previous = self._serving_futures.get(step - 1)
        if previous is None:
            previous = loop.create_future()
            previous.set_result(None)
        serving = loop.create_future()
        serving.add_done_callback(consume_future_exception)
        self._serving_futures[step] = serving
        serving.add_done_callback(
            lambda done: _retire_completed(self._serving_futures, step, done)
        )
        previous_publication = self._publication_tasks.get(step - 1)
        publication = (
            self._publish_generation(
                generation,
                durable=durable,
                publication_waiter=publication_waiter,
                previous_publication=previous_publication,
                publication_targets=publication_targets,
                transfer_manager=transfer_manager,
                previous_serving=previous,
                serving=serving,
            )
            if publication_targets
            else self._publish_generation(
                generation,
                durable=durable,
                publication_waiter=publication_waiter,
                previous_publication=previous_publication,
                previous_serving=previous,
                serving=serving,
            )
        )
        task = asyncio.create_task(publication)
        self._publication_tasks[step] = task
        self._prepared_adapter_transfers.pop(generation.generation_id, None)
        task.add_done_callback(consume_future_exception)
        task.add_done_callback(
            lambda done: _retire_completed(self._publication_tasks, step, done)
        )

    async def _resolve_durable_publication(
        self,
        generation: TrainerGeneration,
        *,
        durable: DurableTrainerPublication | None,
        publication_waiter: Awaitable[tuple[TrainerRankPublication, ...]] | None,
        previous_publication: asyncio.Task[None] | None,
    ) -> tuple[DurableTrainerPublication, float]:
        started = time.monotonic()
        records = (
            asyncio.ensure_future(publication_waiter)
            if publication_waiter is not None
            else None
        )
        if records is not None:
            records.add_done_callback(consume_future_exception)
        try:
            if previous_publication is not None:
                await asyncio.shield(previous_publication)
            if records is not None:
                rank_publications = await asyncio.shield(records)
                async with self._durability_lock:
                    durable = await asyncio.to_thread(
                        commit_trainer_publication,
                        self._optimizer_state_path,
                        generation,
                        rank_publications,
                    )
        finally:
            if records is not None and not records.done():
                records.cancel()
        return cast(DurableTrainerPublication, durable), time.monotonic() - started

    async def _publish_generation(
        self,
        generation: TrainerGeneration,
        *,
        durable: DurableTrainerPublication | None,
        publication_waiter: Awaitable[tuple[TrainerRankPublication, ...]] | None,
        previous_publication: asyncio.Task[None] | None,
        publication_targets: tuple[Any, ...] = (),
        transfer_manager: Any = None,
        previous_serving: asyncio.Future[None],
        serving: asyncio.Future[None],
    ) -> None:
        metrics = self._publication_metrics.setdefault(generation.policy_step, {})
        manager = transfer_manager
        transfer_owned = manager is not None
        durable_task = asyncio.create_task(
            self._resolve_durable_publication(
                generation,
                durable=durable,
                publication_waiter=publication_waiter,
                previous_publication=previous_publication,
            )
        )
        self._durability_tasks.add(durable_task)
        durable_task.add_done_callback(consume_future_exception)
        durable_task.add_done_callback(self._durability_tasks.discard)
        try:
            materialization_started = time.monotonic()
            if manager is None:
                durable_result, _ = await asyncio.shield(durable_task)
                adapter = durable_result.adapter
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
                metrics["adapter_transport_pool_wait_s"] = max(
                    result.pool_wait_s for result in received
                )
                metrics["adapter_transport_prepare_s"] = max(
                    result.prepare_s for result in received
                )
                metrics["adapter_transport_registration_s"] = max(
                    result.registration_s for result in received
                )
                metrics["adapter_transport_sender_staging_s"] = max(
                    result.sender_staging_s for result in received
                )
                metrics["adapter_transport_sender_registration_s"] = max(
                    result.sender_registration_s for result in received
                )
                metrics["adapter_transport_capacity_bytes"] = float(
                    sum(result.capacity_bytes for result in received)
                )
                metrics["adapter_transport_capacity_utilization"] = sum(
                    result.used_bytes for result in received
                ) / sum(result.capacity_bytes for result in received)
            await previous_serving
            self._raise_publication_failure()
            activation_started = time.monotonic()
            async with self._mutation_lock:
                self._published_adapters[generation.policy_step] = adapter
            async with self._serving_lock:
                await self._register_lora_for_step_locked(
                    generation.policy_step,
                    checkpoint,
                )
                if (
                    manager is not None
                    and self.rollout_weight_update_mode != "in_flight_lora"
                ):
                    self._loaded_adapter_transfers[generation.policy_step] = (
                        manager,
                        generation.generation_id,
                    )
                    transfer_owned = False
            metrics["serving_activation_s"] = time.monotonic() - activation_started
            if manager is None and not serving.done():
                serving.set_result(None)
            if manager is not None and transfer_owned:
                interrupted = await self._release_adapter_transfer(
                    manager, generation.generation_id
                )
                transfer_owned = False
                if interrupted is not None:
                    raise interrupted
            if manager is not None:
                if not serving.done():
                    serving.set_result(None)

            durable_result, durable_s = await asyncio.shield(durable_task)
            if durable_result.adapter != adapter:
                raise RuntimeError("Durable and serving adapter manifests differ")
            async with self._mutation_lock:
                self._durable_step = max(self._durable_step, durable_result.resume_step)
                self._durable_optimizer_step = max(
                    self._durable_optimizer_step, durable_result.optimizer_step
                )
                metrics["durable_checkpoint_s"] = durable_s
                metrics["durable_checkpoint_lag_steps"] = float(
                    self._latest_step - self._durable_optimizer_step
                )
            logger.info(
                "Published trainer generation session=%s step=%d generation=%s "
                "launch=%.3fs activate=%.3fs durable=%.3fs durable_lag=%d",
                generation.training_session_id,
                generation.policy_step,
                generation.generation_id,
                metrics["snapshot_launch_s"],
                metrics["serving_activation_s"],
                metrics["durable_checkpoint_s"],
                self._latest_step - self._durable_optimizer_step,
            )
        except BaseException as error:
            if manager is not None and transfer_owned:
                try:
                    interrupted = await self._release_adapter_transfer(
                        manager, generation.generation_id, error
                    )
                except BaseException as cleanup_error:
                    error = cleanup_error
                else:
                    transfer_owned = False
                    if interrupted is not None:
                        error.add_note("adapter transfer cleanup observed cancellation")
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
            raise error

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

        service = self._model_service_spec()
        server_args = dict((config or {}).get("server_args", {}))
        if "port" in server_args:
            from .runtime.local import with_local_serving_port

            self.runtime.topology = with_local_serving_port(
                self.runtime.topology,
                model_name=self.model_name,
                port=cast(int, server_args["port"]),
            )
            service = self._model_service_spec()
        template = ReplicaLaunchTemplate(
            served_model_name=self._serving_lora_name(step),
            lora_path=lora_path,
            initial_generation_id=self._generation_id_for_step(step),
            initial_policy_version=step,
            engine_args=self._engine_args(config),
            server_args=self._server_args(config),
            serving_profile_identity=self._serving_profile_identity(),
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
            self._validate_serving_profile(
                capabilities,
                template.serving_profile_identity,
                service,
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
        self._vllm_sleeping = False

    async def _replica_failed(self, failure: ReplicaFailure) -> None:
        if self._closed or failure.replica_id != self._managed_service_name:
            return
        task = asyncio.create_task(self._recover_failed_replica(failure))
        self._recovery_tasks.add(task)
        task.add_done_callback(self._recovery_tasks.discard)
        task.add_done_callback(consume_future_exception)

    async def _recover_failed_replica(self, failure: ReplicaFailure) -> None:
        async with self._train_lock:
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
        current_lora_name = self._current_lora_name or self._serving_lora_name(
            serving_step
        )
        bootstrap_name = self._serving_lora_name(serving_step)
        base_url = service.leader_endpoint.url
        exact_steps = tuple(sorted(self._loaded_exact_adapter_steps))
        try:
            state = await manager.restart(
                served_model_name=bootstrap_name,
                lora_path=checkpoint,
                initial_generation_id=generation_id,
                initial_policy_version=serving_step,
            )
            self._vllm_sleeping = False
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
                await self._load_adapter_at(
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
            await self._release_loaded_adapter_transfers()
        except BaseException as error:
            manager.quarantine(f"replica recovery failed: {error}")
            try:
                await manager.stop()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "replica recovery and teardown failed", [error, cleanup_error]
                ) from None
            try:
                await self._release_loaded_adapter_transfers()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "replica recovery and transfer cleanup failed",
                    [error, cleanup_error],
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

    def _serving_profile_identity(self) -> ServingProfileIdentity:
        spec = self._runtime_spec()
        retained_routes = self.runtime.retained_route_prefetch_enabled
        return ServingProfileIdentity(
            base_model=self.base_model,
            model_identifier=spec.model_identifier,
            model_revision=spec.model_revision,
            model_support_key=spec.model_support_key,
            handler_name=spec.handler_name,
            lora_rank=spec.lora_rank,
            lora_alpha=spec.lora_alpha,
            lora_target_modules=spec.lora_target_modules,
            trainer_dtype=spec.dtype,
            route_replay=spec.enable_moe_routing_replay,
            lora_transport="nixl" if self.runtime.nixl_transport else "local",
            retained_route_transport=("caios_lota" if retained_routes else "none"),
            retained_route_max_bytes=(
                self.runtime.config.route_bundle_prefetch_capacity_bytes
                if retained_routes
                else 0
            ),
            retained_route_max_bundles=(
                self.runtime.config.route_bundle_prefetch_max_bundles
                if retained_routes
                else 0
            ),
        )

    @staticmethod
    def _validate_serving_profile(
        capabilities: ServingCapabilities,
        identity: ServingProfileIdentity | None,
        service: ModelServiceSpec,
    ) -> None:
        profile = capabilities.profile
        if identity is None or profile is None or profile.identity != identity:
            raise RuntimeError("vLLM returned the wrong serving profile identity")
        parallel = service.parallel
        actual_parallel = (
            profile.tensor_parallel_size,
            profile.pipeline_parallel_size,
            profile.data_parallel_size,
            profile.enable_expert_parallel,
        )
        expected_parallel = (
            parallel.tp,
            parallel.pp,
            parallel.dp,
            parallel.enable_expert_parallel,
        )
        if actual_parallel != expected_parallel:
            raise RuntimeError("vLLM returned the wrong serving parallel geometry")
        if (
            profile.runtime_model != service.base_model
            or profile.runtime_revision != service.model_revision
        ):
            raise RuntimeError("vLLM returned the wrong serving model identity")
        exact_features = (
            capabilities.inplace_lora_load,
            capabilities.in_flight_lora_updates,
            capabilities.policy_token_spans,
            capabilities.fast_metrics is not None,
            capabilities.binary_routed_experts == identity.route_replay,
        )
        if not all(exact_features):
            raise RuntimeError("vLLM returned incomplete exact serving capabilities")

    async def _rollback_server_start(
        self, service_name: str | None
    ) -> list[BaseException]:
        if service_name is None:
            return []
        try:
            await self.runtime.stop_model_service(service_name)
        except BaseException as error:
            return [error]
        try:
            await self._release_loaded_adapter_transfers()
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
        for key, value in handler.vllm_engine_args().items():
            values.setdefault(key, value)
        values["enable_sleep_mode"] = self._temporal_gpu_sharing
        values["enable_lora"] = True
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
        external = get_external_vllm_runtime_config(self.config)
        if external is not None:
            if value is not None and value != external.api_key:
                raise ValueError(
                    "OpenAI server api_key conflicts with external vLLM credentials"
                )
            return external.api_key
        if value is not None:
            return cast(str, value)
        return self._api_key_value

    async def _sleep_for_training_locked(self) -> None:
        if not self._temporal_gpu_sharing or self._base_url is None:
            return
        if self._vllm_sleeping:
            return
        self._vllm_sleeping = True
        await self._sleep_vllm_at(self._base_url, self._api_key())

    async def _wake_for_serving_locked(self) -> None:
        if not self._vllm_sleeping or self._base_url is None:
            return
        await self._wake_vllm_at(self._base_url, self._api_key())
        self._vllm_sleeping = False

    @staticmethod
    async def _sleep_vllm_at(base_url: str, api_key: str | None) -> None:
        response = await _post_vllm(
            f"{base_url}/sleep",
            api_key=api_key,
            params={"level": 1, "mode": "wait"},
            timeout_s=300.0,
        )
        response.raise_for_status()

    @staticmethod
    async def _wake_vllm_at(base_url: str, api_key: str | None) -> None:
        response = await _post_vllm(
            f"{base_url}/wake_up", api_key=api_key, timeout_s=300.0
        )
        response.raise_for_status()

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
            else self._serving_lora_name(step)
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
                "model_name": name,
                "lora_slot": name,
                "lora_path": path,
                "generation_id": self._generation_id_for_step(step),
                "expected_generation_id": self._generation_id_for_step(active_step),
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
        await self._await_trainer_preparation()
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
        await self._await_trainer_preparation()
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
                previous = self._publication_tasks.get(expected_step)
                source = self._learner_generation
                if source is None or source.policy_step != expected_step:
                    raise RuntimeError("no-op transition has no immutable source")
                trainer = self._resident_trainer_for_generation(source)
            if previous is not None:
                await asyncio.shield(previous)
            source_adapter = self._published_adapters.get(expected_step)
            if source_adapter is None:
                raise RuntimeError("no-op source generation is not durably published")

            async with self._mutation_lock:
                self._published_adapters[expected_step] = source_adapter
                generation = TrainerGeneration(
                    training_session_id=self._training_session_id,
                    policy_step=learner_version,
                    generation_id=new_optimizer_generation(learner_version),
                    adapter_path=get_step_checkpoint_dir(
                        self.output_dir, learner_version
                    ),
                )
            snapshot_metrics: dict[str, float] = {}
            if trainer is not None:
                try:
                    snapshot_metrics.update(
                        await trainer.advance_without_training(
                            source=source,
                            output=generation,
                            optimizer_state_path=self._optimizer_state_path,
                            adapter=None,
                        )
                    )
                except BaseException as error:
                    await self._cleanup_failed_trainer_transaction(trainer, None, error)
                    raise

            async def commit() -> None:
                prepare_started = time.monotonic()
                published = await asyncio.to_thread(
                    _commit_adapter_alias,
                    self._optimizer_state_path,
                    self.output_dir,
                    expected_step,
                    source_adapter,
                    generation,
                    f"{self.output_dir}/megatron_runtime/staging/"
                    f"{generation.generation_id}",
                )
                snapshot_metrics["snapshot_launch_s"] = (
                    time.monotonic() - prepare_started
                )
                async with self._mutation_lock:
                    if self._latest_step != expected_step:
                        raise RuntimeError(
                            "learner lineage changed during no-op commit"
                        )
                    self._published_adapters[learner_version] = published
                    self._latest_step = learner_version
                    self._learner_generation = generation
                    self._trainer_resident_generation = (
                        generation if trainer is not None else None
                    )
                    self._publication_metrics[learner_version] = snapshot_metrics
                    pointer = read_committed_optimizer_pointer(
                        self._optimizer_state_path
                    )
                    self._schedule_publication(
                        generation,
                        durable=DurableTrainerPublication(
                            adapter=published,
                            resume_step=learner_version,
                            optimizer_step=0 if pointer is None else pointer.step,
                        ),
                    )

            try:
                _, cancelled = await complete_task(asyncio.create_task(commit()))
            except BaseException as error:
                if trainer is not None:
                    await self._cleanup_failed_trainer_transaction(trainer, None, error)
                raise
            metrics.update(self.drain_publication_metrics())
            if cancelled is not None:
                raise cancelled
            return metrics

    async def _register_lora_for_step_locked(
        self,
        step: int,
        checkpoint: str,
    ) -> None:
        if self._base_url is None:
            self._serving_step = step
            self._record_serving_activation(step)
            return
        await self._wake_for_serving_locked()
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
            lora_name = self._serving_lora_name(step)
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
                    raise RuntimeError("model service rejected its policy update")
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
                    "policy publication and serving rollback failed", [error, *cleanup]
                ) from None
            raise
        if self.rollout_weight_update_mode != "in_flight_lora":
            self._loaded_adapter_steps.add(step)
        self._current_lora_name = lora_name
        self._serving_step = step
        self._record_serving_activation(step)

    def _generation_id_for_step(self, step: int) -> str:
        published = self._published_adapters.get(step)
        if published is None:
            raise RuntimeError(f"No immutable generation is registered for step {step}")
        return published.generation_id

    async def acquire_exact_adapter(self, step: int, checkpoint: str) -> str:
        self._require_open()
        async with self._mutation_lock:
            published = step in self._published_adapters
            generation = self._learner_generation
            materialization = (
                self.checkpoint_materialization(step)
                if self.rollout_weight_update_mode == "in_flight_lora"
                and generation is not None
                and generation.policy_step == step
                else None
            )
        if materialization is not None:
            await asyncio.shield(materialization)
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
                if self.rollout_weight_update_mode == "in_flight_lora":
                    await self._unload_adapter(f"{self.model_name}:eval@{step}")
                self._exact_adapter_refcounts.pop(step, None)
                self._loaded_exact_adapter_steps.discard(step)
            else:
                self._exact_adapter_refcounts[step] = count - 1

    async def prune_loaded_adapters(self, *, retain_steps: set[int]) -> None:
        async with self._serving_lock:
            self._require_open()
            for step in sorted(self._loaded_exact_adapter_steps - retain_steps):
                if self._exact_adapter_refcounts.get(step, 0) == 0:
                    name = (
                        f"{self.model_name}:eval@{step}"
                        if self.rollout_weight_update_mode == "in_flight_lora"
                        else f"{self.model_name}@{step}"
                    )
                    await self._unload_adapter(name)
                    self._loaded_exact_adapter_steps.discard(step)
            if self.rollout_weight_update_mode == "in_flight_lora":
                return
            for step in sorted(
                self._loaded_adapter_steps - retain_steps - {self._serving_step}
            ):
                await self._unload_adapter(f"{self.model_name}@{step}")
                await self._release_loaded_adapter_transfer(step)
                self._loaded_adapter_steps.discard(step)

    @asynccontextmanager
    async def checkpoint_retention_lease(self) -> AsyncIterator[frozenset[int]]:
        # Disk pruning keeps mutation, not serving, held after ordered acquisition.
        async with self._serving_lock:
            await self._mutation_lock.acquire()
        try:
            self._require_open()
            protected = frozenset((self._latest_step, self._serving_step))
            yield protected
        finally:
            self._mutation_lock.release()

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
        return self._vllm_sleeping

    async def train_sft(
        self, batches: list[Any], config: Any, verbose: bool = False
    ) -> AsyncIterator[dict[str, float]]:
        del verbose
        payload = tuple(
            SFTBatchData(
                trajectory_tensors=tuple(batch.trajectory_tensors),
                learning_rate=float(batch.learning_rate),
                num_trajectories=int(batch.num_trajectories),
                num_tokens=int(batch.num_tokens),
                num_trainable_tokens=int(batch.num_trainable_tokens),
            )
            for batch in batches
        )
        if not payload:
            return

        def build_job(fields: _TrainerJobFields) -> TrainerJobSpec:
            return SFTJobSpec(
                **fields,
                batch_id=uuid.uuid4().hex,
                num_batches=len(payload),
                config=CurrentSFTConfig.model_validate(config.model_dump()),
            )

        async for metrics in self._run_train_job(
            build_job,
            lambda trainer, job: trainer.train_sft(job, payload),
            lineage_error="learner lineage changed during SFT",
            wait_for_serving=True,
        ):
            yield metrics

    async def aclose(self) -> None:
        if self._close_task is not None and self._close_task.done():
            try:
                self._close_task.result()
            except BaseException:
                self._close_task = None
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(consume_future_exception)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        failures: list[BaseException] = []
        preparation = self._trainer_preparation_task
        if preparation is not None:
            try:
                _, interrupted = await complete_task(preparation)
            except BaseException as error:
                failures.append(error)
            else:
                if interrupted is not None:
                    failures.append(interrupted)
            self._trainer_preparation_task = None
            self._trainer_preparation_step = None
        async with self._train_lock:
            publications = tuple(self._publication_tasks.values())
            durability_tasks = tuple(self._durability_tasks)
            recovery_tasks = tuple(self._recovery_tasks)
            for task in recovery_tasks:
                task.cancel()
            if recovery_tasks:
                await asyncio.gather(*recovery_tasks, return_exceptions=True)
            self._recovery_tasks.clear()
            async with self._mutation_lock:
                trainer = self._trainer
            shutdown = [*publications, *durability_tasks]
            trainer_task = None
            if trainer is not None:
                trainer_task = asyncio.create_task(self.runtime.stop_trainer(trainer))
                shutdown.append(trainer_task)
            results = await asyncio.gather(*shutdown, return_exceptions=True)
            if trainer_task is not None and not isinstance(results[-1], BaseException):
                async with self._mutation_lock:
                    if self._trainer is trainer:
                        self._trainer = None
            publication_failures = [
                result
                for result in results[: len(publications)]
                if isinstance(result, BaseException)
            ]
            failures.extend(
                result for result in results if isinstance(result, BaseException)
            )
            if self._publication_failure is not None and not publication_failures:
                failures.append(self._publication_failure)
            async with self._mutation_lock:
                self._publication_tasks.clear()
                self._durability_tasks.clear()
                self._serving_futures.clear()
                self._publication_metrics.clear()
                self._emitted_publication_metrics.clear()
                self._trainer_completion_times.clear()
                self._serving_activation_times.clear()
            try:
                await self._discard_next_publication_preparation()
            except BaseException as error:
                failures.append(error)
            try:
                await self._release_prepared_adapter_transfers()
            except BaseException as error:
                failures.append(error)
            async with self._serving_lock:
                serving_stopped = False
                if self._managed_service_name is not None:
                    result = await asyncio.gather(
                        self.runtime.stop_model_service(self._managed_service_name),
                        return_exceptions=True,
                    )
                    serving_failures = [
                        value for value in result if isinstance(value, BaseException)
                    ]
                    failures.extend(serving_failures)
                    if not serving_failures:
                        serving_stopped = True
                        self._clear_serving_state()
                elif (
                    get_external_vllm_runtime_config(self.config) is not None
                    and self._base_url is not None
                ):
                    names = {
                        *(
                            self._serving_lora_name(step)
                            for step in self._loaded_adapter_steps
                        ),
                        *(
                            f"{self.model_name}:eval@{step}"
                            if self.rollout_weight_update_mode == "in_flight_lora"
                            else f"{self.model_name}@{step}"
                            for step in self._loaded_exact_adapter_steps
                        ),
                    }
                    if self._current_lora_name is not None:
                        names.add(self._current_lora_name)
                    results = await asyncio.gather(
                        *(
                            self._unload_adapter_at(name, self._base_url)
                            for name in sorted(names)
                        ),
                        return_exceptions=True,
                    )
                    serving_failures = [
                        value for value in results if isinstance(value, BaseException)
                    ]
                    failures.extend(serving_failures)
                    if not serving_failures:
                        serving_stopped = True
                        self._clear_serving_state()
                else:
                    serving_stopped = True
                    self._clear_serving_state()
                if serving_stopped:
                    try:
                        await self._release_loaded_adapter_transfers()
                    except BaseException as error:
                        failures.append(error)
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


def _commit_adapter_alias(
    optimizer_state_path: str,
    output_dir: str,
    expected_step: int,
    source: OptimizerAdapter,
    generation: TrainerGeneration,
    staging_path: str,
) -> OptimizerAdapter:
    try:
        published = _publish_adapter_alias(source, generation, staging_path)
        commit_optimizer_policy_advance(
            optimizer_state_path,
            initial_adapter_path=get_step_checkpoint_dir(output_dir, 0),
            expected_step=expected_step,
            adapter=published,
        )
        return published
    except BaseException as error:
        try:
            policy = resolve_committed_optimizer_policy(
                optimizer_state_path,
                initial_adapter_path=get_step_checkpoint_dir(output_dir, 0),
            )
        except BaseException as state_error:
            raise BaseExceptionGroup(
                "no-op policy commit state is ambiguous", [error, state_error]
            ) from None
        if policy.policy_adapter.generation_id == generation.generation_id:
            return policy.policy_adapter
        failures: list[BaseException] = []
        latest = Path(output_dir) / "megatron_runtime/latest-adapter.json"
        try:
            if latest.is_file():
                adapter = OptimizerAdapter.model_validate_json(
                    latest.read_text("utf-8")
                )
                if adapter.generation_id == generation.generation_id:
                    latest.unlink()
        except BaseException as cleanup_error:
            failures.append(cleanup_error)
        for path in (
            staging_path,
            generation.adapter_path,
        ):
            try:
                _remove_staging_checkpoint(path)
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
        if failures:
            raise BaseExceptionGroup(
                "no-op policy commit and rollback failed", [error, *failures]
            ) from None
        raise


def _headers(api_key: str | None) -> dict[str, str] | None:
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def _host_port(base_url: str) -> tuple[str, int]:
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    assert parsed.hostname is not None
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
