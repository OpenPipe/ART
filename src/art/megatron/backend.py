import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import math
import os
import time
from typing import Any, Iterable, cast

from mp_actors import move_to_child_process

from ..backend import AnyTrainableModel
from ..distill import CurrentStep, PreparedTrainingBatch, TrainingObjectives
from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..metrics_taxonomy import average_metric_samples
from ..model import Model, TrainableModel
from ..trajectories import TrajectoryGroup
from ..types import LocalTrainResult, TrainConfig
from ..utils.lifecycle import process_shutdown_timeout
from ..utils.output_dirs import get_model_dir, get_step_checkpoint_dir
from .distillation import (
    CispoObjectiveConfig,
    DistillationObjectiveConfig,
    PolicyPackingConfig,
    pack_prepared_batch,
    validate_prepared_forward_kl,
    validate_standalone_forward_kl,
)
from .migrations import apply_megatron_migrations, optimizer_state_path
from .optimizer_state import (
    format_megatron_resume_message,
    prepare_megatron_resume_state,
    read_optimizer_commit,
)
from .runtime_config import get_megatron_runtime_config


@dataclass(slots=True)
class _ActiveCurrentStep:
    consistency: CurrentStep
    session: "_CurrentStepSession"
    capability: bytes
    heartbeat_task: asyncio.Task[None]
    heartbeat_error: BaseException | None = None
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class _CurrentStepSession:
    """Opaque backend-issued authority. Only revision and session ID are public."""

    revision: int
    session_id: str


def _summarize_distillation_metrics(
    metric_samples: list[dict[str, float]],
    *,
    target_count: int,
    policy_count: int | None,
) -> dict[str, float]:
    """Average diagnostics while preserving exact immutable job denominators."""

    metrics = average_metric_samples(metric_samples)
    metrics["data/step_distillation_target_tokens"] = float(target_count)
    if policy_count is not None:
        metrics["data/step_policy_tokens"] = float(policy_count)
    return metrics


class MegatronBackend(LocalBackend):
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        enable_expert_replay: bool = True,
    ) -> None:
        super().__init__(
            in_process=in_process,
            path=path,
            enable_expert_replay=enable_expert_replay,
        )
        self._requires_explicit_packed_sequence_length = True
        self._packed_sequence_length_requires_chunk_alignment = False
        self._supports_result_packing = True
        self._resume_prepared_models: set[tuple[str, str]] = set()
        self._active_current_steps: dict[tuple[str, str], _ActiveCurrentStep] = {}

    async def register(self, model: Model) -> None:
        await super().register(model)
        if model.trainable:
            # Keep durable Megatron state migrations centralized behind this call.
            apply_megatron_migrations(get_model_dir(model=model, art_path=self._path))

    async def train(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup] | PreparedTrainingBatch,
        **kwargs: Any,
    ) -> LocalTrainResult:
        for removed_kwarg in ("packed_sequence_length", "megatron_topology"):
            if removed_kwarg in kwargs:
                raise TypeError(
                    f"MegatronBackend.train gets {removed_kwarg} from "
                    "art.init_megatron_runtime_config(...)."
                )
        if isinstance(trajectory_groups, PreparedTrainingBatch):
            return await self._train_prepared_distillation(
                model, trajectory_groups, **kwargs
            )
        if any(name in kwargs for name in ("objectives", "idempotency_key", "session")):
            raise TypeError(
                "objectives, idempotency_key, and session are only valid with "
                "a PreparedTrainingBatch"
            )
        return await super().train(
            model,
            trajectory_groups,
            packed_sequence_length=get_megatron_runtime_config().packed_sequence_length,
            **kwargs,
        )

    async def _train_prepared_distillation(
        self,
        model: AnyTrainableModel,
        batch: PreparedTrainingBatch,
        **kwargs: Any,
    ) -> LocalTrainResult:
        allowed = {
            "objectives",
            "idempotency_key",
            "learning_rate",
            "grad_accumulation_sequences",
            "optimizer_save_interval",
            "save_checkpoint",
            "verbose",
            "epsilon",
            "epsilon_high",
            "importance_sampling_level",
            "scale_rewards",
            "advantage_balance",
            "session",
        }
        unexpected = sorted(set(kwargs) - allowed)
        if unexpected:
            raise TypeError(
                "unsupported prepared-batch training arguments: "
                + ", ".join(unexpected)
            )
        objectives = kwargs.get("objectives")
        if not isinstance(objectives, TrainingObjectives):
            raise TypeError(
                "PreparedTrainingBatch training requires objectives="
                "distill.TrainingObjectives(...)"
            )
        idempotency_key = kwargs.get("idempotency_key")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise TypeError(
                "PreparedTrainingBatch training requires a stable, non-empty "
                "idempotency_key"
            )
        learning_rate = float(kwargs.get("learning_rate", 5e-6))
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        grad_accumulation_sequences = kwargs.get("grad_accumulation_sequences")
        if grad_accumulation_sequences is not None and (
            not isinstance(grad_accumulation_sequences, int)
            or isinstance(grad_accumulation_sequences, bool)
            or grad_accumulation_sequences < 1
        ):
            raise ValueError("grad_accumulation_sequences must be a positive integer")
        optimizer_save_interval = kwargs.get("optimizer_save_interval", 1)
        if (
            not isinstance(optimizer_save_interval, int)
            or isinstance(optimizer_save_interval, bool)
            or optimizer_save_interval != 1
        ):
            raise ValueError(
                "prepared-batch training requires optimizer_save_interval=1 "
                "so every ART revision has durable optimizer state"
            )
        save_checkpoint = kwargs.get("save_checkpoint", True)
        if not isinstance(save_checkpoint, bool):
            raise TypeError("save_checkpoint must be a bool")
        verbose = kwargs.get("verbose", False)
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a bool")

        if objectives.policy not in (None, "cispo"):
            raise ValueError("prepared Megatron training supports CISPO only")
        if objectives.distillation is None:
            raise ValueError("prepared Megatron training requires distillation")
        policy_objective: CispoObjectiveConfig | None = None
        policy_packing: PolicyPackingConfig | None = None
        if objectives.policy is not None:
            importance_sampling_level = kwargs.get("importance_sampling_level", "token")
            if importance_sampling_level != "token":
                raise ValueError(
                    "prepared Megatron CISPO currently supports "
                    "importance_sampling_level='token' only"
                )
            epsilon = kwargs.get("epsilon")
            epsilon_high = kwargs.get("epsilon_high")
            policy_objective = CispoObjectiveConfig(
                epsilon=1.0 if epsilon is None else epsilon,
                epsilon_high=4.0 if epsilon_high is None else epsilon_high,
                importance_sampling_level=importance_sampling_level,
                scale_rewards=kwargs.get("scale_rewards", True),
                advantage_balance=kwargs.get("advantage_balance", 0.0),
            )
            policy_packing = PolicyPackingConfig(
                scale_rewards=policy_objective.scale_rewards,
                advantage_balance=policy_objective.advantage_balance,
            )
        objective = objectives.distillation
        objective_config = DistillationObjectiveConfig(
            coefficient=objective.coefficient,
            compensate_temperature_squared=(objective.compensate_temperature_squared),
            policy=policy_objective,
        )
        current_step = (
            batch.constraints.consistency
            if isinstance(batch.constraints.consistency, CurrentStep)
            else None
        )
        active_current: _ActiveCurrentStep | None = None
        if current_step is not None:
            active_current = self._require_active_current_step(
                model,
                current_step,
                kwargs.get("session"),
            )
            if active_current.consumed:
                raise RuntimeError(
                    "a current-step writer session permits exactly one backend.train call"
                )
            if active_current.heartbeat_error is not None:
                raise RuntimeError("the current-step writer heartbeat failed") from (
                    active_current.heartbeat_error
                )
            # The service owns heartbeat renewal while the bound optimizer job runs.
            # Stop the orchestration heartbeat before handing over the capability.
            active_current.heartbeat_task.cancel()
            try:
                await active_current.heartbeat_task
            except asyncio.CancelledError:
                pass
            active_current.consumed = True
        elif kwargs.get("session") is not None:
            raise ValueError("session= is valid only for CurrentStep prepared batches")
        config = TrainConfig(
            learning_rate=learning_rate,
            grad_accumulation_sequences=grad_accumulation_sequences,
            optimizer_save_interval=optimizer_save_interval,
        )
        artifact_source_revision = batch.constraints.learner_revision
        service = await self._get_service(cast(TrainableModel, model))
        if active_current is not None:
            await cast(Any, service).heartbeat_current_step(
                session_id=active_current.consistency.session_id,
                capability=active_current.capability,
                ttl_s=600.0,
            )
        committed_step = await cast(Any, service).committed_distillation_step(
            idempotency_key=idempotency_key,
            expected_source_revision=artifact_source_revision,
            preparation_id=batch.preparation_id,
            payload_sha256=batch.payload_sha256,
            objective=objective_config,
            config=config,
        )
        if committed_step is not None:
            checkpoint_path: str | None = None
            if save_checkpoint:
                candidate = get_step_checkpoint_dir(
                    get_model_dir(model=model, art_path=self._path), committed_step
                )
                if os.path.exists(candidate):
                    checkpoint_path = candidate
            return LocalTrainResult(
                step=committed_step,
                metrics={
                    "distill/idempotent_replay": 1.0,
                    "distill/committed_step": float(committed_step),
                    "data/step_num_gradient_steps": 0.0,
                },
                checkpoint_path=checkpoint_path,
            )

        runtime = get_megatron_runtime_config()
        topology = runtime.topology
        expected_source_revision = await self._get_step(model)
        if objectives.policy is None:
            payload = validate_standalone_forward_kl(
                batch=batch,
                objectives=objectives,
                expected_source_revision=expected_source_revision,
                packed_sequence_length=runtime.packed_sequence_length,
                tensor_parallel_size=topology.tp,
                context_parallel_size=topology.cp,
                pipeline_parallel_size=topology.pp,
                expert_parallel_size=topology.ep,
                expert_tensor_parallel_size=topology.etp,
            )
        else:
            topology_tuple = (
                topology.tp,
                topology.cp,
                topology.pp,
                topology.ep,
                topology.etp,
            )
            tp, cp, pp, ep, etp = topology_tuple
            if min(tp, cp) <= 0 or (pp, ep, etp) != (1, 1, 1):
                raise ValueError(
                    "additive prepared Megatron training supports positive "
                    "TP/CP with PP=EP=ETP=1; "
                    f"received {topology_tuple}"
                )
            payload = validate_prepared_forward_kl(
                batch=batch,
                objectives=objectives,
                expected_source_revision=expected_source_revision,
                packed_sequence_length=runtime.packed_sequence_length,
                policy_config=policy_packing,
            )
        capabilities = model._serving_capabilities
        if capabilities is None:
            raise RuntimeError(
                "prepared-batch training requires discovered student token-space "
                "capabilities"
            )
        if (
            capabilities.token_space_fingerprint
            != payload.constraints.token_space_fingerprint
            or capabilities.logical_vocab_size != payload.constraints.logical_vocab_size
        ):
            raise ValueError(
                "prepared token-space identity does not match the learner runtime"
            )

        tensor_dir = os.path.join(
            get_model_dir(model=model, art_path=self._path),
            "tensors",
            "distillation",
            batch.preparation_id,
        )
        disk_tensors = pack_prepared_batch(
            batch=batch,
            payload=payload,
            sequence_length=runtime.packed_sequence_length,
            output_dir=tensor_dir,
            objectives=objectives,
            policy_config=policy_packing,
        )
        metric_samples: list[dict[str, float]] = []
        started = time.monotonic()
        async for metrics in cast(Any, service).train_distillation(
            disk_tensors,
            config,
            objective=objective_config,
            expected_source_revision=expected_source_revision,
            idempotency_key=idempotency_key,
            preparation_id=batch.preparation_id,
            payload_sha256=batch.payload_sha256,
            current_step_session_id=(
                current_step.session_id if current_step is not None else None
            ),
            current_step_capability=(
                active_current.capability if active_current is not None else None
            ),
            verbose=verbose,
        ):
            metric_samples.append(metrics)
        metrics = _summarize_distillation_metrics(
            metric_samples,
            target_count=disk_tensors["target_count"],
            policy_count=(
                int(disk_tensors.get("policy_count", 0))
                if objective_config.policy is not None
                else None
            ),
        )
        metrics.setdefault("time/step_backend_train_s", time.monotonic() - started)
        step = await self._get_step(model)
        checkpoint_path: str | None = None
        if save_checkpoint:
            candidate = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), step
            )
            if os.path.exists(candidate):
                checkpoint_path = candidate
        wandb_run = model._get_wandb_run()
        if wandb_run is not None:
            self._record_provenance_nonblocking(wandb_run, "megatron-distillation")
        return LocalTrainResult(
            step=step,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
        )

    @asynccontextmanager
    async def current_step(
        self,
        model: AnyTrainableModel,
        *,
        ttl_s: float = 600.0,
    ) -> AsyncIterator[_CurrentStepSession]:
        """Freeze one learner revision across rollout, preparation, and one update."""

        if not model.trainable:
            raise TypeError("current_step requires a trainable model")
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("ttl_s must be finite and positive")
        storage_key = self._model_storage_key(model)
        if storage_key in self._active_current_steps:
            raise RuntimeError(
                "this backend already has an active current-step session"
            )
        service = await self._get_service(cast(TrainableModel, model))
        revision = await self._get_step(model)
        lease = await cast(Any, service).acquire_current_step(
            revision=revision,
            ttl_s=ttl_s,
        )
        session = _CurrentStepSession(
            revision=revision,
            session_id=lease.session_id,
        )
        consistency = CurrentStep(
            revision=session.revision,
            session_id=session.session_id,
        )
        active: _ActiveCurrentStep

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(max(min(ttl_s / 3.0, 60.0), 0.05))
                    await cast(Any, service).heartbeat_current_step(
                        session_id=session.session_id,
                        capability=lease.capability,
                        ttl_s=ttl_s,
                    )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                active.heartbeat_error = exc

        task = asyncio.create_task(_heartbeat())
        active = _ActiveCurrentStep(
            consistency=consistency,
            session=session,
            capability=lease.capability,
            heartbeat_task=task,
        )
        self._active_current_steps[storage_key] = active
        body_error: BaseException | None = None
        try:
            yield session
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._active_current_steps.pop(storage_key, None)
            release_error: BaseException | None = None
            try:
                await cast(Any, service).release_current_step(
                    session_id=consistency.session_id,
                    capability=lease.capability,
                )
            except BaseException as exc:
                release_error = exc
            if release_error is not None:
                if body_error is not None:
                    raise BaseExceptionGroup(
                        "Current-step work failed and its writer outcome is ambiguous.",
                        [body_error, release_error],
                    ) from None
                raise release_error
            if active.heartbeat_error is not None and body_error is None:
                raise RuntimeError("the current-step writer heartbeat failed") from (
                    active.heartbeat_error
                )

    def _require_active_current_step(
        self,
        model: AnyTrainableModel,
        consistency: CurrentStep,
        session: Any,
    ) -> _ActiveCurrentStep:
        active = self._active_current_steps.get(self._model_storage_key(model))
        if (
            active is None
            or active.consistency != consistency
            or session is not active.session
        ):
            raise ValueError(
                "CurrentStep and session= must come from the same active "
                "backend.current_step(model) context owned by this backend and model"
            )
        return active

    async def _validate_current_step(
        self,
        model: AnyTrainableModel,
        consistency: CurrentStep,
    ) -> None:
        active = self._active_current_steps.get(self._model_storage_key(model))
        if active is None or active.consistency != consistency:
            raise ValueError(
                "CurrentStep must come from an active backend.current_step(model) "
                "context owned by this backend and model"
            )
        if active.consumed:
            raise RuntimeError("the current-step session has already been consumed")
        if active.heartbeat_error is not None:
            raise RuntimeError("the current-step writer heartbeat failed") from (
                active.heartbeat_error
            )

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from .service import MegatronService

        storage_key = self._model_storage_key(model)
        if storage_key not in self._services:
            output_dir = get_model_dir(model=model, art_path=self._path)
            config = get_model_config(
                base_model=model.base_model,
                output_dir=output_dir,
                config=model._internal_config,
                lora_config=model.lora_config,
            )
            self._services[storage_key] = MegatronService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=output_dir,
                enable_expert_replay=self._enable_expert_replay,
            )
            if not self._in_process:
                self._services[storage_key] = move_to_child_process(
                    self._services[storage_key],
                    process_name="megatron-service",
                )
        return self._services[storage_key]

    async def _get_step(self, model: AnyTrainableModel) -> int:
        if not model.trainable:
            return 0
        storage_key = self._model_storage_key(model)
        if storage_key in self._resume_prepared_models:
            return await super()._get_step(model)
        output_dir = get_model_dir(model=model, art_path=self._path)
        info = prepare_megatron_resume_state(
            output_dir=output_dir,
            optimizer_state_path=optimizer_state_path(output_dir),
        )
        print(format_megatron_resume_message(info))
        self._resume_prepared_models.add(storage_key)
        return await super()._get_step(model)

    async def finalize_training_session(self, model: AnyTrainableModel) -> None:
        service = self._services.get(self._model_storage_key(model))
        if service is not None:
            await cast(Any, service).finalize_training_session()

    async def _delete_checkpoint_files(
        self,
        model: AnyTrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        output_dir = get_model_dir(model=model, art_path=self._path)
        commit = read_optimizer_commit(optimizer_state_path(output_dir))
        if commit is not None:
            steps_to_keep = sorted(set(steps_to_keep) | {commit.step})
        await super()._delete_checkpoint_files(model, steps_to_keep)

    async def close(self) -> None:
        failures: list[BaseException] = []
        for service in self._services.values():
            try:
                await asyncio.wait_for(
                    cast(Any, service).finalize_training_session(),
                    timeout=process_shutdown_timeout(1),
                )
            except BaseException as exc:
                failures.append(exc)
        await super().close()
        if failures:
            raise BaseExceptionGroup(
                "Failed to persist Megatron optimizer state during shutdown",
                failures,
            )

    def _default_sft_batch_size(self) -> int:
        import torch

        num_gpus = max(int(torch.cuda.device_count()), 1)
        tensor_parallel_size = min(2, num_gpus)
        return max(num_gpus // tensor_parallel_size, 1)
