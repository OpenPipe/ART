import asyncio
import math
import os
import time
from typing import Any, Iterable, cast

from mp_actors import move_to_child_process

from ..backend import AnyTrainableModel
from ..distill import PreparedTrainingBatch, TrainingObjectives
from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..metrics_taxonomy import average_metric_samples
from ..model import Model, TrainableModel
from ..trajectories import TrajectoryGroup
from ..types import LocalTrainResult, TrainConfig
from ..utils.lifecycle import process_shutdown_timeout
from ..utils.output_dirs import get_model_dir, get_step_checkpoint_dir
from .distillation import (
    DistillationObjectiveConfig,
    pack_prepared_batch,
    validate_standalone_forward_kl,
)
from .migrations import apply_megatron_migrations, optimizer_state_path
from .optimizer_state import (
    format_megatron_resume_message,
    prepare_megatron_resume_state,
    read_optimizer_commit,
)
from .runtime_config import get_megatron_runtime_config


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
        if "objectives" in kwargs or "idempotency_key" in kwargs:
            raise TypeError(
                "objectives and idempotency_key are only valid with a "
                "PreparedTrainingBatch"
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
        optimizer_save_interval = kwargs.get("optimizer_save_interval", 5)
        if (
            not isinstance(optimizer_save_interval, int)
            or isinstance(optimizer_save_interval, bool)
            or optimizer_save_interval < 1
        ):
            raise ValueError("optimizer_save_interval must be a positive integer")
        save_checkpoint = kwargs.get("save_checkpoint", True)
        if not isinstance(save_checkpoint, bool):
            raise TypeError("save_checkpoint must be a bool")
        verbose = kwargs.get("verbose", False)
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a bool")

        if objectives.policy is not None or objectives.distillation is None:
            raise ValueError(
                "M3 Megatron prepared-batch training supports standalone "
                "distillation only"
            )
        objective = objectives.distillation
        objective_config = DistillationObjectiveConfig(
            coefficient=objective.coefficient,
            compensate_temperature_squared=(objective.compensate_temperature_squared),
        )
        artifact_source_revision = batch.constraints.learner_revision
        service = await self._get_service(cast(TrainableModel, model))
        committed_step = await cast(Any, service).committed_distillation_step(
            idempotency_key=idempotency_key,
            expected_source_revision=artifact_source_revision,
            preparation_id=batch.preparation_id,
            payload_sha256=batch.payload_sha256,
            objective=objective_config,
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
        )
        config = TrainConfig(
            learning_rate=learning_rate,
            grad_accumulation_sequences=grad_accumulation_sequences,
            optimizer_save_interval=optimizer_save_interval,
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
            verbose=verbose,
        ):
            metric_samples.append(metrics)
        metrics = average_metric_samples(metric_samples)
        metrics.setdefault("time/step_backend_train_s", time.monotonic() - started)
        metrics.setdefault(
            "data/step_distillation_target_tokens",
            float(disk_tensors["target_count"]),
        )
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
