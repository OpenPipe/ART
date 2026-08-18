from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
import gc
import hashlib
import math
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock
import time
from typing import TYPE_CHECKING, Any, Iterator

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.distributed.object_store import (
    BinaryObjectTarget,
    S3BinaryObjectStore,
)
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_VLLM,
    encode_adapter_config,
)
from art.megatron.optimizer_state import (
    CheckpointFile,
    OptimizerAdapter,
    OptimizerShard,
    OptimizerTopology,
)
from art.utils.safetensors import PreparedSafetensors, SafetensorsLayout

from ..tensor_snapshot import PinnedCpuSnapshotStager
from ..training.gradient_accumulator import GradientAccumulator
from .data_plane import InMemoryPackedBatch, SFTBatchData, validate_packed_batch
from .publication import (
    TrainerPublicationFailed,
    TrainerPublicationSucceeded,
    TrainerRankPublication,
)
from .residency import ResidencyKey
from .run_residency import RunResidencyManager
from .specs import (
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    GenerationSnapshotJobSpec,
    LoadStateJobSpec,
    OptimizerJobSpec,
    ResidentLoraInspectionShard,
    ResidentLoraInspectionSpec,
    ResidentScoreJobSpec,
    ResidentScoreShard,
    SftForwardBackwardJobSpec,
    SftForwardJobSpec,
    SFTJobSpec,
    TrainerGeneration,
    TrainerJobSpec,
    TrainJobSpec,
)
from .trainer_run import EventSink

if TYPE_CHECKING:
    from art.megatron.lora import LoRASlotRef
    from art.trainer_rank import TrainerRankOptimizerState


def _consume_future(future: Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _command_token_logprobs(
    batch: InMemoryPackedBatch, outputs: list[Any] | tuple[Any, ...]
) -> tuple[Any, ...]:
    if batch.ref.training_kind != "tokenized":
        return tuple(
            tuple(float(item) for item in values.flatten().tolist())
            for values in outputs
        )
    output_map = batch.ref.tokenized_output_map
    if output_map is None:
        raise RuntimeError("tokenized batch has no output map")
    target_tokens = batch.tensors.get("target_tokens")
    if target_tokens is None:
        raise RuntimeError("tokenized batch has no target tensor")
    candidate_capacity = int(target_tokens.shape[2])
    physical = torch.cat(
        [values.reshape(-1, candidate_capacity) for values in outputs], dim=0
    )
    expected_rows = batch.ref.num_sequences * batch.ref.sequence_length
    if int(physical.shape[0]) != expected_rows:
        raise RuntimeError(
            "tokenized command did not return every physical packed row: "
            f"returned={physical.shape[0]}, expected={expected_rows}"
        )
    host = physical.detach().cpu()
    logical = []
    for positions, candidates in zip(
        output_map.packed_positions, output_map.candidate_counts, strict=True
    ):
        values = host[list(positions), :candidates]
        if candidates == 1:
            logical.append(tuple(float(item) for item in values[:, 0].tolist()))
        else:
            logical.append(
                tuple(tuple(float(item) for item in row) for row in values.tolist())
            )
    return tuple(logical)


class MegatronTrainJobExecutor:
    """Thin adapter around the warm runtime's in-memory job entrypoint."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._publisher = _GenerationPublisher(
            runtime,
            capacity=int(runtime.snapshot_pool_capacity),
        )
        self._gradients = GradientAccumulator(model_chunks=runtime.model)
        self._gradient_parent_version: int | None = None
        self._python_gc_stabilized = False
        self._closed = False

    def execute(
        self,
        job: TrainJobSpec,
        batch: InMemoryPackedBatch,
        sink: EventSink,
        cancelled: Event,
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        timing = self.runtime.inter_forward_backward_timing
        timing.current_job_start_s = time.monotonic()
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_rl_job

        metrics = execute_megatron_rl_job(
            self.runtime,
            job,
            batch.tensors,
            progress_sink=lambda step_index, num_steps, metrics: sink.progress(
                step_index=step_index,
                num_steps=num_steps,
                metrics=metrics,
            ),
            adapter_ready_sink=lambda: sink.adapter_ready(
                learner_version=job.learner_version,
                adapter_path=job.output_adapter_path,
            ),
            snapshot_sink=lambda job, adapter_dtypes, adapter_config, save_optimizer: (
                self._publisher.stage_and_submit(
                    run_id=job.run_id,
                    generation=job.output.generation,
                    optimizer_state_path=job.output.optimizer_state_path,
                    staging_adapter_path=job.output.staging_adapter_path,
                    publication_targets=job.publication_targets,
                    adapter_dtypes=adapter_dtypes,
                    adapter_config=adapter_config,
                    save_optimizer=save_optimizer,
                    sink=sink,
                )
            ),
            cancelled=cancelled,
        )
        metrics.update(self._stabilize_python_gc())
        timing.previous_job_complete_s = time.monotonic()
        return metrics

    def execute_sft(
        self,
        job: SFTJobSpec,
        batches: tuple[SFTBatchData, ...],
        sink: EventSink,
        cancelled: Event,
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        timing = self.runtime.inter_forward_backward_timing
        timing.current_job_start_s = time.monotonic()
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_sft_job

        metrics = execute_megatron_sft_job(
            self.runtime,
            job,
            batches,
            progress_sink=lambda step_index, num_steps, metrics: sink.progress(
                step_index=step_index,
                num_steps=num_steps,
                metrics=metrics,
            ),
            adapter_ready_sink=lambda: sink.adapter_ready(
                learner_version=job.learner_version,
                adapter_path=job.output_adapter_path,
            ),
            snapshot_sink=lambda snapshot_job, adapter_dtypes, adapter_config, save_optimizer: (
                self._publisher.stage_and_submit(
                    run_id=snapshot_job.run_id,
                    generation=snapshot_job.output.generation,
                    optimizer_state_path=snapshot_job.output.optimizer_state_path,
                    staging_adapter_path=snapshot_job.output.staging_adapter_path,
                    publication_targets=snapshot_job.publication_targets,
                    adapter_dtypes=adapter_dtypes,
                    adapter_config=adapter_config,
                    save_optimizer=save_optimizer,
                    sink=sink,
                )
            ),
            cancelled=cancelled,
        )
        metrics.update(self._stabilize_python_gc())
        timing.previous_job_complete_s = time.monotonic()
        return metrics

    def _stabilize_python_gc(self) -> dict[str, float]:
        if self._python_gc_stabilized or not self.runtime.transformer_layers_compiled:
            return {}
        started = time.perf_counter()
        collected = gc.collect()
        gc.freeze()
        self._python_gc_stabilized = True
        return {
            "python_gc_stabilize_s": time.perf_counter() - started,
            "python_gc_collected_objects": float(collected),
            "python_gc_frozen_objects": float(gc.get_freeze_count()),
        }

    def score(
        self,
        job: ResidentScoreJobSpec,
        batch: InMemoryPackedBatch,
    ) -> ResidentScoreShard:
        self._validate_resident_score(job.run_id, job.learner)
        validate_packed_batch(batch)
        from art.megatron.train import execute_megatron_score_job

        return execute_megatron_score_job(self.runtime, job, batch.tensors)

    def inspect_resident_lora(
        self,
        request: ResidentLoraInspectionSpec,
    ) -> ResidentLoraInspectionShard:
        self._validate_resident_inspection(request.run_id, request.learner)
        from art.megatron.train import inspect_resident_lora

        return inspect_resident_lora(self.runtime, request)

    def _validate_diagnostic_runtime(self) -> None:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        self._publisher.raise_if_failed()

    def _validate_resident_score(self, run_id: str, learner: TrainerGeneration) -> None:
        self._validate_diagnostic_runtime()
        runtime = self.runtime
        if (
            runtime.resident_run_id != run_id
            or runtime.resident_training_session_id != learner.training_session_id
            or runtime.resident_policy_step != learner.policy_step
            or runtime.resident_generation_id != learner.generation_id
            or not runtime.optimizer_state_loaded
            or runtime.optimizer is None
        ):
            raise RuntimeError("resident trainer state does not match score learner")

    def _validate_resident_inspection(
        self, run_id: str, learner: TrainerGeneration
    ) -> None:
        self._validate_diagnostic_runtime()
        runtime = self.runtime
        if runtime.resident_run_id != run_id:
            raise RuntimeError("resident trainer run does not match inspection")
        unhydrated = (
            runtime.resident_training_session_id is None
            and runtime.resident_policy_step is None
            and runtime.resident_generation_id is None
            and not runtime.optimizer_state_loaded
        )
        hydrated = (
            runtime.resident_training_session_id == learner.training_session_id
            and runtime.resident_policy_step == learner.policy_step
            and runtime.resident_generation_id == learner.generation_id
            and runtime.optimizer_state_loaded
        )
        if not (unhydrated or hydrated):
            raise RuntimeError(
                "resident trainer hydration markers are partial or do not match "
                "the inspection learner"
            )

    def execute_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        if self._gradient_parent_version not in {
            None,
            job.expected_learner_version,
        }:
            raise RuntimeError(
                "F/B parent does not match the open gradient accumulator"
            )
        from art.megatron.train import execute_megatron_rl_forward_backward_job

        result = execute_megatron_rl_forward_backward_job(
            self.runtime,
            job,
            batch.tensors,
            gradient_accumulator=self._gradients,
            cancelled=cancelled,
        )
        self._gradient_parent_version = job.expected_learner_version
        step = result.result
        return {
            "operation_id": job.operation_id,
            "metrics": result.metrics(),
            "token_count": int(result.token_count.item()),
            "token_logprobs": _command_token_logprobs(batch, step.new_logprobs),
        }

    def execute_forward(
        self,
        job: ForwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        from art.megatron.train import (
            _prepare_rl_training_state,
            execute_megatron_rl_forward_job,
        )

        _prepare_rl_training_state(self.runtime, job)
        result = execute_megatron_rl_forward_job(
            self.runtime,
            job,
            batch.tensors,
            cancelled=cancelled,
        )
        return {
            **result,
            "token_logprobs": _command_token_logprobs(batch, result["token_logprobs"]),
        }

    def execute_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        if self._gradient_parent_version not in {
            None,
            job.expected_learner_version,
        }:
            raise RuntimeError(
                "SFT F/B parent does not match the open gradient accumulator"
            )
        from art.megatron.train import execute_megatron_sft_forward_backward_job

        result = execute_megatron_sft_forward_backward_job(
            self.runtime,
            job,
            batch,
            gradient_accumulator=self._gradients,
            cancelled=cancelled,
        )
        self._gradient_parent_version = job.expected_learner_version
        return result

    def execute_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_sft_forward_job

        return execute_megatron_sft_forward_job(
            self.runtime,
            job,
            batch,
            cancelled=cancelled,
        )

    def execute_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        if self._gradient_parent_version != job.expected_learner_version:
            raise RuntimeError("optimizer parent does not match accumulated gradients")
        runtime = self.runtime
        if runtime.optimizer is None:
            raise RuntimeError("trainer has no resident optimizer")
        self._gradients.seal(job.contributing_forward_backward_operation_ids)
        self._gradients.prepare_optimizer()
        for group in runtime.optimizer.param_groups:
            group["betas"] = (job.optimizer.beta1, job.optimizer.beta2)
            group["eps"] = job.optimizer.eps
            group["weight_decay"] = job.optimizer.weight_decay
        runtime.optimizer.config.adam_beta1 = job.optimizer.beta1
        runtime.optimizer.config.adam_beta2 = job.optimizer.beta2
        runtime.optimizer.config.adam_eps = job.optimizer.eps
        runtime.optimizer.config.weight_decay = job.optimizer.weight_decay
        runtime.optimizer.config.clip_grad = job.optimizer.grad_clip_norm
        from art.megatron.train import run_megatron_optimizer_step

        started = time.perf_counter()
        result = run_megatron_optimizer_step(
            optimizer=runtime.optimizer,
            learning_rate=job.optimizer.learning_rate,
            model_support_handler=runtime.model_support_handler,
            model_chunks=runtime.model,
            before_step=runtime.optimizer_snapshot_barrier.wait_before_mutation,
        )
        if not result.update_successful or not math.isfinite(result.grad_norm):
            raise RuntimeError(
                "Megatron optimizer rejected the update: "
                f"update_successful={result.update_successful}, "
                f"grad_norm={result.grad_norm}"
            )
        optimizer_step_s = time.perf_counter() - started
        consumed = self._gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_policy_step = job.learner_version
        runtime.resident_generation_id = job.generation.generation_id
        runtime.optimizer_state_loaded = True
        self._gradient_parent_version = None
        if (
            runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError("optimizer output has no adapter export layout")
        snapshot_metrics = self._publisher.stage(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes=runtime.adapter_export_dtypes,
            adapter_config=runtime.adapter_export_config,
            snapshot_optimizer=False,
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": consumed,
            "metrics": {
                "loss/learning_rate": job.optimizer.learning_rate,
                "loss/grad_norm": float(result.grad_norm),
                "optimizer/update_successful": float(result.update_successful),
                "optimizer/num_zeros_in_grad": float(result.num_zeros_in_grad or 0),
                "time/optimizer_step_s": optimizer_step_s,
                **snapshot_metrics,
            },
        }

    def execute_load_state(self, job: LoadStateJobSpec) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        from art.megatron.train import execute_megatron_load_state_job

        execute_megatron_load_state_job(self.runtime, job)
        self._gradient_parent_version = None
        runtime = self.runtime
        if (
            runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError("loaded state has no adapter export layout")
        snapshot_metrics = self._publisher.stage(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes=runtime.adapter_export_dtypes,
            adapter_config=runtime.adapter_export_config,
            snapshot_optimizer=job.restore_optimizer,
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "optimizer_restored": job.restore_optimizer,
            "metrics": snapshot_metrics,
        }

    def execute_snapshot(
        self,
        job: GenerationSnapshotJobSpec,
        sink: EventSink,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        runtime = self.runtime
        if (
            runtime.resident_training_session_id != job.training_session_id
            or runtime.resident_policy_step != job.learner_version
            or runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError("resident trainer state does not match snapshot learner")
        if job.save_optimizer and (
            not runtime.optimizer_state_loaded or runtime.optimizer is None
        ):
            raise RuntimeError("snapshot requested non-resident optimizer state")
        stage_metrics = self._publisher.ensure_generation(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes=runtime.adapter_export_dtypes,
            adapter_config=runtime.adapter_export_config,
            snapshot_optimizer=job.save_optimizer,
        )
        metrics = {
            **stage_metrics,
            **self._publisher.submit(
                generation=job.generation,
                optimizer_state_path=job.optimizer_state_path,
                staging_adapter_path=job.staging_adapter_path,
                existing_adapter=job.existing_adapter,
                publication_targets=job.publication_targets,
                adapter_object_target=job.adapter_object_target,
                save_optimizer=job.save_optimizer,
                sink=sink,
            ),
        }
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "metrics": metrics,
        }

    def discard_open_gradients(self) -> None:
        self._gradients.discard()
        self._gradient_parent_version = None

    def advance_without_training(
        self,
        *,
        source: TrainerGeneration,
        output: TrainerGeneration,
        optimizer_state_path: str,
        adapter: "OptimizerAdapter | None",
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        if (
            output.training_session_id != source.training_session_id
            or output.policy_step != source.policy_step + 1
        ):
            raise ValueError(
                "a no-op transition must preserve session and advance one step"
            )
        runtime = self.runtime
        if (
            runtime.resident_training_session_id != source.training_session_id
            or runtime.resident_policy_step != source.policy_step
            or runtime.resident_generation_id != source.generation_id
            or not runtime.optimizer_state_loaded
            or runtime.optimizer is None
        ):
            raise RuntimeError("resident trainer state does not match no-op transition")
        del optimizer_state_path, adapter
        runtime.resident_policy_step = output.policy_step
        runtime.resident_generation_id = output.generation_id
        return {}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        try:
            self._gradients.discard()
        except BaseException as error:
            failures.append(error)
        try:
            self._publisher.close()
            self.runtime.optimizer_snapshot_barrier.synchronize()
        except BaseException as error:
            failures.append(error)
        controller = getattr(self.runtime, "moe_routing_replay_controller", None)
        if controller is not None:
            try:
                controller.remove_router_patches()
            except BaseException as error:
                failures.append(error)
            finally:
                self.runtime.moe_routing_replay_controller = None
        try:
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Megatron executor close failed", failures)

    def _require_no_open_gradients(self) -> None:
        if self._gradients.contribution_ids:
            raise RuntimeError("operation cannot discard open gradient contributions")


class _PreparedRunLoad(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    operation_id: str
    key: ResidencyKey
    checkpoint: Any
    optimizer: Any | None
    adapter_config: dict[str, Any]

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        optimizer = () if self.optimizer is None else self.optimizer.tensors
        return (*self.checkpoint.parameters, *optimizer)


class _ResidentRunState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tenant_id: str
    run_id: str
    training_session_id: str
    learner_version: int
    adapter_config: dict[str, Any]
    gradients: Any
    desired_key: ResidencyKey
    installed_key: ResidencyKey
    pending_load: _PreparedRunLoad | None = None
    gradient_keys: dict[str, ResidencyKey] = Field(default_factory=dict)
    next_accumulator_revision: int = 1
    registration_complete: bool = False


class MCoreRunSlotExecutor:
    """Train independent exact-shape LoRAs on one warm MCore runtime."""

    def __init__(self, runtime: Any) -> None:
        from art.trainer_rank import TrainerRank

        self.runtime = runtime
        self._slot_trainer = TrainerRank(runtime)
        self._publisher = _GenerationPublisher(
            runtime,
            capacity=int(runtime.snapshot_pool_capacity),
        )
        if runtime.run_residency_config is None:
            raise RuntimeError("multi-run Megatron requires explicit residency limits")
        if runtime.optimizer_layout_fingerprint is None:
            raise RuntimeError("multi-run Megatron has no topology fingerprint")
        self._residency = RunResidencyManager(
            runtime.run_residency_config,
            snapshot_barrier=runtime.optimizer_snapshot_barrier,
        )
        self._load_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-load-prepare"
        )
        self._load_preparations: dict[
            str, tuple[str, str, Future[_PreparedRunLoad]]
        ] = {}
        self._runs: dict[str, _ResidentRunState] = {}
        self._closed = False

    def register_run(
        self,
        *,
        tenant_id: str,
        run_id: str,
        training_session_id: str,
        learner_version: int,
        generation_id: str,
        adapter_path: str,
    ) -> None:
        if self._closed:
            raise RuntimeError("Megatron run slot is closed")
        if run_id in self._runs:
            raise RuntimeError(f"training run is already resident: {run_id!r}")
        from art.megatron.model_support.lora_disk import load_adapter_config
        from art.megatron.training.gradient_accumulator import (
            ParameterGradientAccumulator,
        )
        from art.trainer_rank import MaterializedCheckpoint

        adapter_config = load_adapter_config(adapter_path)
        adapter_layout_fingerprint = hashlib.sha256(
            encode_adapter_config(adapter_config)
        ).hexdigest()
        self._slot_trainer.load_checkpoint_sync(
            MaterializedCheckpoint(path=run_id, directory=adapter_path)
        )
        key = ResidencyKey(
            tenant_id=tenant_id,
            run_id=run_id,
            generation_id=generation_id,
            topology_fingerprint=self.runtime.optimizer_layout_fingerprint,
            adapter_layout_fingerprint=adapter_layout_fingerprint,
        )
        self._runs[run_id] = _ResidentRunState(
            tenant_id=tenant_id,
            run_id=run_id,
            training_session_id=training_session_id,
            learner_version=learner_version,
            adapter_config=adapter_config,
            gradients=ParameterGradientAccumulator(
                parameters=self._slot_trainer.checkpoint_slot_parameters(run_id)
            ),
            desired_key=key,
            installed_key=key,
        )

    def complete_run_registration(self, run_id: str) -> None:
        state = self._require_run(run_id, require_complete=False)
        if state.registration_complete:
            return
        tensors = self._slot_trainer.checkpoint_slot_residency_tensors(run_id).all
        self._residency.register_l1(state.installed_key, tensors)
        state.registration_complete = True

    def optimizer_layout(self, run_id: str) -> Any:
        self._require_run(run_id, require_complete=False)
        return self._slot_trainer.checkpoint_slot_optimizer_layout(run_id)

    def restore_optimizer_state(
        self, run_id: str, optimizer_state: "TrainerRankOptimizerState"
    ) -> None:
        run = self._require_run(run_id, require_complete=False)
        if run.registration_complete:
            raise RuntimeError("cannot restore optimizer after residency registration")
        self._slot_trainer.restore_checkpoint_slot_optimizer_state(
            run_id, optimizer_state
        )

    def execute_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        validate_packed_batch(batch)
        from art.megatron.train import (
            execute_megatron_dynamic_lora_forward_backward_job,
        )

        with self._resident(state, include_gradients=True):
            result = execute_megatron_dynamic_lora_forward_backward_job(
                self.runtime,
                job,
                batch.tensors,
                slot_trainer=self._slot_trainer,
                gradient_accumulator=state.gradients,
                cancelled=cancelled,
            )
            self._register_gradient_contribution(state, job.operation_id)
        step = result.result
        return {
            "operation_id": job.operation_id,
            "metrics": result.metrics(),
            "token_count": int(result.token_count.item()),
            "token_logprobs": _command_token_logprobs(batch, step.new_logprobs),
        }

    def execute_forward(
        self,
        job: ForwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_dynamic_lora_forward_job

        with self._resident(state):
            result = execute_megatron_dynamic_lora_forward_job(
                self.runtime,
                job,
                batch.tensors,
                cancelled=cancelled,
            )
        return {
            **result,
            "token_logprobs": _command_token_logprobs(batch, result["token_logprobs"]),
        }

    def execute_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        from art.megatron.train import (
            execute_megatron_dynamic_lora_sft_forward_backward_job,
        )

        with self._resident(state, include_gradients=True):
            result = execute_megatron_dynamic_lora_sft_forward_backward_job(
                self.runtime,
                job,
                batch,
                slot_trainer=self._slot_trainer,
                gradient_accumulator=state.gradients,
                cancelled=cancelled,
            )
            self._register_gradient_contribution(state, job.operation_id)
            return result

    def execute_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_dynamic_lora_sft_forward_job

        with self._resident(state):
            return execute_megatron_dynamic_lora_sft_forward_job(
                self.runtime,
                job,
                batch,
                cancelled=cancelled,
            )

    def execute_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        state.gradients.seal(job.contributing_forward_backward_operation_ids)
        from art.trainer_rank import AdamParams

        with self._resident(state, include_gradients=True):
            self.runtime.optimizer_snapshot_barrier.wait_before_mutation()
            gradients = state.gradients.prepare_optimizer()
            started = time.perf_counter()
            result = self._slot_trainer.optim_step_reduced(
                job.run_id,
                params=AdamParams(
                    learning_rate=job.optimizer.learning_rate,
                    beta1=job.optimizer.beta1,
                    beta2=job.optimizer.beta2,
                    eps=job.optimizer.eps,
                    weight_decay=job.optimizer.weight_decay,
                    grad_clip_norm=job.optimizer.grad_clip_norm,
                ),
                grads=gradients,
            )
        if not result["update_successful"] or not math.isfinite(result["grad_norm"]):
            raise RuntimeError("dynamic LoRA optimizer rejected the update")
        optimizer_step_s = time.perf_counter() - started
        consumed = state.gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
        parent_key = state.installed_key
        output_key = parent_key.model_copy(
            update={"generation_id": job.generation.generation_id}
        )
        residency_tensors = self._slot_trainer.checkpoint_slot_residency_tensors(
            job.run_id
        )
        optimizer_source = (
            self._slot_trainer.checkpoint_slot_optimizer_residency_source(job.run_id)
        )
        if optimizer_source is None:
            raise RuntimeError("optimizer commit has no immutable optimizer state")
        l2 = self._residency.advance_l1(
            parent_key,
            output_key,
            residency_tensors.all,
            retire_source=True,
        )
        state.desired_key = output_key
        state.installed_key = output_key
        state.learner_version = job.learner_version
        for operation_id in consumed:
            self._residency.retire_async(state.gradient_keys.pop(operation_id))
        from art.megatron.lora import LoRASlotRef

        snapshot_metrics = self._publisher.stage(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes={},
            adapter_config=state.adapter_config,
            slot_ref=LoRASlotRef("checkpoint", job.run_id),
            snapshot_optimizer=False,
        )
        snapshot_metrics.update(
            self._publisher.attach_resident_optimizer(
                generation=job.generation,
                source=optimizer_source,
                l2=l2,
                tensor_offset=len(residency_tensors.weights),
            )
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": consumed,
            "metrics": {
                "loss/learning_rate": job.optimizer.learning_rate,
                "loss/grad_norm": result["grad_norm"],
                "optimizer/update_successful": result["update_successful"],
                "optimizer/num_zeros_in_grad": result["num_zeros_in_grad"],
                "time/optimizer_step_s": optimizer_step_s,
                **snapshot_metrics,
            },
        }

    def optimizer_state(self, run_id: str) -> "TrainerRankOptimizerState | None":
        self._require_run(run_id)
        return self._slot_trainer.checkpoint_slot_optimizer_state(run_id)

    def prepare_load_state(self, job: LoadStateJobSpec) -> _PreparedRunLoad:
        state = self._require_run(job.run_id)
        from art.megatron.model_support.lora_disk import load_adapter_config
        from art.megatron.optimizer_state import load_trainer_rank_optimizer_state
        from art.trainer_rank import MaterializedCheckpoint

        config = load_adapter_config(job.adapter_path)
        self._validate_adapter_layout(state.adapter_config, config)
        checkpoint = self._slot_trainer.prepare_checkpoint_slot_load_sync(
            MaterializedCheckpoint(path=job.run_id, directory=job.adapter_path),
            device="cpu",
        )
        optimizer_state = None
        if job.optimizer_state_path is not None:
            assert job.optimizer_generation_id is not None
            loaded = load_trainer_rank_optimizer_state(
                self.runtime,
                optimizer_state_path=job.optimizer_state_path,
                adapter_path=job.adapter_path,
                adapter_step=job.adapter_step,
                optimizer_generation_id=job.optimizer_generation_id,
                layout=self._slot_trainer.checkpoint_slot_optimizer_layout(job.run_id),
            )
            optimizer_state = (
                self._slot_trainer.prepare_checkpoint_slot_optimizer_for_residency(
                    job.run_id, checkpoint, loaded
                )
            )
        adapter_layout_fingerprint = hashlib.sha256(
            encode_adapter_config(config)
        ).hexdigest()
        prepared = _PreparedRunLoad(
            operation_id=job.operation_id,
            key=ResidencyKey(
                tenant_id=state.tenant_id,
                run_id=job.run_id,
                generation_id=job.generation.generation_id,
                topology_fingerprint=self.runtime.optimizer_layout_fingerprint,
                adapter_layout_fingerprint=adapter_layout_fingerprint,
            ),
            checkpoint=checkpoint,
            optimizer=optimizer_state,
            adapter_config=config,
        )
        if not prepared.tensors or any(
            tensor.device.type != "cpu" for tensor in prepared.tensors
        ):
            raise RuntimeError(
                "prepared checkpoint generation is not entirely CPU resident"
            )
        return prepared

    def start_prepare_load_state(self, job: LoadStateJobSpec) -> None:
        existing = self._load_preparations.get(job.operation_id)
        if existing is not None:
            if existing[:2] != (job.run_id, job.fingerprint):
                raise RuntimeError(
                    "operation_id was reused for another load preparation"
                )
            return
        self._load_preparations[job.operation_id] = (
            job.run_id,
            job.fingerprint,
            self._load_pool.submit(self.prepare_load_state, job),
        )

    def load_state_prepared(self, job: LoadStateJobSpec) -> bool:
        try:
            run_id, fingerprint, future = self._load_preparations[job.operation_id]
        except KeyError as error:
            raise RuntimeError("load preparation was not started") from error
        if (run_id, fingerprint) != (job.run_id, job.fingerprint):
            raise RuntimeError("load preparation identity changed")
        if future.done():
            future.result()
            return True
        return False

    def finish_prepared_load_state(self, job: LoadStateJobSpec) -> dict[str, Any]:
        if not self.load_state_prepared(job):
            raise RuntimeError("load preparation is not complete")
        _run_id, _fingerprint, future = self._load_preparations.pop(job.operation_id)
        return self.commit_load_state(job, future.result())

    def discard_prepared_load_state(self, operation_id: str) -> None:
        preparation = self._load_preparations.pop(operation_id, None)
        if preparation is None:
            return
        future = preparation[2]
        if not future.cancel():
            future.add_done_callback(_consume_future)

    def commit_load_state(
        self, job: LoadStateJobSpec, prepared: _PreparedRunLoad
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        if state.gradients.contribution_ids:
            raise RuntimeError("load_state cannot discard open gradient contributions")
        if (
            prepared.operation_id != job.operation_id
            or prepared.key.generation_id != job.generation.generation_id
            or (prepared.optimizer is not None) != job.restore_optimizer
        ):
            raise RuntimeError("prepared load does not match its ordered command")
        if state.pending_load is not None:
            self._residency.retire(state.pending_load.key)
        image = self._residency.register_l2(prepared.key, prepared.tensors)
        state.desired_key = prepared.key
        state.pending_load = prepared
        state.learner_version = job.learner_version
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "optimizer_restored": job.restore_optimizer,
            "metrics": {"residency/load_l2_bytes": float(image.stats.byte_count)},
        }

    def execute_snapshot(
        self,
        job: GenerationSnapshotJobSpec,
        sink: EventSink,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(state, job.training_session_id, job.learner_version)
        if state.desired_key.generation_id != job.generation.generation_id:
            raise RuntimeError("snapshot does not identify the selected generation")
        if not self._publisher.has_generation(
            job.generation, require_optimizer=job.save_optimizer
        ):
            raise RuntimeError(
                "selected generation has no immutable publication snapshot"
            )
        metrics = self._publisher.submit(
            generation=job.generation,
            optimizer_state_path=job.optimizer_state_path,
            staging_adapter_path=job.staging_adapter_path,
            existing_adapter=job.existing_adapter,
            publication_targets=job.publication_targets,
            adapter_object_target=job.adapter_object_target,
            save_optimizer=job.save_optimizer,
            sink=sink,
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "metrics": metrics,
        }

    def record_archive(
        self,
        run_id: str,
        generation_id: str,
        *,
        immutable_ref: str,
        digest: str,
        byte_count: int,
    ) -> bool:
        self._require_run(run_id)
        keys = tuple(
            key
            for key in self._residency.keys(run_id)
            if key.accumulator_revision == 0 and key.generation_id == generation_id
        )
        if not keys:
            return False
        if len(keys) != 1:
            raise RuntimeError("generation has multiple base residency identities")
        self._residency.record_l4(
            keys[0],
            immutable_ref=immutable_ref,
            digest=digest,
            byte_count=byte_count,
        )
        return True

    def discard_run_gradients(self, run_id: str) -> None:
        self._require_run(run_id).gradients.discard()

    def unregister_run(self, run_id: str) -> None:
        state = self._require_run(run_id)
        for operation_id, (prepared_run_id, _fingerprint, _future) in tuple(
            self._load_preparations.items()
        ):
            if prepared_run_id == run_id:
                self.discard_prepared_load_state(operation_id)
        state.gradients.discard()
        retirements = tuple(
            self._residency.retire_async(key) for key in self._residency.keys(run_id)
        )
        for retirement in retirements:
            retirement.result()
        self._slot_trainer.unload_checkpoint_slot(run_id)
        self._runs.pop(run_id)
        self._publisher.retire_run(run_id)

    def close(self) -> None:
        if self._closed:
            return
        for state in self._runs.values():
            state.gradients.discard()
        futures = tuple(value[2] for value in self._load_preparations.values())
        for future in futures:
            future.cancel()
        _done, pending = wait(
            futures, timeout=self._residency.config.shutdown_timeout_s
        )
        self._load_pool.shutdown(wait=False, cancel_futures=True)
        self._publisher.close()
        self._residency.close()
        self._closed = True
        if pending:
            raise TimeoutError(
                f"{len(pending)} load preparations exceeded shutdown timeout"
            )

    def _require_run(
        self, run_id: str, *, require_complete: bool = True
    ) -> _ResidentRunState:
        if self._closed:
            raise RuntimeError("Megatron run slot is closed")
        try:
            state = self._runs[run_id]
        except KeyError as exc:
            raise RuntimeError(f"training run is not resident: {run_id!r}") from exc
        if require_complete and not state.registration_complete:
            raise RuntimeError("training run residency registration is incomplete")
        return state

    def _register_gradient_contribution(
        self, state: _ResidentRunState, operation_id: str
    ) -> None:
        key = state.installed_key.model_copy(
            update={"accumulator_revision": state.next_accumulator_revision}
        )
        tensors = state.gradients.contribution_residency_tensors(operation_id)
        self._residency.register_l1(key, tensors)
        state.gradient_keys[operation_id] = key
        state.next_accumulator_revision += 1

    @contextmanager
    def _resident(
        self, state: _ResidentRunState, *, include_gradients: bool = False
    ) -> Iterator[None]:
        base_key = state.desired_key
        acquired: list[ResidencyKey] = []
        try:
            self._residency.acquire_l1(base_key)
            acquired.append(base_key)
            if state.installed_key != base_key:
                pending = state.pending_load
                if pending is None or pending.key != base_key:
                    raise RuntimeError("desired learner has no prepared L1 install")
                previous = state.installed_key
                try:
                    self._slot_trainer.install_prepared_checkpoint_slot_load_sync(
                        pending.checkpoint, pending.optimizer
                    )
                except BaseException:
                    self._residency.release_l1(acquired.pop())
                    self._residency.acquire_l1(previous)
                    self._residency.release_l1(previous)
                    raise
                live = self._slot_trainer.checkpoint_slot_residency_tensors(
                    state.run_id
                ).all
                if tuple(map(id, live)) != tuple(map(id, pending.tensors)):
                    raise RuntimeError(
                        "installed checkpoint changed prepared residency tensors"
                    )
                from art.megatron.training.gradient_accumulator import (
                    ParameterGradientAccumulator,
                )

                state.gradients = ParameterGradientAccumulator(
                    parameters=self._slot_trainer.checkpoint_slot_parameters(
                        state.run_id
                    )
                )
                state.adapter_config = pending.adapter_config
                state.installed_key = base_key
                state.pending_load = None
                self._residency.retire_async(previous)
            for key in state.gradient_keys.values() if include_gradients else ():
                self._residency.acquire_l1(key)
                acquired.append(key)
            yield
        finally:
            for key in reversed(acquired):
                self._residency.release_l1(key)

    @staticmethod
    def _validate_parent(
        state: _ResidentRunState,
        training_session_id: str,
        learner_version: int,
    ) -> None:
        if (
            state.training_session_id != training_session_id
            or state.learner_version != learner_version
        ):
            raise RuntimeError("resident run state does not match command parent")

    @staticmethod
    def _validate_adapter_layout(
        current: dict[str, Any], replacement: dict[str, Any]
    ) -> None:
        for key in ("base_model_name_or_path", "r", "lora_alpha"):
            if replacement.get(key) != current.get(key):
                raise ValueError(f"loaded adapter changes immutable {key}")
        if set(replacement.get("target_modules", ())) != set(
            current.get("target_modules", ())
        ):
            raise ValueError("loaded adapter changes immutable target_modules")


class _ResolvedGeneration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lora: Any | None
    optimizer: Any | None
    prepared_tensors: PreparedSafetensors | None


class _CachedGeneration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    generation: TrainerGeneration
    stager: PinnedCpuSnapshotStager
    resolved: Future[_ResolvedGeneration]
    optimizer_upgrade: Future[Any] | None = None
    has_optimizer: bool = False
    consumers: list[Future[TrainerRankPublication]] = Field(default_factory=list)
    object_ids: set[str] = Field(default_factory=set)
    retired: bool = False
    released: bool = False


class _SnapshotTransport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: OptimizerAdapter | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class _RankSnapshotPersistence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: TrainerGeneration
    rank: int = Field(ge=0)
    adapter: OptimizerAdapter | None = None
    shard: OptimizerShard | None = None
    runtime_sha256: str | None = None
    topology: OptimizerTopology | None = None
    saves_optimizer: bool
    metrics: dict[str, float] = Field(default_factory=dict)


class _GenerationPublisher:
    def __init__(
        self,
        runtime: Any,
        *,
        capacity: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("snapshot pool capacity must be positive")
        self.runtime = runtime
        self.capacity = capacity
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._available_stagers = [
            PinnedCpuSnapshotStager(reusable=True) for _ in range(capacity)
        ]
        self._resolution_pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="art-snapshot-resolve"
        )
        self._transport_pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="art-publish-transport"
        )
        self._durability_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-publish-durable"
        )
        self._completion_pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="art-publish-complete"
        )
        self._transport_sender: Any | None = None
        self._transport_sender_lock = Lock()
        self._object_store: S3BinaryObjectStore | None = None
        self._object_publications: dict[str, Future[OptimizerAdapter]] = {}
        self._lora_layouts: dict[tuple[Any, ...], SafetensorsLayout] = {}
        self._cache: dict[str, _CachedGeneration] = {}
        self._latest_by_run: dict[str, str] = {}
        self._failures: list[BaseException] = []
        self._in_flight = 0

    def stage(
        self,
        *,
        run_id: str,
        generation: TrainerGeneration,
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
        slot_ref: "LoRASlotRef | None" = None,
        trainer_rank_optimizer_state: "TrainerRankOptimizerState | None" = None,
        snapshot_optimizer: bool = True,
    ) -> dict[str, float]:
        from art.megatron.optimizer_state import (
            stage_optimizer_state_snapshot,
            stage_trainer_rank_optimizer_state_snapshot,
        )
        from art.megatron.weights.lora_publish import (
            LoraSnapshotTimings,
            stage_vllm_lora_snapshot_from_model,
        )

        self._retire_previous(run_id)
        wait_s, in_flight, stager = self._acquire_slot()
        prepare_started = time.perf_counter()
        try:
            lora_timings = LoraSnapshotTimings()
            lora = stage_vllm_lora_snapshot_from_model(
                model=self.runtime.model,
                adapter_dtypes=adapter_dtypes,
                handler=self.runtime.model_support_handler,
                adapter_config=adapter_config,
                rank=self.runtime.rank,
                world_size=self.runtime.world_size,
                stager=stager,
                slot_ref=slot_ref,
                timings=lora_timings,
            )
            lora_launch_s = time.perf_counter() - prepare_started
            optimizer_started = time.perf_counter()
            if snapshot_optimizer:
                if slot_ref is not None:
                    if trainer_rank_optimizer_state is None:
                        raise RuntimeError(
                            "dynamic LoRA snapshot has no optimizer state"
                        )
                    optimizer = stage_trainer_rank_optimizer_state_snapshot(
                        self.runtime,
                        trainer_rank_optimizer_state,
                        generation_id=generation.generation_id,
                        step=generation.policy_step,
                        stager=stager,
                    )
                else:
                    optimizer = stage_optimizer_state_snapshot(
                        self.runtime,
                        generation_id=generation.generation_id,
                        step=generation.policy_step,
                        stager=stager,
                    )
            else:
                optimizer = None
            if lora is not None:
                self.runtime.optimizer_snapshot_barrier.register(lora)
            if optimizer is not None:
                self.runtime.optimizer_snapshot_barrier.register(optimizer)
            optimizer_launch_s = time.perf_counter() - optimizer_started
            resolved = self._resolution_pool.submit(
                self._resolve_generation, lora, optimizer
            )
            entry = _CachedGeneration(
                run_id=run_id,
                generation=generation,
                stager=stager,
                resolved=resolved,
                has_optimizer=optimizer is not None,
            )
            with self._lock:
                if generation.generation_id in self._cache:
                    raise RuntimeError(
                        f"generation snapshot already exists: {generation.generation_id}"
                    )
                self._cache[generation.generation_id] = entry
                self._latest_by_run[run_id] = generation.generation_id
            resolved.add_done_callback(
                lambda done, cached=entry: self._snapshot_resolved(cached, done)
            )
        except BaseException:
            self._release_slot(stager)
            raise
        metrics = {
            "snapshot_pool_wait_s": wait_s,
            "snapshot_pool_in_use": float(in_flight),
            "snapshot_pool_pressure": in_flight / self.capacity,
            "snapshot_lora_launch_s": lora_launch_s,
            "snapshot_optimizer_launch_s": optimizer_launch_s,
            "snapshot_launch_s": time.perf_counter() - prepare_started,
        }
        metrics.update(lora_timings.metrics())
        return metrics

    def ensure_generation(
        self,
        *,
        run_id: str,
        generation: TrainerGeneration,
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
        slot_ref: "LoRASlotRef | None" = None,
        trainer_rank_optimizer_state: "TrainerRankOptimizerState | None" = None,
        snapshot_optimizer: bool,
    ) -> dict[str, float]:
        with self._lock:
            entry = self._cache.get(generation.generation_id)
            reusable = (
                entry is not None
                and not entry.retired
                and entry.run_id == run_id
                and entry.generation == generation
            )
        if not reusable:
            return self.stage(
                run_id=run_id,
                generation=generation,
                adapter_dtypes=adapter_dtypes,
                adapter_config=adapter_config,
                slot_ref=slot_ref,
                trainer_rank_optimizer_state=trainer_rank_optimizer_state,
                snapshot_optimizer=snapshot_optimizer,
            )
        assert entry is not None
        if not snapshot_optimizer or entry.has_optimizer:
            return {}

        from art.megatron.optimizer_state import (
            stage_optimizer_state_snapshot,
            stage_trainer_rank_optimizer_state_snapshot,
        )

        started = time.perf_counter()
        if slot_ref is None:
            if trainer_rank_optimizer_state is not None:
                raise RuntimeError("static optimizer upgrade received slot state")
            optimizer = stage_optimizer_state_snapshot(
                self.runtime,
                generation_id=generation.generation_id,
                step=generation.policy_step,
                stager=entry.stager,
            )
        else:
            if trainer_rank_optimizer_state is None:
                raise RuntimeError("dynamic LoRA snapshot has no optimizer state")
            optimizer = stage_trainer_rank_optimizer_state_snapshot(
                self.runtime,
                trainer_rank_optimizer_state,
                generation_id=generation.generation_id,
                step=generation.policy_step,
                stager=entry.stager,
            )
        self.runtime.optimizer_snapshot_barrier.register(optimizer)
        resolved = self._resolution_pool.submit(optimizer.resolve)
        with self._lock:
            if entry.retired or entry.released or entry.optimizer_upgrade is not None:
                raise RuntimeError(
                    "generation changed while staging optimizer snapshot"
                )
            entry.optimizer_upgrade = resolved
            entry.has_optimizer = True
        resolved.add_done_callback(
            lambda done, cached=entry: self._optimizer_resolved(cached, done)
        )
        elapsed = time.perf_counter() - started
        return {
            "snapshot_optimizer_launch_s": elapsed,
            "snapshot_launch_s": elapsed,
        }

    def attach_resident_optimizer(
        self,
        *,
        generation: TrainerGeneration,
        source: Any,
        l2: Future[Any],
        tensor_offset: int,
    ) -> dict[str, float]:
        started = time.perf_counter()
        with self._lock:
            entry = self._cache.get(generation.generation_id)
            if entry is None or entry.retired or entry.generation != generation:
                raise RuntimeError("optimizer residency has no staged generation")
            if entry.has_optimizer or entry.optimizer_upgrade is not None:
                raise RuntimeError("generation already has an optimizer snapshot")
            resolved = self._resolution_pool.submit(
                self._resolve_resident_optimizer,
                l2,
                source,
                tensor_offset,
                generation,
            )
            entry.optimizer_upgrade = resolved
            entry.has_optimizer = True
        resolved.add_done_callback(
            lambda done, cached=entry: self._optimizer_resolved(cached, done)
        )
        elapsed = time.perf_counter() - started
        return {
            "snapshot_optimizer_attach_s": elapsed,
        }

    def _resolve_resident_optimizer(
        self,
        l2: Future[Any],
        source: Any,
        tensor_offset: int,
        generation: TrainerGeneration,
    ) -> Any:
        from art.megatron.optimizer_state import (
            trainer_rank_optimizer_snapshot_from_cpu,
        )

        tensors = l2.result().tensors()[tensor_offset:]
        state = source.bind(tensors)
        return trainer_rank_optimizer_snapshot_from_cpu(
            self.runtime,
            state,
            generation_id=generation.generation_id,
            step=generation.policy_step,
        )

    def submit(
        self,
        *,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        existing_adapter: OptimizerAdapter | None = None,
        publication_targets: tuple[Any, ...],
        adapter_object_target: BinaryObjectTarget | None,
        save_optimizer: bool,
        sink: EventSink,
    ) -> dict[str, float]:
        self.raise_if_failed()
        with self._lock:
            entry = self._cache.get(generation.generation_id)
        if entry is None or entry.retired or entry.generation != generation:
            raise RuntimeError(
                f"learner generation is not staged: {generation.generation_id}"
            )
        started = time.perf_counter()
        transport = self._transport_pool.submit(
            self._transfer_cached_snapshot,
            entry,
            publication_targets,
            adapter_object_target,
            generation,
            started,
        )
        durability = self._durability_pool.submit(
            self._persist_cached_snapshot,
            entry,
            generation,
            optimizer_state_path,
            staging_adapter_path,
            existing_adapter if int(self.runtime.rank) == 0 else None,
            save_optimizer,
            started,
        )
        persistence = self._completion_pool.submit(
            self._complete_publication, transport, durability, started
        )
        with self._lock:
            entry.consumers.append(persistence)
        persistence.add_done_callback(
            lambda done: self._completed(
                done,
                entry=entry,
                sink=sink,
                generation=generation,
                submitted_at=started,
            )
        )
        return {
            "snapshot_attach_s": time.perf_counter() - started,
        }

    def stage_and_submit(
        self,
        *,
        run_id: str,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str,
        publication_targets: tuple[Any, ...],
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
        save_optimizer: bool,
        sink: EventSink,
    ) -> dict[str, float]:
        return {
            **self.stage(
                run_id=run_id,
                generation=generation,
                adapter_dtypes=adapter_dtypes,
                adapter_config=adapter_config,
                snapshot_optimizer=save_optimizer,
            ),
            **self.submit(
                generation=generation,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=staging_adapter_path,
                publication_targets=publication_targets,
                adapter_object_target=None,
                save_optimizer=save_optimizer,
                sink=sink,
            ),
        }

    def has_generation(
        self,
        generation: TrainerGeneration,
        *,
        require_optimizer: bool = False,
    ) -> bool:
        with self._lock:
            entry = self._cache.get(generation.generation_id)
            return (
                entry is not None
                and not entry.retired
                and entry.generation == generation
                and (not require_optimizer or entry.has_optimizer)
            )

    def retire_run(self, run_id: str) -> None:
        self._retire_previous(run_id)

    def _acquire_slot(self) -> tuple[float, int, PinnedCpuSnapshotStager]:
        self.raise_if_failed()
        started = time.perf_counter()
        if not self._slots.acquire(blocking=False):
            self._evict_for_capacity()
            self._slots.acquire()
        wait_s = time.perf_counter() - started
        with self._lock:
            stager = self._available_stagers.pop()
            stager.reset()
            self._in_flight += 1
            return wait_s, self._in_flight, stager

    def _evict_for_capacity(self) -> None:
        with self._lock:
            entries = tuple(
                entry for entry in self._cache.values() if not entry.released
            )
            ready = tuple(
                entry
                for entry in entries
                if entry.resolved.done()
                and (entry.optimizer_upgrade is None or entry.optimizer_upgrade.done())
                and all(consumer.done() for consumer in entry.consumers)
            )
            if not entries:
                raise RuntimeError("snapshot pool is full without a cached generation")
            entry = (ready or entries)[0]
            entry.retired = True
            consumers = tuple(entry.consumers)
        try:
            entry.resolved.result()
            if entry.optimizer_upgrade is not None:
                entry.optimizer_upgrade.result()
            for consumer in consumers:
                consumer.result()
        finally:
            self._maybe_release(entry)

    def _retire_previous(self, run_id: str) -> None:
        with self._lock:
            generation_id = self._latest_by_run.pop(run_id, None)
            entry = None if generation_id is None else self._cache[generation_id]
            if entry is not None:
                entry.retired = True
        if entry is not None:
            self._maybe_release(entry)

    def _snapshot_resolved(
        self,
        entry: _CachedGeneration,
        resolved: Future[_ResolvedGeneration],
    ) -> None:
        if not resolved.cancelled() and (error := resolved.exception()) is not None:
            with self._lock:
                self._failures.append(error)
                entry.retired = True
        self._maybe_release(entry)

    def _optimizer_resolved(
        self,
        entry: _CachedGeneration,
        resolved: Future[Any],
    ) -> None:
        if not resolved.cancelled() and (error := resolved.exception()) is not None:
            with self._lock:
                self._failures.append(error)
                entry.retired = True
        self._maybe_release(entry)

    def _maybe_release(self, entry: _CachedGeneration) -> None:
        with self._lock:
            if (
                not entry.retired
                or entry.released
                or not entry.resolved.done()
                or (
                    entry.optimizer_upgrade is not None
                    and not entry.optimizer_upgrade.done()
                )
                or any(not consumer.done() for consumer in entry.consumers)
            ):
                return
            entry.released = True
            self._cache.pop(entry.generation.generation_id, None)
            if self._latest_by_run.get(entry.run_id) == entry.generation.generation_id:
                self._latest_by_run.pop(entry.run_id)
            for object_id in entry.object_ids:
                self._object_publications.pop(object_id, None)
        self._release_slot(entry.stager)

    def _resolve_generation(self, lora: Any, optimizer: Any) -> _ResolvedGeneration:
        lora = None if lora is None else lora.resolve()
        optimizer = None if optimizer is None else optimizer.resolve()
        prepared_tensors = None
        if lora is not None:
            layout_key = tuple(
                (key, tuple(tensor.shape), str(tensor.dtype))
                for key, tensor in sorted(lora.tensors.items())
            )
            with self._lock:
                layout = self._lora_layouts.get(layout_key)
                if layout is None:
                    layout = self._lora_layouts[layout_key] = SafetensorsLayout(
                        lora.tensors
                    )
            prepared_tensors = layout.bind(lora.tensors)
        return _ResolvedGeneration(
            lora=lora,
            optimizer=optimizer,
            prepared_tensors=prepared_tensors,
        )

    def _transfer_cached_snapshot(
        self,
        entry: _CachedGeneration,
        publication_targets: tuple[Any, ...],
        object_target: BinaryObjectTarget | None,
        generation: TrainerGeneration,
        submitted_at: float,
    ) -> _SnapshotTransport:
        started = time.perf_counter()
        if int(self.runtime.rank) != 0:
            return _SnapshotTransport(
                metrics={"time/snapshot_transport_queue_s": started - submitted_at}
            )
        snapshot = entry.resolved.result()
        ready = time.perf_counter()
        if snapshot.lora is None or snapshot.prepared_tensors is None:
            raise RuntimeError("rank zero has no LoRA snapshot to transfer")
        adapter = None
        if object_target is not None:
            adapter = self._publish_lora_object_once(
                entry,
                object_target,
                generation,
                snapshot.lora,
                snapshot.prepared_tensors,
            )
        elif publication_targets:
            self._transfer_lora_snapshot(
                snapshot.lora,
                publication_targets,
                prepared_tensors=snapshot.prepared_tensors,
            )
        return _SnapshotTransport(
            adapter=adapter,
            metrics={
                "time/snapshot_transport_queue_s": started - submitted_at,
                "time/snapshot_transport_wait_s": ready - started,
                "time/snapshot_transport_s": time.perf_counter() - ready,
            },
        )

    def _persist_cached_snapshot(
        self,
        entry: _CachedGeneration,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        adapter: OptimizerAdapter | None,
        save_optimizer: bool,
        submitted_at: float,
    ) -> _RankSnapshotPersistence:
        started = time.perf_counter()
        snapshot = entry.resolved.result()
        optimizer = snapshot.optimizer
        if save_optimizer and optimizer is None and entry.optimizer_upgrade is not None:
            optimizer = entry.optimizer_upgrade.result()
        ready = time.perf_counter()
        if save_optimizer and optimizer is None:
            raise RuntimeError("optimizer persistence requires an optimizer snapshot")
        result = self._persist_generation(
            generation=generation,
            optimizer_state_path=optimizer_state_path,
            staging_adapter_path=staging_adapter_path,
            lora=(
                snapshot.lora
                if adapter is None and staging_adapter_path is not None
                else None
            ),
            adapter=adapter,
            optimizer=optimizer if save_optimizer else None,
            prepared_tensors=(
                snapshot.prepared_tensors
                if adapter is None and staging_adapter_path is not None
                else None
            ),
        )
        return result.model_copy(
            update={
                "metrics": {
                    "time/snapshot_persistence_queue_s": started - submitted_at,
                    "time/snapshot_persistence_wait_s": ready - started,
                    "time/snapshot_persistence_s": time.perf_counter() - ready,
                }
            }
        )

    @staticmethod
    def _complete_publication(
        transport: Future[_SnapshotTransport],
        durability: Future[_RankSnapshotPersistence],
        submitted_at: float,
    ) -> TrainerRankPublication:
        started = time.perf_counter()
        failures: list[BaseException] = []
        for future in (transport, durability):
            try:
                future.result()
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "adapter persistence and transport failed", failures
            )
        transferred = transport.result()
        persisted = durability.result()
        adapter = persisted.adapter or transferred.adapter
        if persisted.rank == 0 and adapter is None:
            raise RuntimeError("rank zero snapshot requires an adapter output")
        if persisted.rank != 0 and (
            adapter is not None or transferred.adapter is not None
        ):
            raise RuntimeError("nonzero rank unexpectedly published an adapter")
        metrics = {**transferred.metrics, **persisted.metrics}
        completed_at = time.perf_counter()
        metrics.update(
            {
                "time/snapshot_completion_queue_s": started - submitted_at,
                "time/snapshot_completion_wait_s": completed_at - started,
                "time/snapshot_rank_ready_s": completed_at - submitted_at,
                "time/snapshot_publication_s": completed_at - submitted_at,
            }
        )
        return TrainerRankPublication(
            generation=persisted.generation,
            rank=persisted.rank,
            adapter=adapter,
            transport_adapter=transferred.adapter,
            shard=persisted.shard,
            runtime_sha256=persisted.runtime_sha256,
            topology=persisted.topology,
            saves_optimizer=persisted.saves_optimizer,
            metrics=metrics,
        )

    def _transfer_lora_snapshot(
        self,
        lora: Any,
        targets: tuple[Any, ...],
        *,
        prepared_tensors: PreparedSafetensors,
    ) -> None:
        from art.distributed.adapter_transport import AdapterSnapshotSender

        with self._transport_sender_lock:
            if self._transport_sender is None:
                self._transport_sender = AdapterSnapshotSender()
            self._transport_sender.send(
                lora,
                targets,
                prepared_tensors=prepared_tensors,
            )

    def _publish_lora_object_once(
        self,
        entry: _CachedGeneration,
        target: BinaryObjectTarget,
        generation: TrainerGeneration,
        lora: Any,
        prepared_tensors: PreparedSafetensors,
    ) -> OptimizerAdapter:
        with self._lock:
            publication = self._object_publications.get(target.object_id)
            owner = publication is None
            if publication is None:
                publication = Future()
                self._object_publications[target.object_id] = publication
            entry.object_ids.add(target.object_id)
        if not owner:
            return publication.result()
        try:
            adapter = self._publish_lora_object(
                target, generation, lora, prepared_tensors
            )
        except BaseException as error:
            publication.set_exception(error)
            raise
        publication.set_result(adapter)
        return adapter

    def _publish_lora_object(
        self,
        target: BinaryObjectTarget,
        generation: TrainerGeneration,
        lora: Any,
        prepared_tensors: PreparedSafetensors,
    ) -> OptimizerAdapter:
        expected_metadata = {
            "training_session_id": generation.training_session_id,
            "generation_id": generation.generation_id,
            "policy_step": str(generation.policy_step),
        }
        if any(
            target.metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise RuntimeError(
                "adapter object target identifies another learner generation"
            )
        with self._lock:
            if self._object_store is None:
                self._object_store = S3BinaryObjectStore(target.store)
            elif self._object_store.config != target.store:
                raise RuntimeError("generation publisher cannot mix object stores")
            store = self._object_store
        config = encode_adapter_config(
            {
                **lora.adapter_config,
                ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM,
            }
        )
        ref = store.publish(
            target,
            {
                "adapter_model.safetensors": tuple(
                    memoryview(chunk.numpy()).cast("B")
                    for chunk in prepared_tensors.chunks
                ),
                "adapter_config.json": (memoryview(config),),
            },
        )
        files = {file.relative_path: file for file in ref.files}
        expected_files = ("adapter_config.json", "adapter_model.safetensors")
        if set(files) != set(expected_files):
            raise RuntimeError("adapter object manifest has unexpected files")
        return OptimizerAdapter(
            identity=ref.manifest_uri,
            training_session_id=generation.training_session_id,
            step=generation.policy_step,
            generation_id=generation.generation_id,
            files=tuple(
                CheckpointFile(name=name, size_bytes=files[name].byte_count)
                for name in expected_files
            ),
        )

    def _persist_generation(
        self,
        *,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        lora: Any,
        adapter: "OptimizerAdapter | None",
        optimizer: Any,
        prepared_tensors: PreparedSafetensors | None,
    ) -> _RankSnapshotPersistence:
        from art.megatron.optimizer_state import (
            adapter_publication_transaction,
            publish_adapter_checkpoint,
            write_optimizer_snapshot_shard,
        )
        from art.megatron.weights.lora_publish import save_vllm_lora_snapshot

        rank = int(self.runtime.rank)
        if rank == 0:
            if lora is not None:
                if staging_adapter_path is None or adapter is not None:
                    raise RuntimeError("new adapter publication is inconsistent")
                with adapter_publication_transaction(
                    staging_adapter_path,
                    step=generation.policy_step,
                    training_session_id=generation.training_session_id,
                    generation_id=generation.generation_id,
                ) as (staging, existing):
                    adapter = existing
                    if adapter is None:
                        save_vllm_lora_snapshot(
                            lora,
                            str(staging),
                            prepared_tensors=prepared_tensors,
                        )
                        adapter = publish_adapter_checkpoint(
                            staging,
                            step=generation.policy_step,
                            training_session_id=generation.training_session_id,
                            generation_id=generation.generation_id,
                        )
        shard = (
            write_optimizer_snapshot_shard(
                optimizer,
                optimizer_state_path=optimizer_state_path,
            )
            if optimizer is not None
            else None
        )
        return _RankSnapshotPersistence(
            generation=generation,
            rank=rank,
            adapter=adapter,
            shard=shard,
            runtime_sha256=None if optimizer is None else optimizer.runtime_sha256,
            topology=None if optimizer is None else optimizer.topology,
            saves_optimizer=optimizer is not None,
        )

    def _completed(
        self,
        future: Future[TrainerRankPublication],
        *,
        entry: _CachedGeneration,
        sink: EventSink,
        generation: TrainerGeneration,
        submitted_at: float,
    ) -> None:
        callback_started = time.perf_counter()
        try:
            record = future.result()
            callback_delay_s = max(
                0.0,
                callback_started
                - submitted_at
                - record.metrics["time/snapshot_rank_ready_s"],
            )
            event = TrainerPublicationSucceeded(
                record=record.model_copy(
                    update={
                        "metrics": {
                            **record.metrics,
                            "time/snapshot_callback_delay_s": callback_delay_s,
                        }
                    }
                )
            )
        except BaseException as error:
            self._failed(error, entry=entry, sink=sink, generation=generation)
            return
        try:
            sink.publication(event)
        except BaseException as error:
            with self._lock:
                self._failures.append(error)
        finally:
            self._maybe_release(entry)

    def _failed(
        self,
        error: BaseException,
        *,
        entry: _CachedGeneration,
        sink: EventSink,
        generation: TrainerGeneration,
    ) -> None:
        self._report_failure(
            error,
            entry=entry,
            sink=sink,
            generation=generation,
            remember=True,
        )

    def _report_failure(
        self,
        error: BaseException,
        *,
        entry: _CachedGeneration,
        sink: EventSink,
        generation: TrainerGeneration,
        remember: bool,
    ) -> None:
        if remember:
            with self._lock:
                self._failures.append(error)
        event = TrainerPublicationFailed(
            generation_id=generation.generation_id,
            rank=int(self.runtime.rank),
            error_type=type(error).__name__,
            message=str(error) or type(error).__name__,
        )
        try:
            sink.publication(event)
        except BaseException as sink_error:
            with self._lock:
                self._failures.append(sink_error)
        finally:
            self._maybe_release(entry)

    def _release_slot(self, stager: PinnedCpuSnapshotStager) -> None:
        with self._lock:
            self._available_stagers.append(stager)
            self._in_flight -= 1
        self._slots.release()

    def raise_if_failed(self) -> None:
        with self._lock:
            failures = tuple(self._failures)
        if failures:
            raise BaseExceptionGroup("trainer generation publication failed", failures)

    def close(self) -> None:
        with self._lock:
            entries = tuple(self._cache.values())
            for entry in entries:
                entry.retired = True
        for entry in entries:
            self._maybe_release(entry)
        self._resolution_pool.shutdown(wait=True)
        self._transport_pool.shutdown(wait=True)
        self._durability_pool.shutdown(wait=True)
        self._completion_pool.shutdown(wait=True)
        for entry in entries:
            self._maybe_release(entry)
        if self._transport_sender is not None:
            self._transport_sender.close()
            self._transport_sender = None
        if self._object_store is not None:
            self._object_store.close()
            self._object_store = None
        with self._lock:
            in_flight = self._in_flight
        if in_flight:
            raise RuntimeError(f"publication close retained {in_flight} snapshots")
        self.raise_if_failed()
