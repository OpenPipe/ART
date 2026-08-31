from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock
import time
from typing import TYPE_CHECKING, Any, Literal

import torch

from art.distributed.adapter_transport import AdapterTransferTarget
from art.training.contracts import TokenLogprobs
from art.utils.safetensors import PreparedSafetensors, SafetensorsLayout

from ..tensor_snapshot import PendingCpuSnapshot, PinnedCpuSnapshotStager
from ..training.finalize_grads import (
    finalize_accumulated_model_grads,
    flush_param_grads_to_main_grads,
)
from ..training.gradient_accumulator import GradientAccumulator
from .data_plane import InMemoryPackedBatch, SFTBatchData, validate_packed_batch
from .device_usage import measure_cuda_call
from .publication import (
    TrainerPublicationEvent,
    TrainerPublicationFailed,
    TrainerPublicationSucceeded,
    TrainerRankPublication,
)
from .residency import ResidencyKey
from .run_residency import RunResidencyConfig, RunResidencyManager
from .run_slots import MegatronRunSlots, OptimizerConfig
from .specs import (
    CommandPublicationSpec,
    ForwardBackwardJobSpec,
    ForwardJobSpec,
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
    TrainingRunSpec,
    TrainJobSpec,
)
from .trainer_run import EventSink

if TYPE_CHECKING:
    from art.megatron.lora import LoRASlotRef
    from art.megatron.optimizer_state import OptimizerAdapter
    from art.megatron.runtime.numerical_capture import (
        ForwardBackwardNumericalRankReceipt,
    )
    from art.megatron.runtime.portable_snapshot import (
        PortableSnapshotArchive,
        PortableSnapshotRankReceipt,
        PortableSnapshotReadReceipt,
        PortableSnapshotSink,
        PortableSnapshotSource,
    )
    from art.megatron.weights.external_lora_publish import (
        ExternalLoraPlan,
        ExternalLoraPublication,
        ExternalLoraPublicationSink,
        ExternalLoraTarget,
    )


def _command_token_logprobs(
    batch: InMemoryPackedBatch,
    outputs: tuple[torch.Tensor, ...],
) -> tuple[dict[str, Any], ...]:
    decoder = _command_token_logprob_decoder(batch)
    return decoder(tuple(value.detach().to(device="cpu") for value in outputs))


def _command_token_logprob_decoder(
    batch: InMemoryPackedBatch,
) -> Callable[[tuple[torch.Tensor, ...]], tuple[dict[str, Any], ...]]:
    if batch.ref.training_kind != "tokenized":
        return _sft_token_logprobs
    output_map = batch.ref.tokenized_output_map
    if output_map is None:
        raise RuntimeError("tokenized batch has no output map")
    target_tokens = batch.tensors.get("target_tokens")
    if target_tokens is None:
        raise RuntimeError("tokenized batch has no target tensor")
    candidate_capacity = int(target_tokens.shape[2])
    expected_rows = batch.ref.num_sequences * batch.ref.sequence_length

    def decode(outputs: tuple[torch.Tensor, ...]) -> tuple[dict[str, Any], ...]:
        return _decode_tokenized_logprobs(
            outputs,
            candidate_capacity=candidate_capacity,
            expected_rows=expected_rows,
            packed_positions=output_map.packed_positions,
            candidate_counts=output_map.candidate_counts,
        )

    return decode


def _decode_tokenized_logprobs(
    outputs: tuple[torch.Tensor, ...],
    *,
    candidate_capacity: int,
    expected_rows: int,
    packed_positions: tuple[tuple[int, ...], ...],
    candidate_counts: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    physical = torch.cat(
        [values.reshape(-1, candidate_capacity) for values in outputs], dim=0
    )
    if int(physical.shape[0]) != expected_rows:
        raise RuntimeError(
            "tokenized command did not return every physical packed row: "
            f"returned={physical.shape[0]}, expected={expected_rows}"
        )
    logical = []
    for positions, candidates in zip(packed_positions, candidate_counts, strict=True):
        values = physical[list(positions), :candidates]
        logical.append(
            TokenLogprobs.from_values(
                values.flatten().tolist(), shape=tuple(values.shape)
            ).model_dump(mode="python")
        )
    return tuple(logical)


def _sft_token_logprobs(
    outputs: tuple[torch.Tensor, ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        TokenLogprobs.from_values(
            values.flatten().tolist(), shape=tuple(values.shape)
        ).model_dump(mode="python")
        for values in outputs
    )


@dataclass(slots=True)
class CommandResultLaunch:
    """GPU-complete command whose bounded host result is still materializing."""

    result: dict[str, Any]
    pending: PendingCpuSnapshot[tuple[torch.Tensor, ...]] | None = None
    decoder: Callable[[tuple[torch.Tensor, ...]], tuple[dict[str, Any], ...]] | None = (
        None
    )
    _stager: PinnedCpuSnapshotStager | None = None

    def materialize(self) -> dict[str, Any]:
        if self.pending is not None:
            outputs = self.pending.resolve()
            if self.decoder is None:
                raise RuntimeError("deferred command result has no decoder")
            self.result["token_logprobs"] = self.decoder(outputs)
            self.pending = None
            self.decoder = None
            self._stager = None
        return self.result


def _defer_command_result(
    result: dict[str, Any],
    outputs: tuple[torch.Tensor, ...],
    decoder: Callable[[tuple[torch.Tensor, ...]], tuple[dict[str, Any], ...]] | None,
) -> CommandResultLaunch:
    if not outputs:
        result["token_logprobs"] = ()
        return CommandResultLaunch(result=result)
    if decoder is None:
        raise RuntimeError("deferred command result has no decoder")
    stager = PinnedCpuSnapshotStager()
    builder = stager.begin()
    pending = builder.finish(tuple(builder.stage(value) for value in outputs))
    return CommandResultLaunch(
        result=result,
        pending=pending,
        decoder=decoder,
        _stager=stager,
    )


class MegatronTrainJobExecutor:
    """Thin adapter around the warm runtime's in-memory job entrypoint."""

    def __init__(
        self, runtime: Any, *, accumulator_l1_budget_bytes: int = 16 * 1024**3
    ) -> None:
        self.runtime = runtime
        self._publisher = _GenerationPublisher(
            runtime,
            capacity=int(runtime.snapshot_pool_capacity),
        )
        self._gradients: dict[str, GradientAccumulator] = {}
        self._gradient_parent_versions: dict[str, int] = {}
        self._accumulator_l1_budget_bytes = accumulator_l1_budget_bytes
        self._python_gc_stabilized = False
        self._closed = False

    @property
    def publisher(self) -> "_GenerationPublisher":
        return self._publisher

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
            snapshot_sink=lambda *args: self._publisher.submit(*args, sink=sink),
            cancelled=cancelled,
        )
        metrics.update(self._stabilize_python_gc())
        timing.previous_job_complete_s = time.monotonic()
        return metrics

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
        parent = self._gradient_parent_versions.get(job.run_id)
        if parent not in {
            None,
            job.expected_learner_version,
        }:
            raise RuntimeError(
                "F/B parent does not match the open gradient accumulator"
            )
        self.runtime.inter_forward_backward_timing.current_job_start_s = (
            time.monotonic()
        )
        from art.megatron.train import execute_megatron_rl_forward_backward_job

        gradients = self._gradients.setdefault(
            job.run_id,
            GradientAccumulator(
                self.runtime.model,
                flush_gradients=flush_param_grads_to_main_grads,
            ),
        )
        result = execute_megatron_rl_forward_backward_job(
            self.runtime,
            job,
            batch.tensors,
            gradient_accumulator=gradients,
            cancelled=cancelled,
        )
        gradients.stash_resident()
        self._enforce_accumulator_budget()
        self._gradient_parent_versions[job.run_id] = job.expected_learner_version
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "loss_bearing_token_count": job.expected_global_loss_bearing_tokens,
            "completed_gradient_steps": result.completed_gradient_steps,
            "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
            "executed_token_equivalents": result.executed_token_equivalents,
            "gpu_service_ns": result.gpu_service_ns,
            "token_logprobs": _command_token_logprobs(batch, result.new_logprobs)
            if job.return_token_logprobs
            else (),
            "metrics": result.metrics,
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
        from art.megatron.train import execute_megatron_rl_forward_job

        result = execute_megatron_rl_forward_job(
            self.runtime,
            job,
            batch.tensors,
            cancelled=cancelled,
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
            "executed_token_equivalents": result.executed_token_equivalents,
            "gpu_service_ns": result.gpu_service_ns,
            "token_logprobs": _command_token_logprobs(batch, result.new_logprobs)
            if job.return_token_logprobs
            else (),
            "metrics": result.metrics,
        }

    def execute_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        if (
            self._gradient_parent_versions.get(job.run_id)
            != job.expected_learner_version
        ):
            raise RuntimeError("optimizer parent does not match accumulated gradients")
        gradients = self._gradients[job.run_id]
        runtime = self.runtime
        from art.megatron.train import (
            _prepare_rl_training_state,
            run_megatron_optimizer_step,
        )

        _prepare_rl_training_state(runtime, job)
        if runtime.optimizer is None:
            raise RuntimeError("trainer has no resident optimizer")
        gradients.seal(job.contributing_forward_backward_operation_ids)
        started = time.perf_counter()

        def optimizer_step() -> Any:
            accumulated = gradients.prepare_optimizer()
            finalize_accumulated_model_grads(runtime.model, accumulated)
            for group in runtime.optimizer.param_groups:
                group["betas"] = (job.optimizer.beta1, job.optimizer.beta2)
                group["eps"] = job.optimizer.eps
                group["weight_decay"] = job.optimizer.weight_decay
            runtime.optimizer.config.adam_beta1 = job.optimizer.beta1
            runtime.optimizer.config.adam_beta2 = job.optimizer.beta2
            runtime.optimizer.config.adam_eps = job.optimizer.eps
            runtime.optimizer.config.weight_decay = job.optimizer.weight_decay
            runtime.optimizer.config.clip_grad = job.optimizer.grad_clip_norm
            return run_megatron_optimizer_step(
                optimizer=runtime.optimizer,
                learning_rate=job.optimizer.learning_rate,
                model_support_handler=runtime.model_support_handler,
                model_chunks=runtime.model,
                before_step=runtime.optimizer_snapshot_barrier.wait_before_mutation,
            )

        result, gpu_service_ns = measure_cuda_call(optimizer_step)
        if not result.update_successful or not math.isfinite(result.grad_norm):
            raise RuntimeError(
                "Megatron optimizer rejected the update: "
                f"update_successful={result.update_successful}, "
                f"grad_norm={result.grad_norm}"
            )
        consumed = gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_run_id = job.run_id
        runtime.resident_policy_step = job.learner_version
        runtime.resident_generation_id = job.generation.generation_id
        runtime.optimizer_state_loaded = True
        self._gradients.pop(job.run_id)
        self._gradient_parent_versions.pop(job.run_id)
        runtime.inter_forward_backward_timing.previous_job_complete_s = time.monotonic()
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": consumed,
            "gpu_service_ns": gpu_service_ns,
            "metrics": {
                "loss/learning_rate": job.optimizer.learning_rate,
                "loss/grad_norm": float(result.grad_norm),
                "optimizer/update_successful": 1.0,
                "optimizer/num_zeros_in_grad": float(result.num_zeros_in_grad or 0),
                "time/optimizer_step_s": time.perf_counter() - started,
            },
        }

    def publish_split_generation(
        self,
        job: TrainJobSpec,
        *,
        sink: EventSink,
    ) -> dict[str, float]:
        """Launch the ordinary fast snapshot after a split optimizer command."""

        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        self._publisher.raise_if_failed()
        runtime = self.runtime
        if (
            runtime.resident_run_id != job.run_id
            or runtime.resident_training_session_id != job.training_session_id
            or runtime.resident_policy_step != job.learner_version
            or runtime.resident_generation_id != job.output_generation_id
            or runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError(
                "split publication does not match the resident learner generation"
            )
        from art.megatron.train import _should_snapshot_optimizer

        return self._publisher.submit(
            job,
            runtime.adapter_export_dtypes,
            runtime.adapter_export_config,
            _should_snapshot_optimizer(
                runtime,
                step=job.learner_version,
                optimizer_save_interval=job.config.optimizer_save_interval,
                final_training_step=job.config.final_training_step,
            ),
            sink=sink,
        )

    def execute_sft(
        self,
        job: SFTJobSpec,
        batches: tuple[SFTBatchData, ...],
        sink: EventSink,
        cancelled: Event,
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
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
            snapshot_sink=lambda *args: self._publisher.submit(*args, sink=sink),
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

    def publish_external_lora(
        self,
        target: "ExternalLoraTarget",
        sink: "ExternalLoraPublicationSink",
        *,
        source_topology: str,
    ) -> tuple[
        "ExternalLoraPlan",
        Future["ExternalLoraPublication | None"],
        dict[str, float],
    ]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        runtime = self.runtime
        if (
            runtime.resident_run_id != target.run_id
            or runtime.resident_training_session_id != target.training_session_id
            or runtime.resident_policy_step != target.policy_step
            or runtime.resident_generation_id != target.generation_id
            or runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError(
                "resident trainer state does not match external LoRA target"
            )
        return self._publisher.submit_external(
            target,
            sink,
            source_topology=source_topology,
            adapter_dtypes=runtime.adapter_export_dtypes,
            adapter_config=runtime.adapter_export_config,
        )

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
            self.discard_open_gradients()
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

    def discard_open_gradients(self) -> None:
        for gradients in self._gradients.values():
            gradients.discard()
        self._gradients.clear()
        self._gradient_parent_versions.clear()

    def discard_run_gradients(self, run_id: str) -> tuple[str, ...]:
        gradients = self._gradients.get(run_id)
        if gradients is None:
            return ()
        contributions = gradients.contribution_ids
        gradients.discard()
        self._gradients.pop(run_id)
        self._gradient_parent_versions.pop(run_id, None)
        return contributions

    def run_gradient_ids(self, run_id: str) -> tuple[str, ...]:
        gradients = self._gradients.get(run_id)
        return () if gradients is None else gradients.contribution_ids

    @property
    def has_open_gradients(self) -> bool:
        return any(value.contribution_ids for value in self._gradients.values())

    def _require_no_open_gradients(self) -> None:
        if self.has_open_gradients:
            raise RuntimeError("operation cannot discard open gradient contributions")

    def _enforce_accumulator_budget(self) -> None:
        resident = sum(value.residency_nbytes for value in self._gradients.values())
        if resident > self._accumulator_l1_budget_bytes:
            raise RuntimeError(
                "gradient accumulators exceed the per-rank L1 budget: "
                f"{resident} > {self._accumulator_l1_budget_bytes}"
            )


@dataclass(slots=True)
class _ResidentCommandRun:
    spec: TrainingRunSpec
    learner_version: int
    gradients: Any
    adapter_config: dict[str, Any]
    portable_read: PortableSnapshotReadReceipt | None = None
    weights_key: ResidencyKey | None = None
    optimizer_key: ResidencyKey | None = None
    accumulator_key: ResidencyKey | None = None
    next_accumulator_revision: int = 1


@dataclass(slots=True)
class _PreparedPortableRun:
    receipt: PortableSnapshotReadReceipt
    staging_name: str
    adapter_config: dict[str, Any]
    weights: tuple[torch.nn.Parameter, ...]
    optimizer_tensors: tuple[torch.Tensor, ...]


@dataclass(slots=True)
class _PreparedRunCheckpoint:
    operation_id: str
    fingerprint: str
    run_id: str
    learner_version: int
    previous_learner_version: int
    previous_keys: tuple[ResidencyKey, ...]
    prepared: _PreparedPortableRun
    gradients: Any
    weights_key: ResidencyKey
    optimizer_key: ResidencyKey
    remaining_keys: set[ResidencyKey]
    staging_live: bool = True


class _CommandPublicationSink:
    def __init__(self) -> None:
        self.future: Future[TrainerRankPublication] = Future()

    def progress(
        self, *, step_index: int, num_steps: int, metrics: dict[str, float]
    ) -> None:
        del step_index, num_steps, metrics

    def adapter_ready(self, *, learner_version: int, adapter_path: str) -> None:
        del learner_version, adapter_path

    def publication(self, event: TrainerPublicationEvent) -> None:
        if self.future.done():
            raise RuntimeError("command publication settled twice")
        if isinstance(event, TrainerPublicationSucceeded):
            self.future.set_result(event.record)
            return
        if not isinstance(event, TrainerPublicationFailed):
            raise TypeError("command publication returned an unknown event")
        self.future.set_exception(
            RuntimeError(
                f"rank {event.rank} publication failed "
                f"({event.error_type}): {event.message}"
            )
        )


class MCoreRunSlotExecutor:
    """Execute independent exact-shape LoRAs on one warm MCore rank."""

    def __init__(
        self,
        runtime: Any,
        *,
        accumulator_l1_budget_bytes: int = 16 * 1024**3,
        run_residency_config: RunResidencyConfig,
        topology_fingerprint: str,
        portable_snapshot_source: PortableSnapshotSource | None = None,
        portable_snapshot_sink: PortableSnapshotSink | None = None,
        publisher: "_GenerationPublisher | None" = None,
    ) -> None:
        self.runtime = runtime
        self._slots = MegatronRunSlots(runtime)
        self._runs: dict[str, _ResidentCommandRun] = {}
        self._accumulator_l1_budget_bytes = accumulator_l1_budget_bytes
        self._topology_fingerprint = topology_fingerprint
        device = next(runtime.model[0].parameters()).device
        self._residency = RunResidencyManager(
            run_residency_config.model_copy(update={"device": str(device)}),
            snapshot_barrier=runtime.optimizer_snapshot_barrier,
        )
        self._residency_admission_lock = Lock()
        self._residency_admissions: dict[str, tuple[ResidencyKey, ...]] = {}
        self._checkpoint_hydrations: dict[str, _PreparedRunCheckpoint] = {}
        self._portable_snapshot_source = portable_snapshot_source
        self._portable_snapshot_sink = portable_snapshot_sink
        self._publisher = publisher or _GenerationPublisher(
            runtime, capacity=int(runtime.snapshot_pool_capacity)
        )
        self._owns_publisher = publisher is None
        self._closed = False

    def register_run(self, spec: TrainingRunSpec) -> PortableSnapshotReadReceipt | None:
        if self._closed:
            raise RuntimeError("Megatron run slot executor is closed")
        prior = self._runs.get(spec.run_id)
        if prior is not None:
            if prior.spec != spec:
                raise RuntimeError("run_id was reused with different trainer state")
            return prior.portable_read
        from art.megatron.model_support.lora_disk import (
            load_adapter_config,
            training_target_modules,
        )
        from art.megatron.training.gradient_accumulator import (
            ParameterGradientAccumulator,
        )

        portable_read = None
        prepared_portable: _PreparedPortableRun | None = None
        registered_keys: tuple[ResidencyKey, ...] = ()
        installed = False
        try:
            if spec.initial_portable_snapshot is None:
                adapter_config = load_adapter_config(spec.initial_adapter_path)
                if int(adapter_config.get("r", 0)) != spec.lora_rank or set(
                    training_target_modules(adapter_config)
                ) != set(spec.lora_target_modules):
                    raise RuntimeError(
                        "resident adapter shape differs from run admission"
                    )
                self._slots.load_checkpoint(spec.run_id, spec.initial_adapter_path)
                installed = True
                parameters = self._slots.checkpoint_slot_parameters(spec.run_id)
                optimizer_tensors = self._slots.prepare_checkpoint_slot_optimizer(
                    spec.run_id, OptimizerConfig(learning_rate=0.0)
                )
            else:
                archive = spec.initial_portable_snapshot
                generation = archive.generation
                if (
                    generation.training_session_id != spec.training_session_id
                    or generation.policy_step != spec.initial_learner_version
                    or generation.generation_id != spec.initial_generation_id
                ):
                    raise RuntimeError("portable archive identifies another generation")
                prepared_portable = self._prepare_portable_run(
                    run_id=spec.run_id,
                    generation_id=generation.generation_id,
                    archive=archive,
                    expected_lora_rank=spec.lora_rank,
                    expected_lora_target_modules=spec.lora_target_modules,
                    restore_optimizer=True,
                )
                portable_read = prepared_portable.receipt
                adapter_config = prepared_portable.adapter_config
                parameters = prepared_portable.weights
                optimizer_tensors = prepared_portable.optimizer_tensors
            state = _ResidentCommandRun(
                spec=spec,
                learner_version=spec.initial_learner_version,
                gradients=ParameterGradientAccumulator(parameters),
                adapter_config=adapter_config,
                portable_read=portable_read,
            )
            generation_id = spec.initial_generation_id or (
                "initial-"
                + hashlib.sha256(spec.initial_adapter_path.encode()).hexdigest()
            )
            state.weights_key = self._residency_key(
                state, generation_id=generation_id, representation="weights"
            )
            state.optimizer_key = self._residency_key(
                state, generation_id=generation_id, representation="optimizer"
            )
            if prepared_portable is None:
                weights_l2 = self._residency.register_l1(state.weights_key, parameters)
                registered_keys = (state.weights_key,)
                optimizer_l2 = self._residency.register_l1(
                    state.optimizer_key, optimizer_tensors
                )
                registered_keys = (state.weights_key, state.optimizer_key)
                for future in (weights_l2, optimizer_l2):
                    future.result()
            else:
                self._register_portable_l2_working_set(
                    (
                        (state.weights_key, parameters),
                        (state.optimizer_key, optimizer_tensors),
                    )
                )
                registered_keys = (state.weights_key, state.optimizer_key)
                from .portable_snapshot import commit_prepared_portable_checkpoint

                commit_prepared_portable_checkpoint(
                    self._slots,
                    staging_name=prepared_portable.staging_name,
                    name=spec.run_id,
                )
                installed = True
                prepared_portable = None
            self._runs[spec.run_id] = state
        except BaseException as error:
            retirements = tuple(
                self._residency.retire_async(key) for key in registered_keys
            )
            for retirement in retirements:
                try:
                    retirement.result(timeout=self._residency.config.shutdown_timeout_s)
                except BaseException as cleanup_error:
                    error.add_note(
                        "run registration residency cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            if prepared_portable is not None:
                try:
                    self._discard_prepared_portable_run(prepared_portable)
                except BaseException as cleanup_error:
                    error.add_note(
                        "portable registration staging cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            if installed:
                try:
                    self._slots.release_checkpoint_slot(spec.run_id)
                except BaseException as cleanup_error:
                    error.add_note(
                        "run registration slot cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise
        return portable_read

    def _residency_key(
        self,
        state: _ResidentCommandRun,
        *,
        generation_id: str,
        representation: Literal["weights", "optimizer", "accumulator"],
        accumulator_revision: int = 0,
        adapter_config: dict[str, Any] | None = None,
    ) -> ResidencyKey:
        adapter_layout = json.dumps(
            state.adapter_config if adapter_config is None else adapter_config,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return ResidencyKey(
            training_session_id=state.spec.training_session_id,
            run_id=state.spec.run_id,
            generation_id=generation_id,
            representation=representation,
            accumulator_revision=accumulator_revision,
            topology_fingerprint=self._topology_fingerprint,
            adapter_layout_fingerprint=hashlib.sha256(adapter_layout).hexdigest(),
        )

    def _prepare_portable_run(
        self,
        *,
        run_id: str,
        generation_id: str,
        archive: PortableSnapshotArchive,
        expected_lora_rank: int,
        expected_lora_target_modules: tuple[str, ...],
        restore_optimizer: bool,
    ) -> _PreparedPortableRun:
        source = self._portable_snapshot_source
        if source is None:
            raise RuntimeError("portable checkpoint source is not configured")
        from art.megatron import checkpoint as _checkpoint

        from .portable_snapshot import (
            install_prepared_portable_checkpoint,
            prepare_portable_checkpoint,
        )

        staging_digest = hashlib.sha256(
            (
                run_id
                + "\0"
                + generation_id
                + "\0"
                + archive.archive_sha256
                + "\0"
                + json.dumps(restore_optimizer)
            ).encode()
        ).hexdigest()[:32]
        staging_name = f"__art_portable_{staging_digest}"
        installed = False
        try:
            with prepare_portable_checkpoint(
                self._slots,
                source,
                archive,
                destination_rank=int(self.runtime.rank),
                expected_lora_rank=expected_lora_rank,
                expected_lora_target_modules=expected_lora_target_modules,
                restore_optimizer=restore_optimizer,
            ) as prepared:
                install_prepared_portable_checkpoint(
                    self._slots, prepared, name=staging_name
                )
                installed = True
                prepared_run = None
                preparation_error: BaseException | None = None
                try:
                    if not restore_optimizer:
                        self._slots.prepare_checkpoint_slot_optimizer(
                            staging_name, OptimizerConfig(learning_rate=0.0)
                        )
                    weights = self._slots.checkpoint_slot_parameters(staging_name)
                    optimizer_tensors = self._slots.checkpoint_slot_optimizer_tensors(
                        staging_name
                    )
                    if not weights or not optimizer_tensors:
                        raise RuntimeError(
                            "prepared portable checkpoint lacks a complete working set"
                        )
                    with torch.no_grad():
                        for tensor in (*weights, *optimizer_tensors):
                            tensor.data = tensor.detach().to(device="cpu")
                    if any(
                        tensor.device.type != "cpu"
                        for tensor in (*weights, *optimizer_tensors)
                    ):
                        raise RuntimeError(
                            "prepared portable checkpoint is not entirely CPU resident"
                        )
                    prepared_run = _PreparedPortableRun(
                        receipt=prepared.receipt,
                        staging_name=staging_name,
                        adapter_config=dict(prepared.config),
                        weights=weights,
                        optimizer_tensors=optimizer_tensors,
                    )
                except BaseException as error:
                    preparation_error = error
                _checkpoint.raise_distributed(
                    preparation_error,
                    "prepare portable checkpoint CPU working set",
                    _checkpoint._ensure_group(self._slots),
                )
                assert prepared_run is not None
                return prepared_run
        except BaseException as error:
            if installed:
                try:
                    self._slots.release_checkpoint_slot(staging_name)
                except BaseException as cleanup_error:
                    error.add_note(
                        "portable preparation cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise

    def _discard_prepared_portable_run(self, prepared: _PreparedPortableRun) -> None:
        self._slots.release_checkpoint_slot(prepared.staging_name)

    def _register_portable_l2_working_set(
        self,
        working_set: tuple[
            tuple[ResidencyKey, tuple[torch.Tensor, ...]],
            ...,
        ],
    ) -> None:
        from art.megatron import checkpoint as _checkpoint

        registered = False
        registration_error: BaseException | None = None
        try:
            self._residency.register_l2_working_set(working_set)
            registered = True
        except BaseException as error:
            registration_error = error
        try:
            _checkpoint.raise_distributed(
                registration_error,
                "register portable checkpoint L2 working set",
                _checkpoint._ensure_group(self._slots),
            )
        except BaseException as error:
            if registered:
                for key, _tensors in working_set:
                    try:
                        self._residency.retire_async(key).result(
                            timeout=self._residency.config.shutdown_timeout_s
                        )
                    except BaseException as cleanup_error:
                        error.add_note(
                            "portable L2 registration cleanup failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            raise

    @staticmethod
    def _state_keys(state: _ResidentCommandRun) -> tuple[ResidencyKey, ...]:
        return tuple(
            key
            for key in (
                state.weights_key,
                state.optimizer_key,
                state.accumulator_key,
            )
            if key is not None
        )

    @staticmethod
    def _component_keys(
        state: _ResidentCommandRun, components: tuple[str, ...]
    ) -> tuple[ResidencyKey, ...]:
        by_component = {
            "weights": state.weights_key,
            "optimizer": state.optimizer_key,
            "accumulator": state.accumulator_key,
        }
        unknown = set(components).difference(by_component)
        if unknown:
            raise ValueError(f"unsupported residency components: {sorted(unknown)}")
        return tuple(
            key
            for component in components
            if (key := by_component[component]) is not None
        )

    def prefetch_residency(
        self,
        run_id: str,
        components: tuple[str, ...],
        learner_version: int,
    ) -> dict[str, Any]:
        state = self._require_residency_parent(run_id, learner_version)
        keys = self._component_keys(state, components)
        for key in keys:
            lower = self._residency.prefetch_l2_from_lower(key)
            if lower is not None:
                lower.result()
        return self._residency_evidence(run_id, None, components, keys)

    def admit_residency(
        self,
        operation_id: str,
        run_id: str,
        components: tuple[str, ...],
        learner_version: int,
    ) -> dict[str, Any]:
        state = self._require_residency_parent(run_id, learner_version)
        keys = self._component_keys(state, components)
        with self._residency_admission_lock:
            retained = self._residency_admissions.get(operation_id)
        if retained is not None:
            if retained != keys:
                raise RuntimeError("operation residency admission changed identity")
            return self._residency_evidence(run_id, operation_id, components, keys)
        self._residency.acquire_l1_working_set(keys)
        with self._residency_admission_lock:
            retained = self._residency_admissions.get(operation_id)
            if retained is not None:
                self._residency.release_l1_working_set(keys)
                if retained != keys:
                    raise RuntimeError("operation residency admission changed identity")
            else:
                self._residency_admissions[operation_id] = keys
        return self._residency_evidence(run_id, operation_id, components, keys)

    def release_residency_admission(self, operation_id: str) -> None:
        with self._residency_admission_lock:
            keys = self._residency_admissions.pop(operation_id, None)
        if keys is not None:
            self._residency.release_l1_working_set(keys)

    def _require_residency_parent(
        self, run_id: str, learner_version: int
    ) -> _ResidentCommandRun:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"trainer command run {run_id!r} is absent")
        if state.learner_version != learner_version:
            raise RuntimeError(
                "residency request does not identify the current learner: "
                f"requested={learner_version}, current={state.learner_version}"
            )
        return state

    def _residency_evidence(
        self,
        run_id: str,
        operation_id: str | None,
        requested: tuple[str, ...],
        keys: tuple[ResidencyKey, ...],
    ) -> dict[str, Any]:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"trainer command run {run_id!r} is absent")
        required = set(keys)
        components = []
        for key in self._state_keys(state):
            entry = self._residency.ledger.entry(key)
            l1 = next((copy for copy in entry.copies if copy.tier == "l1_gpu"), None)
            components.append(
                {
                    "component": key.representation,
                    "generation_id": key.generation_id,
                    "required_for_operation": key in required,
                    "byte_count": 0 if l1 is None else l1.byte_count,
                    "tiers": tuple(copy.tier for copy in entry.copies),
                    "l1_ready": l1 is not None,
                }
            )
        return {
            "rank": int(self.runtime.rank),
            "run_id": run_id,
            "operation_id": operation_id,
            "requested_components": requested,
            "components": tuple(components),
        }

    @contextmanager
    def _resident_operation(
        self,
        operation_id: str,
        state: _ResidentCommandRun,
        components: tuple[str, ...],
    ) -> Any:
        expected = self._component_keys(state, components)
        with self._residency_admission_lock:
            admitted = self._residency_admissions.get(operation_id)
        if admitted != expected:
            raise RuntimeError("GPU command has no exact residency admission")
        try:
            yield expected
        finally:
            self.release_residency_admission(operation_id)

    @contextmanager
    def _maintenance_resident(
        self, state: _ResidentCommandRun, components: tuple[str, ...]
    ) -> Any:
        keys = self._component_keys(state, components)
        self._residency.acquire_l1_working_set(keys)
        try:
            yield keys
        finally:
            self._residency.release_l1_working_set(keys)

    def _replace_admission_key(
        self, operation_id: str, source: ResidencyKey, target: ResidencyKey
    ) -> None:
        with self._residency_admission_lock:
            keys = self._residency_admissions.get(operation_id)
            if keys is None or source not in keys:
                raise RuntimeError("residency generation advance lost its admission")
            self._residency_admissions[operation_id] = tuple(
                target if key == source else key for key in keys
            )

    def export_run_checkpoint(
        self,
        run_id: str,
        generation: TrainerGeneration,
        export_id: str,
    ) -> PortableSnapshotRankReceipt | None:
        state = self._runs.get(run_id)
        if state is None or state.learner_version != generation.policy_step:
            raise RuntimeError("portable export generation is not resident")
        if generation.training_session_id != state.spec.training_session_id:
            raise RuntimeError("portable export belongs to another training session")
        if self._portable_snapshot_sink is None:
            raise RuntimeError("portable checkpoint sink is not configured")
        from .portable_snapshot import (
            PortableSnapshotGeneration,
            export_portable_checkpoint,
        )

        with self._maintenance_resident(state, ("weights", "optimizer")):
            return export_portable_checkpoint(
                self._slots,
                self._portable_snapshot_sink,
                PortableSnapshotGeneration(
                    training_session_id=generation.training_session_id,
                    policy_step=generation.policy_step,
                    generation_id=generation.generation_id,
                ),
                export_id=export_id,
                name=run_id,
                rank=int(self.runtime.rank),
            )

    def run_tensor_owners(self, run_id: str) -> tuple[tuple[str, int], ...]:
        if run_id not in self._runs:
            raise KeyError(f"trainer command run {run_id!r} is absent")
        return self._slots.checkpoint_slot_tensor_owners(run_id)

    def prepare_run_checkpoint(
        self,
        operation_id: str,
        run_id: str,
        generation: TrainerGeneration,
        archive: PortableSnapshotArchive,
        *,
        restore_optimizer: bool,
    ) -> PortableSnapshotReadReceipt:
        fingerprint = hashlib.sha256(
            (
                run_id
                + "\0"
                + generation.model_dump_json()
                + "\0"
                + archive.archive_sha256
                + "\0"
                + json.dumps(restore_optimizer)
            ).encode()
        ).hexdigest()
        prior = self._checkpoint_hydrations.get(operation_id)
        if prior is not None:
            if prior.fingerprint != fingerprint or prior.run_id != run_id:
                raise RuntimeError("checkpoint hydration operation was reused")
            return prior.prepared.receipt
        if any(
            pending.run_id == run_id for pending in self._checkpoint_hydrations.values()
        ):
            raise RuntimeError("another checkpoint hydration is pending for this run")
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"trainer command run {run_id!r} is absent")
        if state.gradients.contribution_ids:
            raise RuntimeError("cannot load state with open gradient contributions")
        if generation.training_session_id != state.spec.training_session_id:
            raise RuntimeError("loaded generation belongs to another training session")
        if self._portable_snapshot_source is None:
            raise RuntimeError("portable checkpoint source is not configured")
        from art.megatron.training.gradient_accumulator import (
            ParameterGradientAccumulator,
        )

        prepared: _PreparedPortableRun | None = None
        try:
            prepared = self._prepare_portable_run(
                run_id=run_id,
                generation_id=generation.generation_id,
                archive=archive,
                expected_lora_rank=state.spec.lora_rank,
                expected_lora_target_modules=state.spec.lora_target_modules,
                restore_optimizer=restore_optimizer,
            )
            gradients = ParameterGradientAccumulator(prepared.weights)
            weights_key = self._residency_key(
                state,
                generation_id=generation.generation_id,
                representation="weights",
                adapter_config=prepared.adapter_config,
            )
            optimizer_key = self._residency_key(
                state,
                generation_id=generation.generation_id,
                representation="optimizer",
                adapter_config=prepared.adapter_config,
            )
            self._register_portable_l2_working_set(
                (
                    (weights_key, prepared.weights),
                    (optimizer_key, prepared.optimizer_tensors),
                )
            )
            pending = _PreparedRunCheckpoint(
                operation_id=operation_id,
                fingerprint=fingerprint,
                run_id=run_id,
                learner_version=generation.policy_step,
                previous_learner_version=state.learner_version,
                previous_keys=self._state_keys(state),
                prepared=prepared,
                gradients=gradients,
                weights_key=weights_key,
                optimizer_key=optimizer_key,
                remaining_keys={weights_key, optimizer_key},
            )
            self._checkpoint_hydrations[operation_id] = pending
        except BaseException as error:
            if prepared is not None:
                try:
                    self._discard_prepared_portable_run(prepared)
                except BaseException as cleanup_error:
                    error.add_note(
                        "portable restore staging cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise
        return pending.prepared.receipt

    def commit_prepared_run_checkpoint(
        self, operation_id: str, run_id: str
    ) -> PortableSnapshotReadReceipt:
        pending = self._checkpoint_hydrations.get(operation_id)
        if pending is None:
            raise RuntimeError("prepared checkpoint hydration is absent")
        if pending.run_id != run_id:
            raise RuntimeError("prepared checkpoint hydration changed run identity")
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"trainer command run {run_id!r} is absent")
        if state.gradients.contribution_ids:
            raise RuntimeError("cannot load state with open gradient contributions")
        if (
            state.learner_version != pending.previous_learner_version
            or self._state_keys(state) != pending.previous_keys
        ):
            raise RuntimeError("run state changed after checkpoint hydration")

        from .portable_snapshot import commit_prepared_portable_checkpoint

        commit_prepared_portable_checkpoint(
            self._slots,
            staging_name=pending.prepared.staging_name,
            name=run_id,
        )
        pending.staging_live = False
        state.gradients = pending.gradients
        state.learner_version = pending.learner_version
        state.adapter_config = pending.prepared.adapter_config
        state.portable_read = pending.prepared.receipt
        state.accumulator_key = None
        state.next_accumulator_revision = 1
        state.weights_key = pending.weights_key
        state.optimizer_key = pending.optimizer_key
        pending.remaining_keys.clear()
        self._checkpoint_hydrations.pop(operation_id)
        for key in pending.previous_keys:
            if key not in (pending.weights_key, pending.optimizer_key):
                self._residency.retire_async(key)
        return pending.prepared.receipt

    def discard_prepared_run_checkpoint(
        self, operation_id: str, run_id: str | None = None
    ) -> bool:
        pending = self._checkpoint_hydrations.get(operation_id)
        if pending is None:
            return False
        if run_id is not None and pending.run_id != run_id:
            raise RuntimeError("checkpoint hydration discard changed run identity")
        failures: list[BaseException] = []
        for key in tuple(pending.remaining_keys):
            try:
                self._residency.retire_async(key).result(
                    timeout=self._residency.config.shutdown_timeout_s
                )
                pending.remaining_keys.remove(key)
            except BaseException as error:
                failures.append(error)
        if pending.staging_live:
            try:
                self._discard_prepared_portable_run(pending.prepared)
                pending.staging_live = False
            except BaseException as error:
                failures.append(error)
        if not pending.remaining_keys and not pending.staging_live:
            self._checkpoint_hydrations.pop(operation_id)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "prepared checkpoint hydration cleanup failed", failures
            )
        return True

    def _register_accumulator_residency(self, state: _ResidentCommandRun) -> None:
        if state.accumulator_key is not None:
            return
        tensors = state.gradients.residency_tensors()
        if not tensors:
            raise RuntimeError("F/B completed without a resident gradient image")
        weights_key = state.weights_key
        if weights_key is None:
            raise RuntimeError("gradient accumulator has no parent weight generation")
        key = self._residency_key(
            state,
            generation_id=weights_key.generation_id,
            representation="accumulator",
            accumulator_revision=state.next_accumulator_revision,
        )
        state.next_accumulator_revision += 1
        self._residency.register_mutable_l1(key, tensors)
        state.accumulator_key = key

    def execute_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> CommandResultLaunch:
        state = self._require_parent(job)
        validate_packed_batch(batch)
        from art.megatron.train import (
            execute_megatron_dynamic_lora_forward_backward_job,
        )

        with self._resident_operation(
            job.operation_id, state, ("weights", "accumulator")
        ):
            if state.accumulator_key is not None:
                self._residency.wait_before_mutation_working_set(
                    (state.accumulator_key,)
                )
                self._residency.begin_l1_mutation(state.accumulator_key)
            try:
                result = execute_megatron_dynamic_lora_forward_backward_job(
                    self.runtime,
                    job,
                    batch.tensors,
                    run_slots=self._slots,
                    gradient_accumulator=state.gradients,
                    cancelled=cancelled,
                )
            except BaseException:
                self._slots.clear_checkpoint_slot_grads(job.run_id)
                raise
            self._register_accumulator_residency(state)
        self._enforce_accumulator_budget()
        return _defer_command_result(
            {
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "loss_bearing_token_count": job.expected_global_loss_bearing_tokens,
                "completed_gradient_steps": result.completed_gradient_steps,
                "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
                "executed_token_equivalents": result.executed_token_equivalents,
                "gpu_service_ns": result.gpu_service_ns,
                "metrics": result.metrics,
            },
            result.new_logprobs if job.return_token_logprobs else (),
            _command_token_logprob_decoder(batch)
            if job.return_token_logprobs
            else None,
        )

    def execute_forward(
        self,
        job: ForwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> CommandResultLaunch:
        state = self._require_parent(job)
        validate_packed_batch(batch)
        from art.megatron.lora import LoRASlotRef, use_lora_slot
        from art.megatron.train import execute_megatron_rl_forward_job

        with self._resident_operation(job.operation_id, state, ("weights",)):
            with use_lora_slot(LoRASlotRef("checkpoint", job.run_id)):
                result = execute_megatron_rl_forward_job(
                    self.runtime,
                    job,
                    batch.tensors,
                    cancelled=cancelled,
                    state_is_resident=True,
                )
        return _defer_command_result(
            {
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
                "executed_token_equivalents": result.executed_token_equivalents,
                "gpu_service_ns": result.gpu_service_ns,
                "metrics": result.metrics,
            },
            result.new_logprobs if job.return_token_logprobs else (),
            _command_token_logprob_decoder(batch)
            if job.return_token_logprobs
            else None,
        )

    def execute_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> CommandResultLaunch:
        state = self._require_parent(job)
        from art.megatron.train import (
            execute_megatron_dynamic_lora_sft_forward_backward_job,
        )

        with self._resident_operation(
            job.operation_id, state, ("weights", "accumulator")
        ):
            if state.accumulator_key is not None:
                self._residency.wait_before_mutation_working_set(
                    (state.accumulator_key,)
                )
                self._residency.begin_l1_mutation(state.accumulator_key)
            try:
                result = execute_megatron_dynamic_lora_sft_forward_backward_job(
                    self.runtime,
                    job,
                    batch,
                    run_slots=self._slots,
                    gradient_accumulator=state.gradients,
                    cancelled=cancelled,
                )
            except BaseException:
                self._slots.clear_checkpoint_slot_grads(job.run_id)
                raise
            self._register_accumulator_residency(state)
        self._enforce_accumulator_budget()
        return _defer_command_result(
            {
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "loss_bearing_token_count": job.expected_global_loss_bearing_tokens,
                "completed_gradient_steps": result.completed_gradient_steps,
                "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
                "executed_token_equivalents": result.executed_token_equivalents,
                "gpu_service_ns": result.gpu_service_ns,
                "metrics": result.metrics,
            },
            result.new_logprobs if job.return_token_logprobs else (),
            _sft_token_logprobs,
        )

    def execute_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> CommandResultLaunch:
        state = self._require_parent(job)
        from art.megatron.train import execute_megatron_dynamic_lora_sft_forward_job

        with self._resident_operation(job.operation_id, state, ("weights",)):
            result = execute_megatron_dynamic_lora_sft_forward_job(
                self.runtime, job, batch, cancelled=cancelled
            )
        return _defer_command_result(
            {
                "operation_id": job.operation_id,
                "learner_version": job.expected_learner_version,
                "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
                "executed_token_equivalents": result.executed_token_equivalents,
                "gpu_service_ns": result.gpu_service_ns,
                "metrics": result.metrics,
            },
            result.new_logprobs if job.return_token_logprobs else (),
            _sft_token_logprobs,
        )

    def capture_forward_backward_numerics(
        self,
        *,
        run_id: str,
        operation_id: str,
        batch: InMemoryPackedBatch,
        token_logprobs: tuple[TokenLogprobs, ...],
        root: str,
    ) -> "ForwardBackwardNumericalRankReceipt":
        """Persist exact rank-local evidence for an open F/B contribution."""

        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"trainer command run {run_id!r} is absent")
        contribution_ids = state.gradients.contribution_ids
        if not contribution_ids or contribution_ids[-1] != operation_id:
            raise RuntimeError("numerical capture is not the open F/B suffix")
        validate_packed_batch(batch)
        from .numerical_capture import capture_forward_backward_rank

        with self._maintenance_resident(state, ("accumulator",)):
            gradients = state.gradients.snapshot_local_sums().gradients
            return capture_forward_backward_rank(
                root=root,
                run_id=run_id,
                operation_id=operation_id,
                contribution_ids=contribution_ids,
                rank=int(self.runtime.rank),
                packed_tensors=batch.tensors,
                token_logprobs=token_logprobs,
                gradients=gradients,
            )

    def execute_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        state = self._require_parent(job)
        with self._resident_operation(
            job.operation_id, state, ("weights", "optimizer", "accumulator")
        ):
            weights_key = state.weights_key
            optimizer_key = state.optimizer_key
            accumulator_key = state.accumulator_key
            if weights_key is None or optimizer_key is None or accumulator_key is None:
                raise RuntimeError("optimizer command lacks a complete working set")
            for key in (weights_key, optimizer_key):
                self._residency.ensure_l2(key).result()
            self._residency.wait_before_mutation_working_set(
                (weights_key, optimizer_key, accumulator_key)
            )
            state.gradients.seal(job.contributing_forward_backward_operation_ids)
            local_sums, step_flags = state.gradients.prepare_local_sums()
            expected = local_sums.expected_global_token_count
            if expected is None:
                raise RuntimeError("optimizer gradients lack global token provenance")

            def optimizer_step() -> tuple[dict[str, float], int]:
                from megatron.core import parallel_state as ps
                import torch

                global_tokens = local_sums.local_token_count.detach().clone()
                group = ps.get_data_parallel_group(with_context_parallel=True)
                if torch.distributed.get_world_size(group) > 1:
                    torch.distributed.all_reduce(global_tokens, group=group)
                observed = int(global_tokens.item())
                if observed != expected:
                    raise RuntimeError(
                        "accumulated trainable-token count differs from packed "
                        f"provenance: observed={observed}, expected={expected}"
                    )
                gradients = self._slots.reduce_checkpoint_slot_grads(
                    job.run_id,
                    local_sums.gradients,
                    scale_grads=(
                        1.0 if local_sums.reduction == "sum" else 1.0 / observed
                    ),
                )
                result = self._slots.optim_step_reduced(
                    job.run_id,
                    params=OptimizerConfig(
                        learning_rate=job.optimizer.learning_rate,
                        beta1=job.optimizer.beta1,
                        beta2=job.optimizer.beta2,
                        eps=job.optimizer.eps,
                        weight_decay=job.optimizer.weight_decay,
                        grad_clip_norm=job.optimizer.grad_clip_norm,
                    ),
                    gradients=gradients,
                    step_flags=step_flags,
                )
                return result, observed

            started = time.perf_counter()
            (result, _tokens), gpu_service_ns = measure_cuda_call(optimizer_step)
            if not result["update_successful"] or not math.isfinite(
                result["grad_norm"]
            ):
                raise RuntimeError("dynamic LoRA optimizer rejected the update")
            consumed = state.gradients.consume()
            if consumed != job.contributing_forward_backward_operation_ids:
                raise RuntimeError(
                    "optimizer consumed the wrong gradient contributions"
                )
            output_weights = self._residency_key(
                state,
                generation_id=job.generation.generation_id,
                representation="weights",
            )
            output_optimizer = self._residency_key(
                state,
                generation_id=job.generation.generation_id,
                representation="optimizer",
            )
            self._residency.advance_l1(
                weights_key,
                output_weights,
                self._slots.checkpoint_slot_parameters(job.run_id),
                retire_source=True,
            )
            self._replace_admission_key(job.operation_id, weights_key, output_weights)
            self._residency.advance_l1(
                optimizer_key,
                output_optimizer,
                self._slots.checkpoint_slot_optimizer_tensors(job.run_id),
                retire_source=True,
            )
            self._replace_admission_key(
                job.operation_id, optimizer_key, output_optimizer
            )
            self._residency.retire_async(accumulator_key)
            state.weights_key = output_weights
            state.optimizer_key = output_optimizer
            state.accumulator_key = None
            state.learner_version = job.learner_version
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": consumed,
            "gpu_service_ns": gpu_service_ns,
            "metrics": {
                "loss/learning_rate": job.optimizer.learning_rate,
                "loss/grad_norm": result["grad_norm"],
                "optimizer/update_successful": result["update_successful"],
                "optimizer/num_zeros_in_grad": result["num_zeros_in_grad"],
                "time/optimizer_step_s": time.perf_counter() - started,
            },
        }

    def publish_generation(self, spec: CommandPublicationSpec) -> dict[str, Any]:
        state = self._runs.get(spec.run_id)
        if state is None or state.learner_version != spec.generation.policy_step:
            raise RuntimeError("published generation is not resident")
        if state.spec.training_session_id != spec.generation.training_session_id:
            raise RuntimeError("published generation belongs to another session")
        if state.gradients.contribution_ids:
            raise RuntimeError("cannot publish a generation with open gradients")
        with self._maintenance_resident(state, ("weights",)):
            sink = _CommandPublicationSink()
            metrics = self._publisher.submit_command(
                spec,
                adapter_config=state.adapter_config,
                sink=sink,
            )
            record = sink.future.result()
        return {
            "run_id": spec.run_id,
            "generation_id": spec.generation.generation_id,
            "rank": int(self.runtime.rank),
            "record": record.model_dump(mode="json"),
            "metrics": metrics,
        }

    def run_gradient_ids(self, run_id: str) -> tuple[str, ...]:
        state = self._runs.get(run_id)
        return () if state is None else state.gradients.contribution_ids

    def discard_run_gradients(self, run_id: str) -> tuple[str, ...]:
        state = self._runs.get(run_id)
        if state is None:
            return ()
        contributions = state.gradients.contribution_ids
        state.gradients.discard()
        if state.accumulator_key is not None:
            self._residency.retire_async(state.accumulator_key)
            state.accumulator_key = None
        return contributions

    def release_run(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        if state.gradients.contribution_ids:
            raise RuntimeError("cannot release a run with open gradients")
        with self._residency_admission_lock:
            active = tuple(
                operation_id
                for operation_id, keys in self._residency_admissions.items()
                if any(key.run_id == run_id for key in keys)
            )
        if active:
            raise RuntimeError(
                f"cannot release a run with residency admissions: {active}"
            )
        for operation_id, pending in tuple(self._checkpoint_hydrations.items()):
            if pending.run_id == run_id:
                self.discard_prepared_run_checkpoint(operation_id, run_id)
        retirements = tuple(
            self._residency.retire_async(key) for key in self._residency.keys(run_id)
        )
        for retirement in retirements:
            retirement.result(timeout=self._residency.config.shutdown_timeout_s)
        self._slots.release_checkpoint_slot(run_id)
        self._runs.pop(run_id)

    def discard_open_gradients(self) -> None:
        for state in self._runs.values():
            state.gradients.discard()

    @property
    def has_open_gradients(self) -> bool:
        return any(state.gradients.contribution_ids for state in self._runs.values())

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_publisher:
            self._publisher.close()
        with self._residency_admission_lock:
            operation_ids = tuple(self._residency_admissions)
        for operation_id in operation_ids:
            self.release_residency_admission(operation_id)
        for operation_id in tuple(self._checkpoint_hydrations):
            self.discard_prepared_run_checkpoint(operation_id)
        for state in tuple(self._runs.values()):
            state.gradients.discard()
            retirements = tuple(
                self._residency.retire_async(key)
                for key in self._residency.keys(state.spec.run_id)
            )
            for retirement in retirements:
                retirement.result(timeout=self._residency.config.shutdown_timeout_s)
            self._slots.release_checkpoint_slot(state.spec.run_id)
        self._residency.close()
        self._runs.clear()
        source, self._portable_snapshot_source = self._portable_snapshot_source, None
        sink, self._portable_snapshot_sink = self._portable_snapshot_sink, None
        if source is not None:
            source.close()
        if sink is not None and sink is not source:
            sink.close()
        self._closed = True

    def _require_parent(
        self,
        job: (
            ForwardBackwardJobSpec
            | ForwardJobSpec
            | SftForwardBackwardJobSpec
            | SftForwardJobSpec
            | OptimizerJobSpec
        ),
    ) -> _ResidentCommandRun:
        if self._closed:
            raise RuntimeError("Megatron run slot executor is closed")
        state = self._runs.get(job.run_id)
        if state is None:
            raise RuntimeError(f"training run is not resident: {job.run_id!r}")
        if (
            state.spec.training_session_id != job.training_session_id
            or state.learner_version != job.expected_learner_version
        ):
            raise RuntimeError("resident run state does not match command parent")
        return state

    def _enforce_accumulator_budget(self) -> None:
        resident = sum(
            state.gradients.residency_nbytes for state in self._runs.values()
        )
        if resident > self._accumulator_l1_budget_bytes:
            raise RuntimeError(
                "gradient accumulators exceed the per-rank L1 budget: "
                f"{resident} > {self._accumulator_l1_budget_bytes}"
            )


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
        self._transport_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-publish-transport"
        )
        self._durability_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-publish-durable"
        )
        self._external_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-publish-external"
        )
        self._transport_sender: Any | None = None
        self._lora_layout: SafetensorsLayout | None = None
        self._failures: list[BaseException] = []
        self._in_flight = 0

    def submit(
        self,
        job: TrainerJobSpec,
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
        save_optimizer: bool,
        *,
        sink: EventSink,
    ) -> dict[str, float]:
        return self._submit(
            generation=job.output.generation,
            optimizer_state_path=job.output.optimizer_state_path,
            staging_adapter_path=job.output.staging_adapter_path,
            adapter_dtypes=adapter_dtypes,
            adapter_config=adapter_config,
            save_optimizer=save_optimizer,
            publication_targets=job.publication_targets,
            sink=sink,
        )

    def submit_command(
        self,
        spec: CommandPublicationSpec,
        *,
        adapter_config: dict[str, Any],
        sink: EventSink,
    ) -> dict[str, float]:
        from art.megatron.lora import LoRASlotRef

        return self._submit(
            generation=spec.generation,
            optimizer_state_path=spec.optimizer_state_path,
            staging_adapter_path=spec.staging_adapter_path,
            adapter_dtypes={},
            adapter_config=adapter_config,
            save_optimizer=False,
            publication_targets=spec.publication_targets,
            slot_ref=LoRASlotRef("checkpoint", spec.run_id),
            sink=sink,
        )

    def _submit(
        self,
        *,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str,
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
        save_optimizer: bool,
        publication_targets: tuple[AdapterTransferTarget, ...],
        slot_ref: "LoRASlotRef | None" = None,
        sink: EventSink,
    ) -> dict[str, float]:
        from art.megatron.optimizer_state import stage_optimizer_state_snapshot
        from art.megatron.weights.lora_publish import (
            stage_vllm_lora_snapshot_from_model,
        )

        wait_s, in_flight, stager = self._acquire_slot()
        prepare_started = time.perf_counter()
        optimizer_handoff: Future[Any] = Future()
        transport: Future[Future[TrainerRankPublication]] | None = None
        try:
            lora = stage_vllm_lora_snapshot_from_model(
                model=self.runtime.model,
                adapter_dtypes=adapter_dtypes,
                handler=self.runtime.model_support_handler,
                adapter_config=adapter_config,
                rank=self.runtime.rank,
                world_size=self.runtime.world_size,
                stager=stager,
                slot_ref=slot_ref,
            )
            lora_launch_s = time.perf_counter() - prepare_started
            if lora is not None:
                self.runtime.optimizer_snapshot_barrier.register(lora)
            transport = self._enqueue_transport(
                generation=generation,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=staging_adapter_path,
                lora=lora,
                adapter=None,
                optimizer=optimizer_handoff,
                publication_targets=publication_targets,
            )
            optimizer_started = time.perf_counter()
            optimizer = (
                stage_optimizer_state_snapshot(
                    self.runtime,
                    generation_id=generation.generation_id,
                    step=generation.policy_step,
                    stager=stager,
                )
                if save_optimizer
                else None
            )
            if optimizer is not None:
                self.runtime.optimizer_snapshot_barrier.register(optimizer)
            optimizer_handoff.set_result(optimizer)
            optimizer_launch_s = time.perf_counter() - optimizer_started
            handoff_started = time.perf_counter()
            transport.add_done_callback(
                lambda done: self._transport_ready(
                    done,
                    sink=sink,
                    generation=generation,
                    stager=stager,
                )
            )
            transport_handoff_wait_s = time.perf_counter() - handoff_started
        except BaseException as error:
            publication_error = error
            if transport is not None:
                optimizer_handoff.set_exception(error)
                publication_error = self._drain_transport(transport, error)
            self._report_failure(
                publication_error,
                sink=sink,
                generation=generation,
                remember=False,
                stager=stager,
            )
            raise
        return {
            "snapshot_pool_wait_s": wait_s,
            "snapshot_pool_in_use": float(in_flight),
            "snapshot_pool_pressure": in_flight / self.capacity,
            "snapshot_lora_launch_s": lora_launch_s,
            "snapshot_optimizer_launch_s": optimizer_launch_s,
            "snapshot_transport_handoff_wait_s": transport_handoff_wait_s,
            "snapshot_launch_s": time.perf_counter() - prepare_started,
        }

    def submit_external(
        self,
        target: "ExternalLoraTarget",
        sink: "ExternalLoraPublicationSink",
        *,
        source_topology: str,
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
    ) -> tuple[
        "ExternalLoraPlan",
        Future["ExternalLoraPublication | None"],
        dict[str, float],
    ]:
        from art.megatron.weights.external_lora_publish import (
            stage_external_lora_from_model,
        )

        wait_s, in_flight, stager = self._acquire_slot()
        started = time.perf_counter()
        try:
            pending = stage_external_lora_from_model(
                model=self.runtime.model,
                adapter_dtypes=adapter_dtypes,
                handler=self.runtime.model_support_handler,
                adapter_config=adapter_config,
                target=target,
                source_topology=source_topology,
                stager=stager,
            )
            self.runtime.optimizer_snapshot_barrier.register(pending)
            publication = self._external_pool.submit(
                self._publish_external, pending, sink
            )
            publication.add_done_callback(lambda _done: self._release_slot(stager))
        except BaseException:
            self._release_slot(stager)
            raise
        return (
            pending.payload.plan,
            publication,
            {
                "snapshot_pool_wait_s": wait_s,
                "snapshot_pool_in_use": float(in_flight),
                "snapshot_pool_pressure": in_flight / self.capacity,
                "snapshot_external_prepare_s": time.perf_counter() - started,
            },
        )

    @staticmethod
    def _publish_external(
        pending: Any,
        sink: "ExternalLoraPublicationSink",
    ) -> "ExternalLoraPublication | None":
        from art.megatron.weights.external_lora_publish import (
            publish_external_lora_rank,
        )

        try:
            return publish_external_lora_rank(pending.resolve(), sink)
        finally:
            close = getattr(sink, "close", None)
            if close is not None:
                close()

    def _transport_ready(
        self,
        future: Future[Future[TrainerRankPublication]],
        *,
        sink: EventSink,
        generation: TrainerGeneration,
        stager: PinnedCpuSnapshotStager,
    ) -> None:
        try:
            persistence = future.result()
        except BaseException as error:
            self._failed(error, sink=sink, generation=generation, stager=stager)
            return
        persistence.add_done_callback(
            lambda done: self._completed(
                done,
                sink=sink,
                generation=generation,
                stager=stager,
            )
        )

    def _acquire_slot(self) -> tuple[float, int, PinnedCpuSnapshotStager]:
        self.raise_if_failed()
        started = time.perf_counter()
        self._slots.acquire()
        wait_s = time.perf_counter() - started
        with self._lock:
            stager = self._available_stagers.pop()
            stager.reset()
            self._in_flight += 1
            return wait_s, self._in_flight, stager

    def _enqueue_transport(
        self,
        **kwargs: Any,
    ) -> Future[Future[TrainerRankPublication]]:
        return self._transport_pool.submit(self._transport_snapshot, **kwargs)

    @staticmethod
    def _drain_transport(
        transport: Future[Future[TrainerRankPublication]],
        fallback: BaseException,
    ) -> BaseException:
        try:
            transport.result().result()
        except BaseException as error:
            return error
        return fallback

    def _transport_snapshot(
        self,
        *,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        lora: Any,
        adapter: "OptimizerAdapter | None",
        optimizer: Future[Any],
        publication_targets: tuple[Any, ...],
    ) -> Future[TrainerRankPublication]:
        lora = None if lora is None else lora.resolve()
        prepared_tensors = None
        if lora is not None:
            if self._lora_layout is None:
                self._lora_layout = SafetensorsLayout(lora.tensors)
            prepared_tensors = self._lora_layout.bind(lora.tensors)
        failures: list[BaseException] = []
        if int(self.runtime.rank) == 0 and publication_targets:
            if lora is None or prepared_tensors is None:
                raise RuntimeError("rank zero has no LoRA snapshot to transfer")
            try:
                self._transfer_lora_snapshot(
                    lora,
                    publication_targets,
                    prepared_tensors=prepared_tensors,
                )
            except BaseException as error:
                failures.append(error)
        return self._durability_pool.submit(
            self._persist_snapshot,
            generation=generation,
            optimizer_state_path=optimizer_state_path,
            staging_adapter_path=staging_adapter_path,
            lora=lora,
            adapter=adapter,
            optimizer=optimizer,
            prepared_tensors=prepared_tensors,
            failures=failures,
        )

    def _persist_snapshot(
        self,
        *,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        lora: Any,
        adapter: "OptimizerAdapter | None",
        optimizer: Future[Any],
        prepared_tensors: PreparedSafetensors | None,
        failures: list[BaseException],
    ) -> TrainerRankPublication:
        record: TrainerRankPublication | None = None
        try:
            pending_optimizer = optimizer.result()
            resolved_optimizer = (
                None if pending_optimizer is None else pending_optimizer.resolve()
            )
            record = self._persist_generation(
                generation=generation,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=staging_adapter_path,
                lora=lora,
                adapter=adapter,
                optimizer=resolved_optimizer,
                prepared_tensors=prepared_tensors,
            )
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "adapter persistence and transport failed", failures
            )
        if record is None:
            raise RuntimeError("trainer rank produced no publication record")
        return record

    def _transfer_lora_snapshot(
        self,
        lora: Any,
        targets: tuple[Any, ...],
        *,
        prepared_tensors: PreparedSafetensors,
    ) -> None:
        from art.distributed.adapter_transport import AdapterSnapshotSender

        if self._transport_sender is None:
            self._transport_sender = AdapterSnapshotSender()
        self._transport_sender.send(
            lora,
            targets,
            prepared_tensors=prepared_tensors,
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
    ) -> TrainerRankPublication:
        from art.megatron.optimizer_state import (
            publish_adapter_checkpoint,
            write_optimizer_snapshot_shard,
        )
        from art.megatron.weights.lora_publish import save_vllm_lora_snapshot

        rank = int(self.runtime.rank)
        if rank == 0:
            if lora is not None:
                if staging_adapter_path is None or adapter is not None:
                    raise RuntimeError("new adapter publication is inconsistent")
                staging = Path(staging_adapter_path)
                if staging.exists():
                    raise RuntimeError(f"Adapter staging generation exists: {staging}")
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
            if adapter is None:
                raise RuntimeError("rank zero has no immutable adapter")
        shard = (
            write_optimizer_snapshot_shard(
                optimizer,
                optimizer_state_path=optimizer_state_path,
            )
            if optimizer is not None
            else None
        )
        return TrainerRankPublication(
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
        sink: EventSink,
        generation: TrainerGeneration,
        stager: PinnedCpuSnapshotStager,
    ) -> None:
        try:
            event = TrainerPublicationSucceeded(record=future.result())
        except BaseException as error:
            self._failed(error, sink=sink, generation=generation, stager=stager)
            return
        try:
            sink.publication(event)
        except BaseException as error:
            with self._lock:
                self._failures.append(error)
        finally:
            self._release_slot(stager)

    def _failed(
        self,
        error: BaseException,
        *,
        sink: EventSink,
        generation: TrainerGeneration,
        stager: PinnedCpuSnapshotStager,
    ) -> None:
        self._report_failure(
            error,
            sink=sink,
            generation=generation,
            remember=True,
            stager=stager,
        )

    def _report_failure(
        self,
        error: BaseException,
        *,
        sink: EventSink,
        generation: TrainerGeneration,
        remember: bool,
        stager: PinnedCpuSnapshotStager,
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
            self._release_slot(stager)

    def _release_slot(self, stager: PinnedCpuSnapshotStager) -> None:
        with self._lock:
            self._in_flight -= 1
            self._available_stagers.append(stager)
        self._slots.release()

    def raise_if_failed(self) -> None:
        with self._lock:
            failures = tuple(self._failures)
        if failures:
            raise BaseExceptionGroup("trainer generation publication failed", failures)

    def close(self) -> None:
        self._external_pool.shutdown(wait=True)
        self._transport_pool.shutdown(wait=True)
        self._durability_pool.shutdown(wait=True)
        if self._transport_sender is not None:
            self._transport_sender.close()
            self._transport_sender = None
        with self._lock:
            in_flight = self._in_flight
        if in_flight:
            raise RuntimeError(f"publication close retained {in_flight} snapshots")
        self.raise_if_failed()
