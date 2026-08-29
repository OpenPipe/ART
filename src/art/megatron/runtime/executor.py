from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import gc
import math
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock
import time
from typing import TYPE_CHECKING, Any

import torch

from art.training.contracts import TokenLogprobs
from art.utils.safetensors import PreparedSafetensors, SafetensorsLayout

from ..tensor_snapshot import PinnedCpuSnapshotStager
from ..training.finalize_grads import (
    finalize_accumulated_model_grads,
    flush_param_grads_to_main_grads,
)
from ..training.gradient_accumulator import GradientAccumulator
from .data_plane import InMemoryPackedBatch, SFTBatchData, validate_packed_batch
from .device_usage import measure_cuda_call
from .publication import (
    TrainerPublicationFailed,
    TrainerPublicationSucceeded,
    TrainerRankPublication,
)
from .specs import (
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
    if batch.ref.training_kind != "tokenized":
        return tuple(
            TokenLogprobs.from_values(
                values.detach().to(device="cpu").flatten().tolist(),
                shape=tuple(values.shape),
            ).model_dump(mode="python")
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
    portable_read: PortableSnapshotReadReceipt | None = None


class MCoreRunSlotExecutor:
    """Execute independent exact-shape LoRAs on one warm MCore rank."""

    def __init__(
        self,
        runtime: Any,
        *,
        accumulator_l1_budget_bytes: int = 16 * 1024**3,
        portable_snapshot_source: PortableSnapshotSource | None = None,
        portable_snapshot_sink: PortableSnapshotSink | None = None,
    ) -> None:
        from art.trainer_rank import TrainerRank

        self.runtime = runtime
        self._trainer = TrainerRank(runtime)
        self._runs: dict[str, _ResidentCommandRun] = {}
        self._accumulator_l1_budget_bytes = accumulator_l1_budget_bytes
        self._portable_snapshot_source = portable_snapshot_source
        self._portable_snapshot_sink = portable_snapshot_sink
        self._closed = False

    def register_run(self, spec: TrainingRunSpec) -> PortableSnapshotReadReceipt | None:
        if self._closed:
            raise RuntimeError("Megatron run slot executor is closed")
        prior = self._runs.get(spec.run_id)
        if prior is not None:
            if prior.spec != spec:
                raise RuntimeError("run_id was reused with different trainer state")
            return prior.portable_read
        from art.megatron.model_support.lora_disk import load_adapter_config
        from art.megatron.training.gradient_accumulator import (
            ParameterGradientAccumulator,
        )
        from art.trainer_rank import MaterializedCheckpoint

        portable_read = None
        installed = False
        try:
            if spec.initial_portable_snapshot is None:
                adapter_config = load_adapter_config(spec.initial_adapter_path)
                targets = adapter_config.get("target_modules")
                target_modules = (
                    (targets,) if isinstance(targets, str) else tuple(targets or ())
                )
                if int(adapter_config.get("r", 0)) != spec.lora_rank or set(
                    target_modules
                ) != set(spec.lora_target_modules):
                    raise RuntimeError(
                        "resident adapter shape differs from run admission"
                    )
                with self._trainer.push_checkpoint(
                    MaterializedCheckpoint(
                        path=spec.run_id,
                        directory=spec.initial_adapter_path,
                    )
                ):
                    pass
            else:
                archive = spec.initial_portable_snapshot
                if self._portable_snapshot_source is None:
                    raise RuntimeError(
                        "portable run registration requires a snapshot source"
                    )
                generation = archive.generation
                if (
                    generation.training_session_id != spec.training_session_id
                    or generation.policy_step != spec.initial_learner_version
                    or generation.generation_id != spec.initial_generation_id
                ):
                    raise RuntimeError("portable archive identifies another generation")
                from .portable_snapshot import install_portable_checkpoint

                portable_read, _adapter_config = install_portable_checkpoint(
                    self._trainer,
                    self._portable_snapshot_source,
                    archive,
                    name=spec.run_id,
                    destination_rank=int(self.runtime.rank),
                    expected_lora_rank=spec.lora_rank,
                    expected_lora_target_modules=spec.lora_target_modules,
                )
            installed = True
            parameters = self._trainer.checkpoint_slot_parameters(spec.run_id)
            self._runs[spec.run_id] = _ResidentCommandRun(
                spec=spec,
                learner_version=spec.initial_learner_version,
                gradients=ParameterGradientAccumulator(parameters),
                portable_read=portable_read,
            )
        except BaseException:
            if installed:
                self._trainer.release_checkpoint_slot(spec.run_id)
            raise
        return portable_read

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

        return export_portable_checkpoint(
            self._trainer,
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
        return self._trainer.checkpoint_slot_tensor_owners(run_id)

    def install_run_checkpoint(
        self,
        run_id: str,
        generation: TrainerGeneration,
        archive: PortableSnapshotArchive,
    ) -> PortableSnapshotReadReceipt:
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

        from .portable_snapshot import install_portable_checkpoint

        receipt, _adapter_config = install_portable_checkpoint(
            self._trainer,
            self._portable_snapshot_source,
            archive,
            name=run_id,
            destination_rank=int(self.runtime.rank),
            expected_lora_rank=state.spec.lora_rank,
            expected_lora_target_modules=state.spec.lora_target_modules,
        )
        state.gradients = ParameterGradientAccumulator(
            self._trainer.checkpoint_slot_parameters(run_id)
        )
        state.learner_version = generation.policy_step
        return receipt

    def execute_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_parent(job)
        validate_packed_batch(batch)
        from art.megatron.train import (
            execute_megatron_dynamic_lora_forward_backward_job,
        )

        try:
            result = execute_megatron_dynamic_lora_forward_backward_job(
                self.runtime,
                job,
                batch.tensors,
                slot_trainer=self._trainer,
                gradient_accumulator=state.gradients,
                cancelled=cancelled,
            )
        except BaseException:
            self._trainer.clear_checkpoint_slot_grads(job.run_id)
            raise
        self._enforce_accumulator_budget()
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
        self._require_parent(job)
        validate_packed_batch(batch)
        from art.megatron.lora import LoRASlotRef, use_lora_slot
        from art.megatron.train import execute_megatron_rl_forward_job

        with use_lora_slot(LoRASlotRef("checkpoint", job.run_id)):
            result = execute_megatron_rl_forward_job(
                self.runtime,
                job,
                batch.tensors,
                cancelled=cancelled,
                state_is_resident=True,
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

    def execute_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_parent(job)
        from art.megatron.train import (
            execute_megatron_dynamic_lora_sft_forward_backward_job,
        )

        try:
            result = execute_megatron_dynamic_lora_sft_forward_backward_job(
                self.runtime,
                job,
                batch,
                slot_trainer=self._trainer,
                gradient_accumulator=state.gradients,
                cancelled=cancelled,
            )
        except BaseException:
            self._trainer.clear_checkpoint_slot_grads(job.run_id)
            raise
        self._enforce_accumulator_budget()
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "loss_bearing_token_count": job.expected_global_loss_bearing_tokens,
            "completed_gradient_steps": result.completed_gradient_steps,
            "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
            "executed_token_equivalents": result.executed_token_equivalents,
            "gpu_service_ns": result.gpu_service_ns,
            "token_logprobs": (
                _sft_token_logprobs(result.new_logprobs)
                if job.return_token_logprobs
                else ()
            ),
            "metrics": result.metrics,
        }

    def execute_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        self._require_parent(job)
        from art.megatron.train import execute_megatron_dynamic_lora_sft_forward_job

        result = execute_megatron_dynamic_lora_sft_forward_job(
            self.runtime, job, batch, cancelled=cancelled
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
            "executed_token_equivalents": result.executed_token_equivalents,
            "gpu_service_ns": result.gpu_service_ns,
            "token_logprobs": (
                _sft_token_logprobs(result.new_logprobs)
                if job.return_token_logprobs
                else ()
            ),
            "metrics": result.metrics,
        }

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
        state.gradients.seal(job.contributing_forward_backward_operation_ids)
        local_sums, step_flags = state.gradients.prepare_local_sums()
        expected = local_sums.expected_global_token_count
        if expected is None:
            raise RuntimeError("optimizer gradients lack global token provenance")
        from art.trainer_rank import AdamParams

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
            gradients = self._trainer.reduce_checkpoint_slot_grads(
                job.run_id,
                local_sums.gradients,
                scale_grads=(1.0 if local_sums.reduction == "sum" else 1.0 / observed),
            )
            result = self._trainer.optim_step_reduced(
                job.run_id,
                params=AdamParams(
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
        if not result["update_successful"] or not math.isfinite(result["grad_norm"]):
            raise RuntimeError("dynamic LoRA optimizer rejected the update")
        consumed = state.gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
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

    def run_gradient_ids(self, run_id: str) -> tuple[str, ...]:
        state = self._runs.get(run_id)
        return () if state is None else state.gradients.contribution_ids

    def discard_run_gradients(self, run_id: str) -> tuple[str, ...]:
        state = self._runs.get(run_id)
        if state is None:
            return ()
        contributions = state.gradients.contribution_ids
        state.gradients.discard()
        return contributions

    def release_run(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        if state.gradients.contribution_ids:
            raise RuntimeError("cannot release a run with open gradients")
        self._trainer.release_checkpoint_slot(run_id)
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
        self._closed = True
        for state in self._runs.values():
            state.gradients.discard()
        self._runs.clear()
        source, self._portable_snapshot_source = self._portable_snapshot_source, None
        sink, self._portable_snapshot_sink = self._portable_snapshot_sink, None
        if source is not None:
            source.close()
        if sink is not None and sink is not source:
            sink.close()

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
            )
            lora_launch_s = time.perf_counter() - prepare_started
            if lora is not None:
                self.runtime.optimizer_snapshot_barrier.register(lora)
            transport = self._enqueue_transport(
                generation=job.output.generation,
                optimizer_state_path=job.output.optimizer_state_path,
                staging_adapter_path=job.output.staging_adapter_path,
                lora=lora,
                adapter=None,
                optimizer=optimizer_handoff,
                publication_targets=getattr(job, "publication_targets", ()),
            )
            optimizer_started = time.perf_counter()
            optimizer = (
                stage_optimizer_state_snapshot(
                    self.runtime,
                    generation_id=job.output_generation_id,
                    step=job.learner_version,
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
                    generation=job.output.generation,
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
                generation=job.output.generation,
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
