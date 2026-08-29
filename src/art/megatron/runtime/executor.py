from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import gc
import math
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock
import time
from typing import TYPE_CHECKING, Any

from art.training.contracts import TokenLogprobs
from art.utils.safetensors import PreparedSafetensors, SafetensorsLayout

from ..tensor_snapshot import PinnedCpuSnapshotStager
from ..training.finalize_grads import (
    finalize_accumulated_model_grads,
    flush_param_grads_to_main_grads,
)
from ..training.gradient_accumulator import GradientAccumulator
from .data_plane import InMemoryPackedBatch, SFTBatchData, validate_packed_batch
from .publication import (
    TrainerPublicationFailed,
    TrainerPublicationSucceeded,
    TrainerRankPublication,
)
from .specs import (
    ForwardBackwardJobSpec,
    OptimizerJobSpec,
    ResidentLoraInspectionShard,
    ResidentLoraInspectionSpec,
    ResidentScoreJobSpec,
    ResidentScoreShard,
    SFTJobSpec,
    TrainerGeneration,
    TrainerJobSpec,
    TrainJobSpec,
)
from .trainer_run import EventSink

if TYPE_CHECKING:
    from art.megatron.optimizer_state import OptimizerAdapter


class MegatronTrainJobExecutor:
    """Thin adapter around the warm runtime's in-memory job entrypoint."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._publisher = _GenerationPublisher(
            runtime,
            capacity=int(runtime.snapshot_pool_capacity),
        )
        self._gradients = GradientAccumulator(
            runtime.model,
            flush_gradients=flush_param_grads_to_main_grads,
        )
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
        if self._gradient_parent_version not in {
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

        result = execute_megatron_rl_forward_backward_job(
            self.runtime,
            job,
            batch.tensors,
            gradient_accumulator=self._gradients,
            cancelled=cancelled,
        )
        self._gradient_parent_version = job.expected_learner_version
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "loss_bearing_token_count": job.expected_global_loss_bearing_tokens,
            "completed_gradient_steps": result.completed_gradient_steps,
            "logical_nonpadding_tokens": result.logical_nonpadding_tokens,
            "executed_token_equivalents": result.executed_token_equivalents,
            "token_logprobs": tuple(
                TokenLogprobs.from_values(
                    values.detach()
                    .to(device="cpu", dtype=values.dtype)
                    .flatten()
                    .tolist(),
                    shape=tuple(values.shape),
                ).model_dump(mode="python")
                for values in result.new_logprobs
            )
            if job.return_token_logprobs
            else (),
            "metrics": result.metrics,
        }

    def execute_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        if self._gradient_parent_version != job.expected_learner_version:
            raise RuntimeError("optimizer parent does not match accumulated gradients")
        runtime = self.runtime
        if runtime.optimizer is None:
            raise RuntimeError("trainer has no resident optimizer")
        self._gradients.seal(job.contributing_forward_backward_operation_ids)
        accumulated = self._gradients.prepare_optimizer()
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
        consumed = self._gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_policy_step = job.learner_version
        runtime.resident_generation_id = job.generation.generation_id
        runtime.optimizer_state_loaded = True
        self._gradient_parent_version = None
        runtime.inter_forward_backward_timing.previous_job_complete_s = time.monotonic()
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": consumed,
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

    def discard_open_gradients(self) -> None:
        self._gradients.discard()
        self._gradient_parent_version = None

    def _require_no_open_gradients(self) -> None:
        if self._gradients.contribution_ids:
            raise RuntimeError("operation cannot discard open gradient contributions")


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
            lora_resolve_started = time.perf_counter()
            lora = None if lora is None else lora.resolve()
            lora_resolve_s = time.perf_counter() - lora_resolve_started
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
            "snapshot_lora_resolve_s": lora_resolve_s,
            "snapshot_optimizer_launch_s": optimizer_launch_s,
            "snapshot_transport_handoff_wait_s": transport_handoff_wait_s,
            "snapshot_launch_s": time.perf_counter() - prepare_started,
        }

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
