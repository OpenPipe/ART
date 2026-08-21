from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
import gc
import hashlib
import math
from pathlib import Path
from threading import BoundedSemaphore, Condition, Event, Lock
import time
from typing import TYPE_CHECKING, Any, Iterator, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SkipValidation
import torch

from art.distributed.data_plane import PackedBatchRef
from art.distributed.object_store import (
    BinaryObjectPublicationTarget,
    OrderedBinaryObjectTarget,
    S3BinaryObjectStore,
    binary_object_manifest_uri,
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
    canonical_adapter_path,
    read_adapter_publication,
)
from art.training.contracts import TokenLogprobs
from art.utils.safetensors import (
    FileIdentity,
    PreparedSafetensors,
    SafetensorsLayout,
    prepared_safetensors_identity,
    save_prepared_safetensors,
)

from ..tensor_snapshot import PendingCpuSnapshot, PinnedCpuSnapshotStager
from ..training.command_telemetry import (
    PendingRankCommandTelemetry,
    materialize_rank_telemetry,
)
from ..training.gradient_accumulator import GradientAccumulator
from .data_plane import InMemoryPackedBatch, SFTBatchData, validate_packed_batch
from .publication import (
    SnapshotRankWritePlan,
    SnapshotWriteGrant,
    SnapshotWritePlan,
    TrainerPublicationFailed,
    TrainerPublicationSucceeded,
    TrainerRankPublication,
)
from .residency import ResidencyCapacityUnavailable, ResidencyKey
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
    RunSlotRegistration,
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
    from art.megatron.weights.rank_distributed_lora_publish import (
        PreparedRankDistributedLora,
    )
    from art.trainer_rank import TrainerRankOptimizerState


def _consume_future(future: Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _ordered_sampler_target(
    job: GenerationSnapshotJobSpec,
) -> OrderedBinaryObjectTarget | None:
    target = job.adapter_object_target
    if not isinstance(target, OrderedBinaryObjectTarget):
        return None
    if (
        job.save_optimizer
        or job.staging_adapter_path is not None
        or job.existing_adapter is not None
        or job.publication_targets
    ):
        raise ValueError(
            "ordered sampler publication cannot include local or optimizer writes"
        )
    return target


def _command_token_logprobs(
    batch: InMemoryPackedBatch, outputs: list[Any] | tuple[Any, ...]
) -> tuple[Any, ...]:
    target_tokens = (
        batch.tensors.get("target_tokens")
        if batch.ref.training_kind == "tokenized"
        else None
    )
    return _materialize_command_token_logprobs(
        batch.ref,
        None if target_tokens is None else int(target_tokens.shape[2]),
        tuple(outputs),
    )


def _materialize_command_token_logprobs(
    batch: PackedBatchRef,
    candidate_capacity: int | None,
    outputs: tuple[torch.Tensor, ...],
) -> tuple[Any, ...]:
    if batch.training_kind != "tokenized":
        return tuple(
            _packed_logprobs(values.flatten(), (int(values.numel()),))
            for values in outputs
        )
    output_map = batch.tokenized_output_map
    if output_map is None:
        raise RuntimeError("tokenized batch has no output map")
    if candidate_capacity is None:
        raise RuntimeError("tokenized batch has no target candidate capacity")
    physical = torch.cat(
        [values.reshape(-1, candidate_capacity) for values in outputs], dim=0
    )
    expected_rows = batch.num_sequences * batch.sequence_length
    if int(physical.shape[0]) != expected_rows:
        raise RuntimeError(
            "tokenized command did not return every physical packed row: "
            f"returned={physical.shape[0]}, expected={expected_rows}"
        )
    host = physical.detach().to(device="cpu", dtype=torch.float32)
    logical = []
    for positions, candidates in zip(
        output_map.packed_positions, output_map.candidate_counts, strict=True
    ):
        values = host[list(positions), :candidates]
        if candidates == 1:
            logical.append(_packed_logprobs(values[:, 0], (len(positions),)))
        else:
            logical.append(_packed_logprobs(values, (len(positions), candidates)))
    return tuple(logical)


class _ForwardBackwardResultStagerLease:
    def __init__(
        self,
        pool: "_ForwardBackwardResultStagerPool",
        stager: PinnedCpuSnapshotStager,
    ) -> None:
        self._pool = pool
        self.stager = stager
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool.release(self.stager)


class _ForwardBackwardResultStagerPool:
    """Exclusively lease reusable pinned buffers to unresolved F/B results."""

    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise ValueError("F/B result staging capacity must be at least 2")
        self._available = [
            PinnedCpuSnapshotStager(reusable=True) for _ in range(capacity)
        ]
        self._condition = Condition()

    def acquire(self) -> _ForwardBackwardResultStagerLease:
        with self._condition:
            while not self._available:
                self._condition.wait()
            stager = self._available.pop()
            stager.reset()
            return _ForwardBackwardResultStagerLease(self, stager)

    def release(self, stager: PinnedCpuSnapshotStager) -> None:
        with self._condition:
            self._available.append(stager)
            self._condition.notify()


class ForwardBackwardRankLaunch(BaseModel):
    """Gradient-ready rank result whose host serialization is still pending."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    operation_id: str
    learner_version: int
    batch: SkipValidation[PackedBatchRef]
    candidate_capacity: int | None
    coordinator: bool
    return_token_logprobs: bool
    token_count: int
    telemetry: SkipValidation[PendingRankCommandTelemetry]
    snapshot: SkipValidation[PendingCpuSnapshot[dict[str, Any]]]
    staging: SkipValidation[_ForwardBackwardResultStagerLease]
    _materialize_lock: Lock = PrivateAttr(default_factory=Lock)
    _materialized: dict[str, Any] | None = PrivateAttr(default=None)
    _materialization_error: BaseException | None = PrivateAttr(default=None)
    _finished: bool = PrivateAttr(default=False)

    def materialize(self) -> dict[str, Any]:
        with self._materialize_lock:
            if self._finished:
                if self._materialization_error is not None:
                    raise self._materialization_error
                assert self._materialized is not None
                return self._materialized
            try:
                staged = self.snapshot.resolve()
                materialized = {
                    "operation_id": self.operation_id,
                    "learner_version": self.learner_version,
                    "token_count": self.token_count,
                    "metrics": {},
                    "_rank_telemetry": materialize_rank_telemetry(
                        self.telemetry, staged["statistics"]
                    ),
                    "token_logprobs": (
                        _materialize_command_token_logprobs(
                            self.batch,
                            self.candidate_capacity,
                            tuple(staged["token_logprobs"]),
                        )
                        if self.coordinator and self.return_token_logprobs
                        else ()
                    ),
                }
                self._materialized = materialized
                return materialized
            except BaseException as error:
                self._materialization_error = error
                raise
            finally:
                self._finished = True
                self.staging.release()


class ForwardRankLaunch(BaseModel):
    """GPU-complete rank result whose host serialization is still pending."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    operation_id: str
    learner_version: int
    batch: SkipValidation[PackedBatchRef]
    candidate_capacity: int | None
    coordinator: bool
    telemetry: SkipValidation[PendingRankCommandTelemetry | None]
    snapshot: SkipValidation[PendingCpuSnapshot[dict[str, Any]]]
    staging: SkipValidation[_ForwardBackwardResultStagerLease]
    base_metrics: dict[str, float]
    _materialize_lock: Lock = PrivateAttr(default_factory=Lock)
    _materialized: dict[str, Any] | None = PrivateAttr(default=None)
    _materialization_error: BaseException | None = PrivateAttr(default=None)
    _finished: bool = PrivateAttr(default=False)

    def materialize(self) -> dict[str, Any]:
        with self._materialize_lock:
            if self._finished:
                if self._materialization_error is not None:
                    raise self._materialization_error
                assert self._materialized is not None
                return self._materialized
            try:
                staged = self.snapshot.resolve()
                self._materialized = {
                    "operation_id": self.operation_id,
                    "learner_version": self.learner_version,
                    "metrics": (
                        self.base_metrics
                        if self.coordinator and self.telemetry is None
                        else {}
                    ),
                    "_rank_telemetry": (
                        None
                        if self.telemetry is None
                        else materialize_rank_telemetry(
                            self.telemetry, staged["statistics"]
                        )
                    ),
                    "token_logprobs": (
                        _materialize_command_token_logprobs(
                            self.batch,
                            self.candidate_capacity,
                            tuple(staged["token_logprobs"]),
                        )
                        if self.coordinator
                        else ()
                    ),
                }
                return self._materialized
            except BaseException as error:
                self._materialization_error = error
                raise
            finally:
                self._finished = True
                self.staging.release()


class SftRankLaunch(BaseModel):
    """SFT GPU-complete result whose CPU conversion is still pending."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    operation_id: str
    learner_version: int
    coordinator: bool
    token_count: int
    telemetry: SkipValidation[PendingRankCommandTelemetry]
    logprob_lengths: tuple[int, ...]
    snapshot: SkipValidation[PendingCpuSnapshot[dict[str, Any]]]
    staging: SkipValidation[_ForwardBackwardResultStagerLease]
    _materialize_lock: Lock = PrivateAttr(default_factory=Lock)
    _materialized: dict[str, Any] | None = PrivateAttr(default=None)
    _materialization_error: BaseException | None = PrivateAttr(default=None)
    _finished: bool = PrivateAttr(default=False)

    def materialize(self) -> dict[str, Any]:
        with self._materialize_lock:
            if self._finished:
                if self._materialization_error is not None:
                    raise self._materialization_error
                assert self._materialized is not None
                return self._materialized
            try:
                staged = self.snapshot.resolve()
                token_logprobs = ()
                if self.coordinator and "logprob_values" in staged:
                    present = staged["logprob_present"]
                    if not bool(torch.all(present == 1)):
                        raise RuntimeError(
                            "SFT forward did not materialize every trajectory"
                        )
                    values = staged["logprob_values"]
                    token_logprobs = tuple(
                        tuple(float(value) for value in values[index, :length].tolist())
                        for index, length in enumerate(self.logprob_lengths)
                    )
                self._materialized = {
                    "operation_id": self.operation_id,
                    "learner_version": self.learner_version,
                    "token_count": self.token_count,
                    "metrics": {},
                    "_rank_telemetry": materialize_rank_telemetry(
                        self.telemetry, staged["statistics"]
                    ),
                    "token_logprobs": token_logprobs,
                }
                return self._materialized
            except BaseException as error:
                self._materialization_error = error
                raise
            finally:
                self._finished = True
                self.staging.release()


def _stage_forward_rank_result(
    pool: _ForwardBackwardResultStagerPool,
    job: ForwardJobSpec,
    batch: InMemoryPackedBatch,
    result: dict[str, Any],
    *,
    coordinator: bool,
) -> ForwardRankLaunch:
    staging = pool.acquire()
    try:
        outputs = (
            tuple(result["token_logprobs"])
            if coordinator and job.return_token_logprobs
            else ()
        )
        telemetry = result.get("telemetry")
        if telemetry is None:
            base_metrics = cast(dict[str, float], result["metrics"])
        else:
            base_metrics = {}
        builder = staging.stager.begin()
        target_tokens = (
            batch.tensors.get("target_tokens")
            if batch.ref.training_kind == "tokenized"
            else None
        )
        return ForwardRankLaunch(
            operation_id=job.operation_id,
            learner_version=job.expected_learner_version,
            batch=batch.ref,
            candidate_capacity=(
                None if target_tokens is None else int(target_tokens.shape[2])
            ),
            coordinator=coordinator,
            telemetry=telemetry,
            snapshot=builder.finish(
                {
                    "token_logprobs": builder.stage_group(outputs),
                    **(
                        {"statistics": builder.stage(telemetry.statistics)}
                        if telemetry is not None
                        else {}
                    ),
                }
            ),
            staging=staging,
            base_metrics=base_metrics,
        )
    except BaseException:
        staging.release()
        raise


def _stage_forward_backward_rank_result(
    pool: _ForwardBackwardResultStagerPool,
    job: ForwardBackwardJobSpec,
    batch: InMemoryPackedBatch,
    result: Any,
    *,
    coordinator: bool,
) -> ForwardBackwardRankLaunch:
    staging = pool.acquire()
    try:
        builder = staging.stager.begin()
        # These are command-result allocations, not mutable parameter, optimizer,
        # or accumulator storage. The builder retains and record_streams CUDA
        # sources until its side-stream copy completes.
        staged: dict[str, Any] = {
            "statistics": builder.stage(result.telemetry.statistics),
            "token_logprobs": (
                builder.stage_group(result.new_logprobs)
                if coordinator and job.return_token_logprobs
                else ()
            ),
        }
        target_tokens = (
            batch.tensors.get("target_tokens")
            if batch.ref.training_kind == "tokenized"
            else None
        )
        return ForwardBackwardRankLaunch(
            operation_id=job.operation_id,
            learner_version=job.expected_learner_version,
            batch=batch.ref,
            candidate_capacity=(
                None if target_tokens is None else int(target_tokens.shape[2])
            ),
            coordinator=coordinator,
            return_token_logprobs=job.return_token_logprobs,
            token_count=job.trainable_token_count,
            telemetry=result.telemetry,
            snapshot=builder.finish(staged),
            staging=staging,
        )
    except BaseException:
        staging.release()
        raise


def _stage_sft_rank_result(
    pool: _ForwardBackwardResultStagerPool,
    job: SftForwardBackwardJobSpec | SftForwardJobSpec,
    result: dict[str, Any],
    *,
    coordinator: bool,
) -> SftRankLaunch:
    staging = pool.acquire()
    try:
        builder = staging.stager.begin()
        telemetry = cast(PendingRankCommandTelemetry, result["telemetry"])
        staged: dict[str, Any] = {
            "statistics": builder.stage(telemetry.statistics)
        }
        if coordinator and job.return_token_logprobs:
            values = result["logprob_values"]
            present = result["logprob_present"]
            if values is not None:
                staged["logprob_values"] = builder.stage(values)
                staged["logprob_present"] = builder.stage(present)
        return SftRankLaunch(
            operation_id=job.operation_id,
            learner_version=job.expected_learner_version,
            coordinator=coordinator,
            token_count=job.trainable_token_count,
            telemetry=telemetry,
            logprob_lengths=tuple(result["logprob_lengths"]),
            snapshot=builder.finish(staged),
            staging=staging,
        )
    except BaseException:
        staging.release()
        raise


def _packed_logprobs(values: torch.Tensor, shape: tuple[int, ...]) -> dict[str, Any]:
    host = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return TokenLogprobs(
        shape=shape,
        data=host.numpy().tobytes(order="C"),
    ).model_dump(mode="python")


class MegatronTrainJobExecutor:
    """Thin adapter around the warm runtime's in-memory job entrypoint."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._publisher = _GenerationPublisher(
            runtime,
            capacity=int(runtime.snapshot_pool_capacity),
        )
        self._gradients = GradientAccumulator(model_chunks=runtime.model)
        self._command_result_stagers = _ForwardBackwardResultStagerPool(
            int(runtime.snapshot_pool_capacity)
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
        return _stage_forward_backward_rank_result(
            self._command_result_stagers,
            job,
            batch,
            result,
            coordinator=int(self.runtime.rank) == 0,
        ).materialize()

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
        return _stage_forward_rank_result(
            self._command_result_stagers,
            job,
            batch,
            result,
            coordinator=int(self.runtime.rank) == 0,
        ).materialize()

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
        return _stage_sft_rank_result(
            self._command_result_stagers,
            job,
            result,
            coordinator=int(self.runtime.rank) == 0,
        ).materialize()

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

        result = execute_megatron_sft_forward_job(
            self.runtime,
            job,
            batch,
            cancelled=cancelled,
        )
        return _stage_sft_rank_result(
            self._command_result_stagers,
            job,
            result,
            coordinator=int(self.runtime.rank) == 0,
        ).materialize()

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
        from art.megatron.training.finalize_grads import (
            finalize_accumulated_model_grads,
        )

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
        optimizer_step_s = time.perf_counter() - started
        consumed = self._gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_policy_step = job.learner_version
        runtime.resident_generation_id = job.generation.generation_id
        runtime.optimizer_state_loaded = True
        self._gradient_parent_version = None
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
        ordered_target = _ordered_sampler_target(job)
        if ordered_target is not None:
            rank_plan, metrics = self._publisher.prepare_ordered_sampler(
                operation_id=job.operation_id,
                run_id=job.run_id,
                generation=job.generation,
                optimizer_state_path=job.optimizer_state_path,
                target=ordered_target,
                adapter_dtypes=runtime.adapter_export_dtypes,
                adapter_config=runtime.adapter_export_config,
                slot_ref=None,
                sink=sink,
            )
            return {
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
                "rank_write_plan": rank_plan.model_dump(mode="json"),
                "metrics": metrics,
            }
        stage_metrics = self._publisher.ensure_generation(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes=runtime.adapter_export_dtypes,
            adapter_config=runtime.adapter_export_config,
            snapshot_optimizer=job.save_optimizer,
        )
        rank_plan, prepare_metrics = self._publisher.prepare(
            operation_id=job.operation_id,
            generation=job.generation,
            optimizer_state_path=job.optimizer_state_path,
            staging_adapter_path=job.staging_adapter_path,
            existing_adapter=job.existing_adapter,
            publication_targets=job.publication_targets,
            adapter_object_target=job.adapter_object_target,
            save_optimizer=job.save_optimizer,
            sink=sink,
        )
        metrics = {
            **stage_metrics,
            **prepare_metrics,
        }
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "rank_write_plan": rank_plan.model_dump(mode="json"),
            "metrics": metrics,
        }

    def authorize_snapshot(
        self, plan: SnapshotWritePlan, grant: SnapshotWriteGrant
    ) -> dict[str, float]:
        return self._publisher.authorize(
            operation_id=plan.operation_id,
            plan=plan,
            grant=grant,
        )

    def discard_prepared_snapshot(self, operation_id: str) -> None:
        self._publisher.discard(operation_id)

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
    weights_key: ResidencyKey
    optimizer_key: ResidencyKey | None
    checkpoint: Any
    optimizer: Any | None
    adapter_config: dict[str, Any]
    optimizer_restored: bool = False

    @property
    def weights(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.checkpoint.parameters)

    @property
    def optimizer_tensors(self) -> tuple[torch.Tensor, ...]:
        return () if self.optimizer is None else tuple(self.optimizer.tensors)


class _PreparedRunRegistration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    registration: RunSlotRegistration
    load: _PreparedRunLoad


class GenerationResidency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: ResidencyKey
    optimizer: ResidencyKey | None = None
    accumulator: ResidencyKey | None = None


class _ResidentRunState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tenant_id: str
    run_id: str
    training_session_id: str
    learner_version: int
    adapter_config: dict[str, Any]
    gradients: Any | None
    desired: GenerationResidency
    installed_weights: ResidencyKey | None
    installed_optimizer: ResidencyKey | None = None
    checkpoint_slot_installed: bool = False
    initial_generation: TrainerGeneration | None = None
    pending_load: _PreparedRunLoad | None = None
    next_accumulator_revision: int = 1
    residency_revision: int = 0
    registration_complete: bool = False
    unregistering: bool = False


@contextmanager
def _residency_mutation(state: _ResidentRunState) -> Iterator[None]:
    if state.residency_revision % 2:
        raise RuntimeError("nested run residency mutation")
    state.residency_revision += 1
    try:
        yield
    finally:
        state.residency_revision += 1


class MCoreRunSlotExecutor:
    """Train independent exact-shape LoRAs on one warm MCore runtime."""

    def __init__(self, runtime: Any) -> None:
        from art.trainer_rank import TrainerRank

        self.runtime = runtime
        self._slot_trainer = TrainerRank(runtime)
        if runtime.run_residency_config is None:
            raise RuntimeError("multi-run Megatron requires explicit residency limits")
        if runtime.optimizer_layout_fingerprint is None:
            raise RuntimeError("multi-run Megatron has no topology fingerprint")
        self._residency = RunResidencyManager(
            runtime.run_residency_config,
            snapshot_barrier=runtime.optimizer_snapshot_barrier,
        )
        self._publisher = _GenerationPublisher(
            runtime,
            capacity=int(runtime.snapshot_pool_capacity),
            residency=self._residency,
        )
        self._command_result_stagers = _ForwardBackwardResultStagerPool(
            int(runtime.snapshot_pool_capacity)
        )
        self._load_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-load-prepare"
        )
        self._cleanup_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="art-run-cleanup"
        )
        self._load_preparations: dict[
            str, tuple[str, str, Future[_PreparedRunLoad]]
        ] = {}
        self._registration_preparations: dict[
            str, tuple[str, Future[_PreparedRunRegistration]]
        ] = {}
        self._run_cleanups: dict[str, Future[None]] = {}
        self._runs: dict[str, _ResidentRunState] = {}
        self._closed = False

    def register_run(self, registration: RunSlotRegistration) -> None:
        self.commit_run_registration(self.prepare_run_registration(registration))

    def prepare_run_registration(
        self, registration: RunSlotRegistration
    ) -> _PreparedRunRegistration:
        if self._closed:
            raise RuntimeError("Megatron run slot is closed")
        if registration.run_id in self._runs:
            raise RuntimeError(
                f"training run is already resident: {registration.run_id!r}"
            )
        from art.megatron.model_support.lora_disk import load_adapter_config
        from art.megatron.optimizer_state import load_trainer_rank_optimizer_state
        from art.trainer_rank import MaterializedCheckpoint

        adapter_config = load_adapter_config(registration.adapter.identity)
        adapter_layout_fingerprint = hashlib.sha256(
            encode_adapter_config(adapter_config)
        ).hexdigest()
        checkpoint = self._slot_trainer.prepare_checkpoint_slot_load_sync(
            MaterializedCheckpoint(
                path=registration.run_id, directory=registration.adapter.identity
            ),
            device="cpu",
        )
        if registration.initial_optimizer_state_path is not None:
            assert registration.initial_optimizer_generation_id is not None
            loaded_optimizer = load_trainer_rank_optimizer_state(
                self.runtime,
                optimizer_state_path=registration.initial_optimizer_state_path,
                adapter_path=registration.adapter.identity,
                adapter_step=registration.adapter.step,
                optimizer_generation_id=registration.initial_optimizer_generation_id,
                layout=self.optimizer_layout(checkpoint),
            )
            optimizer = (
                self._slot_trainer.prepare_checkpoint_slot_optimizer_for_residency(
                    registration.run_id, checkpoint, loaded_optimizer
                )
            )
        else:
            optimizer = self._slot_trainer.prepare_fresh_checkpoint_slot_optimizer_for_residency(
                checkpoint
            )
        weights_key = ResidencyKey(
            tenant_id=registration.tenant_id,
            run_id=registration.run_id,
            generation_id=registration.generation_id,
            topology_fingerprint=self.runtime.optimizer_layout_fingerprint,
            adapter_layout_fingerprint=adapter_layout_fingerprint,
        )
        optimizer_key = weights_key.model_copy(update={"representation": "optimizer"})
        prepared = _PreparedRunLoad(
            operation_id=f"register:{registration.generation_id}",
            weights_key=weights_key,
            optimizer_key=optimizer_key,
            checkpoint=checkpoint,
            optimizer=optimizer,
            optimizer_restored=(registration.initial_optimizer_state_path is not None),
            adapter_config=adapter_config,
        )
        tensors = (*prepared.weights, *prepared.optimizer_tensors)
        if (
            not prepared.weights
            or not prepared.optimizer_tensors
            or any(tensor.device.type != "cpu" for tensor in tensors)
        ):
            raise RuntimeError(
                "prepared initial working set is not entirely CPU resident"
            )
        return _PreparedRunRegistration(registration=registration, load=prepared)

    def start_prepare_run_registration(
        self, registration: RunSlotRegistration
    ) -> Future[_PreparedRunRegistration]:
        fingerprint = registration.model_dump_json()
        existing = self._registration_preparations.get(registration.run_id)
        if existing is not None:
            if existing[0] != fingerprint:
                raise RuntimeError("run_id was reused for another registration")
            return existing[1]
        if registration.run_id in self._runs:
            raise RuntimeError(
                f"training run is already resident: {registration.run_id!r}"
            )
        self._registration_preparations[registration.run_id] = (
            fingerprint,
            self._load_pool.submit(self.prepare_run_registration, registration),
        )
        return self._registration_preparations[registration.run_id][1]

    def run_registration_prepared(self, registration: RunSlotRegistration) -> bool:
        try:
            fingerprint, future = self._registration_preparations[registration.run_id]
        except KeyError as error:
            raise RuntimeError("run registration preparation was not started") from error
        if fingerprint != registration.model_dump_json():
            raise RuntimeError("run registration preparation identity changed")
        if future.done():
            future.result()
            return True
        return False

    def finish_prepared_run_registration(
        self, registration: RunSlotRegistration
    ) -> None:
        if not self.run_registration_prepared(registration):
            raise RuntimeError("run registration preparation is not complete")
        _fingerprint, future = self._registration_preparations.pop(
            registration.run_id
        )
        self.commit_run_registration(future.result())

    def discard_prepared_run_registration(self, run_id: str) -> None:
        preparation = self._registration_preparations.pop(run_id, None)
        if preparation is None:
            return
        future = preparation[1]
        if not future.cancel():
            future.add_done_callback(_consume_future)

    def commit_run_registration(self, value: _PreparedRunRegistration) -> None:
        registration, prepared = value.registration, value.load
        run_id = registration.run_id
        if self._closed:
            raise RuntimeError("Megatron run slot is closed")
        if run_id in self._runs:
            raise RuntimeError(f"training run is already resident: {run_id!r}")
        weights_key = prepared.weights_key
        optimizer_key = prepared.optimizer_key
        if optimizer_key is None:
            raise RuntimeError("initial run registration has no optimizer state")
        state = _ResidentRunState(
            tenant_id=registration.tenant_id,
            run_id=run_id,
            training_session_id=registration.training_session_id,
            learner_version=registration.learner_version,
            adapter_config=prepared.adapter_config,
            gradients=None,
            desired=GenerationResidency(weights=weights_key, optimizer=optimizer_key),
            installed_weights=None,
            initial_generation=TrainerGeneration(
                training_session_id=registration.training_session_id,
                policy_step=registration.learner_version,
                generation_id=registration.generation_id,
                adapter_path=registration.adapter.identity,
            ),
            pending_load=prepared,
        )
        self._residency.register_l2_working_set(
            (
                (weights_key, prepared.weights),
                (optimizer_key, prepared.optimizer_tensors),
            )
        )
        try:
            self._runs[run_id] = state
        except BaseException as error:
            failures: list[BaseException] = [error]
            for key in (weights_key, optimizer_key):
                try:
                    self._residency.retire(key)
                except BaseException as cleanup_error:
                    failures.append(cleanup_error)
            if len(failures) > 1:
                raise BaseExceptionGroup(
                    "run admission and cleanup failed", failures
                ) from error
            raise

    def complete_run_registration(self, run_id: str) -> None:
        state = self._require_run(run_id, require_complete=False)
        if state.registration_complete:
            return
        try:
            generation = state.initial_generation
            if generation is None:
                raise RuntimeError(
                    "run registration has no immutable initial generation"
                )
            pending = state.pending_load
            if (
                pending is None
                or pending.optimizer is None
                or pending.optimizer_key is None
            ):
                raise RuntimeError("run registration has no prepared L2 working set")
            self._publisher.register_existing(
                run_id=run_id,
                generation=generation,
                optimizer_source=pending.optimizer.snapshot_source(),
                optimizer_residency_key=pending.optimizer_key,
            )
        except BaseException as error:
            try:
                self.unregister_run(run_id)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "run registration and cleanup failed", [error, cleanup_error]
                ) from error
            raise
        state.initial_generation = None
        state.registration_complete = True

    def optimizer_layout(self, checkpoint: Any) -> Any:
        return self._slot_trainer.prepared_checkpoint_slot_optimizer_layout(checkpoint)

    def prepare_residency(
        self,
        run_id: str,
        command_kind: str,
        expected_learner_version: int,
    ) -> bool:
        """Launch component transfers before the serialized GPU command turn."""
        state = self._require_run(run_id)
        revision = state.residency_revision
        desired = state.desired
        if (
            revision % 2
            or revision != state.residency_revision
            or state.learner_version != expected_learner_version
        ):
            return False
        if command_kind in {"forward", "forward_backward"}:
            keys = (desired.weights,)
            if (
                command_kind == "forward_backward"
                and desired.accumulator is not None
            ):
                keys += (desired.accumulator,)
        elif command_kind == "optim_step":
            keys = tuple(
                key
                for key in (
                    desired.weights,
                    desired.optimizer,
                    desired.accumulator,
                )
                if key is not None
            )
        else:
            raise ValueError(f"unsupported residency prefetch command {command_kind!r}")
        try:
            self._residency.prefetch_l1_working_set(keys)
        except ResidencyCapacityUnavailable:
            return False
        except RuntimeError:
            if revision != state.residency_revision:
                return False
            raise
        return (
            revision == state.residency_revision
            and state.learner_version == expected_learner_version
        )

    def execute_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        return self.start_forward_backward(
            job,
            batch,
            cancelled,
            coordinator=int(self.runtime.rank) == 0,
        ).materialize()

    def start_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
        *,
        coordinator: bool,
    ) -> ForwardBackwardRankLaunch:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        validate_packed_batch(batch)
        from art.megatron.train import (
            execute_megatron_dynamic_lora_forward_backward_job,
        )

        with self._resident(state):
            gradients = self._require_gradients(state)
            result = execute_megatron_dynamic_lora_forward_backward_job(
                self.runtime,
                job,
                batch.tensors,
                slot_trainer=self._slot_trainer,
                gradient_accumulator=gradients,
                accumulator_residency=self._accumulator_resident(state),
                cancelled=cancelled,
            )
            self._register_gradient_contribution(state, job.operation_id)
        return _stage_forward_backward_rank_result(
            self._command_result_stagers,
            job,
            batch,
            result,
            coordinator=coordinator,
        )

    def execute_forward(
        self,
        job: ForwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        return self.start_forward(
            job,
            batch,
            cancelled,
            coordinator=int(self.runtime.rank) == 0,
        ).materialize()

    def start_forward(
        self,
        job: ForwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
        *,
        coordinator: bool,
    ) -> ForwardRankLaunch:
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
        return _stage_forward_rank_result(
            self._command_result_stagers,
            job,
            batch,
            result,
            coordinator=coordinator,
        )

    def execute_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        return self.start_sft_forward_backward(
            job, batch, cancelled, coordinator=int(self.runtime.rank) == 0
        ).materialize()

    def start_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
        *,
        coordinator: bool,
    ) -> SftRankLaunch:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        from art.megatron.train import (
            execute_megatron_dynamic_lora_sft_forward_backward_job,
        )

        with self._resident(state):
            gradients = self._require_gradients(state)
            result = execute_megatron_dynamic_lora_sft_forward_backward_job(
                self.runtime,
                job,
                batch,
                slot_trainer=self._slot_trainer,
                gradient_accumulator=gradients,
                accumulator_residency=self._accumulator_resident(state),
                cancelled=cancelled,
            )
            self._register_gradient_contribution(state, job.operation_id)
        return _stage_sft_rank_result(
            self._command_result_stagers, job, result, coordinator=coordinator
        )

    def execute_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        return self.start_sft_forward(
            job, batch, cancelled, coordinator=self.runtime.rank == 0
        ).materialize()

    def start_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
        *,
        coordinator: bool,
    ) -> SftRankLaunch:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_dynamic_lora_sft_forward_job

        with self._resident(state):
            result = execute_megatron_dynamic_lora_sft_forward_job(
                self.runtime,
                job,
                batch,
                cancelled=cancelled,
            )
        return _stage_sft_rank_result(
            self._command_result_stagers, job, result, coordinator=coordinator
        )

    def execute_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        self._publisher.raise_if_failed()
        from art.trainer_rank import AdamParams

        with self._resident(
            state, include_optimizer=True, include_accumulator=True
        ) as working_set:
            self._residency.wait_before_mutation_working_set(working_set)
            self.runtime.optimizer_snapshot_barrier.wait_before_mutation(key=job.run_id)
            gradients = self._require_gradients(state)
            gradients.seal(job.contributing_forward_backward_operation_ids)
            accumulated = gradients.prepare_optimizer()
            expected_tokens = accumulated.expected_global_token_count
            assert expected_tokens is not None
            from megatron.core import parallel_state as ps

            from art.megatron.training.finalize_grads import (
                finalize_model_grads_extended,
                reduce_accumulated_token_count,
            )

            global_tokens = reduce_accumulated_token_count(
                accumulated.local_token_count,
                expected_global_token_count=expected_tokens,
                group=ps.get_data_parallel_group(with_context_parallel=True),
            )
            reduced_gradients = (
                self._slot_trainer.reduce_checkpoint_slot_gradient_sums(
                    job.run_id, accumulated.gradients
                )
            )
            if accumulated.reduction == "token_mean":
                torch._foreach_div_(reduced_gradients, global_tokens)
            finalize_model_grads_extended(self.runtime.model, num_tokens=None)
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
                grads=reduced_gradients,
            )
            residency_tensors = self._slot_trainer.checkpoint_slot_residency_tensors(
                job.run_id
            )
            optimizer_source = (
                self._slot_trainer.checkpoint_slot_optimizer_residency_source(
                    job.run_id
                )
            )
            if optimizer_source is None:
                raise RuntimeError("optimizer commit has no immutable optimizer state")
            from art.megatron.lora import LoRASlotRef
            from art.megatron.weights.lora_publish import (
                build_local_lora_export_plan,
            )

            export_plan = build_local_lora_export_plan(
                self.runtime.model,
                residency_tensors.weights,
                {},
                packed_expert_groups=(
                    self.runtime.model_support_handler.expert_packed_lora_groups()
                ),
                slot_ref=LoRASlotRef("checkpoint", job.run_id),
            )
        if not result["update_successful"] or not math.isfinite(result["grad_norm"]):
            raise RuntimeError("dynamic LoRA optimizer rejected the update")
        optimizer_step_s = time.perf_counter() - started
        consumed = gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
        parent_weights = state.installed_weights
        if parent_weights is None:
            raise RuntimeError("optimizer has no installed weight generation")
        output_weights = parent_weights.model_copy(
            update={"generation_id": job.generation.generation_id}
        )
        self._residency.advance_l1(
            parent_weights,
            output_weights,
            residency_tensors.weights,
            retire_source=True,
        )
        output_optimizer = output_weights.model_copy(
            update={"representation": "optimizer"}
        )
        if state.installed_optimizer is None:
            self._residency.register_l1(output_optimizer, residency_tensors.optimizer)
        else:
            self._residency.advance_l1(
                state.installed_optimizer,
                output_optimizer,
                residency_tensors.optimizer,
                retire_source=True,
            )
        accumulator = state.desired.accumulator
        with _residency_mutation(state):
            state.desired = GenerationResidency(
                weights=output_weights,
                optimizer=output_optimizer,
            )
            state.installed_weights = output_weights
            state.installed_optimizer = output_optimizer
            state.learner_version = job.learner_version
        if accumulator is not None:
            self._residency.retire_async(accumulator)
        snapshot_metrics = self._publisher.register_resident_generation(
            run_id=job.run_id,
            generation=job.generation,
            weights_key=output_weights,
            export_plan=export_plan,
            adapter_config=state.adapter_config,
            optimizer_source=optimizer_source,
            optimizer_key=output_optimizer,
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
        if job.optimizer_state_path is not None:
            assert job.optimizer_generation_id is not None
            loaded = load_trainer_rank_optimizer_state(
                self.runtime,
                optimizer_state_path=job.optimizer_state_path,
                adapter_path=job.adapter_path,
                adapter_step=job.adapter_step,
                optimizer_generation_id=job.optimizer_generation_id,
                layout=self.optimizer_layout(checkpoint),
            )
            optimizer = (
                self._slot_trainer.prepare_checkpoint_slot_optimizer_for_residency(
                    job.run_id, checkpoint, loaded
                )
            )
        else:
            optimizer = self._slot_trainer.prepare_fresh_checkpoint_slot_optimizer_for_residency(
                checkpoint
            )
        adapter_layout_fingerprint = hashlib.sha256(
            encode_adapter_config(config)
        ).hexdigest()
        weights_key = ResidencyKey(
            tenant_id=state.tenant_id,
            run_id=job.run_id,
            generation_id=job.generation.generation_id,
            topology_fingerprint=self.runtime.optimizer_layout_fingerprint,
            adapter_layout_fingerprint=adapter_layout_fingerprint,
        )
        prepared = _PreparedRunLoad(
            operation_id=job.operation_id,
            weights_key=weights_key,
            optimizer_key=weights_key.model_copy(
                update={"representation": "optimizer"}
            ),
            checkpoint=checkpoint,
            optimizer=optimizer,
            optimizer_restored=job.restore_optimizer,
            adapter_config=config,
        )
        tensors = (*prepared.weights, *prepared.optimizer_tensors)
        if (
            not prepared.weights
            or not prepared.optimizer_tensors
            or any(tensor.device.type != "cpu" for tensor in tensors)
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
        if state.gradients is not None and state.gradients.contribution_ids:
            raise RuntimeError("load_state cannot discard open gradient contributions")
        if (
            prepared.operation_id != job.operation_id
            or prepared.weights_key.generation_id != job.generation.generation_id
            or prepared.optimizer is None
            or prepared.optimizer_key is None
            or prepared.optimizer_restored != job.restore_optimizer
        ):
            raise RuntimeError("prepared load does not match its ordered command")
        previous = state.pending_load
        optimizer_key = prepared.optimizer_key
        weights_image, optimizer_image = self._residency.register_l2_working_set(
            (
                (prepared.weights_key, prepared.weights),
                (optimizer_key, prepared.optimizer_tensors),
            )
        )
        adapter = read_adapter_publication(
            job.generation.adapter_path,
            step=job.generation.policy_step,
        )
        if adapter is None or (
            adapter.training_session_id != job.generation.training_session_id
            or adapter.generation_id != job.generation.generation_id
        ):
            self._residency.retire(prepared.weights_key)
            self._residency.retire(optimizer_key)
            raise RuntimeError("loaded generation has no matching immutable adapter")
        try:
            snapshot_metrics = self._publisher.register_existing(
                run_id=job.run_id,
                generation=job.generation,
                optimizer_source=prepared.optimizer.snapshot_source(),
                optimizer_residency_key=optimizer_key,
            )
        except BaseException:
            self._residency.retire(prepared.weights_key)
            self._residency.retire(optimizer_key)
            raise
        if previous is not None:
            if previous.weights_key != state.installed_weights:
                self._residency.retire_async(previous.weights_key)
            if (
                previous.optimizer_key is not None
                and previous.optimizer_key != state.installed_optimizer
            ):
                self._residency.retire_async(previous.optimizer_key)
        with _residency_mutation(state):
            state.desired = GenerationResidency(
                weights=prepared.weights_key,
                optimizer=optimizer_key,
            )
            state.pending_load = prepared
            state.learner_version = job.learner_version
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "optimizer_restored": job.restore_optimizer,
            "metrics": {
                "residency/load_l2_bytes": float(
                    weights_image.stats.byte_count + optimizer_image.stats.byte_count
                ),
                **snapshot_metrics,
            },
        }

    def execute_snapshot(
        self,
        job: GenerationSnapshotJobSpec,
        sink: EventSink,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(state, job.training_session_id, job.learner_version)
        if state.desired.weights.generation_id != job.generation.generation_id:
            raise RuntimeError("snapshot does not identify the selected generation")
        ordered_target = _ordered_sampler_target(job)
        if ordered_target is not None:
            from art.megatron.lora import LoRASlotRef

            rank_plan, metrics = self._publisher.prepare_ordered_sampler(
                operation_id=job.operation_id,
                run_id=job.run_id,
                generation=job.generation,
                optimizer_state_path=job.optimizer_state_path,
                target=ordered_target,
                adapter_dtypes={},
                adapter_config=state.adapter_config,
                slot_ref=LoRASlotRef("checkpoint", job.run_id),
                sink=sink,
            )
            return {
                "operation_id": job.operation_id,
                "learner_version": job.learner_version,
                "rank_write_plan": rank_plan.model_dump(mode="json"),
                "metrics": metrics,
            }
        if not self._publisher.has_generation(job.generation):
            raise RuntimeError("selected generation has no immutable weights snapshot")
        if job.save_optimizer and not self._publisher.has_generation(
            job.generation, require_optimizer=True
        ):
            raise RuntimeError(
                "selected generation has no immutable optimizer snapshot"
            )
        rank_plan, metrics = self._publisher.prepare(
            operation_id=job.operation_id,
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
            "rank_write_plan": rank_plan.model_dump(mode="json"),
            "metrics": metrics,
        }

    def authorize_snapshot(
        self, plan: SnapshotWritePlan, grant: SnapshotWriteGrant
    ) -> dict[str, float]:
        return self._publisher.authorize(
            operation_id=plan.operation_id,
            plan=plan,
            grant=grant,
        )

    def discard_prepared_snapshot(self, operation_id: str) -> None:
        self._publisher.discard(operation_id)

    def discard_run_gradients(self, run_id: str) -> None:
        state = self._require_run(run_id)
        if state.gradients is not None:
            state.gradients.discard()
        self._retire_accumulator(state)

    def unregister_run(self, run_id: str) -> None:
        self.start_unregister_run(run_id)
        self.finish_unregister_run(run_id)

    def start_unregister_run(self, run_id: str) -> None:
        if run_id in self._run_cleanups:
            return
        state = self._require_run(
            run_id, require_complete=False, allow_unregistering=True
        )
        state.unregistering = True
        failures: list[BaseException] = []
        for operation_id, (prepared_run_id, _fingerprint, _future) in tuple(
            self._load_preparations.items()
        ):
            if prepared_run_id == run_id:
                try:
                    self.discard_prepared_load_state(operation_id)
                except BaseException as error:
                    failures.append(error)
        if state.gradients is not None:
            try:
                state.gradients.discard()
            except BaseException as error:
                failures.append(error)
            else:
                state.gradients = None
        try:
            if state.checkpoint_slot_installed or state.installed_weights is not None:
                self._slot_trainer.unload_checkpoint_slot(run_id)
                state.checkpoint_slot_installed = False
                state.installed_weights = None
                state.installed_optimizer = None
                state.gradients = None
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Megatron run detach failed", failures)
        self._run_cleanups[run_id] = self._cleanup_pool.submit(
            self._finish_run_cleanup, run_id
        )

    def finish_unregister_run(self, run_id: str) -> None:
        try:
            cleanup = self._run_cleanups[run_id]
        except KeyError as error:
            raise RuntimeError("Megatron run cleanup was not started") from error
        try:
            cleanup.result()
        finally:
            self._run_cleanups.pop(run_id)

    def _finish_run_cleanup(self, run_id: str) -> None:
        failures: list[BaseException] = []
        try:
            self._publisher.retire_run(run_id)
        except BaseException as error:
            failures.append(error)
        retirements = tuple(
            self._residency.retire_async(key) for key in self._residency.keys(run_id)
        )
        _done, pending = wait(
            retirements, timeout=self._residency.config.shutdown_timeout_s
        )
        failures.extend(
            error
            for retirement in retirements
            if retirement.done() and (error := retirement.exception()) is not None
        )
        if pending:
            failures.append(
                TimeoutError(f"{len(pending)} run residency retirements timed out")
            )
        if not failures and self._residency.keys(run_id):
            failures.append(RuntimeError("Megatron run cleanup left resident resources"))
        if failures:
            if len(failures) == 1:
                raise failures[0]
            raise BaseExceptionGroup("Megatron run cleanup failed", failures)
        self._runs.pop(run_id)

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        for state in self._runs.values():
            if state.gradients is not None:
                try:
                    state.gradients.discard()
                except BaseException as error:
                    failures.append(error)
        futures = (
            *(value[2] for value in self._load_preparations.values()),
            *(value[1] for value in self._registration_preparations.values()),
            *self._run_cleanups.values(),
        )
        for future in futures:
            future.cancel()
        _done, pending = wait(
            futures, timeout=self._residency.config.shutdown_timeout_s
        )
        self._load_pool.shutdown(wait=False, cancel_futures=True)
        self._cleanup_pool.shutdown(wait=False, cancel_futures=True)
        try:
            self._publisher.close()
        except BaseException as error:
            failures.append(error)
        try:
            self._residency.close()
        except BaseException as error:
            failures.append(error)
        self._closed = True
        if pending:
            failures.append(
                TimeoutError(
                    f"{len(pending)} state preparations exceeded shutdown timeout"
                )
            )
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Megatron run slot close failed", failures)

    def _require_run(
        self,
        run_id: str,
        *,
        require_complete: bool = True,
        allow_unregistering: bool = False,
    ) -> _ResidentRunState:
        if self._closed:
            raise RuntimeError("Megatron run slot is closed")
        try:
            state = self._runs[run_id]
        except KeyError as exc:
            raise RuntimeError(f"training run is not resident: {run_id!r}") from exc
        if state.unregistering and not allow_unregistering:
            raise RuntimeError("training run is being unregistered")
        if require_complete and not state.registration_complete:
            raise RuntimeError("training run residency registration is incomplete")
        return state

    def _register_gradient_contribution(
        self, state: _ResidentRunState, operation_id: str
    ) -> None:
        del operation_id
        accumulator = state.desired.accumulator
        if accumulator is not None:
            self._residency.touch(accumulator)
            return
        tensors = self._require_gradients(state).residency_tensors()
        installed_weights = state.installed_weights
        if installed_weights is None:
            raise RuntimeError("gradient contribution has no installed weights")
        key = installed_weights.model_copy(
            update={
                "representation": "accumulator",
                "accumulator_revision": state.next_accumulator_revision,
            }
        )
        self._residency.register_mutable_l1(key, tensors)
        with _residency_mutation(state):
            state.desired = state.desired.model_copy(update={"accumulator": key})
            state.next_accumulator_revision += 1

    def _retire_accumulator(self, state: _ResidentRunState) -> None:
        key = state.desired.accumulator
        if key is None:
            return
        with _residency_mutation(state):
            state.desired = state.desired.model_copy(update={"accumulator": None})
        self._residency.retire_async(key)

    @contextmanager
    def _accumulator_resident(self, state: _ResidentRunState) -> Iterator[None]:
        key = state.desired.accumulator
        if key is None:
            yield
            return
        self._residency.acquire_l1(key)
        try:
            self._residency.wait_before_mutation_working_set((key,))
            self._residency.begin_l1_mutation(key)
            yield
        finally:
            self._residency.release_l1(key)

    @contextmanager
    def _resident(
        self,
        state: _ResidentRunState,
        *,
        include_optimizer: bool = False,
        include_accumulator: bool = False,
    ) -> Iterator[tuple[ResidencyKey, ...]]:
        weights_key = state.desired.weights
        optimizer_key = state.desired.optimizer if include_optimizer else None
        accumulator_key = state.desired.accumulator if include_accumulator else None
        working_set = tuple(
            key
            for key in (weights_key, optimizer_key, accumulator_key)
            if key is not None
        )
        acquired = False
        try:
            self._residency.acquire_l1_working_set(working_set)
            acquired = True
            if state.installed_weights != weights_key:
                pending = state.pending_load
                if pending is None or pending.weights_key != weights_key:
                    raise RuntimeError("desired weights have no prepared L1 install")
                previous_weights = state.installed_weights
                previous_optimizer = state.installed_optimizer
                from art.megatron.training.gradient_accumulator import (
                    ParameterGradientAccumulator,
                )

                try:
                    # TrainerRank binds these exact parameters or rolls back the slot.
                    gradients = ParameterGradientAccumulator(parameters=pending.weights)
                    self._slot_trainer.install_prepared_checkpoint_slot_load_sync(
                        pending.checkpoint
                    )
                except BaseException:
                    self._residency.release_l1_working_set(working_set)
                    acquired = False
                    if previous_weights is not None:
                        self._residency.acquire_l1(previous_weights)
                        self._residency.release_l1(previous_weights)
                    raise
                state.checkpoint_slot_installed = True
                state.gradients = gradients
                state.adapter_config = pending.adapter_config
                state.installed_weights = weights_key
                state.installed_optimizer = None
                if pending.optimizer is None:
                    state.pending_load = None
                if previous_weights is not None:
                    self._residency.retire_async(previous_weights)
                if previous_optimizer is not None:
                    self._residency.retire_async(previous_optimizer)
            if optimizer_key is not None:
                if state.installed_optimizer != optimizer_key:
                    pending = state.pending_load
                    if (
                        pending is None
                        or pending.optimizer_key != optimizer_key
                        or pending.optimizer is None
                    ):
                        raise RuntimeError(
                            "desired optimizer has no prepared L1 install"
                        )
                    self._slot_trainer.install_prepared_checkpoint_slot_optimizer(
                        state.run_id, pending.optimizer
                    )
                    live_optimizer = (
                        self._slot_trainer.checkpoint_slot_residency_tensors(
                            state.run_id
                        ).optimizer
                    )
                    if tuple(map(id, live_optimizer)) != tuple(
                        map(id, pending.optimizer_tensors)
                    ):
                        raise RuntimeError(
                            "installed optimizer changed prepared residency tensors"
                        )
                    state.installed_optimizer = optimizer_key
                    state.pending_load = None
            yield working_set
        finally:
            if acquired:
                self._residency.release_l1_working_set(working_set)

    @staticmethod
    def _require_gradients(state: _ResidentRunState) -> Any:
        if state.gradients is None:
            raise RuntimeError("run weights have not been installed")
        return state.gradients

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
        if replacement.get("moe_parameterization", "per_expert") != current.get(
            "moe_parameterization", "per_expert"
        ):
            raise ValueError("loaded adapter changes immutable moe_parameterization")


class _ResolvedGeneration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lora: Any | None
    optimizer: Any | None
    prepared_tensors: PreparedSafetensors | None
    lora_residency_key: ResidencyKey | None = None


class _ResidentOptimizerSource(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    key: ResidencyKey
    source: Any


class _ResidentLoraSource(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    key: ResidencyKey
    prepared: SkipValidation[Future[Any]]


class _CachedGeneration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    generation: TrainerGeneration
    stager: PinnedCpuSnapshotStager | None
    resolved: Future[_ResolvedGeneration]
    optimizer_upgrade: Future[Any] | None = None
    resident_lora: _ResidentLoraSource | None = None
    resident_optimizer: _ResidentOptimizerSource | None = None
    has_optimizer: bool = False
    release_stager_on_resolve: bool = False
    consumers: list[Future[TrainerRankPublication]] = Field(default_factory=list)
    object_ids: set[str] = Field(default_factory=set)
    residency_retirement: Future[None] | None = None
    admission_order: int = Field(default=0, ge=0)
    retired: bool = False
    released: bool = False
    ephemeral: bool = False


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


class _PreparedAdapterPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    lora: Any
    tensors: PreparedSafetensors
    config: bytes
    model_identity: FileIdentity | None
    config_identity: FileIdentity


class _PreparedRankSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    operation_id: str
    entry: _CachedGeneration
    plan: SnapshotRankWritePlan
    adapter: _PreparedAdapterPayload | None
    distributed_adapter: Any | None = None
    optimizer: Any | None
    optimizer_archive: Any | None
    optimizer_identity: FileIdentity | None
    optimizer_state_path: str
    staging_adapter_path: str | None
    publication_targets: tuple[Any, ...]
    adapter_object_target: BinaryObjectPublicationTarget | None
    contexts: Any
    completion: Future[TrainerRankPublication]
    sink: SkipValidation[EventSink]
    prepared_at: float
    authorized: bool = False


class _GenerationPublisher:
    def __init__(
        self,
        runtime: Any,
        *,
        capacity: int,
        residency: RunResidencyManager | None = None,
    ) -> None:
        if capacity < 2:
            raise ValueError(
                "snapshot pool capacity must be at least 2 for transactional replacement"
            )
        self.runtime = runtime
        self.capacity = capacity
        self._residency = residency
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._available_stagers = [
            PinnedCpuSnapshotStager(reusable=True) for _ in range(capacity)
        ]
        self._resolution_pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="art-snapshot-resolve"
        )
        self._sampler_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-sampler-prepare"
        )
        self._transport_pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="art-publish-transport"
        )
        self._ordered_transport_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-publish-ordered"
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
        # Keep one active publication cohort and one recent shape cohort.
        self._lora_layout_capacity = 2 * capacity
        self._lora_layouts: OrderedDict[
            tuple[Any, ...], SafetensorsLayout
        ] = OrderedDict()
        self._cache: dict[str, _CachedGeneration] = {}
        self._latest_by_run: dict[str, str] = {}
        self._prepared: dict[str, _PreparedRankSnapshot] = {}
        self._prepared_order: deque[str] = deque()
        self._residency_retirements: set[Future[None]] = set()
        self._failures: list[BaseException] = []
        self._next_admission_order = 0
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
        residency_key: ResidencyKey | None = None,
    ) -> dict[str, float]:
        from art.megatron.optimizer_state import (
            stage_optimizer_state_snapshot,
            stage_trainer_rank_optimizer_state_snapshot,
        )
        from art.megatron.weights.lora_publish import (
            LoraSnapshotTimings,
            stage_vllm_lora_snapshot_from_model,
        )

        wait_s, in_flight, stager = self._acquire_slot(protected_run_id=run_id)
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
                self.runtime.optimizer_snapshot_barrier.register(lora, key=run_id)
            if optimizer is not None:
                self.runtime.optimizer_snapshot_barrier.register(optimizer, key=run_id)
            optimizer_launch_s = time.perf_counter() - optimizer_started
            lora_residency_key = (
                None
                if residency_key is None
                else residency_key.model_copy(update={"representation": "sampler"})
            )
            resolved = self._resolution_pool.submit(
                self._resolve_generation, lora, optimizer, lora_residency_key
            )
            entry = _CachedGeneration(
                run_id=run_id,
                generation=generation,
                stager=stager,
                resolved=resolved,
                has_optimizer=optimizer is not None,
                release_stager_on_resolve=lora_residency_key is not None,
            )
            self._cache_staging(entry)
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

    def register_existing(
        self,
        *,
        run_id: str,
        generation: TrainerGeneration,
        optimizer_source: Any | None,
        optimizer_residency_key: ResidencyKey | None,
    ) -> dict[str, float]:
        """Register an existing adapter and L2 state without reading mutable GPU state."""
        self.raise_if_failed()
        started = time.perf_counter()
        with self._lock:
            if generation.generation_id in self._cache:
                raise RuntimeError(
                    f"generation snapshot already exists: {generation.generation_id}"
                )
        resolved: Future[_ResolvedGeneration] = Future()
        resolved.set_result(
            _ResolvedGeneration(lora=None, optimizer=None, prepared_tensors=None)
        )
        if self._residency is None:
            raise RuntimeError("resident generation requires a residency manager")
        if (optimizer_source is None) != (optimizer_residency_key is None):
            raise RuntimeError(
                "optimizer source and residency key must be provided together"
            )
        if optimizer_residency_key is not None:
            self._residency.ensure_l2(optimizer_residency_key)
        resident_optimizer = (
            None
            if optimizer_source is None
            else _ResidentOptimizerSource(
                key=optimizer_residency_key,
                source=optimizer_source,
            )
        )
        entry = _CachedGeneration(
            run_id=run_id,
            generation=generation,
            stager=None,
            resolved=resolved,
            resident_optimizer=resident_optimizer,
            has_optimizer=resident_optimizer is not None,
        )
        self._cache_staging(entry)
        self._activate_generation(entry)
        return {
            "snapshot_optimizer_attach_s": time.perf_counter() - started,
        }

    def register_resident_generation(
        self,
        *,
        run_id: str,
        generation: TrainerGeneration,
        weights_key: ResidencyKey,
        export_plan: Any,
        adapter_config: dict[str, Any],
        optimizer_source: Any,
        optimizer_key: ResidencyKey,
    ) -> dict[str, float]:
        """Prepare one exact sampler generation from immutable L2 state."""
        if self._residency is None:
            raise RuntimeError("resident generation requires a residency manager")
        started = time.perf_counter()
        with self._lock:
            if generation.generation_id in self._cache:
                raise RuntimeError(
                    f"generation snapshot already exists: {generation.generation_id}"
                )
        self._residency.ensure_l2(weights_key)
        self._residency.ensure_l2(optimizer_key)
        sampler_key = weights_key.model_copy(update={"representation": "sampler"})
        prepared = self._sampler_pool.submit(
            self._prepare_resident_lora,
            weights_key,
            sampler_key,
            export_plan,
            adapter_config,
        )
        resolved: Future[_ResolvedGeneration] = Future()
        resolved.set_result(
            _ResolvedGeneration(lora=None, optimizer=None, prepared_tensors=None)
        )
        entry = _CachedGeneration(
            run_id=run_id,
            generation=generation,
            stager=None,
            resolved=resolved,
            resident_lora=_ResidentLoraSource(
                key=sampler_key,
                prepared=prepared,
            ),
            resident_optimizer=_ResidentOptimizerSource(
                key=optimizer_key,
                source=optimizer_source,
            ),
            has_optimizer=True,
        )
        self._cache_staging(entry)
        prepared.add_done_callback(
            lambda done, cached=entry: self._resident_lora_prepared(cached, done)
        )
        return {
            "snapshot_resident_attach_s": time.perf_counter() - started,
        }

    def _prepare_resident_lora(
        self,
        weights_key: ResidencyKey,
        sampler_key: ResidencyKey,
        export_plan: Any,
        adapter_config: dict[str, Any],
    ) -> Any:
        if self._residency is None:
            raise RuntimeError("resident generation requires a residency manager")
        from art.megatron.lora import _block_for_key
        from art.megatron.weights.rank_distributed_lora_publish import (
            prepare_rank_distributed_vllm_lora_source,
        )

        with self._residency.borrow_l2(weights_key) as image:
            regular, regular_meta, packed, packed_meta = export_plan.materialize(
                image.tensors(), owner_rank=int(self.runtime.rank)
            )
        pending = prepare_rank_distributed_vllm_lora_source(
            local_tensors=regular,
            local_metadata=regular_meta,
            local_packed_tensors=packed,
            local_packed_metadata=packed_meta,
            handler=self.runtime.model_support_handler,
            adapter_config=adapter_config,
            conversion_group_for_key=_block_for_key,
            group=self.runtime.publication_group,
            metadata_group=self.runtime.publication_metadata_group,
            exchange_device=torch.device("cpu"),
            stager=PinnedCpuSnapshotStager(reusable=True),
        )
        source = pending.resolve()
        names = tuple(sorted(source.tensors))
        self._residency.register_l2(
            sampler_key, tuple(source.tensors[name] for name in names)
        )
        return source.model_copy(update={"tensors": {}})

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
        residency_key: ResidencyKey | None = None,
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
                residency_key=residency_key,
            )
        assert entry is not None
        if not snapshot_optimizer or entry.has_optimizer:
            return {}
        stager = entry.stager
        if stager is None:
            raise RuntimeError("loaded generation has no optimizer state to publish")

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
                stager=stager,
            )
        else:
            if trainer_rank_optimizer_state is None:
                raise RuntimeError("dynamic LoRA snapshot has no optimizer state")
            optimizer = stage_trainer_rank_optimizer_state_snapshot(
                self.runtime,
                trainer_rank_optimizer_state,
                generation_id=generation.generation_id,
                step=generation.policy_step,
                stager=stager,
            )
        self.runtime.optimizer_snapshot_barrier.register(optimizer, key=entry.run_id)
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
        residency_key: ResidencyKey,
    ) -> dict[str, float]:
        started = time.perf_counter()
        with self._lock:
            entry = self._cache.get(generation.generation_id)
            if entry is None or entry.retired or entry.generation != generation:
                raise RuntimeError("optimizer residency has no staged generation")
            if entry.has_optimizer or entry.optimizer_upgrade is not None:
                raise RuntimeError("generation already has an optimizer snapshot")
            if self._residency is None:
                raise RuntimeError("resident optimizer requires a residency manager")
            entry.resident_optimizer = _ResidentOptimizerSource(
                key=residency_key,
                source=source,
            )
            entry.has_optimizer = True
        elapsed = time.perf_counter() - started
        return {
            "snapshot_optimizer_attach_s": elapsed,
        }

    def prepare_ordered_sampler(
        self,
        *,
        operation_id: str,
        run_id: str,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        target: OrderedBinaryObjectTarget,
        adapter_dtypes: dict[str, torch.dtype],
        adapter_config: dict[str, Any],
        slot_ref: "LoRASlotRef | None",
        sink: EventSink,
    ) -> tuple[SnapshotRankWritePlan, dict[str, float]]:
        """Prepare rank-owned sampler bytes while the selected learner is resident."""
        self.raise_if_failed()
        with self._lock:
            cached = self._prepared.get(operation_id)
        if cached is not None:
            if (
                cached.plan.generation != generation
                or cached.adapter_object_target != target
                or cached.distributed_adapter is None
            ):
                raise RuntimeError(
                    "ordered sampler operation was reused for another snapshot"
                )
            return cached.plan, {}
        expected_metadata = {
            "run_id": run_id,
            "training_session_id": generation.training_session_id,
            "generation_id": generation.generation_id,
            "policy_step": str(generation.policy_step),
        }
        if any(target.metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(
                "ordered sampler target identifies another learner generation"
            )

        with self._lock:
            generation_entry = self._cache.get(generation.generation_id)
        if (
            generation_entry is not None
            and not generation_entry.retired
            and generation_entry.generation == generation
            and generation_entry.resident_lora is not None
        ):
            return self._prepare_ordered_resident_sampler(
                operation_id=operation_id,
                generation=generation,
                optimizer_state_path=optimizer_state_path,
                target=target,
                entry=generation_entry,
                sink=sink,
            )
        if slot_ref is not None:
            raise RuntimeError(
                "dynamic LoRA generation has no immutable resident sampler source"
            )

        from art.megatron.lora import _block_for_key
        from art.megatron.weights.lora_publish import (
            collect_local_lora_entries,
            collect_local_packed_expert_entries,
        )
        from art.megatron.weights.rank_distributed_lora_publish import (
            prepare_rank_distributed_vllm_lora,
        )

        wait_s = 0.0
        in_flight = 0
        stager: PinnedCpuSnapshotStager | None = None
        started = time.perf_counter()
        entry: _CachedGeneration | None = None
        resolved: Future[_ResolvedGeneration] | None = None
        handler: Any | None = None
        local_tensors: dict[str, torch.Tensor] = {}
        local_metadata = ()
        local_packed_tensors: dict[str, torch.Tensor] = {}
        local_packed_metadata = ()
        exchange_device: torch.device | None = None
        local_error: BaseException | None = None
        try:
            try:
                wait_s, in_flight, stager = self._acquire_slot(
                    protected_run_id=run_id
                )
                handler = self.runtime.model_support_handler
                packed_groups = handler.expert_packed_lora_groups()
                local_tensors, local_metadata = collect_local_lora_entries(
                    self.runtime.model,
                    adapter_dtypes,
                    owner_rank=int(self.runtime.rank),
                    packed_expert_groups=packed_groups,
                    slot_ref=slot_ref,
                )
                local_packed_tensors, local_packed_metadata = (
                    collect_local_packed_expert_entries(
                        self.runtime.model,
                        adapter_dtypes,
                        owner_rank=int(self.runtime.rank),
                        packed_expert_groups=packed_groups,
                        slot_ref=slot_ref,
                    )
                )
                source_devices = {
                    tensor.device
                    for tensor in (
                        *local_tensors.values(),
                        *local_packed_tensors.values(),
                    )
                }
                if len(source_devices) > 1:
                    raise RuntimeError(
                        "one rank-distributed sampler source spans multiple devices"
                    )
                exchange_device = (
                    source_devices.pop()
                    if source_devices
                    else torch.device("cuda", torch.cuda.current_device())
                )
            except BaseException as error:
                local_error = error
            pending = prepare_rank_distributed_vllm_lora(
                target=target,
                local_tensors=local_tensors,
                local_metadata=local_metadata,
                local_packed_tensors=local_packed_tensors,
                local_packed_metadata=local_packed_metadata,
                handler=handler,
                adapter_config=adapter_config,
                conversion_group_for_key=_block_for_key,
                metadata_group=self.runtime.publication_metadata_group,
                exchange_device=exchange_device,
                stager=stager,
                local_error=local_error,
            )
            if stager is None:
                raise RuntimeError(
                    "LoRA readiness collective accepted a missing snapshot stager"
                )
            self.runtime.optimizer_snapshot_barrier.register(pending, key=run_id)
            resolved = self._resolution_pool.submit(
                self._resolve_ordered_sampler, pending
            )
            entry = _CachedGeneration(
                run_id=run_id,
                generation=generation,
                stager=stager,
                resolved=resolved,
                ephemeral=True,
            )
            resolved.add_done_callback(
                lambda _done, transient=entry: self._maybe_release(transient)
            )
            ref = pending.payload.layout.ref
            expected_files = {"adapter_config.json", "adapter_model.safetensors"}
            if {file.relative_path for file in ref.files} != expected_files:
                raise RuntimeError("ordered sampler object has an invalid LoRA layout")
            transport_adapter = OptimizerAdapter(
                identity=ref.manifest_uri,
                training_session_id=generation.training_session_id,
                step=generation.policy_step,
                generation_id=generation.generation_id,
                files=tuple(
                    CheckpointFile(
                        name=cast(
                            Literal[
                                "adapter_config.json",
                                "adapter_model.safetensors",
                            ],
                            file.relative_path,
                        ),
                        size_bytes=file.byte_count,
                        sha256=file.sha256,
                    )
                    for file in ref.files
                ),
            )
            rank = int(self.runtime.rank)
            rank_plan = SnapshotRankWritePlan(
                rank=rank,
                generation=generation,
                transport_adapter=transport_adapter if rank == 0 else None,
                saves_optimizer=False,
            )
            completion: Future[TrainerRankPublication] = Future()
            completion.add_done_callback(_consume_future)
            contexts = ExitStack()
            prepared = _PreparedRankSnapshot(
                operation_id=operation_id,
                entry=entry,
                plan=rank_plan,
                adapter=None,
                distributed_adapter=pending.payload,
                optimizer=None,
                optimizer_archive=None,
                optimizer_identity=None,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=None,
                publication_targets=(),
                adapter_object_target=target,
                contexts=contexts,
                completion=completion,
                sink=sink,
                prepared_at=started,
            )
            with self._lock:
                if operation_id in self._prepared:
                    raise RuntimeError(
                        f"snapshot operation already prepared: {operation_id}"
                    )
                self._prepared[operation_id] = prepared
                self._prepared_order.append(operation_id)
                entry.consumers.append(completion)
        except BaseException:
            if entry is None:
                if resolved is not None:
                    resolved.result()
                if stager is not None:
                    self._release_slot(stager)
            else:
                entry.retired = True
                self._maybe_release(entry)
            raise
        metrics = {
            "snapshot_pool_wait_s": wait_s,
            "snapshot_pool_in_use": float(in_flight),
            "snapshot_pool_pressure": in_flight / self.capacity,
            "snapshot_ordered_prepare_s": time.perf_counter() - started,
            **{
                f"snapshot_ordered_{key}": float(value)
                for key, value in pending.payload.stats.model_dump().items()
            },
        }
        return rank_plan, metrics

    def _prepare_ordered_resident_sampler(
        self,
        *,
        operation_id: str,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        target: OrderedBinaryObjectTarget,
        entry: _CachedGeneration,
        sink: EventSink,
    ) -> tuple[SnapshotRankWritePlan, dict[str, float]]:
        if self._residency is None or entry.resident_lora is None:
            raise RuntimeError("ordered sampler generation has no resident source")
        started = time.perf_counter()
        source = entry.resident_lora.prepared.result()
        contexts = ExitStack()
        try:
            image = contexts.enter_context(
                self._residency.borrow_l2(entry.resident_lora.key)
            )
            names = tuple(
                tensor.name
                for tensor in source.metadata
                if tensor.owner_rank == int(self.runtime.rank)
            )
            tensors = image.tensors()
            if len(names) != len(tensors):
                raise RuntimeError("resident sampler tensor identity changed")
            distributed = source.model_copy(
                update={"tensors": dict(zip(names, tensors, strict=True))}
            ).bind_target(target)
            ref = distributed.layout.ref
            expected_files = {"adapter_config.json", "adapter_model.safetensors"}
            if {file.relative_path for file in ref.files} != expected_files:
                raise RuntimeError("ordered sampler object has an invalid LoRA layout")
            transport_adapter = OptimizerAdapter(
                identity=ref.manifest_uri,
                training_session_id=generation.training_session_id,
                step=generation.policy_step,
                generation_id=generation.generation_id,
                files=tuple(
                    CheckpointFile(
                        name=cast(
                            Literal[
                                "adapter_config.json",
                                "adapter_model.safetensors",
                            ],
                            file.relative_path,
                        ),
                        size_bytes=file.byte_count,
                        sha256=file.sha256,
                    )
                    for file in ref.files
                ),
            )
            rank = int(self.runtime.rank)
            rank_plan = SnapshotRankWritePlan(
                rank=rank,
                generation=generation,
                transport_adapter=transport_adapter if rank == 0 else None,
                saves_optimizer=False,
            )
            completion: Future[TrainerRankPublication] = Future()
            completion.add_done_callback(_consume_future)
            prepared = _PreparedRankSnapshot(
                operation_id=operation_id,
                entry=entry,
                plan=rank_plan,
                adapter=None,
                distributed_adapter=distributed,
                optimizer=None,
                optimizer_archive=None,
                optimizer_identity=None,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=None,
                publication_targets=(),
                adapter_object_target=target,
                contexts=contexts,
                completion=completion,
                sink=sink,
                prepared_at=started,
            )
            with self._lock:
                if operation_id in self._prepared:
                    raise RuntimeError(
                        f"snapshot operation already prepared: {operation_id}"
                    )
                self._prepared[operation_id] = prepared
                self._prepared_order.append(operation_id)
                entry.consumers.append(completion)
        except BaseException:
            contexts.close()
            raise
        return rank_plan, {
            "snapshot_ordered_prepare_s": time.perf_counter() - started,
            **{
                f"snapshot_ordered_{key}": float(value)
                for key, value in distributed.stats.model_dump().items()
            },
        }

    @staticmethod
    def _resolve_ordered_sampler(
        pending: PendingCpuSnapshot["PreparedRankDistributedLora"],
    ) -> _ResolvedGeneration:
        pending.resolve()
        return _ResolvedGeneration(lora=None, optimizer=None, prepared_tensors=None)

    def prepare(
        self,
        *,
        operation_id: str,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        existing_adapter: OptimizerAdapter | None = None,
        publication_targets: tuple[Any, ...],
        adapter_object_target: BinaryObjectPublicationTarget | None,
        save_optimizer: bool,
        sink: EventSink,
    ) -> tuple[SnapshotRankWritePlan, dict[str, float]]:
        if isinstance(adapter_object_target, OrderedBinaryObjectTarget):
            raise RuntimeError(
                "ordered sampler objects require rank-distributed preparation"
            )
        self.raise_if_failed()
        with self._lock:
            entry = self._cache.get(generation.generation_id)
            cached = self._prepared.get(operation_id)
        if cached is not None:
            if cached.plan.generation != generation:
                raise RuntimeError(
                    "snapshot operation was reused for another generation"
                )
            return cached.plan, {}
        if entry is None or entry.retired or entry.generation != generation:
            raise RuntimeError(
                f"learner generation is not staged: {generation.generation_id}"
            )
        started = time.perf_counter()
        contexts = ExitStack()
        try:
            rank = int(self.runtime.rank)
            adapter = existing_adapter if rank == 0 else None
            transport_adapter = None
            adapter_payload = None
            wants_adapter_payload = (
                (existing_adapter is None and staging_adapter_path is not None)
                or bool(publication_targets)
                or adapter_object_target is not None
            )
            consolidated_lora = (
                self._consolidate_resident_lora(entry)
                if wants_adapter_payload and entry.resident_lora is not None
                else None
            )
            needs_adapter_payload = rank == 0 and (
                wants_adapter_payload
            )
            if needs_adapter_payload:
                lora, tensors = contexts.enter_context(
                    self._lora_snapshot(entry, adapter, consolidated_lora)
                )
                config = encode_adapter_config(
                    {
                        **lora.adapter_config,
                        ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM,
                    }
                )
                exact_model_identity = (
                    staging_adapter_path is not None
                    or bool(publication_targets)
                    or (
                        adapter_object_target is not None
                        and not isinstance(
                            adapter_object_target, OrderedBinaryObjectTarget
                        )
                    )
                )
                # Ordered shards are fenced by their plan, range, ETag, and
                # commit. Hashing the whole adapter here would serialize every
                # serverless publication before its first shard can upload.
                model_identity = (
                    prepared_safetensors_identity(tensors)
                    if exact_model_identity
                    else None
                )
                config_identity = FileIdentity(
                    size_bytes=len(config),
                    sha256=hashlib.sha256(config).hexdigest(),
                )
                exact_files = (
                    CheckpointFile(
                        name="adapter_config.json", **config_identity.model_dump()
                    ),
                    CheckpointFile(
                        name="adapter_model.safetensors",
                        size_bytes=tensors.nbytes,
                        sha256=(
                            None if model_identity is None else model_identity.sha256
                        ),
                    ),
                )
                adapter_payload = _PreparedAdapterPayload(
                    lora=lora,
                    tensors=tensors,
                    config=config,
                    model_identity=model_identity,
                    config_identity=config_identity,
                )
                if adapter is None and staging_adapter_path is not None:
                    if model_identity is None:
                        raise RuntimeError("local adapter plan has no exact identity")
                    adapter = OptimizerAdapter(
                        identity=str(
                            canonical_adapter_path(
                                staging_adapter_path, generation.policy_step
                            )
                        ),
                        training_session_id=generation.training_session_id,
                        step=generation.policy_step,
                        generation_id=generation.generation_id,
                        files=exact_files,
                    )
                if adapter_object_target is not None:
                    transport_files = (
                        tuple(
                            file.model_copy(update={"sha256": None})
                            for file in exact_files
                        )
                        if isinstance(
                            adapter_object_target, OrderedBinaryObjectTarget
                        )
                        else exact_files
                    )
                    transport_adapter = OptimizerAdapter(
                        identity=binary_object_manifest_uri(adapter_object_target),
                        training_session_id=generation.training_session_id,
                        step=generation.policy_step,
                        generation_id=generation.generation_id,
                        files=transport_files,
                    )
            if rank == 0 and adapter is None and transport_adapter is None:
                raise RuntimeError("rank-zero snapshot has no planned adapter output")

            optimizer = contexts.enter_context(
                self._optimizer_snapshot(entry, generation, required=save_optimizer)
            )
            optimizer_archive = None
            optimizer_identity = None
            shard = None
            runtime_sha256 = None
            topology = None
            if optimizer is not None:
                from art.megatron.optimizer_archive import prepare_optimizer_archive

                optimizer_archive = prepare_optimizer_archive(optimizer.state_dict)
                optimizer_identity = optimizer_archive.identity()
                shard = OptimizerShard(
                    rank=optimizer.rank,
                    size_bytes=optimizer_identity.size_bytes,
                    layout_sha256=optimizer.layout_sha256,
                    sha256=optimizer_identity.sha256,
                    serialization="art_safetensors_v1",
                )
                runtime_sha256 = optimizer.runtime_sha256
                topology = optimizer.topology
            rank_plan = SnapshotRankWritePlan(
                rank=rank,
                generation=generation,
                adapter=adapter,
                transport_adapter=transport_adapter,
                optimizer_shard=shard,
                runtime_sha256=runtime_sha256,
                topology=topology,
                saves_optimizer=save_optimizer,
            )
            completion: Future[TrainerRankPublication] = Future()
            completion.add_done_callback(_consume_future)
            prepared = _PreparedRankSnapshot(
                operation_id=operation_id,
                entry=entry,
                plan=rank_plan,
                adapter=adapter_payload,
                distributed_adapter=None,
                optimizer=optimizer,
                optimizer_archive=optimizer_archive,
                optimizer_identity=optimizer_identity,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=staging_adapter_path,
                publication_targets=publication_targets,
                adapter_object_target=adapter_object_target,
                contexts=contexts,
                completion=completion,
                sink=sink,
                prepared_at=started,
            )
            with self._lock:
                if operation_id in self._prepared:
                    raise RuntimeError(
                        f"snapshot operation already prepared: {operation_id}"
                    )
                self._prepared[operation_id] = prepared
                self._prepared_order.append(operation_id)
                entry.consumers.append(completion)
        except BaseException:
            contexts.close()
            raise
        return rank_plan, {"snapshot_prepare_s": time.perf_counter() - started}

    def authorize(
        self,
        *,
        operation_id: str,
        plan: SnapshotWritePlan,
        grant: SnapshotWriteGrant,
    ) -> dict[str, float]:
        self.raise_if_failed()
        grant.validate_plan(plan)
        rank = int(self.runtime.rank)
        try:
            rank_plan = plan.ranks[rank]
        except IndexError as error:
            raise RuntimeError(
                "snapshot write plan does not cover this rank"
            ) from error
        with self._lock:
            prepared = self._prepared.get(operation_id)
            if prepared is None:
                raise RuntimeError(
                    f"snapshot operation is not prepared: {operation_id}"
                )
            if prepared.authorized:
                if prepared.plan != rank_plan:
                    raise RuntimeError("authorized snapshot plan changed")
                return {}
            if not self._prepared_order or self._prepared_order[0] != operation_id:
                raise RuntimeError(
                    "snapshot authorization would overtake an earlier save"
                )
            if prepared.plan != rank_plan:
                raise RuntimeError("rank snapshot differs from the authorized plan")
            prepared.authorized = True
            self._prepared_order.popleft()
        started = time.perf_counter()
        try:
            transport_pool = (
                self._ordered_transport_pool
                if prepared.distributed_adapter is not None
                else self._transport_pool
            )
            transport = transport_pool.submit(
                self._transfer_prepared_snapshot, prepared, started
            )
            durability = self._start_durability(prepared, started)
            publication = self._completion_pool.submit(
                self._complete_publication,
                transport,
                durability,
                started,
                prepared.plan,
                grant,
            )
            publication.add_done_callback(
                lambda done: self._authorized_completed(done, prepared=prepared)
            )
        except BaseException as error:
            self._authorization_failed(prepared, error)
            raise
        return {"snapshot_authorize_s": time.perf_counter() - started}

    @staticmethod
    def _requires_durable_write(prepared: _PreparedRankSnapshot) -> bool:
        return prepared.optimizer is not None or (
            prepared.adapter is not None and prepared.staging_adapter_path is not None
        )

    def _start_durability(
        self, prepared: _PreparedRankSnapshot, submitted_at: float
    ) -> Future[_RankSnapshotPersistence]:
        if self._requires_durable_write(prepared):
            return self._durability_pool.submit(
                self._persist_prepared_snapshot, prepared, submitted_at
            )
        completed: Future[_RankSnapshotPersistence] = Future()
        completed.set_result(self._persist_prepared_snapshot(prepared, submitted_at))
        return completed

    def discard(self, operation_id: str) -> None:
        with self._lock:
            prepared = self._prepared.get(operation_id)
            if prepared is None:
                return
            if prepared.authorized:
                raise RuntimeError("cannot discard an authorized snapshot write")
            self._prepared.pop(operation_id)
            self._prepared_order.remove(operation_id)
        error = RuntimeError("snapshot write authorization was rejected")
        try:
            prepared.contexts.close()
        except BaseException as cleanup_error:
            error.add_note(
                "snapshot context cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        self._report_failure(
            error,
            entry=prepared.entry,
            sink=prepared.sink,
            generation=prepared.plan.generation,
            remember=False,
        )
        if not prepared.completion.done():
            prepared.completion.set_exception(error)
        self._retire_prepared(prepared)

    def stage_and_submit(self, **_kwargs: Any) -> dict[str, float]:
        raise RuntimeError(
            "fused snapshot publication was removed; use prepare/authorize"
        )

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
        with self._lock:
            entries = tuple(
                entry for entry in self._cache.values() if entry.run_id == run_id
            )
            self._latest_by_run.pop(run_id, None)
            for entry in entries:
                entry.retired = True
        failures: list[BaseException] = []
        for entry in entries:
            try:
                entry.resolved.result()
                if entry.resident_lora is not None:
                    entry.resident_lora.prepared.result()
                if entry.optimizer_upgrade is not None:
                    entry.optimizer_upgrade.result()
                for consumer in tuple(entry.consumers):
                    consumer.result()
            except BaseException as error:
                failures.append(error)
            finally:
                self._maybe_release(entry)
                if entry.residency_retirement is not None:
                    try:
                        entry.residency_retirement.result()
                    except BaseException as error:
                        failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("generation retirement failed", failures)

    def _acquire_slot(
        self, *, protected_run_id: str | None = None
    ) -> tuple[float, int, PinnedCpuSnapshotStager]:
        self.raise_if_failed()
        started = time.perf_counter()
        while not self._slots.acquire(blocking=False):
            self._evict_for_capacity(protected_run_id=protected_run_id)
            self.raise_if_failed()
        wait_s = time.perf_counter() - started
        with self._lock:
            stager = self._available_stagers.pop()
            stager.reset()
            self._in_flight += 1
            return wait_s, self._in_flight, stager

    def _evict_for_capacity(self, *, protected_run_id: str | None = None) -> None:
        with self._lock:
            entries = list(self._cache.values())
            entries.extend(
                prepared.entry
                for prepared in self._prepared.values()
                if prepared.entry.ephemeral
                and all(entry is not prepared.entry for entry in entries)
            )
            entries = tuple(
                entry
                for entry in entries
                if entry.stager is not None and not entry.released
            )
            if not entries:
                raise RuntimeError("snapshot pool is full without a cached generation")
            candidates = tuple(
                entry
                for entry in entries
                if entry.retired or entry.run_id != protected_run_id
            )
            if not candidates:
                raise ResidencyCapacityUnavailable(
                    "snapshot pool capacity is occupied by the run being replaced"
                )
            entry = next(
                (
                    entry
                    for entry in candidates
                    if all(future.done() for future in self._stager_dependencies(entry))
                ),
                None,
            )
            if entry is None:
                pending = tuple(
                    {
                        future
                        for candidate in candidates
                        for future in self._stager_dependencies(candidate)
                        if not future.done()
                    }
                )
            else:
                pending = ()
                consumers = tuple(entry.consumers)
                preserve = entry.release_stager_on_resolve
                if not preserve:
                    entry.retired = True
        if entry is None:
            if not pending:
                raise RuntimeError("snapshot pool has no releasable dependency")
            wait(pending, return_when=FIRST_COMPLETED)
            return
        try:
            entry.resolved.result()
            if not preserve and entry.optimizer_upgrade is not None:
                entry.optimizer_upgrade.result()
            if not preserve:
                for consumer in consumers:
                    consumer.result()
        finally:
            if preserve:
                self._release_entry_stager(entry)
            else:
                self._maybe_release(entry)

    @staticmethod
    def _stager_dependencies(entry: _CachedGeneration) -> tuple[Future[Any], ...]:
        if entry.release_stager_on_resolve:
            return (entry.resolved,)
        optimizer = (
            () if entry.optimizer_upgrade is None else (entry.optimizer_upgrade,)
        )
        return entry.resolved, *optimizer, *entry.consumers

    def _cache_staging(self, entry: _CachedGeneration) -> None:
        if entry.ephemeral:
            raise RuntimeError("ephemeral publication entered the generation cache")
        with self._lock:
            generation_id = entry.generation.generation_id
            if generation_id in self._cache:
                raise RuntimeError(
                    f"generation snapshot already exists: {generation_id}"
                )
            self._next_admission_order += 1
            entry.admission_order = self._next_admission_order
            self._cache[generation_id] = entry

    def _activate_generation(self, entry: _CachedGeneration) -> None:
        if entry.ephemeral:
            raise RuntimeError("ephemeral publication cannot become a generation")
        # Callback order must not let an older admission replace a newer success.
        with self._lock:
            if entry.retired or entry.released:
                return
            latest_id = self._latest_by_run.get(entry.run_id)
            latest = None if latest_id is None else self._cache.get(latest_id)
            if latest is not None and latest.admission_order > entry.admission_order:
                predecessors = (entry,)
            else:
                self._latest_by_run[entry.run_id] = entry.generation.generation_id
                predecessors = tuple(
                    candidate
                    for candidate in self._cache.values()
                    if candidate is not entry
                    and candidate.run_id == entry.run_id
                    and candidate.admission_order < entry.admission_order
                    and not candidate.retired
                )
            for predecessor in predecessors:
                predecessor.retired = True
        for predecessor in predecessors:
            self._maybe_release(predecessor)

    def _snapshot_resolved(
        self,
        entry: _CachedGeneration,
        resolved: Future[_ResolvedGeneration],
    ) -> None:
        if resolved.cancelled():
            with self._lock:
                entry.retired = True
        elif (error := resolved.exception()) is not None:
            with self._lock:
                self._failures.append(error)
                entry.retired = True
        else:
            self._activate_generation(entry)
        if entry.release_stager_on_resolve:
            self._release_entry_stager(entry)
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

    def _resident_lora_prepared(
        self,
        entry: _CachedGeneration,
        prepared: Future[Any],
    ) -> None:
        if prepared.cancelled():
            with self._lock:
                entry.retired = True
        elif (error := prepared.exception()) is not None:
            with self._lock:
                self._failures.append(error)
                entry.retired = True
        else:
            self._activate_generation(entry)
        self._maybe_release(entry)

    def _maybe_release(self, entry: _CachedGeneration) -> None:
        retirement: Future[None] | None = None
        retirement_started = False
        with self._lock:
            if (
                not entry.retired
                or entry.released
                or not entry.resolved.done()
                or (
                    entry.resident_lora is not None
                    and not entry.resident_lora.prepared.done()
                )
                or (
                    entry.optimizer_upgrade is not None
                    and not entry.optimizer_upgrade.done()
                )
                or any(not consumer.done() for consumer in entry.consumers)
            ):
                return
            entry.released = True
            if not entry.ephemeral:
                if self._cache.get(entry.generation.generation_id) is entry:
                    self._cache.pop(entry.generation.generation_id)
                if (
                    self._latest_by_run.get(entry.run_id)
                    == entry.generation.generation_id
                ):
                    self._latest_by_run.pop(entry.run_id)
                for object_id in entry.object_ids:
                    self._object_publications.pop(object_id, None)
            resident_lora = entry.resident_lora
            if (
                resident_lora is not None
                and not resident_lora.prepared.cancelled()
                and resident_lora.prepared.exception() is None
            ):
                if self._residency is None:
                    raise RuntimeError(
                        "resident LoRA source has no residency manager"
                    )
                try:
                    retirement = self._residency.retire_async(resident_lora.key)
                    retirement_started = True
                except BaseException as error:
                    retirement = Future()
                    retirement.set_exception(error)
                    self._failures.append(error)
                entry.residency_retirement = retirement
                if retirement_started:
                    self._residency_retirements.add(retirement)
            elif not entry.resolved.cancelled() and entry.resolved.exception() is None:
                lora_key = entry.resolved.result().lora_residency_key
                if lora_key is not None:
                    if self._residency is None:
                        raise RuntimeError(
                            "resident LoRA source has no residency manager"
                        )
                    try:
                        retirement = self._residency.retire_async(lora_key)
                        retirement_started = True
                    except BaseException as error:
                        retirement = Future()
                        retirement.set_exception(error)
                        self._failures.append(error)
                    entry.residency_retirement = retirement
                    if retirement_started:
                        self._residency_retirements.add(retirement)
        self._release_entry_stager(entry)
        if retirement is not None and retirement_started:
            retirement.add_done_callback(self._residency_retired)

    def _release_entry_stager(self, entry: _CachedGeneration) -> None:
        with self._lock:
            stager = entry.stager
            entry.stager = None
        if stager is not None:
            self._release_slot(stager)

    def _resolve_generation(
        self,
        lora: Any,
        optimizer: Any,
        lora_residency_key: ResidencyKey | None = None,
    ) -> _ResolvedGeneration:
        lora = None if lora is None else lora.resolve()
        optimizer = None if optimizer is None else optimizer.resolve()
        prepared_tensors = None
        if lora is not None:
            if lora_residency_key is None:
                prepared_tensors = self._prepare_lora_tensors(lora)
            else:
                if self._residency is None:
                    raise RuntimeError("LoRA residency requires a residency manager")
                self._residency.register_l2(
                    lora_residency_key,
                    tuple(lora.tensors[key] for key in sorted(lora.tensors)),
                )
        return _ResolvedGeneration(
            lora=lora,
            optimizer=optimizer,
            prepared_tensors=prepared_tensors,
            lora_residency_key=lora_residency_key if lora is not None else None,
        )

    def _prepare_lora_tensors(self, lora: Any) -> PreparedSafetensors:
        storages: dict[tuple[int, int], int] = {}
        layout_key = []
        for key, tensor in sorted(lora.tensors.items()):
            storage = tensor.untyped_storage()
            identity = storage.data_ptr(), storage.nbytes()
            storage_index = storages.setdefault(identity, len(storages))
            layout_key.append(
                (
                    key,
                    tuple(tensor.shape),
                    str(tensor.dtype),
                    storage_index,
                    storage.nbytes(),
                    tensor.data_ptr() - storage.data_ptr(),
                )
            )
        cache_key = tuple(layout_key)
        with self._lock:
            layout = self._lora_layouts.get(cache_key)
            if layout is None:
                layout = self._lora_layouts[cache_key] = SafetensorsLayout(lora.tensors)
                if len(self._lora_layouts) > self._lora_layout_capacity:
                    self._lora_layouts.popitem(last=False)
            else:
                self._lora_layouts.move_to_end(cache_key)
        return layout.bind(lora.tensors)

    @contextmanager
    def _lora_snapshot(
        self,
        entry: _CachedGeneration,
        existing_adapter: OptimizerAdapter | None,
        consolidated_lora: Any | None = None,
    ) -> Iterator[tuple[Any, PreparedSafetensors]]:
        if entry.resident_lora is not None:
            if consolidated_lora is None:
                raise RuntimeError("resident LoRA consolidation returned no rank-zero value")
            yield consolidated_lora, self._prepare_lora_tensors(consolidated_lora)
            return
        snapshot = entry.resolved.result()
        if snapshot.lora is None:
            if existing_adapter is None:
                raise RuntimeError("rank zero has no LoRA snapshot to publish")
            yield self._load_existing_adapter(existing_adapter)
            return
        if snapshot.lora_residency_key is None:
            if snapshot.prepared_tensors is None:
                raise RuntimeError("LoRA snapshot has no serialized tensor layout")
            yield snapshot.lora, snapshot.prepared_tensors
            return
        if self._residency is None:
            raise RuntimeError("resident LoRA snapshot has no residency manager")
        with self._residency.borrow_l2(snapshot.lora_residency_key) as image:
            keys = tuple(sorted(snapshot.lora.tensors))
            tensors = image.tensors()
            if len(keys) != len(tensors):
                raise RuntimeError("resident LoRA tensor identity changed")
            lora = snapshot.lora.model_copy(
                update={"tensors": dict(zip(keys, tensors, strict=True))}
            )
            yield lora, self._prepare_lora_tensors(lora)

    def _consolidate_resident_lora(self, entry: _CachedGeneration) -> Any | None:
        resident = entry.resident_lora
        if resident is None or self._residency is None:
            raise RuntimeError("generation has no resident LoRA source")
        return self._sampler_pool.submit(
            self._consolidate_resident_lora_sync, resident
        ).result()

    def _consolidate_resident_lora_sync(
        self, resident: _ResidentLoraSource
    ) -> Any | None:
        from art.megatron.weights.lora_publish import LoraSnapshot
        from art.megatron.weights.rank_distributed_lora_publish import (
            consolidate_rank_distributed_vllm_lora_source,
        )

        if self._residency is None:
            raise RuntimeError("generation has no residency manager")
        source = resident.prepared.result()
        with self._residency.borrow_l2(resident.key) as image:
            names = tuple(
                tensor.name
                for tensor in source.metadata
                if tensor.owner_rank == int(self.runtime.rank)
            )
            tensors = image.tensors()
            if len(names) != len(tensors):
                raise RuntimeError("resident sampler tensor identity changed")
            consolidated = consolidate_rank_distributed_vllm_lora_source(
                source.model_copy(
                    update={"tensors": dict(zip(names, tensors, strict=True))}
                ),
                group=self.runtime.publication_group,
            )
        if consolidated is None:
            return None
        return LoraSnapshot(
            tensors=consolidated.tensors,
            adapter_config=consolidated.adapter_config,
        )

    @contextmanager
    def _optimizer_snapshot(
        self,
        entry: _CachedGeneration,
        generation: TrainerGeneration,
        *,
        required: bool,
    ) -> Iterator[Any | None]:
        if not required:
            yield None
            return
        snapshot = entry.resolved.result()
        if snapshot.optimizer is not None:
            yield snapshot.optimizer
            return
        if entry.optimizer_upgrade is not None:
            yield entry.optimizer_upgrade.result()
            return
        resident = entry.resident_optimizer
        if resident is None or self._residency is None:
            raise RuntimeError("optimizer persistence requires an optimizer snapshot")
        from art.megatron.optimizer_state import (
            trainer_rank_optimizer_snapshot_from_cpu,
        )

        with self._residency.borrow_l2(resident.key) as image:
            state = resident.source.bind(image.tensors())
            yield trainer_rank_optimizer_snapshot_from_cpu(
                self.runtime,
                state,
                generation_id=generation.generation_id,
                step=generation.policy_step,
            )

    def _transfer_prepared_snapshot(
        self, prepared: _PreparedRankSnapshot, submitted_at: float
    ) -> _SnapshotTransport:
        started = time.perf_counter()
        distributed = prepared.distributed_adapter
        if distributed is not None:
            from art.megatron.weights.rank_distributed_lora_publish import (
                PreparedRankDistributedLora,
                publish_rank_distributed_vllm_lora,
            )

            if not isinstance(distributed, PreparedRankDistributedLora):
                raise RuntimeError("ordered publication has an invalid distributed payload")
            target = prepared.adapter_object_target
            if not isinstance(target, OrderedBinaryObjectTarget):
                raise RuntimeError(
                    "rank-distributed adapter has no ordered object target"
                )
            local_error: BaseException | None = None
            store: S3BinaryObjectStore | None = None
            try:
                prepared.entry.resolved.result()
                store = self._object_store_for(target)
            except BaseException as error:
                local_error = error
            ready = time.perf_counter()
            group = self.runtime.publication_group
            if distributed.layout.world_size > 1 and group is None:
                raise RuntimeError(
                    "multi-rank ordered publication has no independent control group"
                )
            ref = publish_rank_distributed_vllm_lora(
                distributed,
                store,
                group=group,
                local_error=local_error,
            )
            if ref != distributed.layout.ref:
                raise RuntimeError(
                    "rank-distributed adapter publication changed identity"
                )
            adapter = (
                prepared.plan.transport_adapter
                if int(self.runtime.rank) == distributed.layout.coordinator_rank
                else None
            )
            if (adapter is None) != (
                int(self.runtime.rank) != distributed.layout.coordinator_rank
            ):
                raise RuntimeError(
                    "rank-distributed adapter plan changed coordinator ownership"
                )
            return _SnapshotTransport(
                adapter=adapter,
                metrics={
                    "time/snapshot_transport_queue_s": started - submitted_at,
                    "time/snapshot_transport_wait_s": ready - started,
                    "time/snapshot_transport_s": time.perf_counter() - ready,
                },
            )
        if int(self.runtime.rank) != 0:
            return _SnapshotTransport(
                metrics={"time/snapshot_transport_queue_s": started - submitted_at}
            )
        if not prepared.publication_targets and prepared.adapter_object_target is None:
            return _SnapshotTransport(
                metrics={"time/snapshot_transport_queue_s": started - submitted_at}
            )
        ready = time.perf_counter()
        payload = prepared.adapter
        if payload is None:
            raise RuntimeError("authorized adapter transport has no prepared payload")
        adapter = None
        if prepared.adapter_object_target is not None:
            planned = prepared.plan.transport_adapter
            if planned is None:
                raise RuntimeError("authorized object transport has no adapter plan")
            adapter = self._publish_lora_object_once(
                prepared.entry,
                prepared.adapter_object_target,
                prepared.plan.generation,
                payload,
                planned,
            )
        elif prepared.publication_targets:
            if payload.model_identity is None:
                raise RuntimeError("adapter transport has no exact file identity")
            self._transfer_lora_snapshot(
                payload.lora,
                prepared.publication_targets,
                prepared_tensors=payload.tensors,
                model_identity=payload.model_identity,
            )
        return _SnapshotTransport(
            adapter=adapter,
            metrics={
                "time/snapshot_transport_queue_s": started - submitted_at,
                "time/snapshot_transport_wait_s": ready - started,
                "time/snapshot_transport_s": time.perf_counter() - ready,
            },
        )

    def _persist_prepared_snapshot(
        self, prepared: _PreparedRankSnapshot, submitted_at: float
    ) -> _RankSnapshotPersistence:
        started = time.perf_counter()
        result = self._persist_generation(prepared)
        return result.model_copy(
            update={
                "metrics": {
                    "time/snapshot_persistence_queue_s": started - submitted_at,
                    "time/snapshot_persistence_wait_s": 0.0,
                    "time/snapshot_persistence_s": time.perf_counter() - started,
                }
            }
        )

    @staticmethod
    def _load_existing_adapter(
        adapter: OptimizerAdapter,
    ) -> tuple[Any, PreparedSafetensors]:
        from art.megatron.model_support.lora_disk import (
            ART_LORA_FORMAT_CONFIG_KEY,
            ART_LORA_FORMAT_VLLM,
            load_adapter_config,
            load_vllm_lora_tensors,
        )
        from art.megatron.weights.lora_publish import LoraSnapshot

        if read_adapter_publication(adapter.identity, step=adapter.step) != adapter:
            raise RuntimeError("existing adapter publication changed before transport")
        config = load_adapter_config(adapter.identity)
        if config.get(ART_LORA_FORMAT_CONFIG_KEY) != ART_LORA_FORMAT_VLLM:
            raise RuntimeError("existing adapter is not in vLLM format")
        tensors = load_vllm_lora_tensors(adapter.identity)
        return LoraSnapshot(tensors=tensors, adapter_config=config), (
            SafetensorsLayout(tensors).bind(tensors)
        )

    @staticmethod
    def _complete_publication(
        transport: Future[_SnapshotTransport],
        durability: Future[_RankSnapshotPersistence],
        submitted_at: float,
        plan: SnapshotRankWritePlan,
        grant: SnapshotWriteGrant,
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
            plan=plan,
            grant=grant,
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
        model_identity: FileIdentity,
    ) -> None:
        from art.distributed.adapter_transport import AdapterSnapshotSender

        with self._transport_sender_lock:
            if self._transport_sender is None:
                self._transport_sender = AdapterSnapshotSender()
            self._transport_sender.send(
                lora,
                targets,
                prepared_tensors=prepared_tensors,
                model_identity=model_identity,
            )

    def _publish_lora_object_once(
        self,
        entry: _CachedGeneration,
        target: BinaryObjectPublicationTarget,
        generation: TrainerGeneration,
        payload: _PreparedAdapterPayload,
        planned: OptimizerAdapter,
    ) -> OptimizerAdapter:
        with self._lock:
            publication = self._object_publications.get(target.object_id)
            owner = publication is None
            if publication is None:
                publication = Future()
                self._object_publications[target.object_id] = publication
            entry.object_ids.add(target.object_id)
        if not owner:
            adapter = publication.result()
            if adapter != planned:
                raise RuntimeError("cached object publication differs from its plan")
            return adapter
        try:
            adapter = self._publish_lora_object(target, generation, payload, planned)
        except BaseException as error:
            publication.set_exception(error)
            raise
        publication.set_result(adapter)
        return adapter

    def _publish_lora_object(
        self,
        target: BinaryObjectPublicationTarget,
        generation: TrainerGeneration,
        payload: _PreparedAdapterPayload,
        planned: OptimizerAdapter,
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
        store = self._object_store_for(target)
        planned_files = {file.name: file for file in planned.files}
        source_files: dict[str, tuple[memoryview, ...]] = {
            "adapter_model.safetensors": tuple(
                memoryview(chunk.numpy()).cast("B")
                for chunk in payload.tensors.chunks
            ),
            "adapter_config.json": (memoryview(payload.config),),
        }
        if isinstance(target, OrderedBinaryObjectTarget):
            ref = store.publish_ordered(target, source_files)
        else:
            if any(file.sha256 is None for file in planned_files.values()):
                raise RuntimeError("adapter object plan has no exact file identity")
            ref = store.publish(
                target,
                source_files,
                file_sha256={
                    name: cast(str, file.sha256)
                    for name, file in planned_files.items()
                },
            )
        published_files = {file.relative_path: file for file in ref.files}
        expected_files = ("adapter_config.json", "adapter_model.safetensors")
        if set(published_files) != set(expected_files):
            raise RuntimeError("adapter object manifest has unexpected files")
        if ref.manifest_uri != planned.identity or tuple(
            (published_files[name].byte_count, published_files[name].sha256)
            for name in expected_files
        ) != tuple(
            (planned_files[name].size_bytes, planned_files[name].sha256)
            for name in expected_files
        ):
            raise RuntimeError("adapter object publication differs from its plan")
        return planned

    def _object_store_for(
        self, target: BinaryObjectPublicationTarget
    ) -> S3BinaryObjectStore:
        with self._lock:
            if self._object_store is None:
                self._object_store = S3BinaryObjectStore(target.store)
            elif self._object_store.config != target.store:
                raise RuntimeError("generation publisher cannot mix object stores")
            return self._object_store

    def _persist_generation(
        self, prepared: _PreparedRankSnapshot
    ) -> _RankSnapshotPersistence:
        from art.megatron.optimizer_state import (
            adapter_publication_transaction,
            publish_adapter_checkpoint,
            write_optimizer_snapshot_shard,
        )

        rank = int(self.runtime.rank)
        generation = prepared.plan.generation
        adapter = prepared.plan.adapter
        if rank == 0:
            payload = prepared.adapter
            if payload is not None and prepared.staging_adapter_path is not None:
                if adapter is None:
                    raise RuntimeError("local adapter write has no authorized plan")
                with adapter_publication_transaction(
                    prepared.staging_adapter_path,
                    step=generation.policy_step,
                    training_session_id=generation.training_session_id,
                    generation_id=generation.generation_id,
                ) as (staging, existing):
                    if existing is not None:
                        if existing != adapter:
                            raise RuntimeError(
                                "existing adapter differs from authorized plan"
                            )
                    else:
                        staging.mkdir(parents=True)
                        if payload.model_identity is None:
                            raise RuntimeError(
                                "local adapter write has no exact file identity"
                            )
                        model_identity = save_prepared_safetensors(
                            payload.tensors,
                            staging / "adapter_model.safetensors",
                            identity=payload.model_identity,
                        )
                        config_path = staging / "adapter_config.json"
                        config_path.write_bytes(payload.config)
                        if (
                            model_identity != payload.model_identity
                            or config_path.stat().st_size
                            != payload.config_identity.size_bytes
                        ):
                            raise RuntimeError(
                                "adapter bytes changed after write authorization"
                            )
                        published = publish_adapter_checkpoint(
                            staging,
                            step=generation.policy_step,
                            files=adapter.files,
                            training_session_id=generation.training_session_id,
                            generation_id=generation.generation_id,
                        )
                        if published != adapter:
                            raise RuntimeError(
                                "adapter publication differs from authorized plan"
                            )
        optimizer = prepared.optimizer
        if optimizer is None:
            shard = None
        else:
            archive = prepared.optimizer_archive
            identity = prepared.optimizer_identity
            if archive is None or identity is None:
                raise RuntimeError("authorized optimizer has no prepared archive")
            shard = write_optimizer_snapshot_shard(
                optimizer,
                optimizer_state_path=prepared.optimizer_state_path,
                prepared=archive,
                identity=identity,
            )
        if shard != prepared.plan.optimizer_shard:
            raise RuntimeError("optimizer shard differs from authorized plan")
        return _RankSnapshotPersistence(
            generation=generation,
            rank=rank,
            adapter=adapter,
            shard=shard,
            runtime_sha256=prepared.plan.runtime_sha256,
            topology=prepared.plan.topology,
            saves_optimizer=optimizer is not None,
        )

    def _authorized_completed(
        self,
        future: Future[TrainerRankPublication],
        *,
        prepared: _PreparedRankSnapshot,
    ) -> None:
        callback_started = time.perf_counter()
        try:
            record = future.result()
            callback_delay_s = max(
                0.0,
                callback_started
                - prepared.prepared_at
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
            prepared.contexts.close()
            prepared.sink.publication(event)
        except BaseException as error:
            try:
                prepared.contexts.close()
            except BaseException as cleanup_error:
                error.add_note(
                    "snapshot context cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            self._report_failure(
                error,
                entry=prepared.entry,
                sink=prepared.sink,
                generation=prepared.plan.generation,
                remember=True,
            )
            if not prepared.completion.done():
                prepared.completion.set_exception(error)
            self._retire_prepared(prepared)
            return
        prepared.completion.set_result(record)
        self._retire_prepared(prepared)

    def _authorization_failed(
        self, prepared: _PreparedRankSnapshot, error: BaseException
    ) -> None:
        try:
            prepared.contexts.close()
        except BaseException as cleanup_error:
            error.add_note(
                "snapshot context cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        self._report_failure(
            error,
            entry=prepared.entry,
            sink=prepared.sink,
            generation=prepared.plan.generation,
            remember=True,
        )
        if not prepared.completion.done():
            prepared.completion.set_exception(error)
        self._retire_prepared(prepared)

    def _retire_prepared(self, prepared: _PreparedRankSnapshot) -> None:
        with self._lock:
            if self._prepared.get(prepared.operation_id) is prepared:
                self._prepared.pop(prepared.operation_id)
            try:
                self._prepared_order.remove(prepared.operation_id)
            except ValueError:
                pass
            if prepared.entry.ephemeral:
                prepared.entry.retired = True
        self._maybe_release(prepared.entry)

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

    def _residency_retired(self, future: Future[None]) -> None:
        with self._lock:
            self._residency_retirements.discard(future)
            if not future.cancelled() and (error := future.exception()) is not None:
                self._failures.append(error)

    def raise_if_failed(self) -> None:
        with self._lock:
            failures = tuple(self._failures)
        if failures:
            raise BaseExceptionGroup("trainer generation publication failed", failures)

    def close(self) -> None:
        failures: list[BaseException] = []
        with self._lock:
            dormant = tuple(
                prepared
                for prepared in self._prepared.values()
                if not prepared.authorized
            )
        for prepared in dormant:
            self._authorization_failed(
                prepared,
                RuntimeError("snapshot publisher closed before write authorization"),
            )
        with self._lock:
            entries = tuple(self._cache.values())
            for entry in entries:
                entry.retired = True
        for entry in entries:
            self._maybe_release(entry)
        self._sampler_pool.shutdown(wait=True)
        self._resolution_pool.shutdown(wait=True)
        self._transport_pool.shutdown(wait=True)
        self._ordered_transport_pool.shutdown(wait=True)
        self._durability_pool.shutdown(wait=True)
        self._completion_pool.shutdown(wait=True)
        for entry in entries:
            self._maybe_release(entry)
        with self._lock:
            retirements = tuple(self._residency_retirements)
        _done, pending = wait(
            retirements,
            timeout=(
                None
                if self._residency is None
                else self._residency.config.shutdown_timeout_s
            ),
        )
        if pending:
            failures.append(
                TimeoutError(
                    f"{len(pending)} sampler residency retirements exceeded shutdown timeout"
                )
            )
        if self._transport_sender is not None:
            try:
                self._transport_sender.close()
            except BaseException as error:
                failures.append(error)
            self._transport_sender = None
        if self._object_store is not None:
            try:
                self._object_store.close()
            except BaseException as error:
                failures.append(error)
            self._object_store = None
        with self._lock:
            in_flight = self._in_flight
        if in_flight:
            failures.append(
                RuntimeError(f"publication close retained {in_flight} snapshots")
            )
        try:
            self.raise_if_failed()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("generation publisher close failed", failures)
