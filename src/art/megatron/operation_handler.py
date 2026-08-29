from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from art.distributed.art_runtime import ArtRuntime, DistributedPackedBatch
from art.distributed.packing import PackingRequest
from art.distributed.rollout import RolloutModelSpec
from art.training import (
    CheckpointRef,
    CommandExecutionUsage,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    OperationExecutionError,
    OperationRef,
    OperationResultType,
    OperationWorker,
    OptimStepRequest,
    OptimStepResult,
    PackedInputCaptureRef,
    PackingOutcome,
    RlTrajectoryBatch,
    RunCommand,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
    TokenLogprobs,
    UsageMeasurement,
    bootstrap_operation_worker,
)

from .runtime.specs import (
    CurrentTrainConfig,
    ExperimentalTrainConfig,
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    OptimizerJobSpec,
    TrainerGeneration,
)


class MegatronCheckpointOperations(Protocol):
    """ART-owned persistence seam implemented by the selected slot runtime."""

    async def save_weights_for_sampler(
        self,
        request: SaveWeightsForSamplerRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> SamplerWeightsResult: ...

    async def save_state(
        self,
        request: SaveStateRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> SaveStateResult: ...

    async def load_state(
        self,
        request: LoadStateRequest,
        operation: OperationRef,
    ) -> "MegatronLoadedState": ...


class MegatronLoadedState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: LoadStateResult
    generation: TrainerGeneration
    optimizer_state_path: str = Field(min_length=1)


class MegatronOperationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    source: TrainerGeneration
    optimizer_state_path: str = Field(min_length=1)
    rollout_model: RolloutModelSpec
    train_config: CurrentTrainConfig = CurrentTrainConfig()
    output_adapter_root: str = Field(min_length=1)
    max_retained_inputs: int = Field(default=65, ge=1, le=256)


@dataclass(slots=True)
class _CapturedInput:
    ref: PackedInputCaptureRef
    request_fingerprint: str
    control_fingerprint: str
    packed: DistributedPackedBatch
    packing: PackingOutcome
    owners: set[str] = field(default_factory=set)


class _ResidentTrainer(Protocol):
    runtime_spec: Any

    async def forward(self, job: ForwardJobSpec, batch: Any) -> dict[str, Any]: ...

    async def forward_backward(
        self, job: ForwardBackwardJobSpec, batch: Any
    ) -> dict[str, Any]: ...

    async def optim_step(self, job: OptimizerJobSpec) -> dict[str, Any]: ...


class MegatronOperationHandler:
    """Concrete command boundary for one persistent Megatron training run."""

    def __init__(
        self,
        runtime: ArtRuntime,
        trainer: _ResidentTrainer,
        config: MegatronOperationConfig,
        *,
        checkpoints: MegatronCheckpointOperations | None = None,
    ) -> None:
        if config.source.training_session_id != config.training_session_id:
            raise ValueError("source generation belongs to another training session")
        self.runtime = runtime
        self.trainer = trainer
        self.config = config
        self.checkpoints = checkpoints
        self._generation = config.source
        self._optimizer_state_path = config.optimizer_state_path
        self._captures: dict[str, _CapturedInput] = {}
        self._contributions: dict[str, str] = {}
        self._capture_lock = asyncio.Lock()
        self._release_failures: dict[str, BaseException] = {}

    @property
    def generation(self) -> TrainerGeneration:
        return self._generation

    async def prepare_input(
        self,
        request: ForwardRequest | ForwardBackwardRequest,
        operation: OperationRef,
    ) -> PackedInputCaptureRef:
        """Pack one raw request once and retain its exact slot-local lease."""

        if operation.kind not in {"forward", "forward_backward"}:
            raise ValueError("only forward operations own packed input")
        expected_kind = (
            "forward_backward"
            if isinstance(request, ForwardBackwardRequest)
            else "forward"
        )
        if operation.kind != expected_kind:
            raise ValueError("request and operation kinds differ")
        if (
            request.run_id != operation.run_id
            or request.sequence_id != operation.sequence_id
            or operation.run_id != self.config.run_id
        ):
            raise ValueError("request and operation identities differ")
        if isinstance(request.batch, PackedInputCaptureRef):
            captured = await self._require_capture(request.batch, operation)
            if captured.control_fingerprint != _input_control_fingerprint(
                request, operation
            ):
                raise ValueError("packed-input command controls changed")
            return captured.ref
        if not isinstance(request.batch, RlTrajectoryBatch):
            raise OperationExecutionError(
                "invalid_request",
                "the persistent Megatron packer currently requires an RL batch",
                usage=CommandExecutionUsage.no_work(),
            )
        fingerprint = _input_fingerprint(request, operation)
        capture_id = operation.operation_id
        async with self._capture_lock:
            prior = self._captures.get(capture_id)
            if prior is not None:
                if prior.request_fingerprint != fingerprint:
                    raise RuntimeError("packed-input operation was reused")
                return prior.ref
            if len(self._captures) >= self.config.max_retained_inputs:
                raise OperationExecutionError(
                    "capacity_exhausted",
                    "retained packed-input capacity is exhausted",
                    usage=CommandExecutionUsage.no_work(),
                )
            packed = await self.runtime.pack(self._packing_request(request, operation))
            if packed is None:
                raise ValueError("training input produced no packed sequence")
            packing = self._packing_outcome(packed, request)
            ref = PackedInputCaptureRef(
                run_id=operation.run_id,
                capture_id=capture_id,
                manifest_sha256=_capture_manifest_sha256(
                    operation, packed, packing, fingerprint
                ),
                input_kind="rl",
                min_source_version=request.batch.min_source_version,
                max_source_version=request.batch.max_source_version,
            )
            self._captures[capture_id] = _CapturedInput(
                ref=ref,
                request_fingerprint=fingerprint,
                control_fingerprint=_input_control_fingerprint(request, operation),
                packed=packed,
                packing=packing,
            )
            return ref

    async def __call__(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributing_forward_backward_operation_ids: tuple[str, ...],
    ) -> OperationResultType:
        if operation.run_id != self.config.run_id:
            raise ValueError("operation belongs to another run")
        if isinstance(request, ForwardBackwardRequest):
            return await self._forward(request, operation, backward=True)
        if isinstance(request, ForwardRequest):
            return await self._forward(request, operation, backward=False)
        if isinstance(request, OptimStepRequest):
            return await self._optim_step(
                request, operation, contributing_forward_backward_operation_ids
            )
        if self.checkpoints is None:
            raise OperationExecutionError(
                "execution_failed",
                "Megatron checkpoint operations are not configured",
                usage=CommandExecutionUsage.no_work(),
            )
        if isinstance(request, SaveWeightsForSamplerRequest):
            return await self.checkpoints.save_weights_for_sampler(
                request, operation, self._generation
            )
        if isinstance(request, SaveStateRequest):
            return await self.checkpoints.save_state(
                request, operation, self._generation
            )
        if isinstance(request, LoadStateRequest):
            loaded = await self.checkpoints.load_state(request, operation)
            if loaded.result.operation_id != operation.operation_id:
                raise RuntimeError("checkpoint loader changed operation identity")
            self._generation = loaded.generation
            self._optimizer_state_path = loaded.optimizer_state_path
            return loaded.result
        raise TypeError(f"unsupported command type {type(request).__name__}")

    async def release_operation_input(self, operation_id: str) -> None:
        """Release a completed forward capture; F/B releases at optimizer commit."""

        capture_id = self._contributions.pop(operation_id, None)
        if capture_id is None:
            capture_id = operation_id if operation_id in self._captures else None
        if capture_id is None:
            return
        captured = self._captures[capture_id]
        captured.owners.discard(operation_id)
        await self._release_if_unowned(capture_id)

    async def discard_prepared_input(self, ref: PackedInputCaptureRef) -> None:
        captured = await self._require_capture(ref, None)
        if captured.owners:
            raise RuntimeError("cannot discard packed input owned by an operation")
        await self._release_if_unowned(ref.capture_id)

    async def retry_releases(self) -> None:
        for capture_id in tuple(self._release_failures):
            await self._release_if_unowned(capture_id)

    def retained_contribution_inputs(
        self,
    ) -> tuple[tuple[str, PackedInputCaptureRef], ...]:
        return tuple(
            (operation_id, self._captures[capture_id].ref)
            for operation_id, capture_id in self._contributions.items()
        )

    async def _forward(
        self,
        request: ForwardRequest | ForwardBackwardRequest,
        operation: OperationRef,
        *,
        backward: bool,
    ) -> ForwardResult | ForwardBackwardResult:
        capture_ref = await self.prepare_input(request, operation)
        captured = await self._require_capture(capture_ref, operation)
        try:
            common = {
                "operation": operation,
                "training_session_id": self.config.training_session_id,
                "source": self._generation,
                "optimizer_state_path": self._optimizer_state_path,
                "batch": captured.packed.leases.ref,
                "expected_global_loss_bearing_tokens": (
                    captured.packing.loss_bearing_tokens
                ),
                "config": _train_config(self.config.train_config, request),
                "experimental_config": _experimental_config(request),
                "return_token_logprobs": request.return_token_logprobs,
            }
            if backward:
                raw = await self.trainer.forward_backward(
                    ForwardBackwardJobSpec(**common), captured.packed.leases
                )
            else:
                raw = await self.trainer.forward(
                    ForwardJobSpec(**common), captured.packed.leases
                )
        except BaseException as error:
            await self._release_after_failed_execution(capture_ref.capture_id, error)
            raise
        captured.owners.add(operation.operation_id)
        if backward:
            self._contributions[operation.operation_id] = capture_ref.capture_id
        usage = CommandExecutionUsage(
            logical_nonpadding_tokens=UsageMeasurement.complete(
                int(raw["logical_nonpadding_tokens"])
            ),
            executed_token_equivalents=UsageMeasurement.complete(
                int(raw["executed_token_equivalents"])
            ),
            gpu_service_ns=UsageMeasurement.unknown(),
        )
        result_type = ForwardBackwardResult if backward else ForwardResult
        return result_type(
            operation_id=operation.operation_id,
            packing=captured.packing,
            packed_input_capture=capture_ref,
            token_logprobs=tuple(
                TokenLogprobs.model_validate(value)
                for value in raw.get("token_logprobs", ())
            ),
            metrics={**_packing_metrics(captured.packed), **raw.get("metrics", {})},
            usage=usage,
        )

    async def _optim_step(
        self,
        request: OptimStepRequest,
        operation: OperationRef,
        contributions: tuple[str, ...],
    ) -> OptimStepResult:
        missing = [item for item in contributions if item not in self._contributions]
        if missing:
            raise ValueError(f"optimizer input captures are missing: {missing}")
        generation = _next_generation(self.config, operation)
        raw = await self.trainer.optim_step(
            OptimizerJobSpec(
                operation=operation,
                training_session_id=self.config.training_session_id,
                generation=generation,
                contributing_forward_backward_operation_ids=contributions,
                optimizer=request.optimizer,
            )
        )
        consumed = tuple(raw["contributing_forward_backward_operation_ids"])
        if consumed != contributions:
            raise RuntimeError("trainer consumed the wrong packed-input captures")
        self._generation = generation
        release_ids = set()
        for contribution in contributions:
            capture_id = self._contributions.pop(contribution)
            self._captures[capture_id].owners.discard(contribution)
            release_ids.add(capture_id)
        for capture_id in release_ids:
            await self._release_if_unowned(capture_id)
        return OptimStepResult(
            operation_id=operation.operation_id,
            contributing_forward_backward_operation_ids=contributions,
            checkpoint=CheckpointRef(
                run_id=operation.run_id,
                learner_version=generation.policy_step,
                checkpoint_id=generation.generation_id,
            ),
            metrics=raw.get("metrics", {}),
            usage=CommandExecutionUsage.not_applicable(),
        )

    async def _require_capture(
        self,
        ref: PackedInputCaptureRef,
        operation: OperationRef | None,
    ) -> _CapturedInput:
        if ref.run_id != self.config.run_id:
            raise ValueError("packed input belongs to another run")
        captured = self._captures.get(ref.capture_id)
        if captured is None or captured.ref != ref:
            raise ValueError("packed-input capture is absent or changed")
        if operation is not None and ref.capture_id != operation.operation_id:
            raise ValueError("packed input belongs to another operation")
        return captured

    async def _release_after_failed_execution(
        self, capture_id: str, primary: BaseException
    ) -> None:
        try:
            await self._release_if_unowned(capture_id)
        except BaseException as cleanup:
            primary.add_note(
                f"packed-input cleanup also failed: {type(cleanup).__name__}: {cleanup}"
            )

    async def _release_if_unowned(self, capture_id: str) -> None:
        captured = self._captures.get(capture_id)
        if captured is None or captured.owners:
            return
        try:
            await self.runtime.release_batch(captured.packed)
        except BaseException as error:
            self._release_failures[capture_id] = error
            return
        self._release_failures.pop(capture_id, None)
        self._captures.pop(capture_id, None)

    def _packing_request(
        self,
        request: ForwardRequest | ForwardBackwardRequest,
        operation: OperationRef,
    ) -> PackingRequest:
        assert isinstance(request.batch, RlTrajectoryBatch)
        experimental = _experimental_config(request)
        return PackingRequest(
            model=self.config.rollout_model,
            generation_id=operation.operation_id,
            trajectory_groups=request.batch.groups,
            advantage_balance=experimental.advantage_balance,
            allow_training_without_logprobs=bool(
                experimental.allow_training_without_logprobs
            ),
            scale_rewards=experimental.scale_rewards,
            plot_tensors=bool(experimental.plot_tensors),
            packed_sequence_length=self.trainer.runtime_spec.packed_sequence_length,
            logprob_calculation_chunk_size=(
                experimental.logprob_calculation_chunk_size or 1024
            ),
            include_moe_routing=self.trainer.runtime_spec.enable_moe_routing_replay,
            collect_packing_shapes=request.collect_packing_shapes,
            group_ids=tuple(
                f"{operation.operation_id}:{index}"
                for index in range(len(request.batch.groups))
            ),
            min_source_version=request.batch.min_source_version,
            max_source_version=request.batch.max_source_version,
        )

    def _packing_outcome(
        self,
        packed: DistributedPackedBatch,
        request: ForwardRequest | ForwardBackwardRequest,
    ) -> PackingOutcome:
        ref = packed.leases.ref
        stats = ref.prefix_tree_packing_stats
        if stats is None:
            raise RuntimeError("packed input has no exact packing statistics")
        config = _train_config(self.config.train_config, request)
        target = config.grad_accumulation_sequences or _data_parallel_size(self.trainer)
        return PackingOutcome(
            packed_sequence_length=ref.sequence_length,
            packed_sequences=ref.num_sequences,
            target_packed_sequences=target,
            physical_tokens=stats.physical_tokens,
            non_padding_tokens=packed.non_padding_tokens,
            loss_bearing_tokens=packed.loss_bearing_tokens,
            trainable_assistant_tokens=packed.trainable_assistant_tokens,
            group_shapes=tuple(
                cast(Any, shape)
                for shape in packed.packed_group_shapes
                if shape is not None
            ),
        )


@dataclass(frozen=True, slots=True)
class MegatronOperationRuntime:
    handler: MegatronOperationHandler
    worker: OperationWorker


def bootstrap_megatron_operation_worker(
    runtime: ArtRuntime,
    trainer: _ResidentTrainer,
    config: MegatronOperationConfig,
    *,
    checkpoints: MegatronCheckpointOperations | None = None,
    max_retained_operations: int = 128,
) -> MegatronOperationRuntime:
    handler = MegatronOperationHandler(
        runtime, trainer, config, checkpoints=checkpoints
    )
    return MegatronOperationRuntime(
        handler=handler,
        worker=bootstrap_operation_worker(
            handler, max_retained_operations=max_retained_operations
        ),
    )


def _train_config(
    base: CurrentTrainConfig,
    request: ForwardRequest | ForwardBackwardRequest,
) -> CurrentTrainConfig:
    updates = {
        name: value
        for name, value in request.loss.values.items()
        if name in CurrentTrainConfig.model_fields
    }
    return CurrentTrainConfig.model_validate(
        {**base.model_dump(mode="python"), **updates}
    )


def _experimental_config(
    request: ForwardRequest | ForwardBackwardRequest,
) -> ExperimentalTrainConfig:
    values = {
        name: value
        for name, value in request.loss.values.items()
        if name in ExperimentalTrainConfig.model_fields
    }
    values["ppo"] = request.loss.name == "ppo"
    values["scale_rewards"] = bool(values.get("scale_rewards", True))
    return ExperimentalTrainConfig.model_validate(values)


def _data_parallel_size(trainer: _ResidentTrainer) -> int:
    mesh = trainer.runtime_spec.trainer_mesh
    topology = mesh.topology
    return len(mesh.ranks) // (topology.tp * topology.cp * topology.pp)


def _next_generation(
    config: MegatronOperationConfig, operation: OperationRef
) -> TrainerGeneration:
    version = operation.reserved_output_learner_version
    if version is None:
        raise ValueError("optimizer operation has no reserved learner version")
    suffix = hashlib.sha256(operation.operation_id.encode()).hexdigest()[:32]
    generation_id = f"step-{version:08d}-{suffix}"
    return TrainerGeneration(
        training_session_id=config.training_session_id,
        policy_step=version,
        generation_id=generation_id,
        adapter_path=f"{config.output_adapter_root.rstrip('/')}/{generation_id}",
    )


def _input_fingerprint(
    request: ForwardRequest | ForwardBackwardRequest, operation: OperationRef
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "operation": operation.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _input_control_fingerprint(
    request: ForwardRequest | ForwardBackwardRequest, operation: OperationRef
) -> str:
    command = request.model_dump(mode="json", exclude={"batch"})
    return hashlib.sha256(
        json.dumps(
            {"command": command, "operation": operation.model_dump(mode="json")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _capture_manifest_sha256(
    operation: OperationRef,
    packed: DistributedPackedBatch,
    packing: PackingOutcome,
    request_fingerprint: str,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "operation_id": operation.operation_id,
                "request_fingerprint": request_fingerprint,
                "packed": packed.leases.ref.model_dump(mode="json"),
                "packing": packing.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _packing_metrics(packed: DistributedPackedBatch) -> dict[str, float]:
    return {
        "time/step_trajectory_fetch_s": packed.trajectory_fetch_s,
        "time/step_packing_core_s": packed.packing_core_s,
        "time/step_trajectory_log_wait_s": packed.trajectory_log_wait_s,
        "time/step_packed_batch_finalize_s": packed.packed_batch_finalize_s,
        "time/step_packing_rpc_s": packed.packing_rpc_s,
        "time/step_packed_batch_fanout_s": packed.packed_batch_fanout_s,
    }
