from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from art.distributed.art_runtime import ArtRuntime, DistributedPackedBatch
from art.distributed.packing import PackingRequest
from art.distributed.rollout import RolloutModelSpec
from art.distributed.trajectory_store import retained_route_bundles_from_bundles
from art.preprocessing.sft import SftBatchTokenizer
from art.training import (
    AdapterSpec,
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
    SupervisedTrajectoryBatch,
    TokenizedTrainingBatch,
    TokenLogprobs,
    UsageMeasurement,
    bootstrap_operation_worker,
)
from art.vllm_route_transport import RetainedRouteBundleRef

from .route_retention import (
    RouteBundleOwnershipHandle,
    RouteBundleOwnershipProvider,
)
from .runtime.data_plane import SFTBatchData
from .runtime.numerical_capture import ForwardBackwardNumericalCaptureReceipt
from .runtime.publication import TrainerRankPublication
from .runtime.specs import (
    CommandPublicationSpec,
    CurrentTrainConfig,
    ExperimentalTrainConfig,
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    OptimizerJobSpec,
    SftForwardBackwardJobSpec,
    SftForwardJobSpec,
    TrainerGeneration,
)

POLICY_ACTIVATION_LAG_METRIC = "publication/policy_activation_lag_s"


class MegatronCheckpointOperations(Protocol):
    """ART-owned persistence seam implemented by the selected slot runtime."""

    async def save_weights_for_sampler(
        self,
        request: SaveWeightsForSamplerRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> SamplerWeightsResult | "MegatronSamplerPublicationReceipt": ...

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

    async def plan_artifacts(
        self,
        request: SaveWeightsForSamplerRequest | SaveStateRequest | LoadStateRequest,
        generation: TrainerGeneration,
    ) -> "MegatronArtifactResourcePlan": ...


class MegatronPairedPublisher(Protocol):
    async def aclose(self) -> None: ...

    async def save_weights_for_sampler(
        self,
        request: SaveWeightsForSamplerRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
        *,
        template_adapter_path: str,
        optimizer_state_path: str,
        staging_adapter_path: str,
    ) -> "MegatronSamplerPublicationReceipt": ...

    async def plan_artifacts(
        self,
        request: SaveWeightsForSamplerRequest,
        generation: TrainerGeneration,
        *,
        template_adapter_path: str,
    ) -> "MegatronArtifactResourcePlan": ...


class MegatronArtifactResourcePlan(BaseModel):
    """Checkpoint bytes reserved before a command enters service admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    basis: Literal["exact", "bounded"]
    checkpoint_objects: int = Field(ge=0)
    lora_bytes: int = Field(ge=0)
    transfer_bytes: int = Field(ge=0)
    storage_bytes: int = Field(ge=0)


class MegatronRetainedState(BaseModel):
    """Exact durable owner transferred into service retention after success."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str = Field(min_length=1, max_length=255)
    resource: Literal["lora", "storage"]
    bytes: int = Field(gt=0, le=(1 << 63) - 1)
    work_fingerprint: str = Field(min_length=1, max_length=128)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_expiry(self) -> "MegatronRetainedState":
        if self.expires_at is not None and self.expires_at.utcoffset() is None:
            raise ValueError("retained-state expiry must include a timezone")
        return self


class MegatronPolicyActivationTiming(BaseModel):
    """Exact holder-local endpoints for one learner's serving activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trainer_completed_monotonic_s: FiniteFloat = Field(ge=0)
    serving_activated_monotonic_s: FiniteFloat = Field(ge=0)

    @model_validator(mode="after")
    def _validate_order(self) -> "MegatronPolicyActivationTiming":
        if self.serving_activated_monotonic_s < self.trainer_completed_monotonic_s:
            raise ValueError("serving activation preceded trainer completion")
        return self

    @property
    def activation_lag_s(self) -> float:
        return float(
            self.serving_activated_monotonic_s - self.trainer_completed_monotonic_s
        )


class MegatronInferenceUpdateUsage(BaseModel):
    """Disjoint service-visible time spent staging and applying one LoRA update."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    staging_s: FiniteFloat = Field(ge=0)
    apply_s: FiniteFloat = Field(ge=0)


class MegatronSamplerPublicationReceipt(BaseModel):
    """Private durable proof for one operation-keyed serving publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1, max_length=64)
    request_id: str = Field(min_length=1, max_length=255)
    publication_mode: Literal[
        "versioned_lora", "in_flight_lora", "external_lora", "merged_weights"
    ]
    requested_public_alias: str = Field(min_length=1, max_length=255)
    runtime_model_name: str = Field(min_length=1, max_length=512)
    runtime_lora_name: str | None = Field(default=None, min_length=1, max_length=512)
    serving_generation_id: str = Field(min_length=1, max_length=255)
    learner_version: int = Field(ge=0)
    policy_activation_timing: MegatronPolicyActivationTiming | None = None
    inference_update_usage: MegatronInferenceUpdateUsage | None = None
    holder_update_sequence: int | None = Field(default=None, ge=0)
    holder_update_id: str | None = Field(default=None, min_length=1, max_length=255)
    retained: tuple[MegatronRetainedState, ...] = Field(min_length=1, max_length=8)
    result: SamplerWeightsResult

    def validate_command(
        self,
        request: SaveWeightsForSamplerRequest,
        operation: OperationRef,
        generation: TrainerGeneration,
    ) -> None:
        alias = request.publication.model_alias
        if (
            request.publication.mode == "none"
            or alias is None
            or self.operation_id != operation.operation_id
            or self.request_id != request.request_id
            or self.publication_mode != request.publication.mode
            or self.requested_public_alias != alias
            or self.serving_generation_id != generation.generation_id
            or self.learner_version != generation.policy_step
            or self.result.operation_id != operation.operation_id
            or self.result.checkpoint.run_id != operation.run_id
            or self.result.checkpoint.learner_version != generation.policy_step
        ):
            raise RuntimeError("sampler publication receipt changed command identity")
        paired_lora = self.publication_mode in {"versioned_lora", "in_flight_lora"}
        external_lora = self.publication_mode == "external_lora"
        holder_backed = not external_lora
        holder_update = (
            self.holder_update_sequence is not None
            and self.holder_update_id is not None
        )
        if (
            paired_lora != (self.runtime_lora_name is not None)
            or holder_backed != holder_update
            or external_lora != (self.result.external_lora is not None)
            or any(
                item.resource != ("lora" if paired_lora or external_lora else "storage")
                for item in self.retained
            )
        ):
            raise RuntimeError("sampler publication retained the wrong resource kind")
        timing = self.policy_activation_timing
        if paired_lora and timing is None:
            raise RuntimeError("paired publication omitted policy activation timing")
        if paired_lora != (self.inference_update_usage is not None):
            raise RuntimeError("sampler publication returned the wrong update usage")
        if (
            timing is not None
            and self.result.metrics.get(POLICY_ACTIVATION_LAG_METRIC)
            != timing.activation_lag_s
        ):
            raise RuntimeError("public policy activation lag changed receipt evidence")
        if not holder_backed and (
            self.holder_update_sequence is not None or self.holder_update_id is not None
        ):
            raise RuntimeError("non-holder publication returned holder update evidence")
        if len({item.owner_id for item in self.retained}) != len(self.retained):
            raise RuntimeError("sampler publication owner IDs must be unique")


class MegatronLoadedState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: LoadStateResult
    generation: TrainerGeneration
    optimizer_state_path: str = Field(min_length=1)


class MegatronOperationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    adapter: AdapterSpec
    source: TrainerGeneration
    initial_operation_sequence: int = Field(default=0, ge=0)
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
    packed: DistributedPackedBatch | None
    sft: SFTBatchData | None
    packing: PackingOutcome
    route_ownership: RouteBundleOwnershipHandle | None = None
    packed_released: bool = False
    owners: set[str] = field(default_factory=set)
    retained_for_replay: bool = False
    input_metrics: dict[str, float] = field(default_factory=dict)


class _ResidentTrainer(Protocol):
    runtime_spec: Any

    async def forward(self, job: ForwardJobSpec, batch: Any) -> dict[str, Any]: ...

    async def start_forward(
        self, job: ForwardJobSpec, batch: Any
    ) -> "_TrainerCommandLaunch": ...

    async def forward_backward(
        self, job: ForwardBackwardJobSpec, batch: Any
    ) -> dict[str, Any]: ...

    async def start_forward_backward(
        self, job: ForwardBackwardJobSpec, batch: Any
    ) -> "_TrainerCommandLaunch": ...

    async def sft_forward(
        self, job: SftForwardJobSpec, batch: SFTBatchData
    ) -> dict[str, Any]: ...

    async def start_sft_forward(
        self, job: SftForwardJobSpec, batch: SFTBatchData
    ) -> "_TrainerCommandLaunch": ...

    async def sft_forward_backward(
        self, job: SftForwardBackwardJobSpec, batch: SFTBatchData
    ) -> dict[str, Any]: ...

    async def start_sft_forward_backward(
        self, job: SftForwardBackwardJobSpec, batch: SFTBatchData
    ) -> "_TrainerCommandLaunch": ...

    async def optim_step(self, job: OptimizerJobSpec) -> dict[str, Any]: ...

    async def publish_command_generation(
        self, spec: CommandPublicationSpec
    ) -> tuple[tuple[TrainerRankPublication, ...], dict[str, float]]: ...

    async def capture_forward_backward_numerics(
        self,
        run_id: str,
        operation_id: str,
        batch: Any,
        root: str,
    ) -> ForwardBackwardNumericalCaptureReceipt: ...

    async def record_control_command(
        self,
        operation: OperationRef,
        learner_version: int,
    ) -> None: ...

    async def record_no_work_command(
        self,
        operation: OperationRef,
        learner_version: int,
    ) -> None: ...


class _TrainerCommandLaunch(Protocol):
    @property
    def completion(self) -> asyncio.Future[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class MegatronOperationLaunch:
    completion: asyncio.Future[OperationResultType]


class MegatronOperationHandler:
    """Concrete command boundary for one persistent Megatron training run."""

    def __init__(
        self,
        runtime: ArtRuntime,
        trainer: _ResidentTrainer,
        config: MegatronOperationConfig,
        *,
        checkpoints: MegatronCheckpointOperations | None = None,
        publisher: MegatronPairedPublisher | None = None,
        route_ownership: RouteBundleOwnershipProvider | None = None,
    ) -> None:
        if config.source.training_session_id != config.training_session_id:
            raise ValueError("source generation belongs to another training session")
        self.runtime = runtime
        self.trainer = trainer
        self.config = config
        self.checkpoints = checkpoints
        self.publisher = publisher
        self.route_ownership = route_ownership
        self._generation = config.source
        self._optimizer_state_path = config.optimizer_state_path
        self._captures: dict[str, _CapturedInput] = {}
        self._released_captures: dict[str, PackedInputCaptureRef] = {}
        self._contributions: dict[str, str] = {}
        self._capture_lock = asyncio.Lock()
        self._release_failures: dict[str, BaseException] = {}
        self._sampler_publications: dict[str, MegatronSamplerPublicationReceipt] = {}
        self._sft_tokenizer = SftBatchTokenizer()

    @property
    def generation(self) -> TrainerGeneration:
        return self._generation

    @property
    def optimizer_state_path(self) -> str:
        return self._optimizer_state_path

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
            if request.batch.capture_id != operation.operation_id and (
                not captured.retained_for_replay or captured.ref.content_sha256 is None
            ):
                raise ValueError("packed input was not retained for replay")
            if captured.control_fingerprint != _input_control_fingerprint(
                request, operation
            ):
                raise ValueError("packed-input command controls changed")
            return captured.ref
        if not isinstance(
            request.batch,
            (RlTrajectoryBatch, SupervisedTrajectoryBatch, TokenizedTrainingBatch),
        ):
            raise OperationExecutionError(
                "invalid_request",
                "the persistent Megatron runtime requires an RL, SFT, or tokenized batch",
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
            if capture_id in self._released_captures:
                raise RuntimeError("packed-input operation was already released")
            if len(self._captures) >= self.config.max_retained_inputs:
                raise OperationExecutionError(
                    "capacity_exhausted",
                    "retained packed-input capacity is exhausted",
                    usage=CommandExecutionUsage.no_work(),
                )
            if isinstance(request.batch, SupervisedTrajectoryBatch):
                tokenized = self._sft_tokenizer.tokenize(
                    self.config.rollout_model.build(), request.batch
                )
                packing = _sft_packing_outcome(
                    tokenized,
                    configured_sequence_length=self.trainer.runtime_spec.packed_sequence_length,
                    target_sequences=(
                        _train_config(
                            self.config.train_config, request
                        ).grad_accumulation_sequences
                        or _data_parallel_size(self.trainer)
                    ),
                )
                sft = (
                    SFTBatchData(
                        trajectory_tensors=tuple(tokenized.trajectory_tensors),
                        learning_rate=tokenized.learning_rate,
                        num_trajectories=tokenized.num_trajectories,
                        num_tokens=tokenized.num_tokens,
                        num_trainable_tokens=tokenized.num_trainable_tokens,
                        num_dropped_trajectories=(tokenized.num_dropped_trajectories),
                    )
                    if tokenized.num_trainable_tokens > 0
                    else None
                )
                content_sha256 = None if sft is None else sft.fingerprint
                ref = PackedInputCaptureRef(
                    run_id=operation.run_id,
                    capture_id=capture_id,
                    manifest_sha256=_sft_capture_manifest_sha256(
                        operation,
                        packing,
                        fingerprint,
                        content_sha256,
                    ),
                    content_sha256=content_sha256,
                    input_kind="sft",
                )
                self._captures[capture_id] = _CapturedInput(
                    ref=ref,
                    request_fingerprint=fingerprint,
                    control_fingerprint=_input_control_fingerprint(request, operation),
                    packed=None,
                    sft=sft,
                    packing=packing,
                    retained_for_replay=request.retain_packed_input and sft is not None,
                    input_metrics={
                        "data/dropped_sft_trajectories": float(
                            tokenized.num_dropped_trajectories
                        )
                    },
                )
                return ref
            bundles = (
                retained_route_bundles_from_bundles(request.batch.groups)
                if isinstance(request.batch, RlTrajectoryBatch)
                else ()
            )
            ownership = await self._acquire_route_ownership(operation, bundles)
            packed = None
            try:
                packed = await self.runtime.pack(
                    self._packing_request(
                        request,
                        operation,
                        retained_route_bundles=bundles,
                    )
                )
                if packed is None:
                    raise ValueError("training input produced no packed sequence")
                packing = self._packing_outcome(packed, request)
                ref = PackedInputCaptureRef(
                    run_id=operation.run_id,
                    capture_id=capture_id,
                    manifest_sha256=_capture_manifest_sha256(
                        operation, packed, packing, fingerprint
                    ),
                    content_sha256=packed.leases.ref.content_sha256,
                    input_kind=request.batch.kind,
                    min_source_version=(
                        request.batch.min_source_version
                        if isinstance(request.batch, RlTrajectoryBatch)
                        else 0
                    ),
                    max_source_version=(
                        request.batch.max_source_version
                        if isinstance(request.batch, RlTrajectoryBatch)
                        else 0
                    ),
                )
                if request.retain_packed_input and ref.content_sha256 is None:
                    raise RuntimeError("replayable packed input has no content digest")
            except BaseException as error:
                if packed is not None:
                    try:
                        await self.runtime.release_batch(packed)
                    except BaseException as cleanup_error:
                        error.add_note(
                            "packed-input cleanup also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                await self._release_route_ownership(ownership, error)
                raise
            self._captures[capture_id] = _CapturedInput(
                ref=ref,
                request_fingerprint=fingerprint,
                control_fingerprint=_input_control_fingerprint(request, operation),
                packed=packed,
                sft=None,
                packing=packing,
                route_ownership=ownership,
                retained_for_replay=request.retain_packed_input,
            )
            return ref

    async def __call__(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributing_forward_backward_operation_ids: tuple[str, ...],
    ) -> OperationResultType:
        launch = await self.launch(
            request, operation, contributing_forward_backward_operation_ids
        )
        if launch is not None:
            return await asyncio.shield(launch.completion)
        return await self._execute_control(
            request, operation, contributing_forward_backward_operation_ids
        )

    async def launch(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributing_forward_backward_operation_ids: tuple[str, ...],
    ) -> MegatronOperationLaunch | None:
        if operation.run_id != self.config.run_id:
            raise ValueError("operation belongs to another run")
        if isinstance(request, ForwardBackwardRequest):
            return await self._start_forward(request, operation, backward=True)
        if isinstance(request, ForwardRequest):
            return await self._start_forward(request, operation, backward=False)
        return None

    async def _execute_control(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributing_forward_backward_operation_ids: tuple[str, ...],
    ) -> OperationResultType:
        if isinstance(request, OptimStepRequest):
            return await self._optim_step(
                request, operation, contributing_forward_backward_operation_ids
            )
        paired_publication = isinstance(
            request, SaveWeightsForSamplerRequest
        ) and request.publication.mode in {"versioned_lora", "in_flight_lora"}
        if self.checkpoints is None and not (paired_publication and self.publisher):
            raise OperationExecutionError(
                "execution_failed",
                "Megatron checkpoint operations are not configured",
                usage=CommandExecutionUsage.no_work(),
            )
        if isinstance(request, SaveWeightsForSamplerRequest):
            saved = (
                await self.publisher.save_weights_for_sampler(
                    request,
                    operation,
                    self._generation,
                    template_adapter_path=self.config.source.adapter_path,
                    optimizer_state_path=self._optimizer_state_path,
                    staging_adapter_path=_staging_adapter_path(
                        self.config, self._generation.generation_id
                    ),
                )
                if paired_publication and self.publisher is not None
                else await cast(
                    MegatronCheckpointOperations, self.checkpoints
                ).save_weights_for_sampler(request, operation, self._generation)
            )
            if isinstance(saved, MegatronSamplerPublicationReceipt):
                saved.validate_command(request, operation, self._generation)
                result = saved.result
            else:
                if request.publication.mode != "none":
                    raise RuntimeError(
                        "sampler publication returned no durable holder receipt"
                    )
                result = saved
            await self.trainer.record_control_command(
                operation, self._generation.policy_step
            )
            if isinstance(saved, MegatronSamplerPublicationReceipt):
                self._sampler_publications[operation.operation_id] = saved
            return result
        if isinstance(request, SaveStateRequest):
            result = await cast(
                MegatronCheckpointOperations, self.checkpoints
            ).save_state(request, operation, self._generation)
            await self.trainer.record_control_command(
                operation, self._generation.policy_step
            )
            return result
        if isinstance(request, LoadStateRequest):
            loaded = await cast(
                MegatronCheckpointOperations, self.checkpoints
            ).load_state(request, operation)
            if loaded.result.operation_id != operation.operation_id:
                raise RuntimeError("checkpoint loader changed operation identity")
            if (
                loaded.generation.training_session_id != self.config.training_session_id
                or loaded.generation.policy_step
                != operation.reserved_output_learner_version
            ):
                raise RuntimeError("checkpoint loader changed learner lineage")
            await self.trainer.record_control_command(
                operation, loaded.generation.policy_step
            )
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
        captured = self._captures.get(ref.capture_id)
        if captured is None:
            if self._released_captures.get(ref.capture_id) == ref:
                return
            raise ValueError("packed-input capture is absent or changed")
        if captured.ref != ref or ref.run_id != self.config.run_id:
            raise ValueError("packed-input capture is absent or changed")
        if captured.owners:
            raise RuntimeError("cannot discard packed input owned by an operation")
        captured.retained_for_replay = False
        await self._release_if_unowned(ref.capture_id)

    async def packing_for(self, ref: PackedInputCaptureRef) -> PackingOutcome:
        return (await self._require_capture(ref, None)).packing

    async def capture_forward_backward_numerics(
        self, operation_id: str, root: str
    ) -> ForwardBackwardNumericalCaptureReceipt:
        capture_id = self._contributions.get(operation_id)
        if capture_id is None:
            raise RuntimeError("numerical capture operation is not an open F/B")
        captured = self._captures.get(capture_id)
        if captured is None:
            raise RuntimeError("numerical capture packed input is absent")
        if captured.packed is None:
            raise RuntimeError("SFT numerical capture is not yet supported")
        return await self.trainer.capture_forward_backward_numerics(
            self.config.run_id,
            operation_id,
            captured.packed.leases,
            root,
        )

    async def retry_releases(self) -> None:
        for capture_id in tuple(self._release_failures):
            await self._release_if_unowned(capture_id)

    async def transfer_route_ownership(
        self,
        ref: PackedInputCaptureRef,
        *,
        transfer_id: str,
        target_owner_id: str,
    ) -> RouteBundleOwnershipHandle | None:
        """Create a target owner while preserving this capture's source lease."""

        if not transfer_id or not target_owner_id:
            raise ValueError("route ownership transfer identities must not be empty")
        captured = await self._require_capture(ref, None)
        handle = captured.route_ownership
        if handle is None:
            return None
        provider = self.route_ownership
        if provider is None:
            raise RuntimeError("retained route ownership provider is absent")
        target = await provider.transfer(
            handle,
            transfer_id=transfer_id,
            target_owner_id=target_owner_id,
        )
        if target is None:
            raise RuntimeError("retained route ownership transfer returned no handle")
        return target

    def retained_contribution_inputs(
        self,
    ) -> tuple[tuple[str, PackedInputCaptureRef], ...]:
        return tuple(
            (operation_id, self._captures[capture_id].ref)
            for operation_id, capture_id in self._contributions.items()
        )

    def sampler_publication_receipt(
        self, operation_id: str
    ) -> MegatronSamplerPublicationReceipt | None:
        return self._sampler_publications.get(operation_id)

    def retire_operation(self, operation_id: str) -> None:
        self._sampler_publications.pop(operation_id, None)

    async def plan_artifacts(self, request: RunCommand) -> MegatronArtifactResourcePlan:
        if not isinstance(
            request,
            (SaveWeightsForSamplerRequest, SaveStateRequest, LoadStateRequest),
        ):
            return MegatronArtifactResourcePlan(
                basis="exact",
                checkpoint_objects=0,
                lora_bytes=0,
                transfer_bytes=0,
                storage_bytes=0,
            )
        if (
            isinstance(request, SaveWeightsForSamplerRequest)
            and request.publication.mode in {"versioned_lora", "in_flight_lora"}
            and self.publisher is not None
        ):
            return await self.publisher.plan_artifacts(
                request,
                self._generation,
                template_adapter_path=self.config.source.adapter_path,
            )
        if self.checkpoints is None:
            raise RuntimeError("Megatron checkpoint operations are not configured")
        return await self.checkpoints.plan_artifacts(request, self._generation)

    async def aclose(self) -> None:
        if self._contributions:
            raise RuntimeError("cannot close a run with open F/B contributions")
        for captured in self._captures.values():
            captured.owners.clear()
            captured.retained_for_replay = False
        for capture_id in tuple(self._captures):
            await self._release_if_unowned(capture_id)
        if self._captures:
            raise RuntimeError("packed inputs remain after run drain")
        self._sampler_publications.clear()

    async def release_after_migration(self) -> None:
        """Release source-local replay leases after target activation."""

        self._contributions.clear()
        for captured in self._captures.values():
            captured.owners.clear()
            captured.retained_for_replay = False
        for capture_id in tuple(self._captures):
            await self._release_if_unowned(capture_id)
        if self._captures:
            raise RuntimeError("packed inputs remain after migration source release")
        self._sampler_publications.clear()

    async def _start_forward(
        self,
        request: ForwardRequest | ForwardBackwardRequest,
        operation: OperationRef,
        *,
        backward: bool,
    ) -> MegatronOperationLaunch:
        capture_ref = await self.prepare_input(request, operation)
        captured = await self._require_capture(capture_ref, operation)
        if captured.packing.loss_bearing_tokens == 0:
            await self.trainer.record_no_work_command(
                operation, self._generation.policy_step
            )
            captured.retained_for_replay = False
            await self._release_if_unowned(capture_ref.capture_id)
            metrics = {
                **captured.input_metrics,
                "data/sft_zero_work": 1.0,
            }
            if backward:
                result: OperationResultType = ForwardBackwardResult(
                    operation_id=operation.operation_id,
                    packing=captured.packing,
                    packed_input_capture=None,
                    token_logprobs=(),
                    metrics=metrics,
                    usage=_complete_zero_usage(),
                    produced_gradient=False,
                )
            else:
                result = ForwardResult(
                    operation_id=operation.operation_id,
                    packing=captured.packing,
                    packed_input_capture=None,
                    token_logprobs=(),
                    metrics=metrics,
                    usage=_complete_zero_usage(),
                )
            completion = asyncio.get_running_loop().create_future()
            completion.set_result(result)
            return MegatronOperationLaunch(completion=completion)
        try:
            if captured.sft is not None:
                sft_common = {
                    "operation": operation,
                    "training_session_id": self.config.training_session_id,
                    "source": self._generation,
                    "optimizer_state_path": self._optimizer_state_path,
                    "batch_fingerprint": captured.sft.fingerprint,
                    "expected_global_nonpadding_tokens": (
                        captured.packing.non_padding_tokens
                    ),
                    "expected_global_loss_bearing_tokens": (
                        captured.packing.loss_bearing_tokens
                    ),
                    "config": _train_config(self.config.train_config, request),
                    "return_token_logprobs": request.return_token_logprobs,
                }
                if backward:
                    trainer_launch = await self.trainer.start_sft_forward_backward(
                        SftForwardBackwardJobSpec(**sft_common),
                        captured.sft,
                    )
                else:
                    trainer_launch = await self.trainer.start_sft_forward(
                        SftForwardJobSpec(**sft_common),
                        captured.sft,
                    )
            else:
                if captured.packed is None:
                    raise RuntimeError("prepared input has no executable payload")
                packed_common = {
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
                    "loss": (
                        request.loss if captured.ref.input_kind == "tokenized" else None
                    ),
                    "return_token_logprobs": request.return_token_logprobs,
                }
                if backward:
                    trainer_launch = await self.trainer.start_forward_backward(
                        ForwardBackwardJobSpec(**packed_common),
                        captured.packed.leases,
                    )
                else:
                    trainer_launch = await self.trainer.start_forward(
                        ForwardJobSpec(**packed_common),
                        captured.packed.leases,
                    )
        except BaseException as error:
            await self._release_after_failed_execution(capture_ref.capture_id, error)
            if isinstance(error, OperationExecutionError):
                raise
            raise _dispatched_execution_error(error, self.trainer) from error
        completion = asyncio.create_task(
            self._complete_forward(
                trainer_launch.completion,
                capture_ref,
                captured,
                operation,
                backward=backward,
            ),
            name=f"megatron-operation-result-{operation.operation_id}",
        )
        return MegatronOperationLaunch(completion=completion)

    async def _complete_forward(
        self,
        completion: asyncio.Future[dict[str, Any]],
        capture_ref: PackedInputCaptureRef,
        captured: _CapturedInput,
        operation: OperationRef,
        *,
        backward: bool,
    ) -> OperationResultType:
        try:
            raw = await asyncio.shield(completion)
        except BaseException as error:
            await self._release_after_failed_execution(capture_ref.capture_id, error)
            if isinstance(error, OperationExecutionError):
                raise
            raise _dispatched_execution_error(error, self.trainer) from error
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
            gpu_count=UsageMeasurement.complete(int(raw["gpu_count"])),
            gpu_service_ns=UsageMeasurement.complete(int(raw["gpu_service_ns"])),
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
            metrics={
                **(
                    _packing_metrics(captured.packed)
                    if captured.packed is not None
                    else captured.input_metrics
                ),
                **raw.get("metrics", {}),
            },
            usage=usage,
            **({"produced_gradient": True} if backward else {}),
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
        try:
            raw = await self.trainer.optim_step(
                OptimizerJobSpec(
                    operation=operation,
                    training_session_id=self.config.training_session_id,
                    source=self._generation,
                    optimizer_state_path=self._optimizer_state_path,
                    generation=generation,
                    contributing_forward_backward_operation_ids=contributions,
                    optimizer=request.optimizer,
                )
            )
        except BaseException as error:
            if isinstance(error, OperationExecutionError):
                raise
            raise _dispatched_execution_error(error, self.trainer) from error
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
            usage=CommandExecutionUsage(
                logical_nonpadding_tokens=UsageMeasurement.not_applicable(),
                executed_token_equivalents=UsageMeasurement.not_applicable(),
                gpu_count=UsageMeasurement.complete(int(raw["gpu_count"])),
                gpu_service_ns=UsageMeasurement.complete(int(raw["gpu_service_ns"])),
            ),
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
        return captured

    async def _release_after_failed_execution(
        self, capture_id: str, primary: BaseException
    ) -> None:
        captured = self._captures.get(capture_id)
        if captured is not None:
            captured.retained_for_replay = False
        try:
            await self._release_if_unowned(capture_id)
        except BaseException as cleanup:
            primary.add_note(
                f"packed-input cleanup also failed: {type(cleanup).__name__}: {cleanup}"
            )

    async def _release_if_unowned(self, capture_id: str) -> None:
        captured = self._captures.get(capture_id)
        if captured is None or captured.owners or captured.retained_for_replay:
            return
        try:
            if captured.packed is not None and not captured.packed_released:
                await self.runtime.release_batch(captured.packed)
                captured.packed_released = True
            await self._release_route_ownership(captured.route_ownership)
            captured.route_ownership = None
        except BaseException as error:
            self._release_failures[capture_id] = error
            return
        self._release_failures.pop(capture_id, None)
        self._captures.pop(capture_id, None)
        self._released_captures[capture_id] = captured.ref
        while len(self._released_captures) > self.config.max_retained_inputs:
            self._released_captures.pop(next(iter(self._released_captures)))

    async def _acquire_route_ownership(
        self,
        operation: OperationRef,
        bundles: tuple[RetainedRouteBundleRef, ...],
    ) -> RouteBundleOwnershipHandle | None:
        if not bundles:
            return None
        provider = self.route_ownership
        if provider is None:
            raise OperationExecutionError(
                "execution_failed",
                "retained route ownership is not configured",
                usage=CommandExecutionUsage.no_work(),
            )
        handle = await provider.acquire(operation=operation, bundles=bundles)
        if handle is None:
            raise RuntimeError(
                "retained route ownership acquisition returned no handle"
            )
        return handle

    async def _release_route_ownership(
        self,
        handle: RouteBundleOwnershipHandle | None,
        primary: BaseException | None = None,
    ) -> None:
        if handle is None:
            return
        provider = self.route_ownership
        if provider is None:
            raise RuntimeError("retained route ownership provider is absent")
        try:
            await provider.release(handle)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                f"route ownership cleanup also failed: {type(error).__name__}: {error}"
            )

    def _packing_request(
        self,
        request: ForwardRequest | ForwardBackwardRequest,
        operation: OperationRef,
        *,
        retained_route_bundles: tuple[RetainedRouteBundleRef, ...],
    ) -> PackingRequest:
        if isinstance(request.batch, TokenizedTrainingBatch):
            return PackingRequest(
                model=self.config.rollout_model,
                generation_id=operation.operation_id,
                tokenized_batch=request.batch,
                tokenized_loss=cast(Any, request.loss.name),
                packed_sequence_length=self.trainer.runtime_spec.packed_sequence_length,
                compute_content_sha256=request.retain_packed_input,
            )
        assert isinstance(request.batch, RlTrajectoryBatch)
        experimental = _experimental_config(request)
        return PackingRequest(
            model=self.config.rollout_model,
            generation_id=operation.operation_id,
            trajectory_groups=request.batch.groups,
            retained_route_bundles=retained_route_bundles,
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
            compute_content_sha256=request.retain_packed_input,
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
    route_ownership: RouteBundleOwnershipProvider | None = None,
    max_retained_operations: int = 128,
) -> MegatronOperationRuntime:
    handler = MegatronOperationHandler(
        runtime,
        trainer,
        config,
        checkpoints=checkpoints,
        route_ownership=route_ownership,
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
        adapter_path=f"{config.output_adapter_root.rstrip('/')}/{version:04d}",
    )


def _staging_adapter_path(config: MegatronOperationConfig, generation_id: str) -> str:
    output_root = Path(config.output_adapter_root).absolute().parent
    return str(output_root / "megatron_runtime" / "staging" / generation_id)


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
    command = request.model_dump(
        mode="json",
        exclude={
            "batch",
            "run_id",
            "request_id",
            "sequence_id",
            "retain_packed_input",
        },
    )
    return hashlib.sha256(
        json.dumps(
            {"command": command, "operation_kind": operation.kind},
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
                "packed": packed.leases.ref.model_dump(
                    mode="json",
                    exclude={
                        "owner_actor_id",
                        "lease_id",
                        "shared_memory_name",
                        "owner_process_id",
                    },
                ),
                "packing": packing.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _sft_capture_manifest_sha256(
    operation: OperationRef,
    packing: PackingOutcome,
    request_fingerprint: str,
    content_sha256: str | None,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "operation_id": operation.operation_id,
                "request_fingerprint": request_fingerprint,
                "content_sha256": content_sha256,
                "packing": packing.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _sft_packing_outcome(
    batch: Any,
    *,
    configured_sequence_length: int,
    target_sequences: int,
) -> PackingOutcome:
    sequence_length = max(
        (int(tensors["input_ids"].numel()) for tensors in batch.trajectory_tensors),
        default=configured_sequence_length,
    )
    return PackingOutcome(
        packed_sequence_length=sequence_length,
        packed_sequences=batch.num_trajectories,
        target_packed_sequences=target_sequences,
        physical_tokens=batch.num_tokens,
        non_padding_tokens=batch.num_tokens,
        loss_bearing_tokens=batch.num_trainable_tokens,
        trainable_assistant_tokens=batch.num_trainable_tokens,
    )


def _complete_zero_usage() -> CommandExecutionUsage:
    return CommandExecutionUsage(
        logical_nonpadding_tokens=UsageMeasurement.complete(0),
        executed_token_equivalents=UsageMeasurement.complete(0),
        gpu_count=UsageMeasurement.complete(0),
        gpu_service_ns=UsageMeasurement.complete(0),
    )


def _dispatched_execution_error(
    error: BaseException, trainer: _ResidentTrainer
) -> OperationExecutionError:
    gpu_count = len(trainer.runtime_spec.trainer_mesh.ranks)
    return OperationExecutionError(
        "cancelled"
        if isinstance(error, asyncio.CancelledError)
        else "execution_failed",
        str(error).strip() or type(error).__name__,
        usage=CommandExecutionUsage(
            logical_nonpadding_tokens=UsageMeasurement.unknown(),
            executed_token_equivalents=UsageMeasurement.unknown(),
            gpu_count=UsageMeasurement.complete(gpu_count),
            gpu_service_ns=UsageMeasurement.unknown(),
        ),
    )


def _packing_metrics(packed: DistributedPackedBatch) -> dict[str, float]:
    return {
        "time/step_trajectory_fetch_s": packed.trajectory_fetch_s,
        "time/step_route_fetch_s": packed.route_fetch_s,
        "time/step_packing_core_s": packed.packing_core_s,
        "time/step_trajectory_log_wait_s": packed.trajectory_log_wait_s,
        "time/step_packed_batch_finalize_s": packed.packed_batch_finalize_s,
        "time/step_packing_rpc_s": packed.packing_rpc_s,
        "time/step_packed_batch_fanout_s": packed.packed_batch_fanout_s,
    }
