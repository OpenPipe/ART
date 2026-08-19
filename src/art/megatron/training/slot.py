from __future__ import annotations

import asyncio
import math
from pathlib import Path
import time
from typing import Any, Literal, cast
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.distributed.art_runtime import ArtRuntime, DistributedPackedBatch
from art.distributed.object_store import (
    BinaryObjectTarget,
    S3ObjectStoreConfig,
    vllm_lora_object_target,
)
from art.distributed.packing import PackingRequest
from art.distributed.rollout import RolloutModelSpec
from art.megatron.optimizer_state import (
    OptimizerAdapter,
    OptimizerGenerationManifest,
    link_adapter_generation,
    optimizer_adapter,
    optimizer_generation_nbytes,
    read_adapter_publication,
    read_committed_optimizer_pointer,
    read_optimizer_generation_manifest,
)
from art.megatron.runtime.data_plane import SFTBatchData
from art.megatron.runtime.publication import (
    PreparedSave,
    SnapshotRankWritePlan,
    SnapshotWriteGrant,
    SnapshotWritePlan,
    build_snapshot_write_plan,
    commit_trainer_publication,
)
from art.megatron.runtime.specs import (
    ExperimentalTrainConfig,
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    GenerationSnapshotJobSpec,
    LoadStateJobSpec,
    OptimizerJobSpec,
    ResolvedCheckpointState,
    RlForwardBackwardConfig,
    RunSlotRegistration,
    SftForwardBackwardJobSpec,
    SftForwardJobSpec,
    TrainerGeneration,
    TrainerRuntimeSpec,
)
from art.preprocessing.sft import SftBatchTokenizer
from art.training.contracts import (
    Contract,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    LossConfig,
    LossFnOutput,
    OperationRef,
    OptimStepRequest,
    OptimStepResult,
    PackingOutcome,
    RlTrajectoryBatch,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
    operation_generation_id,
)
from art.utils.output_dirs import get_step_checkpoint_dir

from .commands import (
    checkpoint_ref,
    experimental_train_config,
    forward_backward_config,
    packing_metrics,
    packing_outcome,
    sft_batch_data,
    sft_packing_outcome,
)


class PreparedForward(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    ref: OperationRef
    packing: PackingOutcome
    kind: Literal["rl", "sft", "tokenized"]

    @property
    def token_cost(self) -> int:
        return self.packing.physical_tokens


class PreparedPackedForward(PreparedForward):
    kind: Literal["rl", "tokenized"] = "rl"
    packed: DistributedPackedBatch
    config: RlForwardBackwardConfig
    experimental_config: ExperimentalTrainConfig
    loss: LossConfig | None = None

    @model_validator(mode="after")
    def _validate_loss(self) -> "PreparedPackedForward":
        if (self.kind == "tokenized") != (self.loss is not None):
            raise ValueError("tokenized prepared batches require their named loss")
        return self


class PreparedSftForward(PreparedForward):
    kind: Literal["sft"] = "sft"
    batch: SFTBatchData
    global_grad_accumulation_sequences: int = Field(ge=1)
    tokenization_s: float = Field(ge=0)


class PreparedLoadState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: OperationRef
    request: LoadStateRequest
    source: ResolvedCheckpointState
    job: LoadStateJobSpec
    adapter: OptimizerAdapter


class _ResidentRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    registration: RunSlotRegistration
    model: RolloutModelSpec
    output_dir: str = Field(min_length=1)
    generation: TrainerGeneration


class _PendingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    generation: TrainerGeneration
    optimizer_state_path: str
    plan: SnapshotWritePlan
    publication: Any
    started: float


class _ExistingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: OptimizerAdapter
    optimizer_bytes: int | None
    plan: SnapshotWritePlan


class _PreparedSaveOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    fingerprint: str
    prepared: PreparedSave
    ref: OperationRef
    request: SaveWeightsForSamplerRequest | SaveStateRequest
    snapshot: _ExistingSnapshot | _PendingSnapshot


class MegatronTrainingSlot:
    """Canonical pack-and-command facade for one persistent Megatron mesh."""

    def __init__(
        self,
        *,
        runtime: ArtRuntime,
        trainer: Any,
        runtime_spec: TrainerRuntimeSpec,
        artifact_root: str,
        sampler_store: S3ObjectStoreConfig | None = None,
    ) -> None:
        if trainer.runtime_spec != runtime_spec:
            raise ValueError("trainer slot does not match its runtime spec")
        self.runtime = runtime
        self.trainer = trainer
        self.runtime_spec = runtime_spec
        self.artifact_root = str(Path(artifact_root).resolve())
        self.sampler_store = sampler_store
        self._runs: dict[str, _ResidentRun] = {}
        self._results: dict[str, tuple[str, Contract]] = {}
        self._pending_results: dict[str, tuple[str, asyncio.Future[Contract]]] = {}
        self._prepared_saves: dict[str, _PreparedSaveOperation] = {}
        self._sft_tokenizer = SftBatchTokenizer()
        self._batch_releases: set[asyncio.Task[None]] = set()
        self._batch_release_failures: list[BaseException] = []
        self._closed = False

    @property
    def valid(self) -> bool:
        return (
            not self._closed and self.trainer.valid and not self._batch_release_failures
        )

    async def register_run(
        self,
        registration: RunSlotRegistration,
        *,
        model: RolloutModelSpec,
        output_dir: str,
    ) -> None:
        self._require_open()
        output_dir = self._require_managed_path(output_dir)
        registration = registration.model_copy(
            update={
                "adapter_path": self._require_managed_path(registration.adapter_path),
                "optimizer_state_path": self._require_managed_path(
                    registration.optimizer_state_path
                ),
                "initial_optimizer_state_path": (
                    self._require_managed_path(
                        registration.initial_optimizer_state_path
                    )
                    if registration.initial_optimizer_state_path is not None
                    else None
                ),
            }
        )
        prior = self._runs.get(registration.run_id)
        if prior is not None:
            candidate = (registration, model, output_dir)
            if candidate != (prior.registration, prior.model, prior.output_dir):
                raise RuntimeError("run_id was reused for another resident run")
            return
        adapter = read_adapter_publication(
            registration.adapter_path,
            step=registration.adapter_step,
        ) or optimizer_adapter(
            registration.adapter_path,
            registration.adapter_step,
            training_session_id=registration.adapter_training_session_id,
        )
        if adapter.training_session_id != registration.adapter_training_session_id:
            raise ValueError("adapter belongs to another training session")
        if adapter.generation_id != registration.adapter_generation_id:
            raise ValueError("adapter generation differs from run registration")
        await self.trainer.register_run(registration)
        self._runs[registration.run_id] = _ResidentRun(
            registration=registration,
            model=model,
            output_dir=output_dir,
            generation=TrainerGeneration(
                training_session_id=registration.training_session_id,
                policy_step=registration.learner_version,
                generation_id=registration.generation_id,
                adapter_path=adapter.identity,
            ),
        )

    async def unregister_run(self, run_id: str) -> None:
        self._require_open()
        if run_id not in self._runs:
            return
        if self.trainer.valid:
            try:
                await self.trainer.unregister_run(run_id)
            except BaseException:
                if self.trainer.valid:
                    raise
        self._runs.pop(run_id)

    async def prepare_forward_backward(
        self,
        ref: OperationRef,
        request: ForwardBackwardRequest,
    ) -> PreparedForward:
        return await self._prepare_forward(ref, request)

    async def prepare_forward(
        self,
        ref: OperationRef,
        request: ForwardRequest,
    ) -> PreparedForward:
        return await self._prepare_forward(ref, request)

    async def _prepare_forward(
        self,
        ref: OperationRef,
        request: ForwardRequest | ForwardBackwardRequest,
    ) -> PreparedForward:
        state = self._require_run(ref.run_id)
        if request.batch.kind == "sft":
            started = time.perf_counter()
            tokenized = await asyncio.to_thread(
                self._sft_tokenizer.tokenize,
                state.model.build(),
                request.batch.trajectories,
                assistant_turns=request.batch.assistant_turns,
            )
            if tokenized.num_trainable_tokens < 1:
                raise ValueError("supervised batch produced no trainable tokens")
            batch = sft_batch_data(tokenized)
            topology = self.runtime_spec.trainer_mesh.topology
            data_parallel_size = len(self.runtime_spec.trainer_mesh.ranks) // (
                topology.tp * topology.cp * topology.pp
            )
            grad_sequences = (
                math.ceil(batch.num_trajectories / data_parallel_size)
                * data_parallel_size
            )
            return PreparedSftForward(
                ref=ref,
                batch=batch,
                global_grad_accumulation_sequences=grad_sequences,
                tokenization_s=time.perf_counter() - started,
                packing=sft_packing_outcome(batch),
            )
        if request.batch.kind == "tokenized":
            packed = await self.runtime.pack(
                PackingRequest(
                    model=state.model,
                    generation_id=uuid.uuid4().hex,
                    tokenized_batch=request.batch,
                    tokenized_loss=request.loss.name,
                    packed_sequence_length=self.runtime_spec.packed_sequence_length,
                )
            )
            if packed is None:
                raise ValueError("tokenized batch produced no packed sequence")
            return PreparedPackedForward(
                ref=ref,
                kind="tokenized",
                packed=packed,
                packing=packing_outcome(
                    packed,
                    target_packed_sequences=self._data_parallel_size(),
                ),
                config=RlForwardBackwardConfig(),
                experimental_config=ExperimentalTrainConfig(),
                loss=request.loss,
            )
        if not isinstance(request.batch, RlTrajectoryBatch):
            raise TypeError("unknown forward batch")
        config = experimental_train_config(request)
        training_config = forward_backward_config(request)
        if (
            config.packed_sequence_length is not None
            and config.packed_sequence_length
            != self.runtime_spec.packed_sequence_length
        ):
            raise ValueError("loss packed_sequence_length differs from trainer runtime")
        packed = await self.runtime.pack(
            PackingRequest(
                model=state.model,
                generation_id=uuid.uuid4().hex,
                trajectory_groups=request.batch.groups,
                moe_route_groups=request.batch.local_moe_route_groups(),
                moe_route_object_transfer=(
                    request.batch.local_moe_route_object_transfer()
                ),
                trajectory_annotations=request.batch.local_group_annotations(),
                advantage_balance=config.advantage_balance,
                allow_training_without_logprobs=bool(
                    config.allow_training_without_logprobs
                ),
                scale_rewards=config.scale_rewards,
                plot_tensors=bool(config.plot_tensors),
                packed_sequence_length=self.runtime_spec.packed_sequence_length,
                logprob_calculation_chunk_size=(
                    config.logprob_calculation_chunk_size or 1024
                ),
                include_moe_routing=self.runtime_spec.enable_moe_routing_replay,
                collect_packing_shapes=request.collect_packing_shapes,
                group_ids=tuple(
                    f"{ref.operation_id}:{index}"
                    for index in range(len(request.batch.groups))
                ),
                min_source_version=request.batch.min_source_version,
                max_source_version=request.batch.max_source_version,
            )
        )
        if packed is None:
            raise ValueError("trajectory batch produced no trainable packed sequence")
        return PreparedPackedForward(
            ref=ref,
            packed=packed,
            packing=packing_outcome(
                packed,
                target_packed_sequences=(
                    training_config.grad_accumulation_sequences
                    or self._data_parallel_size()
                ),
            ),
            config=training_config,
            experimental_config=config,
        )

    def _data_parallel_size(self) -> int:
        topology = self.runtime_spec.trainer_mesh.topology
        return len(self.runtime_spec.trainer_mesh.ranks) // (
            topology.tp * topology.cp * topology.pp
        )

    async def discard_prepared(self, prepared: PreparedForward) -> None:
        if isinstance(prepared, PreparedPackedForward):
            await self.runtime.release_batch(prepared.packed)

    async def prepare_residency(
        self,
        run_id: str,
        command_kind: Literal["forward", "forward_backward", "optim_step"],
    ) -> dict[str, float]:
        self._require_run(run_id)
        return await self.trainer.prepare_residency(run_id, command_kind)

    async def forward(self, prepared: PreparedForward) -> ForwardResult:
        ref = prepared.ref
        state = self._require_parent(ref)
        if isinstance(prepared, PreparedSftForward):
            job = SftForwardJobSpec(
                operation_id=ref.operation_id,
                run_id=ref.run_id,
                sequence_id=ref.sequence_id,
                training_session_id=state.registration.training_session_id,
                expected_learner_version=ref.learner_parent_version,
                source=state.generation,
                optimizer_state_path=state.registration.optimizer_state_path,
                batch_fingerprint=prepared.batch.fingerprint,
                trainable_token_count=prepared.batch.num_trainable_tokens,
                global_grad_accumulation_sequences=(
                    prepared.global_grad_accumulation_sequences
                ),
            )
            raw = await self.trainer.sft_forward(job, prepared.batch)
            return self._sft_forward_result(prepared, raw, backward=False)
        if not isinstance(prepared, PreparedPackedForward):
            raise TypeError("unknown prepared forward payload")
        job = ForwardJobSpec(
            operation_id=ref.operation_id,
            run_id=ref.run_id,
            sequence_id=ref.sequence_id,
            training_session_id=state.registration.training_session_id,
            expected_learner_version=ref.learner_parent_version,
            source=state.generation,
            optimizer_state_path=state.registration.optimizer_state_path,
            batch=prepared.packed.leases.ref,
            config=prepared.config,
            experimental_config=prepared.experimental_config,
            loss=prepared.loss,
            tokenized_trainable_token_count=(
                prepared.packed.loss_bearing_tokens
                if prepared.kind == "tokenized"
                else None
            ),
        )
        try:
            raw = await self.trainer.forward(job, prepared.packed.leases)
        finally:
            self._release_batch_soon(prepared.packed)
        return ForwardResult(
            operation_id=ref.operation_id,
            packing=prepared.packing,
            loss_fn_outputs=tuple(
                LossFnOutput(token_logprobs=values) for values in raw["token_logprobs"]
            ),
            metrics={**packing_metrics(prepared.packed), **raw["metrics"]},
        )

    async def forward_backward(
        self, prepared: PreparedForward
    ) -> ForwardBackwardResult:
        ref = prepared.ref
        state = self._require_parent(ref)
        if isinstance(prepared, PreparedSftForward):
            job = SftForwardBackwardJobSpec(
                operation_id=ref.operation_id,
                run_id=ref.run_id,
                sequence_id=ref.sequence_id,
                training_session_id=state.registration.training_session_id,
                expected_learner_version=ref.learner_parent_version,
                source=state.generation,
                optimizer_state_path=state.registration.optimizer_state_path,
                batch_fingerprint=prepared.batch.fingerprint,
                trainable_token_count=prepared.batch.num_trainable_tokens,
                global_grad_accumulation_sequences=(
                    prepared.global_grad_accumulation_sequences
                ),
            )
            raw = await self.trainer.sft_forward_backward(job, prepared.batch)
            return cast(
                ForwardBackwardResult,
                self._sft_forward_result(prepared, raw, backward=True),
            )
        if not isinstance(prepared, PreparedPackedForward):
            raise TypeError("unknown prepared F/B payload")
        job = ForwardBackwardJobSpec(
            operation_id=ref.operation_id,
            run_id=ref.run_id,
            sequence_id=ref.sequence_id,
            training_session_id=state.registration.training_session_id,
            expected_learner_version=ref.learner_parent_version,
            source=state.generation,
            optimizer_state_path=state.registration.optimizer_state_path,
            batch=prepared.packed.leases.ref,
            config=prepared.config,
            experimental_config=prepared.experimental_config,
            loss=prepared.loss,
            tokenized_trainable_token_count=(
                prepared.packed.loss_bearing_tokens
                if prepared.kind == "tokenized"
                else None
            ),
        )
        try:
            raw = await self.trainer.forward_backward(job, prepared.packed.leases)
        finally:
            self._release_batch_soon(prepared.packed)
        return ForwardBackwardResult(
            operation_id=ref.operation_id,
            packing=prepared.packing,
            loss_fn_outputs=tuple(
                LossFnOutput(token_logprobs=values) for values in raw["token_logprobs"]
            ),
            metrics={**packing_metrics(prepared.packed), **raw["metrics"]},
        )

    @staticmethod
    def _sft_forward_result(
        prepared: PreparedSftForward,
        raw: dict[str, Any],
        *,
        backward: bool,
    ) -> ForwardResult:
        result = ForwardBackwardResult if backward else ForwardResult
        return result(
            operation_id=prepared.ref.operation_id,
            packing=prepared.packing,
            loss_fn_outputs=tuple(
                LossFnOutput(token_logprobs=values) for values in raw["token_logprobs"]
            ),
            metrics={
                "time/step_tokenize_trajectory_groups_s": prepared.tokenization_s,
                "data/step_num_dropped_trajectories": float(
                    prepared.batch.num_dropped_trajectories
                ),
                **raw["metrics"],
            },
        )

    async def optim_step(
        self,
        ref: OperationRef,
        request: OptimStepRequest,
        contributions: tuple[str, ...],
    ) -> OptimStepResult:
        completion = await self.start_optim_step(ref, request, contributions)
        return await asyncio.shield(completion)

    async def start_optim_step(
        self,
        ref: OperationRef,
        request: OptimStepRequest,
        contributions: tuple[str, ...],
    ) -> asyncio.Future[OptimStepResult]:
        fingerprint = self._fingerprint(ref, request, *contributions)
        if cached := self._cached_result(ref.operation_id, fingerprint):
            return _resolved_future(cast(OptimStepResult, cached))
        state = self._require_parent(ref)
        output_version = ref.reserved_output_learner_version
        if output_version is None:
            raise ValueError("optimizer operation has no reserved learner version")
        job = OptimizerJobSpec(
            operation_id=ref.operation_id,
            run_id=ref.run_id,
            sequence_id=ref.sequence_id,
            training_session_id=state.registration.training_session_id,
            expected_learner_version=ref.learner_parent_version,
            learner_version=output_version,
            generation=TrainerGeneration(
                training_session_id=state.registration.training_session_id,
                policy_step=output_version,
                generation_id=operation_generation_id(ref.operation_id, output_version),
                adapter_path=get_step_checkpoint_dir(state.output_dir, output_version),
            ),
            contributing_forward_backward_operation_ids=contributions,
            optimizer=request.optimizer,
        )
        raw = await self.trainer.optim_step(job)
        generation = job.generation
        state.generation = generation
        state.registration = state.registration.model_copy(
            update={"learner_version": output_version}
        )
        result = OptimStepResult(
            operation_id=ref.operation_id,
            contributing_forward_backward_operation_ids=contributions,
            metrics=raw["metrics"],
        )
        self._results[ref.operation_id] = (fingerprint, result)
        return _resolved_future(result)

    async def prepare_load_state(
        self,
        ref: OperationRef,
        request: LoadStateRequest,
        source: ResolvedCheckpointState,
    ) -> PreparedLoadState:
        fingerprint = self._fingerprint(ref, request, source.model_dump_json())
        if self._cached_result(ref.operation_id, fingerprint) is not None:
            raise RuntimeError("completed load_state cannot be prepared again")
        state = self._require_parent(ref)
        output_version = ref.reserved_output_learner_version
        if output_version is None:
            raise ValueError("load operation has no reserved learner version")
        if request.restore_optimizer and source.optimizer_state_path is None:
            raise ValueError("optimizer-exact load has no optimizer state")
        generation = TrainerGeneration(
            training_session_id=state.registration.training_session_id,
            policy_step=output_version,
            generation_id=operation_generation_id(ref.operation_id, output_version),
            adapter_path=get_step_checkpoint_dir(state.output_dir, output_version),
        )
        job = LoadStateJobSpec(
            operation_id=ref.operation_id,
            run_id=ref.run_id,
            sequence_id=ref.sequence_id,
            training_session_id=state.registration.training_session_id,
            expected_learner_version=ref.learner_parent_version,
            learner_version=output_version,
            generation=generation,
            adapter_path=self._require_managed_path(source.adapter_path),
            adapter_step=source.adapter_step,
            optimizer_state_path=(
                self._require_managed_path(source.optimizer_state_path)
                if request.restore_optimizer and source.optimizer_state_path is not None
                else None
            ),
            optimizer_generation_id=(
                source.optimizer_generation_id if request.restore_optimizer else None
            ),
            restore_optimizer=request.restore_optimizer,
        )
        await self.trainer.prepare_load_state(job)
        try:
            adapter = await asyncio.to_thread(
                link_adapter_generation,
                job.adapter_path,
                source_step=job.adapter_step,
                staging_path=(
                    Path(state.output_dir)
                    / "megatron_runtime"
                    / "staging"
                    / generation.generation_id
                ),
                step=generation.policy_step,
                training_session_id=generation.training_session_id,
                generation_id=generation.generation_id,
            )
        except BaseException as error:
            try:
                await self.trainer.discard_prepared_load_state(job.operation_id)
            except BaseException as cleanup:
                error.add_note(
                    "prepared load cleanup also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            raise
        return PreparedLoadState(
            ref=ref,
            request=request,
            source=source,
            job=job,
            adapter=adapter,
        )

    async def load_state(self, prepared: PreparedLoadState) -> LoadStateResult:
        ref = prepared.ref
        request = prepared.request
        source = prepared.source
        job = prepared.job
        fingerprint = self._fingerprint(ref, request, source.model_dump_json())
        if cached := self._cached_result(ref.operation_id, fingerprint):
            return cast(LoadStateResult, cached)
        state = self._require_parent(ref)
        raw = await self.trainer.load_state(job)
        generation = job.generation
        state.generation = generation
        state.registration = state.registration.model_copy(
            update={
                "learner_version": job.learner_version,
                "generation_id": generation.generation_id,
                "adapter_path": prepared.adapter.identity,
                "adapter_step": prepared.adapter.step,
                "adapter_training_session_id": prepared.adapter.training_session_id,
                "adapter_generation_id": prepared.adapter.generation_id,
            }
        )
        adapter = prepared.adapter
        if (
            adapter.training_session_id != generation.training_session_id
            or adapter.generation_id != generation.generation_id
        ):
            raise RuntimeError("loaded adapter changed checkpoint identity")
        result = LoadStateResult(
            operation_id=ref.operation_id,
            checkpoint=checkpoint_ref(
                ref.run_id, job.learner_version, generation.generation_id
            ),
            lora=adapter.identity,
            training_session_id=state.registration.training_session_id,
            generation_id=generation.generation_id,
            lora_bytes=sum(file.size_bytes for file in adapter.files),
            optimizer_restored=bool(raw["optimizer_restored"]),
        )
        self._results[ref.operation_id] = (fingerprint, result)
        return result

    async def discard_prepared_load_state(self, operation_id: str) -> None:
        await self.trainer.discard_prepared_load_state(operation_id)

    async def save_weights_for_sampler(
        self,
        ref: OperationRef,
        request: SaveWeightsForSamplerRequest,
    ) -> SamplerWeightsResult:
        completion = await self.start_save_weights_for_sampler(ref, request)
        return await asyncio.shield(completion)

    async def start_save_weights_for_sampler(
        self,
        ref: OperationRef,
        request: SaveWeightsForSamplerRequest,
    ) -> asyncio.Future[SamplerWeightsResult]:
        prepared = await self.prepare_save(ref, request)
        return cast(
            asyncio.Future[SamplerWeightsResult],
            self.authorize_save(prepared, SnapshotWriteGrant.local(prepared.plan)),
        )

    async def save_state(
        self,
        ref: OperationRef,
        request: SaveStateRequest,
    ) -> SaveStateResult:
        completion = await self.start_save_state(ref, request)
        return await asyncio.shield(completion)

    async def start_save_state(
        self,
        ref: OperationRef,
        request: SaveStateRequest,
    ) -> asyncio.Future[SaveStateResult]:
        prepared = await self.prepare_save(ref, request)
        return cast(
            asyncio.Future[SaveStateResult],
            self.authorize_save(prepared, SnapshotWriteGrant.local(prepared.plan)),
        )

    async def prepare_save(
        self,
        ref: OperationRef,
        request: SaveWeightsForSamplerRequest | SaveStateRequest,
    ) -> PreparedSave:
        fingerprint = self._fingerprint(ref, request)
        expected_kind = (
            "sampler_weights"
            if isinstance(request, SaveWeightsForSamplerRequest)
            else "state"
        )
        expected_ref_kind = (
            "save_sampler" if expected_kind == "sampler_weights" else "save_state"
        )
        if ref.kind != expected_ref_kind:
            raise ValueError("save request does not match its operation kind")
        existing = self._prepared_saves.get(ref.operation_id)
        if existing is not None:
            if existing.fingerprint != fingerprint:
                raise RuntimeError("operation_id was reused for a different command")
            return existing.prepared
        if self._cached_result(ref.operation_id, fingerprint) is not None:
            raise RuntimeError("completed save has no retained exact write plan")
        if self._pending_result(ref.operation_id, fingerprint) is not None:
            raise RuntimeError("authorized save has no retained exact write plan")
        snapshot = await self._start_snapshot(
            ref,
            save_optimizer=expected_kind == "state",
            publish_sampler=expected_kind == "sampler_weights",
        )
        prepared = PreparedSave(
            operation_id=ref.operation_id,
            kind=expected_kind,
            generation=snapshot.plan.generation,
            plan=snapshot.plan,
            plan_digest=snapshot.plan.digest,
        )
        self._prepared_saves[ref.operation_id] = _PreparedSaveOperation(
            fingerprint=fingerprint,
            prepared=prepared,
            ref=ref,
            request=request,
            snapshot=snapshot,
        )
        return prepared

    def authorize_save(
        self,
        prepared: PreparedSave,
        grant: SnapshotWriteGrant,
    ) -> asyncio.Future[SamplerWeightsResult | SaveStateResult]:
        operation = self._prepared_saves.get(prepared.operation_id)
        if operation is None or operation.prepared != prepared:
            raise RuntimeError("save is not prepared with this exact write plan")
        grant.validate_plan(prepared.plan)
        if cached := self._cached_result(prepared.operation_id, operation.fingerprint):
            return _resolved_future(
                cast(SamplerWeightsResult | SaveStateResult, cached)
            )
        if pending := self._pending_result(
            prepared.operation_id, operation.fingerprint
        ):
            return cast(asyncio.Future[SamplerWeightsResult | SaveStateResult], pending)

        async def complete() -> SamplerWeightsResult | SaveStateResult:
            authorization_metrics: dict[str, float] = {}
            if isinstance(operation.snapshot, _PendingSnapshot):
                authorization_metrics = await self.trainer.authorize_snapshot(
                    prepared.plan, grant
                )
            adapter, metrics, optimizer_bytes = await self._finish_snapshot(
                operation.snapshot,
                grant=grant,
                authorization_metrics=authorization_metrics,
            )
            request = operation.request
            if isinstance(request, SaveWeightsForSamplerRequest):
                result: SamplerWeightsResult | SaveStateResult = SamplerWeightsResult(
                    operation_id=prepared.operation_id,
                    checkpoint=checkpoint_ref(
                        operation.ref.run_id, adapter.step, request.checkpoint_name
                    ),
                    lora=adapter.identity,
                    training_session_id=adapter.training_session_id,
                    generation_id=adapter.generation_id,
                    lora_bytes=sum(file.size_bytes for file in adapter.files),
                    publication_metrics=metrics,
                )
            else:
                if optimizer_bytes is None:
                    raise RuntimeError("save_state completed without optimizer bytes")
                optimizer_state_path = self._require_run(
                    operation.ref.run_id
                ).registration.optimizer_state_path
                result = SaveStateResult(
                    operation_id=prepared.operation_id,
                    checkpoint=checkpoint_ref(
                        operation.ref.run_id, adapter.step, request.checkpoint_name
                    ),
                    lora=adapter.identity,
                    training_session_id=adapter.training_session_id,
                    generation_id=adapter.generation_id,
                    lora_bytes=sum(file.size_bytes for file in adapter.files),
                    optimizer_state=optimizer_state_path,
                    optimizer_bytes=optimizer_bytes,
                    metrics=metrics,
                )
            self._results[prepared.operation_id] = (operation.fingerprint, result)
            return result

        return cast(
            asyncio.Future[SamplerWeightsResult | SaveStateResult],
            self._start_pending_result(
                prepared.operation_id, operation.fingerprint, complete()
            ),
        )

    async def discard_prepared_save(self, operation_id: str) -> None:
        operation = self._prepared_saves.pop(operation_id, None)
        if operation is None:
            return
        if operation_id in self._pending_results:
            self._prepared_saves[operation_id] = operation
            raise RuntimeError("cannot discard an authorized snapshot write")
        if isinstance(operation.snapshot, _PendingSnapshot):
            await self.trainer.discard_prepared_snapshot(operation_id)

    async def _start_snapshot(
        self,
        ref: OperationRef,
        *,
        save_optimizer: bool,
        publish_sampler: bool,
        sequence_continuation_of: str | None = None,
    ) -> _ExistingSnapshot | _PendingSnapshot:
        state = self._require_parent(ref)
        generation = state.generation
        object_target = (
            self._sampler_object_target(ref.run_id, generation)
            if publish_sampler and self.sampler_store is not None
            else None
        )
        existing = read_adapter_publication(
            generation.adapter_path, step=generation.policy_step
        )
        if not save_optimizer and existing is not None and object_target is None:
            return self._existing_snapshot(ref, generation, existing)
        optimizer_state_path = state.registration.optimizer_state_path
        pointer = read_committed_optimizer_pointer(optimizer_state_path)
        persist_optimizer = save_optimizer
        if (
            save_optimizer
            and pointer is not None
            and (pointer.generation == generation.generation_id)
        ):
            if (
                pointer.step != generation.policy_step
                or pointer.adapter.training_session_id != generation.training_session_id
            ):
                raise RuntimeError(
                    "committed optimizer generation disagrees with resident learner"
                )
            if object_target is None:
                manifest = read_optimizer_generation_manifest(
                    optimizer_state_path, pointer.generation
                )
                return self._existing_snapshot(
                    ref, generation, pointer.adapter, manifest=manifest
                )
            existing = pointer.adapter
            persist_optimizer = False
        staging = (
            None
            if existing is not None
            or (object_target is not None and not save_optimizer)
            else str(
                Path(state.output_dir)
                / "megatron_runtime"
                / "staging"
                / generation.generation_id
            )
        )
        started = time.perf_counter()
        plan = await self.trainer.prepare_snapshot(
            GenerationSnapshotJobSpec(
                operation_id=ref.operation_id,
                sequence_continuation_of=sequence_continuation_of,
                run_id=ref.run_id,
                sequence_id=ref.sequence_id,
                training_session_id=state.registration.training_session_id,
                learner_version=ref.learner_parent_version,
                generation=generation,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=staging,
                existing_adapter=existing,
                adapter_object_target=object_target,
                save_optimizer=persist_optimizer,
            )
        )
        return _PendingSnapshot(
            run_id=ref.run_id,
            generation=generation,
            optimizer_state_path=optimizer_state_path,
            plan=plan,
            publication=self.trainer.wait_for_publication(ref.operation_id),
            started=started,
        )

    def _existing_snapshot(
        self,
        ref: OperationRef,
        generation: TrainerGeneration,
        adapter: OptimizerAdapter,
        *,
        manifest: OptimizerGenerationManifest | None = None,
    ) -> _ExistingSnapshot:
        shards = (
            {} if manifest is None else {shard.rank: shard for shard in manifest.shards}
        )
        ranks = tuple(
            SnapshotRankWritePlan(
                rank=rank,
                generation=generation,
                adapter=adapter if rank == 0 else None,
                optimizer_shard=shards.get(rank),
                runtime_sha256=None if manifest is None else manifest.runtime_sha256,
                topology=None if manifest is None else manifest.topology,
                saves_optimizer=manifest is not None,
            )
            for rank in range(len(self.runtime_spec.trainer_mesh.ranks))
        )
        plan = build_snapshot_write_plan(
            operation_id=ref.operation_id, generation=generation, ranks=ranks
        )
        return _ExistingSnapshot(
            adapter=adapter,
            optimizer_bytes=(
                None if manifest is None else optimizer_generation_nbytes(manifest)
            ),
            plan=plan,
        )

    def _sampler_object_target(
        self, run_id: str, generation: TrainerGeneration
    ) -> BinaryObjectTarget:
        assert self.sampler_store is not None
        return vllm_lora_object_target(
            self.sampler_store,
            run_id=run_id,
            training_session_id=generation.training_session_id,
            generation_id=generation.generation_id,
            policy_step=generation.policy_step,
        )

    async def _finish_snapshot(
        self,
        snapshot: _ExistingSnapshot | _PendingSnapshot,
        *,
        grant: SnapshotWriteGrant,
        authorization_metrics: dict[str, float],
    ) -> tuple[OptimizerAdapter, dict[str, float], int | None]:
        grant.validate_plan(snapshot.plan)
        if isinstance(snapshot, _ExistingSnapshot):
            return snapshot.adapter, authorization_metrics, snapshot.optimizer_bytes
        records = await snapshot.publication
        state = self._require_run(snapshot.run_id)
        if (
            state.generation.training_session_id
            != snapshot.generation.training_session_id
        ):
            raise RuntimeError("snapshot completed for another training session")
        durable = await asyncio.to_thread(
            commit_trainer_publication,
            snapshot.optimizer_state_path,
            snapshot.generation,
            records,
            plan=snapshot.plan,
            grant=grant,
        )
        if state.generation.policy_step < snapshot.generation.policy_step:
            raise RuntimeError("resident learner regressed behind completed snapshot")
        rank_metrics = {
            key: max(record.metrics.get(key, 0.0) for record in records)
            for key in {key for record in records for key in record.metrics}
        }
        return (
            durable.transport_adapter or durable.adapter,
            {
                **authorization_metrics,
                **rank_metrics,
                "time/snapshot_durable_s": time.perf_counter() - snapshot.started,
            },
            durable.optimizer_bytes,
        )

    def retire_operation(self, operation_id: str) -> None:
        if operation_id in self._pending_results:
            raise RuntimeError("cannot retire an incomplete training operation")
        self.trainer.retire_operation(operation_id)
        self._prepared_saves.pop(operation_id, None)
        self._results.pop(operation_id, None)

    async def close(self) -> None:
        if self._closed:
            return
        primary: BaseException | None = None
        try:
            if self._pending_results:
                outcomes = await asyncio.gather(
                    *(pending[1] for pending in self._pending_results.values()),
                    return_exceptions=True,
                )
                failures = [
                    outcome
                    for outcome in outcomes
                    if isinstance(outcome, BaseException)
                ]
                if len(failures) == 1:
                    raise failures[0]
                if failures:
                    raise BaseExceptionGroup(
                        "pending snapshot completion failed", failures
                    )
            if self._batch_releases:
                await asyncio.gather(
                    *tuple(self._batch_releases), return_exceptions=True
                )
                await asyncio.sleep(0)
            self._raise_batch_release_failures()
        except BaseException as error:
            primary = error
        self._closed = True
        try:
            await self.trainer.close()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note(
                    f"trainer close also failed: {type(error).__name__}: {error}"
                )
        self._prepared_saves.clear()
        if primary is not None:
            raise primary

    def _release_batch_soon(self, packed: DistributedPackedBatch) -> None:
        task = asyncio.create_task(
            self.runtime.release_batch(packed),
            name=f"release-packed-batch-{packed.packing_generation_id}",
        )
        self._batch_releases.add(task)
        task.add_done_callback(self._batch_release_done)

    def _batch_release_done(self, task: asyncio.Task[None]) -> None:
        self._batch_releases.discard(task)
        if task.cancelled():
            self._batch_release_failures.append(
                RuntimeError("packed-batch release was cancelled")
            )
        elif error := task.exception():
            self._batch_release_failures.append(error)

    def _raise_batch_release_failures(self) -> None:
        if self._batch_release_failures:
            raise BaseExceptionGroup(
                "packed-batch release failed", self._batch_release_failures
            )

    def _require_run(self, run_id: str) -> _ResidentRun:
        self._require_open()
        try:
            return self._runs[run_id]
        except KeyError as error:
            raise RuntimeError(f"training run is not resident: {run_id!r}") from error

    def _require_parent(self, ref: OperationRef) -> _ResidentRun:
        state = self._require_run(ref.run_id)
        if state.generation.policy_step != ref.learner_parent_version:
            raise RuntimeError(
                "resident learner does not match operation parent: "
                f"resident={state.generation.policy_step}, "
                f"operation={ref.learner_parent_version}"
            )
        return state

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Megatron training slot is closed")
        self._raise_batch_release_failures()

    @staticmethod
    def _fingerprint(ref: Contract, request: Contract, *parts: str) -> str:
        return "\0".join((ref.model_dump_json(), request.model_dump_json(), *parts))

    def _cached_result(self, operation_id: str, fingerprint: str) -> Contract | None:
        cached = self._results.get(operation_id)
        if cached is None:
            return None
        if cached[0] != fingerprint:
            raise RuntimeError("operation_id was reused for a different command")
        return cached[1]

    def _pending_result(
        self, operation_id: str, fingerprint: str
    ) -> asyncio.Future[Contract] | None:
        pending = self._pending_results.get(operation_id)
        if pending is None:
            return None
        if pending[0] != fingerprint:
            raise RuntimeError("operation_id was reused for a different command")
        return pending[1]

    def _start_pending_result(
        self, operation_id: str, fingerprint: str, completion: Any
    ) -> asyncio.Future[Contract]:
        task = asyncio.create_task(completion, name=f"megatron-save-{operation_id}")
        self._pending_results[operation_id] = (fingerprint, task)

        def completed(done: asyncio.Future[Contract]) -> None:
            current = self._pending_results.get(operation_id)
            if current is not None and current[1] is done:
                self._pending_results.pop(operation_id)

        task.add_done_callback(completed)
        return task

    def _require_managed_path(self, path: str) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.artifact_root):
            raise ValueError(f"training run path leaves artifact root: {resolved}")
        return str(resolved)


def _resolved_future(value: Any) -> asyncio.Future[Any]:
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future
