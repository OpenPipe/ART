from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, cast
import uuid

from pydantic import BaseModel, ConfigDict, Field

from art.distributed.art_runtime import ArtRuntime, DistributedPackedBatch
from art.distributed.packing import PackingRequest
from art.distributed.rollout import RolloutModelSpec
from art.megatron.optimizer_state import (
    OptimizerAdapter,
    optimizer_adapter,
    read_adapter_publication,
)
from art.megatron.runtime.publication import commit_trainer_publication
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
    TrainerGeneration,
    TrainerRuntimeSpec,
)
from art.training.contracts import (
    Contract,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    LossFnOutput,
    OperationRef,
    OptimStepRequest,
    OptimStepResult,
    PackingOutcome,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)
from art.utils.output_dirs import get_step_checkpoint_dir

from .commands import (
    checkpoint_ref,
    experimental_train_config,
    forward_backward_config,
    packing_metrics,
    packing_outcome,
)


class PreparedForward(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    ref: OperationRef
    packed: DistributedPackedBatch
    packing: PackingOutcome
    config: RlForwardBackwardConfig
    experimental_config: ExperimentalTrainConfig

    @property
    def token_cost(self) -> int:
        return self.packing.physical_tokens


class PreparedForwardBackward(PreparedForward):
    pass


class _ResidentRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    registration: RunSlotRegistration
    model: RolloutModelSpec
    output_dir: str = Field(min_length=1)
    generation: TrainerGeneration


class MegatronTrainingSlot:
    """Canonical pack-and-command facade for one persistent Megatron mesh."""

    def __init__(
        self,
        *,
        runtime: ArtRuntime,
        trainer: Any,
        runtime_spec: TrainerRuntimeSpec,
        artifact_root: str,
    ) -> None:
        if trainer.runtime_spec != runtime_spec:
            raise ValueError("trainer slot does not match its runtime spec")
        self.runtime = runtime
        self.trainer = trainer
        self.runtime_spec = runtime_spec
        self.artifact_root = str(Path(artifact_root).resolve())
        self._runs: dict[str, _ResidentRun] = {}
        self._results: dict[str, tuple[str, Contract]] = {}
        self._closed = False

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
            step=registration.learner_version,
        ) or optimizer_adapter(
            registration.adapter_path,
            registration.learner_version,
            training_session_id=registration.training_session_id,
        )
        if adapter.training_session_id != registration.training_session_id:
            raise ValueError("adapter belongs to another training session")
        await self.trainer.register_run(registration)
        self._runs[registration.run_id] = _ResidentRun(
            registration=registration,
            model=model,
            output_dir=output_dir,
            generation=TrainerGeneration(
                training_session_id=registration.training_session_id,
                policy_step=registration.learner_version,
                generation_id=adapter.generation_id,
                adapter_path=adapter.identity,
            ),
        )

    async def unregister_run(self, run_id: str) -> None:
        self._require_open()
        if run_id not in self._runs:
            return
        await self.trainer.unregister_run(run_id)
        self._runs.pop(run_id)

    async def prepare_forward_backward(
        self,
        ref: OperationRef,
        request: ForwardBackwardRequest,
    ) -> PreparedForwardBackward:
        prepared = await self._prepare_forward(ref, request)
        return PreparedForwardBackward.model_validate(prepared.model_dump())

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
        if request.batch.kind != "rl":
            raise ValueError("Megatron forward requires an RL trajectory batch")
        config = experimental_train_config(request)
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
        return PreparedForward(
            ref=ref,
            packed=packed,
            packing=packing_outcome(packed),
            config=forward_backward_config(request),
            experimental_config=config,
        )

    async def discard_prepared(self, prepared: PreparedForward) -> None:
        await self.runtime.release_batch(prepared.packed)

    async def forward(self, prepared: PreparedForward) -> ForwardResult:
        ref = prepared.ref
        state = self._require_parent(ref)
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
        )
        try:
            raw = await self.trainer.forward(job, prepared.packed.leases)
        finally:
            primary = sys.exception()
            try:
                await self.runtime.release_batch(prepared.packed)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "packed-batch release also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        return ForwardResult(
            operation_id=ref.operation_id,
            packing=prepared.packing,
            loss_fn_outputs=tuple(
                LossFnOutput(token_logprobs=values) for values in raw["token_logprobs"]
            ),
            metrics={**packing_metrics(prepared.packed), **raw["metrics"]},
        )

    async def forward_backward(
        self, prepared: PreparedForwardBackward
    ) -> ForwardBackwardResult:
        ref = prepared.ref
        state = self._require_parent(ref)
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
        )
        try:
            raw = await self.trainer.forward_backward(job, prepared.packed.leases)
        finally:
            primary = sys.exception()
            try:
                await self.runtime.release_batch(prepared.packed)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "packed-batch release also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        return ForwardBackwardResult(
            operation_id=ref.operation_id,
            packing=prepared.packing,
            loss_fn_outputs=tuple(
                LossFnOutput(token_logprobs=values) for values in raw["token_logprobs"]
            ),
            metrics={**packing_metrics(prepared.packed), **raw["metrics"]},
        )

    async def optim_step(
        self,
        ref: OperationRef,
        request: OptimStepRequest,
        contributions: tuple[str, ...],
    ) -> OptimStepResult:
        fingerprint = self._fingerprint(ref, request, *contributions)
        if cached := self._cached_result(ref.operation_id, fingerprint):
            return cast(OptimStepResult, cached)
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
            contributing_forward_backward_operation_ids=contributions,
            optimizer=request.optimizer,
        )
        raw = await self.trainer.optim_step(job)
        generation = TrainerGeneration(
            training_session_id=state.registration.training_session_id,
            policy_step=output_version,
            generation_id=(
                f"step-{output_version:08d}-"
                f"{hashlib.sha256(ref.operation_id.encode()).hexdigest()[:32]}"
            ),
            adapter_path=get_step_checkpoint_dir(state.output_dir, output_version),
        )
        state.generation = generation
        state.registration = state.registration.model_copy(
            update={"learner_version": output_version}
        )
        commit_operation_id = f"{ref.operation_id}:optim-commit"
        _adapter, snapshot_metrics = await self._snapshot(
            OperationRef(
                run_id=ref.run_id,
                operation_id=commit_operation_id,
                sequence_id=ref.sequence_id,
                learner_parent_version=output_version,
                kind="save_state",
            ),
            save_optimizer=True,
        )
        self.trainer.retire_operation(commit_operation_id)
        result = OptimStepResult(
            operation_id=ref.operation_id,
            contributing_forward_backward_operation_ids=contributions,
            checkpoint=checkpoint_ref(
                ref.run_id, output_version, generation.generation_id
            ),
            metrics={**raw["metrics"], **snapshot_metrics},
        )
        self._results[ref.operation_id] = (fingerprint, result)
        return result

    async def load_state(
        self,
        ref: OperationRef,
        request: LoadStateRequest,
        source: ResolvedCheckpointState,
    ) -> LoadStateResult:
        fingerprint = self._fingerprint(ref, request, source.model_dump_json())
        if cached := self._cached_result(ref.operation_id, fingerprint):
            return cast(LoadStateResult, cached)
        state = self._require_parent(ref)
        output_version = ref.reserved_output_learner_version
        if output_version is None:
            raise ValueError("load operation has no reserved learner version")
        if request.restore_optimizer and source.optimizer_state_path is None:
            raise ValueError("optimizer-exact load has no optimizer state")
        job = LoadStateJobSpec(
            operation_id=ref.operation_id,
            run_id=ref.run_id,
            sequence_id=ref.sequence_id,
            training_session_id=state.registration.training_session_id,
            expected_learner_version=ref.learner_parent_version,
            learner_version=output_version,
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
        raw = await self.trainer.load_state(job)
        generation = TrainerGeneration(
            training_session_id=state.registration.training_session_id,
            policy_step=output_version,
            generation_id=(
                f"step-{output_version:08d}-"
                f"{hashlib.sha256(ref.operation_id.encode()).hexdigest()[:32]}"
            ),
            adapter_path=get_step_checkpoint_dir(state.output_dir, output_version),
        )
        state.generation = generation
        state.registration = state.registration.model_copy(
            update={"learner_version": output_version}
        )
        commit_ref = OperationRef(
            run_id=ref.run_id,
            operation_id=f"{ref.operation_id}:load-commit",
            sequence_id=ref.sequence_id,
            learner_parent_version=output_version,
            kind="save_state",
        )
        await self._snapshot(commit_ref, save_optimizer=True)
        self.trainer.retire_operation(commit_ref.operation_id)
        result = LoadStateResult(
            operation_id=ref.operation_id,
            checkpoint=checkpoint_ref(
                ref.run_id, output_version, generation.generation_id
            ),
            optimizer_restored=bool(raw["optimizer_restored"]),
        )
        self._results[ref.operation_id] = (fingerprint, result)
        return result

    async def save_weights_for_sampler(
        self,
        ref: OperationRef,
        request: SaveWeightsForSamplerRequest,
    ) -> SamplerWeightsResult:
        fingerprint = self._fingerprint(ref, request)
        if cached := self._cached_result(ref.operation_id, fingerprint):
            return cast(SamplerWeightsResult, cached)
        adapter, metrics = await self._snapshot(ref, save_optimizer=False)
        result = SamplerWeightsResult(
            operation_id=ref.operation_id,
            checkpoint=checkpoint_ref(
                ref.run_id, adapter.step, request.checkpoint_name
            ),
            lora=adapter.identity,
            publication_metrics=metrics,
        )
        self._results[ref.operation_id] = (fingerprint, result)
        return result

    async def save_state(
        self,
        ref: OperationRef,
        request: SaveStateRequest,
    ) -> SaveStateResult:
        fingerprint = self._fingerprint(ref, request)
        if cached := self._cached_result(ref.operation_id, fingerprint):
            return cast(SaveStateResult, cached)
        adapter, metrics = await self._snapshot(ref, save_optimizer=True)
        state = self._require_run(ref.run_id)
        optimizer_state_path = state.registration.optimizer_state_path
        result = SaveStateResult(
            operation_id=ref.operation_id,
            checkpoint=checkpoint_ref(
                ref.run_id, adapter.step, request.checkpoint_name
            ),
            optimizer_state=optimizer_state_path,
            metrics=metrics,
        )
        self._results[ref.operation_id] = (fingerprint, result)
        return result

    async def _snapshot(
        self,
        ref: OperationRef,
        *,
        save_optimizer: bool,
    ) -> tuple[OptimizerAdapter, dict[str, float]]:
        state = self._require_parent(ref)
        generation = state.generation
        existing = read_adapter_publication(
            generation.adapter_path, step=generation.policy_step
        )
        optimizer_state_path = state.registration.optimizer_state_path
        staging = (
            None
            if existing is not None
            else str(
                Path(state.output_dir)
                / "megatron_runtime"
                / "staging"
                / generation.generation_id
            )
        )
        started = time.perf_counter()
        raw = await self.trainer.snapshot(
            GenerationSnapshotJobSpec(
                operation_id=ref.operation_id,
                run_id=ref.run_id,
                sequence_id=ref.sequence_id,
                training_session_id=state.registration.training_session_id,
                learner_version=ref.learner_parent_version,
                generation=generation,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=staging,
                existing_adapter=existing,
                save_optimizer=save_optimizer,
            )
        )
        records = await self.trainer.wait_for_publication(ref.operation_id)
        durable = await asyncio.to_thread(
            commit_trainer_publication,
            optimizer_state_path,
            generation,
            records,
        )
        state.generation = TrainerGeneration(
            training_session_id=durable.adapter.training_session_id,
            policy_step=durable.adapter.step,
            generation_id=durable.adapter.generation_id,
            adapter_path=durable.adapter.identity,
        )
        return durable.adapter, {
            **raw["metrics"],
            "time/snapshot_durable_s": time.perf_counter() - started,
        }

    def retire_operation(self, operation_id: str) -> None:
        self.trainer.retire_operation(operation_id)
        self._results.pop(operation_id, None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.trainer.close()

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

    def _require_managed_path(self, path: str) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.artifact_root):
            raise ValueError(f"training run path leaves artifact root: {resolved}")
        return str(resolved)
