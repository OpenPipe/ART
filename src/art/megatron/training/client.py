from __future__ import annotations

import asyncio
from contextlib import suppress
import time
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, cast

from art.training.client import TrainingOperation
from art.training.contracts import (
    CheckpointRef,
    Contract,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    LossFnOutput,
    OperationKind,
    OptimStepRequest,
    OptimStepResult,
    PackingOutcome,
    PolicyTokenCount,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)
from art.training.sequencing import CommandAdmission, RunCommandLedger

from ..runtime.specs import ExperimentalTrainConfig, RlForwardBackwardConfig

if TYPE_CHECKING:
    from art.megatron.backend import MegatronBackend
    from art.megatron.distributed_service import DistributedMegatronService
    from art.model import TrainableModel

ResultT = TypeVar("ResultT", bound=Contract)


def _consume_future(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


class _DeferredResult(NamedTuple, Generic[ResultT]):
    completion: asyncio.Task[ResultT]


class LocalTrainingOperation(Generic[ResultT]):
    def __init__(
        self,
        admission: CommandAdmission,
        ordered: asyncio.Task[None],
        result: asyncio.Future[ResultT],
    ) -> None:
        self._admission = admission
        self._ordered = ordered
        self._result = result

    @property
    def ref(self):
        return self._admission.ref

    async def result(self) -> ResultT:
        return await asyncio.shield(self._result)

    async def cancel(self) -> None:
        self._ordered.cancel()
        self._result.cancel()
        with suppress(asyncio.CancelledError):
            await self._ordered


class LocalMegatronTrainingClient:
    """Run the canonical command API against one local ART Megatron service."""

    def __init__(
        self,
        *,
        run_id: str,
        learner_version: int,
        backend: MegatronBackend,
        model: TrainableModel,
        service: DistributedMegatronService,
    ) -> None:
        self._ledger = RunCommandLedger(run_id, learner_version=learner_version)
        self._backend = backend
        self._model = model
        self._service = service
        self._operations: dict[str, LocalTrainingOperation[Any]] = {}
        self._retired_operation_ids: set[str] = set()
        self._completion_tasks: set[asyncio.Task[Any]] = set()
        self._tail: asyncio.Task[Any] | None = None
        self._closed = False

    @property
    def projected_learner_version(self) -> int:
        return self._ledger.projected_learner_version

    @property
    def run_id(self) -> str:
        return self._ledger.run_id

    @property
    def next_sequence_id(self) -> int:
        return self._ledger.next_sequence_id

    async def _submit(
        self,
        request: Any,
        *,
        kind: OperationKind,
        execute: Any,
    ) -> TrainingOperation[Any]:
        if self._closed:
            raise RuntimeError("local training client is closed")
        admission = await self._ledger.admit(request, kind=kind)
        if operation := self._operations.get(admission.ref.operation_id):
            return operation
        if admission.ref.operation_id in self._retired_operation_ids:
            raise RuntimeError("completed F/B result is no longer retained")
        predecessor = self._tail

        result = asyncio.get_running_loop().create_future()
        result.add_done_callback(_consume_future)

        async def settle(completion: asyncio.Task[Any]) -> None:
            try:
                value = await asyncio.shield(completion)
            except BaseException as error:
                if not result.done():
                    result.set_exception(error)
            else:
                if not result.done():
                    result.set_result(value)

        async def run() -> None:
            try:
                if predecessor is not None:
                    await asyncio.shield(predecessor)
                value = await execute(admission)
            except BaseException as error:
                if not result.done():
                    result.set_exception(error)
                raise
            if isinstance(value, _DeferredResult):
                completion = asyncio.create_task(
                    settle(value.completion),
                    name=f"megatron-{kind}-result-{request.sequence_id}",
                )
                self._completion_tasks.add(completion)
                completion.add_done_callback(self._completion_tasks.discard)
            elif not result.done():
                result.set_result(value)

        ordered = asyncio.create_task(
            run(), name=f"megatron-{kind}-{request.sequence_id}"
        )
        ordered.add_done_callback(_consume_future)
        operation = LocalTrainingOperation(admission, ordered, result)
        self._operations[admission.ref.operation_id] = operation
        self._tail = ordered
        return operation

    async def forward(
        self, request: ForwardRequest
    ) -> TrainingOperation[ForwardResult]:
        del request
        raise NotImplementedError("Megatron forward command is not implemented")

    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> TrainingOperation[ForwardBackwardResult]:
        async def execute(admission: CommandAdmission) -> ForwardBackwardResult:
            batch = await self._prepare_rl_batch(request)
            payload = batch.payload
            from art.megatron.backend import _DistributedBatchPayload

            if not isinstance(payload, _DistributedBatchPayload):
                raise RuntimeError("local command did not use the typed data plane")
            release_started = time.perf_counter()
            await self._backend._release_trajectory_sources(batch, payload)
            raw = await self._service.forward_backward_command(
                admission.ref,
                payload.packed,
                _forward_backward_config(request),
                _experimental_config(request),
            )
            return ForwardBackwardResult(
                operation_id=admission.ref.operation_id,
                packing=_packing_outcome(batch),
                loss_fn_outputs=tuple(
                    LossFnOutput(token_logprobs=values)
                    for values in raw["token_logprobs"]
                ),
                metrics={
                    **_packing_metrics(payload.packed),
                    "time/step_source_lease_release_s": (
                        time.perf_counter() - release_started
                    ),
                    **raw["metrics"],
                },
            )

        return cast(
            TrainingOperation[ForwardBackwardResult],
            await self._submit(
                request,
                kind="forward_backward",
                execute=execute,
            ),
        )

    async def optim_step(
        self, request: OptimStepRequest
    ) -> TrainingOperation[OptimStepResult]:
        async def execute(admission: CommandAdmission) -> OptimStepResult:
            raw, generation = await self._service.optimizer_command(
                admission.ref,
                request.optimizer,
                admission.contributing_forward_backward_operation_ids,
            )
            result = OptimStepResult(
                operation_id=admission.ref.operation_id,
                contributing_forward_backward_operation_ids=(
                    admission.contributing_forward_backward_operation_ids
                ),
                checkpoint=_checkpoint_ref(
                    admission.ref.run_id,
                    generation.policy_step,
                    generation.generation_id,
                ),
                metrics=raw["metrics"],
            )
            for operation_id in admission.contributing_forward_backward_operation_ids:
                if self._operations.pop(operation_id, None) is not None:
                    self._retired_operation_ids.add(operation_id)
            return result

        return cast(
            TrainingOperation[OptimStepResult],
            await self._submit(request, kind="optim_step", execute=execute),
        )

    async def save_weights_for_sampler(
        self, request: SaveWeightsForSamplerRequest
    ) -> TrainingOperation[SamplerWeightsResult]:
        _validate_publication_mode(
            request,
            update_mode=self._service.rollout_weight_update_mode,
            weights_mode=self._service.rollout_weights_mode,
        )

        async def execute(
            admission: CommandAdmission,
        ) -> SamplerWeightsResult | _DeferredResult[SamplerWeightsResult]:
            launch = await self._service.snapshot_command(
                admission.ref,
                save_optimizer=False,
                activate_serving=request.publication.mode != "none",
            )

            async def complete() -> SamplerWeightsResult:
                durable = await launch.completion
                return SamplerWeightsResult(
                    operation_id=admission.ref.operation_id,
                    checkpoint=_checkpoint_ref(
                        admission.ref.run_id,
                        durable.adapter.step,
                        request.checkpoint_name,
                    ),
                    lora=durable.adapter.identity,
                    publication_metrics=launch.metrics,
                )

            return _DeferredResult(completion=asyncio.create_task(complete()))

        return cast(
            TrainingOperation[SamplerWeightsResult],
            await self._submit(request, kind="save_sampler", execute=execute),
        )

    async def save_state(
        self, request: SaveStateRequest
    ) -> TrainingOperation[SaveStateResult]:
        async def execute(
            admission: CommandAdmission,
        ) -> SaveStateResult | _DeferredResult[SaveStateResult]:
            launch = await self._service.snapshot_command(
                admission.ref,
                save_optimizer=True,
                activate_serving=False,
            )

            async def complete() -> SaveStateResult:
                durable = await launch.completion
                return SaveStateResult(
                    operation_id=admission.ref.operation_id,
                    checkpoint=_checkpoint_ref(
                        admission.ref.run_id,
                        durable.adapter.step,
                        request.checkpoint_name,
                    ),
                    optimizer_state=self._service.optimizer_state_path,
                    metrics=launch.metrics,
                )

            return _DeferredResult(completion=asyncio.create_task(complete()))

        return cast(
            TrainingOperation[SaveStateResult],
            await self._submit(request, kind="save_state", execute=execute),
        )

    async def load_state(
        self, request: LoadStateRequest
    ) -> TrainingOperation[LoadStateResult]:
        del request
        raise NotImplementedError("Megatron load_state command is not implemented")

    async def load_state_with_optimizer(
        self, request: LoadStateRequest
    ) -> TrainingOperation[LoadStateResult]:
        del request
        raise NotImplementedError(
            "Megatron load_state_with_optimizer command is not implemented"
        )

    async def _prepare_rl_batch(self, request: ForwardBackwardRequest):
        if request.batch.kind != "rl":
            raise ValueError("Megatron RL F/B requires an RL trajectory batch")
        request.batch.require_local_groups()
        return request.batch.require_local_packed_batch()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending = tuple(
            operation._ordered
            for operation in self._operations.values()
            if not operation._ordered.done()
        ) + tuple(self._completion_tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def _forward_backward_config(
    request: ForwardBackwardRequest,
) -> RlForwardBackwardConfig:
    values = request.loss.values
    kl_penalty_coef = values.get("kl_penalty_coef", 0.0)
    if not isinstance(kl_penalty_coef, int | float):
        raise TypeError("kl_penalty_coef must be numeric")
    return RlForwardBackwardConfig(
        kl_penalty_coef=float(kl_penalty_coef),
        kl_penalty_source=cast(Any, values.get("kl_penalty_source", "current_learner")),
        grad_accumulation_sequences=cast(
            int | None, values.get("grad_accumulation_sequences")
        ),
    )


def _experimental_config(request: ForwardBackwardRequest) -> ExperimentalTrainConfig:
    values = {
        name: value
        for name, value in request.loss.values.items()
        if name in ExperimentalTrainConfig.model_fields
    }
    values["ppo"] = request.loss.name == "ppo"
    values["scale_rewards"] = bool(values.get("scale_rewards", True))
    return ExperimentalTrainConfig.model_validate(values)


def _packing_outcome(batch: Any) -> PackingOutcome:
    ref = batch.payload.packed.leases.ref
    stats = ref.prefix_tree_packing_stats
    if stats is None or stats.policy_token_counts is None:
        raise RuntimeError("RL packed batch has no exact policy-token provenance")
    shapes = tuple(
        shape for shape in batch.payload.packed.packed_group_shapes if shape is not None
    )
    return PackingOutcome(
        packed_sequence_length=batch.sequence_length,
        packed_sequences=batch.num_sequences,
        target_packed_sequences=batch.num_sequences,
        nominal_capacity_tokens=batch.num_sequences * batch.sequence_length,
        physical_tokens=stats.physical_tokens,
        non_padding_tokens=batch.non_padding_tokens,
        loss_bearing_tokens=batch.loss_bearing_tokens,
        trainable_assistant_tokens=batch.trainable_assistant_tokens,
        policy_token_counts=tuple(
            PolicyTokenCount(
                policy_version=version,
                trainable_assistant_tokens=count,
            )
            for version, count in sorted(stats.policy_token_counts.items())
        ),
        group_shapes=shapes,
    )


def _checkpoint_ref(run_id: str, learner_version: int, checkpoint_id: str):
    return CheckpointRef(
        run_id=run_id,
        learner_version=learner_version,
        checkpoint_id=checkpoint_id,
    )


def _validate_publication_mode(
    request: SaveWeightsForSamplerRequest,
    *,
    update_mode: str,
    weights_mode: str,
) -> None:
    requested = request.publication.mode
    if requested == "none":
        return
    expected = (
        "merged_weights"
        if weights_mode == "merged"
        else "in_flight_lora"
        if update_mode == "in_flight_lora"
        else "versioned_lora"
    )
    if requested != expected:
        raise ValueError(
            f"sampler publication mode {requested!r} conflicts with "
            f"weights={weights_mode!r}, update={update_mode!r}"
        )


def _packing_metrics(packed: Any) -> dict[str, float]:
    from art.megatron.backend import _packing_metrics as build_metrics

    return build_metrics(packed)
