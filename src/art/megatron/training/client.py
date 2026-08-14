from __future__ import annotations

import asyncio
from contextlib import suppress
import time
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, cast

from art.preprocessing.sft import SftBatchTokenizer
from art.training.client import TrainingOperation
from art.training.contracts import (
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
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)
from art.training.sequencing import CommandAdmission, RunCommandLedger
from art.types import TrainConfig

from ..runtime.specs import ResolvedCheckpointState
from .commands import (
    checkpoint_ref,
    experimental_train_config,
    forward_backward_config,
    packing_metrics,
    packing_outcome,
    sft_batch_data,
    sft_packing_outcome,
)

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
        self._checkpoints: dict[str, ResolvedCheckpointState] = {}
        self._sft_tokenizer = SftBatchTokenizer()
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
        return await self._forward(request, backward=False)

    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> TrainingOperation[ForwardBackwardResult]:
        return cast(
            TrainingOperation[ForwardBackwardResult],
            await self._forward(request, backward=True),
        )

    async def _forward(
        self,
        request: ForwardRequest | ForwardBackwardRequest,
        *,
        backward: bool,
    ) -> TrainingOperation[ForwardResult | ForwardBackwardResult]:
        async def execute(
            admission: CommandAdmission,
        ) -> ForwardResult | ForwardBackwardResult:
            if request.batch.kind == "sft":
                started = time.perf_counter()
                tokenized = await asyncio.to_thread(
                    self._sft_tokenizer.tokenize,
                    self._model,
                    request.batch.trajectories,
                    assistant_turns=request.batch.assistant_turns,
                )
                if tokenized.num_trainable_tokens < 1:
                    raise ValueError("supervised batch produced no trainable tokens")
                batch = sft_batch_data(tokenized)
                grad_sequences = (
                    self._service.resolve_sft_global_grad_accumulation_sequences(
                        batch.num_trajectories
                    )
                )
                raw = await (
                    self._service.sft_forward_backward_command(
                        admission.ref, batch, grad_sequences
                    )
                    if backward
                    else self._service.sft_forward_command(
                        admission.ref, batch, grad_sequences
                    )
                )
                result_type = ForwardBackwardResult if backward else ForwardResult
                return result_type(
                    operation_id=admission.ref.operation_id,
                    packing=sft_packing_outcome(batch),
                    loss_fn_outputs=tuple(
                        LossFnOutput(token_logprobs=values)
                        for values in raw["token_logprobs"]
                    ),
                    metrics={
                        "time/step_tokenize_trajectory_groups_s": (
                            time.perf_counter() - started
                        ),
                        "data/step_num_dropped_trajectories": float(
                            batch.num_dropped_trajectories
                        ),
                        **raw["metrics"],
                    },
                )
            batch = await self._prepare_rl_batch(request)
            payload = batch.payload
            from art.megatron.backend import _DistributedBatchPayload

            if not isinstance(payload, _DistributedBatchPayload):
                raise RuntimeError("local command did not use the typed data plane")
            release_started = time.perf_counter()
            await self._backend._release_trajectory_sources(batch, payload)
            training_config = forward_backward_config(request)
            raw = await (
                self._service.forward_backward_command(
                    admission.ref,
                    payload.packed,
                    training_config,
                    experimental_train_config(request),
                )
                if backward
                else self._service.forward_command(
                    admission.ref,
                    payload.packed,
                    training_config,
                    experimental_train_config(request),
                )
            )
            result_type = ForwardBackwardResult if backward else ForwardResult
            return result_type(
                operation_id=admission.ref.operation_id,
                packing=packing_outcome(
                    payload.packed,
                    target_packed_sequences=(
                        await self._service.resolve_global_grad_accumulation_sequences(
                            TrainConfig(
                                grad_accumulation_sequences=(
                                    training_config.grad_accumulation_sequences
                                )
                            )
                        )
                    ),
                ),
                loss_fn_outputs=tuple(
                    LossFnOutput(token_logprobs=values)
                    for values in raw["token_logprobs"]
                ),
                metrics={
                    **packing_metrics(payload.packed),
                    "time/step_source_lease_release_s": (
                        time.perf_counter() - release_started
                    ),
                    **raw["metrics"],
                },
            )

        kind: OperationKind = "forward_backward" if backward else "forward"
        return await self._submit(
            request,
            kind=kind,
            execute=execute,
        )

    async def optim_step(
        self, request: OptimStepRequest
    ) -> TrainingOperation[OptimStepResult]:
        async def execute(admission: CommandAdmission) -> OptimStepResult:
            raw, _generation = await self._service.optimizer_command(
                admission.ref,
                request.optimizer,
                admission.contributing_forward_backward_operation_ids,
            )
            result = OptimStepResult(
                operation_id=admission.ref.operation_id,
                contributing_forward_backward_operation_ids=(
                    admission.contributing_forward_backward_operation_ids
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
                self._remember_checkpoint(
                    request.checkpoint_name,
                    ResolvedCheckpointState(
                        adapter_path=durable.adapter.identity,
                        adapter_step=durable.adapter.step,
                    ),
                )
                return SamplerWeightsResult(
                    operation_id=admission.ref.operation_id,
                    checkpoint=checkpoint_ref(
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
                self._remember_checkpoint(
                    request.checkpoint_name,
                    ResolvedCheckpointState(
                        adapter_path=durable.adapter.identity,
                        adapter_step=durable.adapter.step,
                        optimizer_state_path=self._service.optimizer_state_path,
                        optimizer_generation_id=durable.adapter.generation_id,
                    ),
                )
                return SaveStateResult(
                    operation_id=admission.ref.operation_id,
                    checkpoint=checkpoint_ref(
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
        return await self._load_state(request, restore_optimizer=False)

    async def load_state_with_optimizer(
        self, request: LoadStateRequest
    ) -> TrainingOperation[LoadStateResult]:
        return await self._load_state(request, restore_optimizer=True)

    async def _load_state(
        self,
        request: LoadStateRequest,
        *,
        restore_optimizer: bool,
    ) -> TrainingOperation[LoadStateResult]:
        async def execute(admission: CommandAdmission) -> LoadStateResult:
            try:
                source = self._checkpoints[request.checkpoint]
            except KeyError as error:
                raise ValueError(
                    f"unknown local checkpoint: {request.checkpoint!r}"
                ) from error
            raw, generation, _metrics = await self._service.load_state_command(
                admission.ref,
                source,
                restore_optimizer=restore_optimizer,
            )
            checkpoint = checkpoint_ref(
                admission.ref.run_id,
                generation.policy_step,
                generation.generation_id,
            )
            self._remember_checkpoint(
                checkpoint.checkpoint_id,
                ResolvedCheckpointState(
                    adapter_path=generation.adapter_path,
                    adapter_step=generation.policy_step,
                    optimizer_state_path=self._service.optimizer_state_path,
                    optimizer_generation_id=generation.generation_id,
                ),
            )
            return LoadStateResult(
                operation_id=admission.ref.operation_id,
                checkpoint=checkpoint,
                optimizer_restored=bool(raw["optimizer_restored"]),
            )

        return cast(
            TrainingOperation[LoadStateResult],
            await self._submit(request, kind="load_state", execute=execute),
        )

    def _remember_checkpoint(
        self, checkpoint_id: str, checkpoint: ResolvedCheckpointState
    ) -> None:
        existing = self._checkpoints.get(checkpoint_id)
        if existing is not None and (
            existing.adapter_path != checkpoint.adapter_path
            or existing.adapter_step != checkpoint.adapter_step
        ):
            raise RuntimeError(
                f"checkpoint name {checkpoint_id!r} identifies different learners"
            )
        if existing is not None and checkpoint.optimizer_state_path is None:
            return
        self._checkpoints[checkpoint_id] = checkpoint

    async def _prepare_rl_batch(self, request: ForwardRequest | ForwardBackwardRequest):
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
