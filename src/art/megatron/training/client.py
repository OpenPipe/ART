from __future__ import annotations

import asyncio
from collections.abc import Callable
import heapq
import time
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, cast
import uuid

from art.distributed.packing import PackingRequest
from art.distributed.rollout import RolloutModelSpec
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
from art.training.sequencing import (
    CommandAdmission,
    CommandAdmissionPolicy,
    RunCommandLedger,
)
from art.types import TrainConfig
from art.utils.lifecycle import process_shutdown_timeout

from ..runtime.specs import (
    ExperimentalTrainConfig,
    ResolvedCheckpointState,
    RlForwardBackwardConfig,
)
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
_MAX_RETAINED_COMPLETED_OPERATIONS = 1024
_MAX_RETAINED_COMPLETED_RESULT_BYTES = 512 << 20
_TASK_DRAIN_TIMEOUT_S = process_shutdown_timeout(2)
_DEFERRED_RESULT_KINDS = frozenset({"save_sampler", "save_state"})
_SEQUENCE_POISONING_KINDS = frozenset({"forward_backward", "optim_step", "load_state"})


def _consume_future(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _remaining_time(deadline: float, *, context: str) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError(f"{context} exceeded its shutdown deadline")
    return remaining


async def _cancel_and_drain(
    tasks: tuple[asyncio.Task[Any], ...], *, deadline: float, context: str
) -> None:
    pending = tuple(dict.fromkeys(task for task in tasks if not task.done()))
    for task in pending:
        task.cancel()
    await _drain_tasks(pending, deadline=deadline, context=context)


async def _drain_tasks(
    tasks: tuple[asyncio.Task[Any], ...], *, deadline: float, context: str
) -> None:
    pending = tuple(dict.fromkeys(task for task in tasks if not task.done()))
    if pending:
        done, remaining = await asyncio.wait(
            pending, timeout=_remaining_time(deadline, context=context)
        )
        for task in done:
            _consume_future(task)
        if remaining:
            raise TimeoutError(f"{len(remaining)} {context} tasks did not stop")


class _DeferredResult(NamedTuple, Generic[ResultT]):
    completion: asyncio.Task[ResultT]
    owned: tuple[asyncio.Task[Any], ...] = ()


class _SequenceReleaseGate:
    def __init__(self) -> None:
        self._future: asyncio.Future[BaseException | _SequenceReleaseGate | None] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def done(self) -> bool:
        return self._future.done()

    async def wait(self) -> BaseException | None:
        gate = self
        while True:
            outcome = await asyncio.shield(gate._future)
            if not isinstance(outcome, _SequenceReleaseGate):
                return outcome
            gate = outcome

    def release(self) -> None:
        if not self._future.done():
            self._future.set_result(None)

    def poison(self, error: BaseException) -> None:
        if not self._future.done():
            self._future.set_result(error)

    def relay(self, predecessor: _SequenceReleaseGate | None) -> None:
        if self._future.done():
            return
        while predecessor is not None and predecessor._future.done():
            outcome = predecessor._future.result()
            if isinstance(outcome, _SequenceReleaseGate):
                predecessor = outcome
                continue
            self._future.set_result(outcome)
            return
        self._future.set_result(predecessor)


class _OperationState:
    def __init__(self) -> None:
        self.execution_started = False
        self.cancel_prepared = False
        self.cancel_requested = False
        self.force_cancelled = False
        self.terminal_recorded = False
        self.owned_tasks: set[asyncio.Task[Any]] = set()

    def pending_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(task for task in self.owned_tasks if not task.done())


class LocalTrainingOperation(Generic[ResultT]):
    def __init__(
        self,
        request_id: str,
        admission: CommandAdmission,
        ordered: asyncio.Task[None],
        result: asyncio.Future[ResultT],
        state: _OperationState,
        prepare_cancel: Callable[[], None],
    ) -> None:
        self._request_id = request_id
        self._admission = admission
        self._ordered = ordered
        self._result = result
        self._state = state
        self._prepare_cancel = prepare_cancel

    @property
    def ref(self):
        return self._admission.ref

    async def result(self) -> ResultT:
        return await asyncio.shield(self._result)

    async def cancel(self) -> None:
        if self._result.done():
            return
        if self._state.execution_started or self._ordered.done():
            await self.result()
            return
        if not self._state.cancel_prepared:
            self._prepare_cancel()
            self._state.cancel_prepared = True
            self._state.cancel_requested = True
            self._result.cancel()
        await asyncio.sleep(0)

    def _force_cancel(self) -> None:
        self._state.force_cancelled = True
        self._ordered.cancel()
        for task in self._state.pending_tasks():
            task.cancel()
        if not self._result.done():
            self._result.cancel()

    def _admission_done(self) -> bool:
        return (
            self._ordered.done()
            and self._result.done()
            and not self._state.pending_tasks()
        )


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
        admission_policy: CommandAdmissionPolicy | None = None,
    ) -> None:
        self._ledger = RunCommandLedger(
            run_id,
            learner_version=learner_version,
            admission_policy=admission_policy,
        )
        self._backend = backend
        self._model = model
        self._service = service
        self._operations: dict[str, LocalTrainingOperation[Any]] = {}
        self._evicted_forward_backward_operations: dict[
            str, tuple[str, CommandAdmission]
        ] = {}
        self._completed_operations: dict[str, int] = {}
        self._completed_operation_order: list[tuple[int, str]] = []
        self._completed_result_bytes = 0
        self._completion_tasks: set[asyncio.Task[Any]] = set()
        self._checkpoints: dict[str, ResolvedCheckpointState] = {}
        self._sft_tokenizer = SftBatchTokenizer()
        self._batch_releases: set[asyncio.Task[None]] = set()
        self._batch_release_failures: list[BaseException] = []
        self._lifecycle_failures: list[BaseException] = []
        self._retirement_failures: dict[str, BaseException] = {}
        self._sequence_tail: _SequenceReleaseGate | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

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
        self._raise_batch_release_failures()
        self._retry_failed_retirements()
        self._raise_lifecycle_failures()
        admission = await self._ledger.admit(request, kind=kind)
        if existing := self._operations.get(admission.ref.operation_id):
            return existing
        if admission.ref.operation_id in self._evicted_forward_backward_operations:
            raise RuntimeError("F/B terminal result is no longer retained")
        predecessor = self._sequence_tail
        sequence_release = _SequenceReleaseGate()
        state = _OperationState()

        result = asyncio.get_running_loop().create_future()
        result.add_done_callback(_consume_future)
        operation: LocalTrainingOperation[Any]

        def owned_done(task: asyncio.Task[Any]) -> None:
            self._completion_tasks.discard(task)
            _consume_future(task)
            self._bound_operation_cache(operation)

        def own_task(task: asyncio.Task[Any]) -> asyncio.Task[Any]:
            if task not in state.owned_tasks:
                state.owned_tasks.add(task)
                self._completion_tasks.add(task)
                task.add_done_callback(owned_done)
            return task

        def deferred_done(task: asyncio.Task[Any]) -> None:
            if not result.done():
                if task.cancelled():
                    result.cancel()
                elif error := task.exception():
                    result.set_exception(error)
                else:
                    result.set_result(task.result())
            self._bound_operation_cache(operation)

        async def run() -> None:
            try:
                if predecessor is not None:
                    predecessor_failure = await predecessor.wait()
                    if predecessor_failure is not None:
                        if not result.done():
                            result.set_exception(predecessor_failure)
                        sequence_release.poison(predecessor_failure)
                        return
                if state.cancel_requested:
                    await self._service.consume_cancelled_command(admission.ref)
                    sequence_release.release()
                    return
                state.execution_started = True
                value = await execute(admission, own_task)
                if isinstance(value, _DeferredResult):
                    if kind not in _DEFERRED_RESULT_KINDS:
                        invalid_tasks = tuple(
                            dict.fromkeys((value.completion, *value.owned))
                        )
                        await _cancel_and_drain(
                            invalid_tasks,
                            deadline=(
                                asyncio.get_running_loop().time()
                                + _TASK_DRAIN_TIMEOUT_S
                            ),
                            context=f"invalid {kind} completion",
                        )
                        raise RuntimeError(f"{kind} cannot return a deferred result")
                    for task in value.owned:
                        own_task(task)
                    completion = own_task(value.completion)
                    completion.add_done_callback(deferred_done)
                elif not result.done():
                    result.set_result(value)
            except BaseException as error:
                if not result.done():
                    if isinstance(error, asyncio.CancelledError):
                        result.cancel()
                    else:
                        result.set_exception(error)
                if state.cancel_requested and not state.force_cancelled:
                    self._ledger.fail(error)
                    self._lifecycle_failures.append(error)
                    sequence_release.poison(error)
                elif not state.execution_started:
                    sequence_release.relay(predecessor)
                elif kind in _SEQUENCE_POISONING_KINDS:
                    sequence_release.poison(error)
                else:
                    sequence_release.release()
                raise
            sequence_release.release()

        ordered = asyncio.create_task(
            run(), name=f"megatron-{kind}-{request.sequence_id}"
        )

        def ordered_done(task: asyncio.Task[None]) -> None:
            _consume_future(task)
            if task.cancelled() and not sequence_release.done:
                if not state.execution_started:
                    sequence_release.relay(predecessor)
                elif kind in _SEQUENCE_POISONING_KINDS:
                    sequence_release.poison(asyncio.CancelledError())
                else:
                    sequence_release.release()
            self._bound_operation_cache(operation)

        def prepare_cancel() -> None:
            if kind == "forward_backward":
                self._ledger.cancel_pending_forward_backward(
                    request.request_id, admission
                )
            elif kind in {"optim_step", "load_state"}:
                raise RuntimeError("learner-transition command cannot be cancelled")
            self._ledger.mark_terminal(
                request.request_id,
                admission,
                error=asyncio.CancelledError(),
                execution_started=False,
            )
            state.terminal_recorded = True

        ordered.add_done_callback(ordered_done)
        operation = LocalTrainingOperation(
            request.request_id,
            admission,
            ordered,
            result,
            state,
            prepare_cancel,
        )
        self._operations[admission.ref.operation_id] = operation
        result.add_done_callback(lambda _: self._bound_operation_cache(operation))
        self._sequence_tail = sequence_release
        self._bound_operation_cache(operation)
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
            _own_task: Callable[[asyncio.Task[Any]], asyncio.Task[Any]],
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
            if request.batch.kind == "tokenized":
                packed = await self._service.runtime.pack(
                    PackingRequest(
                        model=RolloutModelSpec.from_model(self._model),
                        generation_id=uuid.uuid4().hex,
                        tokenized_batch=request.batch,
                        tokenized_loss=request.loss.name,
                        packed_sequence_length=self._service.packed_sequence_length,
                    )
                )
                if packed is None:
                    raise ValueError("tokenized batch produced no packed sequence")
                packing = packing_outcome(
                    packed,
                    target_packed_sequences=(
                        await self._service.resolve_global_grad_accumulation_sequences(
                            TrainConfig()
                        )
                    ),
                )
                metrics = packing_metrics(packed)
                try:
                    raw = await (
                        self._service.forward_backward_command(
                            admission.ref,
                            packed,
                            RlForwardBackwardConfig(),
                            ExperimentalTrainConfig(),
                            loss=request.loss,
                        )
                        if backward
                        else self._service.forward_command(
                            admission.ref,
                            packed,
                            RlForwardBackwardConfig(),
                            ExperimentalTrainConfig(),
                            loss=request.loss,
                        )
                    )
                finally:
                    self._release_batch_soon(packed)
                result_type = ForwardBackwardResult if backward else ForwardResult
                return result_type(
                    operation_id=admission.ref.operation_id,
                    packing=packing,
                    loss_fn_outputs=tuple(
                        LossFnOutput(token_logprobs=values)
                        for values in raw["token_logprobs"]
                    ),
                    metrics={**metrics, **raw["metrics"]},
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
        async def execute(
            admission: CommandAdmission,
            _own_task: Callable[[asyncio.Task[Any]], asyncio.Task[Any]],
        ) -> OptimStepResult:
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
                if operation := self._operations.get(operation_id):
                    try:
                        self._retire_operation(operation)
                    except BaseException as error:
                        self._retirement_failures[operation_id] = error
                elif operation_id in self._evicted_forward_backward_operations:
                    try:
                        self._retire_evicted_forward_backward(operation_id)
                    except BaseException as error:
                        self._retirement_failures[operation_id] = error
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
        )

        async def execute(
            admission: CommandAdmission,
            own_task: Callable[[asyncio.Task[Any]], asyncio.Task[Any]],
        ) -> SamplerWeightsResult | _DeferredResult[SamplerWeightsResult]:
            launch = await self._service.snapshot_command(
                admission.ref,
                save_optimizer=False,
                activate_serving=request.publication.mode != "none",
            )
            raw_completion = own_task(launch.completion)

            async def complete() -> SamplerWeightsResult:
                durable = await asyncio.shield(raw_completion)
                adapter = durable.transport_adapter or durable.adapter
                self._remember_checkpoint(
                    request.checkpoint_name,
                    ResolvedCheckpointState(
                        adapter_path=adapter.identity,
                        adapter_step=adapter.step,
                        adapter_training_session_id=adapter.training_session_id,
                        adapter_generation_id=adapter.generation_id,
                    ),
                )
                return SamplerWeightsResult(
                    operation_id=admission.ref.operation_id,
                    checkpoint=checkpoint_ref(
                        admission.ref.run_id,
                        adapter.step,
                        request.checkpoint_name,
                    ),
                    lora=adapter.identity,
                    training_session_id=adapter.training_session_id,
                    generation_id=adapter.generation_id,
                    lora_bytes=sum(file.size_bytes for file in adapter.files),
                    publication_metrics=launch.metrics,
                )

            completion = own_task(
                asyncio.create_task(
                    complete(),
                    name=f"megatron-save-sampler-{admission.ref.sequence_id}",
                )
            )
            return _DeferredResult(
                completion=completion,
                owned=(raw_completion,),
            )

        return cast(
            TrainingOperation[SamplerWeightsResult],
            await self._submit(request, kind="save_sampler", execute=execute),
        )

    async def save_state(
        self, request: SaveStateRequest
    ) -> TrainingOperation[SaveStateResult]:
        async def execute(
            admission: CommandAdmission,
            own_task: Callable[[asyncio.Task[Any]], asyncio.Task[Any]],
        ) -> SaveStateResult | _DeferredResult[SaveStateResult]:
            launch = await self._service.snapshot_command(
                admission.ref,
                save_optimizer=True,
                activate_serving=False,
            )
            raw_completion = own_task(launch.completion)

            async def complete() -> SaveStateResult:
                durable = await asyncio.shield(raw_completion)
                if durable.optimizer_bytes is None:
                    raise RuntimeError("save_state completed without optimizer bytes")
                self._remember_checkpoint(
                    request.checkpoint_name,
                    ResolvedCheckpointState(
                        adapter_path=durable.adapter.identity,
                        adapter_step=durable.adapter.step,
                        adapter_training_session_id=(
                            durable.adapter.training_session_id
                        ),
                        adapter_generation_id=durable.adapter.generation_id,
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
                    lora=(durable.transport_adapter or durable.adapter).identity,
                    training_session_id=durable.adapter.training_session_id,
                    generation_id=durable.adapter.generation_id,
                    lora_bytes=sum(
                        file.size_bytes
                        for file in (durable.transport_adapter or durable.adapter).files
                    ),
                    optimizer_state=self._service.optimizer_state_path,
                    optimizer_bytes=durable.optimizer_bytes,
                    metrics=launch.metrics,
                )

            completion = own_task(
                asyncio.create_task(
                    complete(),
                    name=f"megatron-save-state-{admission.ref.sequence_id}",
                )
            )
            return _DeferredResult(
                completion=completion,
                owned=(raw_completion,),
            )

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
        async def execute(
            admission: CommandAdmission,
            _own_task: Callable[[asyncio.Task[Any]], asyncio.Task[Any]],
        ) -> LoadStateResult:
            try:
                source = self._checkpoints[request.checkpoint]
            except KeyError as error:
                raise ValueError(
                    f"unknown local checkpoint: {request.checkpoint!r}"
                ) from error
            raw, generation, _metrics, durable = await self._service.load_state_command(
                admission.ref,
                source,
                restore_optimizer=restore_optimizer,
            )
            checkpoint = checkpoint_ref(
                admission.ref.run_id,
                generation.policy_step,
                generation.generation_id,
            )
            adapter = durable.transport_adapter or durable.adapter
            self._remember_checkpoint(
                checkpoint.checkpoint_id,
                ResolvedCheckpointState(
                    adapter_path=generation.adapter_path,
                    adapter_step=generation.policy_step,
                    adapter_training_session_id=adapter.training_session_id,
                    adapter_generation_id=adapter.generation_id,
                    optimizer_state_path=self._service.optimizer_state_path,
                    optimizer_generation_id=generation.generation_id,
                ),
            )
            return LoadStateResult(
                operation_id=admission.ref.operation_id,
                checkpoint=checkpoint,
                lora=adapter.identity,
                training_session_id=adapter.training_session_id,
                generation_id=adapter.generation_id,
                lora_bytes=sum(file.size_bytes for file in adapter.files),
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
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(
                self._close(), name=f"close-megatron-training-client-{self.run_id}"
            )
            self._close_task.add_done_callback(_consume_future)
        task = self._close_task
        try:
            await asyncio.shield(task)
        except BaseException:
            if task.done() and self._close_task is task:
                self._close_task = None
            raise

    async def _close(self) -> None:
        deadline = asyncio.get_running_loop().time() + _TASK_DRAIN_TIMEOUT_S
        failures: list[BaseException] = []
        for operation in tuple(self._operations.values()):
            operation._force_cancel()
        while True:
            owned = tuple(
                dict.fromkeys(
                    task
                    for operation in self._operations.values()
                    for task in (operation._ordered, *operation._state.pending_tasks())
                    if not task.done()
                )
            ) + tuple(task for task in self._completion_tasks if not task.done())
            if not owned:
                await asyncio.sleep(0)
                owned = tuple(
                    task for task in self._completion_tasks if not task.done()
                )
                if not owned:
                    break
            try:
                await _cancel_and_drain(
                    owned,
                    deadline=deadline,
                    context="local training client",
                )
            except BaseException as error:
                failures.append(error)
                break

        if not failures and self._batch_releases:
            try:
                await _drain_tasks(
                    tuple(self._batch_releases),
                    deadline=deadline,
                    context="packed-batch release",
                )
            except BaseException as error:
                failures.append(error)
        if not failures:
            await asyncio.sleep(0)
        try:
            self._raise_batch_release_failures()
        except BaseException as error:
            failures.append(error)
        if self._lifecycle_failures:
            failures.extend(self._lifecycle_failures)
            self._lifecycle_failures.clear()
        if not failures:
            try:
                _remaining_time(deadline, context="local training client close")
            except BaseException as error:
                failures.append(error)

        if not any(isinstance(error, TimeoutError) for error in failures):
            for operation in tuple(self._operations.values()):
                self._record_terminal(operation)
            self._ledger.close()
            for operation in tuple(self._operations.values()):
                try:
                    _remaining_time(deadline, context="local training client close")
                    self._retire_operation(operation)
                    _remaining_time(deadline, context="local training client close")
                except BaseException as error:
                    self._retirement_failures[operation.ref.operation_id] = error
                    failures.append(error)
            for operation_id in tuple(self._evicted_forward_backward_operations):
                try:
                    _remaining_time(deadline, context="local training client close")
                    self._retire_evicted_forward_backward(operation_id)
                    _remaining_time(deadline, context="local training client close")
                except BaseException as error:
                    self._retirement_failures[operation_id] = error
                    failures.append(error)
        self._completion_tasks = {
            task for task in self._completion_tasks if not task.done()
        }
        if not self._operations and not self._evicted_forward_backward_operations:
            self._retirement_failures.clear()
            self._completed_operation_order.clear()
            self._sequence_tail = None
        if failures:
            raise BaseExceptionGroup("local training client close failed", failures)

    def _release_batch_soon(self, packed: Any) -> None:
        task = asyncio.create_task(
            self._service.runtime.release_batch(packed),
            name=f"release-tokenized-batch-{packed.packing_generation_id}",
        )
        self._batch_releases.add(task)

        def completed(done: asyncio.Task[None]) -> None:
            self._batch_releases.discard(done)
            if done.cancelled():
                self._batch_release_failures.append(
                    RuntimeError("tokenized packed-batch release was cancelled")
                )
            elif error := done.exception():
                self._batch_release_failures.append(error)

        task.add_done_callback(completed)

    def _raise_batch_release_failures(self) -> None:
        if self._batch_release_failures:
            failures, self._batch_release_failures = (
                self._batch_release_failures,
                [],
            )
            raise BaseExceptionGroup("tokenized packed-batch release failed", failures)

    def _retry_failed_retirements(self) -> None:
        for operation_id in tuple(self._retirement_failures):
            operation = self._operations.get(operation_id)
            if operation is not None:
                try:
                    self._retire_operation(operation)
                except BaseException as error:
                    self._retirement_failures[operation_id] = error
            elif operation_id in self._evicted_forward_backward_operations:
                try:
                    self._retire_evicted_forward_backward(operation_id)
                except BaseException as error:
                    self._retirement_failures[operation_id] = error
            else:
                self._retirement_failures.pop(operation_id, None)

    def _raise_lifecycle_failures(self) -> None:
        failures = [
            *self._lifecycle_failures,
            *self._retirement_failures.values(),
        ]
        if failures:
            raise BaseExceptionGroup("local training client lifecycle failed", failures)

    def _bound_operation_cache(self, operation: LocalTrainingOperation[Any]) -> None:
        try:
            self._record_terminal(operation)
            self._cache_terminal_operations()
        except BaseException as error:
            self._lifecycle_failures.append(error)

    def _record_terminal(self, operation: LocalTrainingOperation[Any]) -> None:
        if operation._admission_done() and not operation._state.terminal_recorded:
            error = (
                asyncio.CancelledError()
                if operation._result.cancelled()
                else operation._result.exception()
            )
            self._ledger.mark_terminal(
                operation._request_id,
                operation._admission,
                error=error,
                execution_started=operation._state.execution_started,
            )
            operation._state.terminal_recorded = True

    def _cache_terminal_operations(self) -> None:
        for operation in tuple(self._operations.values()):
            operation_id = operation.ref.operation_id
            if (
                operation_id in self._completed_operations
                or not operation._admission_done()
            ):
                continue
            result_bytes = _completed_result_bytes(operation)
            self._completed_operations[operation_id] = result_bytes
            self._completed_result_bytes += result_bytes
            heapq.heappush(
                self._completed_operation_order,
                (operation.ref.sequence_id, operation_id),
            )
        while self._completed_operations and (
            len(self._completed_operations) > _MAX_RETAINED_COMPLETED_OPERATIONS
            or self._completed_result_bytes > _MAX_RETAINED_COMPLETED_RESULT_BYTES
        ):
            while (
                self._completed_operation_order
                and self._completed_operation_order[0][1]
                not in self._completed_operations
            ):
                heapq.heappop(self._completed_operation_order)
            if not self._completed_operation_order:
                raise RuntimeError("terminal operation cache lost its retirement order")
            _, expired = self._completed_operation_order[0]
            try:
                self._retire_operation(self._operations[expired])
            except BaseException as error:
                self._retirement_failures[expired] = error
                return

    def _retire_operation(self, operation: LocalTrainingOperation[Any]) -> None:
        operation_id = operation.ref.operation_id
        if self._operations.get(operation_id) is not operation:
            return
        if not operation._admission_done():
            raise RuntimeError("cannot retire an incomplete local training operation")
        evict_open_forward_backward = (
            operation.ref.kind == "forward_backward"
            and self._ledger.is_open_forward_backward(operation_id)
        )
        self._service.retire_command_operation(operation_id)
        if evict_open_forward_backward:
            self._evicted_forward_backward_operations[operation_id] = (
                operation._request_id,
                operation._admission,
            )
        else:
            self._ledger.retire(operation._request_id, operation._admission)
        self._operations.pop(operation_id)
        self._completed_result_bytes -= self._completed_operations.pop(operation_id, 0)
        self._completed_operation_order = [
            entry
            for entry in self._completed_operation_order
            if entry[1] != operation_id
        ]
        heapq.heapify(self._completed_operation_order)
        self._retirement_failures.pop(operation_id, None)

    def _retire_evicted_forward_backward(self, operation_id: str) -> None:
        tombstone = self._evicted_forward_backward_operations.get(operation_id)
        if tombstone is None:
            return
        request_id, admission = tombstone
        self._ledger.retire(request_id, admission)
        self._evicted_forward_backward_operations.pop(operation_id)
        self._retirement_failures.pop(operation_id, None)


def _completed_result_bytes(operation: LocalTrainingOperation[Any]) -> int:
    if operation._result.cancelled():
        return 0
    try:
        result = operation._result.result()
    except BaseException:
        return 0
    if not isinstance(result, ForwardResult):
        return 0
    return sum(len(output.token_logprobs.data) for output in result.loss_fn_outputs)


def _validate_publication_mode(
    request: SaveWeightsForSamplerRequest,
    *,
    update_mode: str,
) -> None:
    requested = request.publication.mode
    if requested == "none":
        return
    expected = "in_flight_lora" if update_mode == "in_flight_lora" else "versioned_lora"
    if requested != expected:
        raise ValueError(
            f"sampler publication mode {requested!r} conflicts with "
            f"update mode {update_mode!r}"
        )
