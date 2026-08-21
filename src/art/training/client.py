from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sized
from itertools import batched
from typing import Generic, Literal, Protocol, TypeVar

from .contracts import (
    Contract,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    OperationRef,
    OptimStepRequest,
    OptimStepResult,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)

ResultT_co = TypeVar("ResultT_co", bound=Contract, covariant=True)
BatchItemT = TypeVar("BatchItemT")
PreparedGradientDisposition = Literal["contributes", "empty"]


class TrainingOperation(Protocol, Generic[ResultT_co]):
    @property
    def ref(self) -> OperationRef: ...

    async def result(self) -> ResultT_co: ...

    async def cancel(self) -> None: ...

    async def gradient_disposition(self) -> PreparedGradientDisposition:
        """Return the final F/B disposition; non-F/B operations raise TypeError."""
        ...


async def admit_and_settle_gradient_step(
    forward: TrainingOperation[ForwardBackwardResult],
    admit_optimizer: Callable[[], Awaitable[TrainingOperation[OptimStepResult]]],
) -> tuple[ForwardBackwardResult, OptimStepResult] | None:
    """Admit optimization at final disposition and settle both operations."""
    disposition = await forward.gradient_disposition()
    if disposition == "empty":
        result = await forward.result()
        if result.produced_gradient:
            raise RuntimeError("empty F/B preparation returned a gradient contribution")
        return None
    try:
        optimizer = await admit_optimizer()
    except BaseException as primary:
        cleanup = await asyncio.gather(forward.cancel(), return_exceptions=True)
        failures = [value for value in cleanup if isinstance(value, BaseException)]
        if failures:
            raise BaseExceptionGroup(
                "optimizer admission and F/B cleanup failed", [primary, *failures]
            ) from None
        raise
    results = await asyncio.gather(
        forward.result(), optimizer.result(), return_exceptions=True
    )
    failures = [value for value in results if isinstance(value, BaseException)]
    if failures:
        raise BaseExceptionGroup("F/B and optimizer operations failed", failures)
    forward_result, optimizer_result = results
    if not (
        isinstance(forward_result, ForwardBackwardResult)
        and isinstance(optimizer_result, OptimStepResult)
        and forward_result.produced_gradient
    ):
        raise RuntimeError("contributing F/B operations returned invalid results")
    if (
        forward.ref.operation_id
        not in optimizer_result.contributing_forward_backward_operation_ids
    ):
        raise RuntimeError("optimizer did not consume the prepared F/B operation")
    return forward_result, optimizer_result


def iter_sft_batch_schedule(
    values: Iterable[BatchItemT],
    batch_size: int,
    learning_rate: float | list[float],
) -> Iterator[tuple[tuple[BatchItemT, ...], float]]:
    """Stream exact SFT batches and their scalar learning rates."""
    if isinstance(learning_rate, list) and isinstance(values, Sized):
        value_count = len(values)
        batch_count = (value_count + batch_size - 1) // batch_size
        if value_count and len(learning_rate) != batch_count:
            raise ValueError("SFT learning-rate schedule must match batch count")
    batches = iter(batched(values, batch_size))
    first_batch = next(batches, None)
    if first_batch is None:
        return
    if not isinstance(learning_rate, list):
        yield first_batch, float(learning_rate)
        for batch in batches:
            yield batch, float(learning_rate)
        return
    rates = iter(learning_rate)
    first_rate = next(rates, None)
    if first_rate is None:
        raise ValueError("SFT learning-rate schedule must match batch count")
    yield first_batch, float(first_rate)
    try:
        for batch, rate in zip(batches, rates, strict=True):
            yield batch, float(rate)
    except ValueError:
        raise ValueError("SFT learning-rate schedule must match batch count") from None


class TrainingClient(Protocol):
    @property
    def run_id(self) -> str: ...

    @property
    def next_sequence_id(self) -> int: ...

    @property
    def projected_learner_version(self) -> int: ...

    async def forward(
        self, request: ForwardRequest
    ) -> TrainingOperation[ForwardResult]: ...

    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> TrainingOperation[ForwardBackwardResult]: ...

    async def optim_step(
        self, request: OptimStepRequest
    ) -> TrainingOperation[OptimStepResult]: ...

    async def save_weights_for_sampler(
        self, request: SaveWeightsForSamplerRequest
    ) -> TrainingOperation[SamplerWeightsResult]: ...

    async def save_state(
        self, request: SaveStateRequest
    ) -> TrainingOperation[SaveStateResult]: ...

    async def load_state(
        self, request: LoadStateRequest
    ) -> TrainingOperation[LoadStateResult]: ...

    async def load_state_with_optimizer(
        self, request: LoadStateRequest
    ) -> TrainingOperation[LoadStateResult]: ...

    async def close(self) -> None: ...
