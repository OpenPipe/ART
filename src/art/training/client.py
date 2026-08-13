from __future__ import annotations

from typing import Generic, Protocol, TypeVar

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


class TrainingOperation(Protocol, Generic[ResultT_co]):
    @property
    def ref(self) -> OperationRef: ...

    async def result(self) -> ResultT_co: ...

    async def cancel(self) -> None: ...


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
