from __future__ import annotations

from typing import Protocol

from art.trajectories import TrajectoryGroup
from art.types import TrainResult

from .client import TrainingClient, TrainingOperation
from .contracts import (
    ForwardBackwardRequest,
    ForwardBackwardResult,
    OptimStepRequest,
    OptimStepResult,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)


class PipelineCommandContext(Protocol):
    @property
    def client(self) -> TrainingClient: ...

    @property
    def groups(self) -> tuple[TrajectoryGroup, ...]: ...

    @property
    def forward_request(self) -> ForwardBackwardRequest: ...

    @property
    def preparation_metrics(self) -> dict[str, float]: ...

    def optimizer_request(self, sequence_id: int) -> OptimStepRequest: ...

    async def sampler_request(
        self, step: int, sequence_id: int
    ) -> SaveWeightsForSamplerRequest: ...

    def state_request(self, step: int, sequence_id: int) -> SaveStateRequest | None: ...

    async def commands_admitted(
        self,
        *,
        forward: TrainingOperation[ForwardBackwardResult],
        optimizer: TrainingOperation[OptimStepResult],
        sampler: TrainingOperation[SamplerWeightsResult],
        state: TrainingOperation[SaveStateResult] | None,
    ) -> None: ...

    async def complete(
        self,
        *,
        step: int,
        forward: TrainingOperation[ForwardBackwardResult],
        optimizer: TrainingOperation[OptimStepResult],
        forward_submit_s: float,
    ) -> TrainResult: ...

    async def abort(
        self,
        forward: TrainingOperation[ForwardBackwardResult] | None,
        optimizer: TrainingOperation[OptimStepResult] | None,
        sampler: TrainingOperation[SamplerWeightsResult] | None,
        *,
        optimizer_admitted: bool,
    ) -> None: ...
