from typing import Protocol

from .contracts import OperationRef, RawTrainingBatch, TrainingInputObjectRef


class TrainingInputResolver(Protocol):
    """Resolve and authenticate an immutable input object outside ART storage."""

    async def resolve(
        self,
        input_object: TrainingInputObjectRef,
        *,
        operation: OperationRef,
    ) -> RawTrainingBatch: ...
