from typing import Protocol

from .contracts import OperationRef, TrainingInputObjectRef
from .token_matrix import TokenMatrixBatch


class TrainingInputResolver(Protocol):
    """Resolve and authenticate an immutable input object outside ART storage."""

    async def resolve(
        self,
        input_object: TrainingInputObjectRef,
        *,
        operation: OperationRef,
    ) -> TokenMatrixBatch: ...
