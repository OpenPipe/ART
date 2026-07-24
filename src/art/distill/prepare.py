"""Asynchronous preparation boundary for distillation training batches.

This module intentionally contains no teacher-scoring implementation.  It defines the
boundary that synchronous orchestration uses today and that a future asynchronous
pipeline can schedule without coupling preparation to backend training.
"""

from dataclasses import dataclass
from typing import Protocol, TypeVar

from .artifact import PreparedTrainingBatch
from .types import (
    AnyRevision,
    CurrentStep,
    Frozen,
    StudentOnPolicy,
)

CandidateT = TypeVar("CandidateT", contravariant=True)


@dataclass(frozen=True, slots=True)
class PreparationContext:
    """Resolved, immutable constraints for one preparation operation."""

    learner_revision: int
    token_space_fingerprint: str
    logical_vocab_size: int
    rollout_requirement: StudentOnPolicy | AnyRevision
    consistency: Frozen | CurrentStep
    correlation_id: str

    def __post_init__(self) -> None:
        if self.learner_revision < 0:
            raise ValueError("learner_revision must be non-negative")
        if not self.token_space_fingerprint:
            raise ValueError("token_space_fingerprint must not be empty")
        if self.logical_vocab_size <= 0:
            raise ValueError("logical_vocab_size must be positive")
        if not self.correlation_id:
            raise ValueError("correlation_id must not be empty")


class BatchPreparer(Protocol[CandidateT]):
    """Prepare immutable training data without mutating learner weights."""

    async def prepare(
        self,
        candidate_batch: CandidateT,
        context: PreparationContext,
    ) -> PreparedTrainingBatch: ...


async def _prepare_with(
    preparer: BatchPreparer[CandidateT],
    candidate_batch: CandidateT,
    context: PreparationContext,
) -> PreparedTrainingBatch:
    """Delegate preparation without changing failure or cancellation semantics."""

    return await preparer.prepare(candidate_batch, context)
