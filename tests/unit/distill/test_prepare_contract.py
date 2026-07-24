import asyncio
from dataclasses import FrozenInstanceError, dataclass
from typing import cast

import pytest

from art.distill.artifact import PreparedTrainingBatch
from art.distill.prepare import PreparationContext, _prepare_with
from art.distill.types import Frozen, StudentOnPolicy


@dataclass(frozen=True)
class _Candidate:
    value: str


def _context() -> PreparationContext:
    return PreparationContext(
        learner_revision=7,
        token_space_fingerprint="token-space-sha256",
        logical_vocab_size=151_936,
        rollout_requirement=StudentOnPolicy(),
        consistency=Frozen(revision=3),
        correlation_id="preparation-123",
    )


def _prepared_sentinel() -> PreparedTrainingBatch:
    return cast(PreparedTrainingBatch, object())


def test_preparation_context_is_immutable_and_validated() -> None:
    context = _context()

    with pytest.raises(FrozenInstanceError):
        context.learner_revision = 8  # ty: ignore[invalid-assignment]

    invalid_fields = (
        {"learner_revision": -1},
        {"token_space_fingerprint": ""},
        {"logical_vocab_size": 0},
        {"correlation_id": ""},
    )
    for update in invalid_fields:
        values = {
            "learner_revision": context.learner_revision,
            "token_space_fingerprint": context.token_space_fingerprint,
            "logical_vocab_size": context.logical_vocab_size,
            "rollout_requirement": context.rollout_requirement,
            "consistency": context.consistency,
            "correlation_id": context.correlation_id,
        }
        values.update(update)
        with pytest.raises(ValueError):
            PreparationContext(**values)  # type: ignore[arg-type]


async def test_prepare_with_delegates_arguments_and_returns_result_unchanged() -> None:
    candidate = _Candidate("candidate")
    context = _context()
    prepared = _prepared_sentinel()

    class RecordingPreparer:
        call: tuple[_Candidate, PreparationContext] | None = None

        async def prepare(
            self,
            candidate_batch: _Candidate,
            context: PreparationContext,
        ) -> PreparedTrainingBatch:
            self.call = (candidate_batch, context)
            return prepared

    preparer = RecordingPreparer()

    result = await _prepare_with(preparer, candidate, context)

    assert result is prepared
    assert preparer.call == (candidate, context)
    assert preparer.call[0] is candidate
    assert preparer.call[1] is context


async def test_prepare_with_propagates_preparer_errors() -> None:
    error = RuntimeError("teacher preparation failed")

    class FailingPreparer:
        async def prepare(
            self,
            candidate_batch: _Candidate,
            context: PreparationContext,
        ) -> PreparedTrainingBatch:
            raise error

    with pytest.raises(RuntimeError) as raised:
        await _prepare_with(FailingPreparer(), _Candidate("candidate"), _context())

    assert raised.value is error


async def test_prepare_with_propagates_cancellation_to_preparer() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    never_finishes = asyncio.Event()

    class BlockingPreparer:
        async def prepare(
            self,
            candidate_batch: _Candidate,
            context: PreparationContext,
        ) -> PreparedTrainingBatch:
            started.set()
            try:
                await never_finishes.wait()
            finally:
                cancelled.set()
            raise AssertionError("unreachable")

    task = asyncio.create_task(
        _prepare_with(BlockingPreparer(), _Candidate("candidate"), _context())
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
