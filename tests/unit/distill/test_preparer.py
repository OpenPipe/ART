from __future__ import annotations

import asyncio
import math

import pytest

from art.distill.artifact import PreparedTrainingBatch
from art.distill.candidate import CandidateTrainingBatch
from art.distill.prepare import PreparationContext
from art.distill.preparer import (
    AllCapturedGenerations,
    DistillationPreparationError,
    DistillationPreparer,
    FailurePolicy,
    SameContext,
)
from art.distill.scorer import (
    RankedTokenLogprob,
    ScoredPosition,
    TeacherDistributionScorer,
    TeacherScoringRequest,
    TeacherScoringResult,
)
from art.distill.types import (
    AnyRevision,
    CapturedGeneration,
    CurrentStep,
    Example,
    Frozen,
    GenerationPart,
    PartSpan,
    RolloutRevisionSpan,
    StudentOnPolicy,
    TeacherView,
    TopK,
    TrainingGroupSnapshot,
    TrainingTrajectorySnapshot,
)


def _view(label: str, *, hint: str | None = None) -> TeacherView:
    messages = [{"content": label, "role": "user"}]
    if hint is not None:
        messages.append({"content": hint, "role": "system"})
    return TeacherView.from_request("chat_completions", {"messages": messages})


def _generation(
    identity: int,
    *,
    trajectory_fingerprint: str = "trajectory-1",
    trajectory_token_start: int = 1,
    rollout_revision: int = 7,
) -> CapturedGeneration:
    return CapturedGeneration.create(
        generation_id=f"generation-{identity}",
        trajectory_fingerprint=trajectory_fingerprint,
        event_index=identity,
        trajectory_token_start=trajectory_token_start,
        protocol="chat_completions",
        continuation_token_ids=(identity + 2, identity + 3),
        context=_view(f"context-{identity}"),
        part_spans=(
            PartSpan(part=GenerationPart.REASONING, start=0, end=1),
            PartSpan(part=GenerationPart.ASSISTANT_TEXT, start=1, end=2),
        ),
        rollout_spans=(RolloutRevisionSpan(start=0, end=2, revision=rollout_revision),),
    )


def _candidate(*, examples: tuple[Example, ...] = ()) -> CandidateTrainingBatch:
    first = _generation(1, trajectory_token_start=1)
    second = _generation(2, trajectory_token_start=3)
    trajectory = TrainingTrajectorySnapshot(
        trajectory_fingerprint="trajectory-1",
        token_ids=(0, 3, 4, 4, 5),
        logprobs=(None, -0.1, -0.2, -0.3, -0.4),
        token_flags=(0, 1, 1, 1, 1),
        reward=1.0,
        advantage=0.0,
        generations=(first, second),
    )
    return CandidateTrainingBatch.create(
        groups=(
            TrainingGroupSnapshot(
                group_id="group-1",
                trajectories=(trajectory,),
            ),
        ),
        examples=examples,
    )


def _context(
    *,
    consistency: Frozen | CurrentStep | None = None,
    rollout_requirement: StudentOnPolicy | AnyRevision | None = None,
) -> PreparationContext:
    return PreparationContext(
        learner_revision=7,
        token_space_fingerprint="token-space",
        logical_vocab_size=8,
        rollout_requirement=rollout_requirement or StudentOnPolicy(),
        consistency=consistency or Frozen(revision="teacher-1"),
        correlation_id="preparation-1",
    )


def _result(request: TeacherScoringRequest) -> TeacherScoringResult:
    rows = tuple(
        ScoredPosition(
            position=position,
            forced_token_id=request.forced_token_ids[position],
            entries=(
                RankedTokenLogprob(rank=0, token_id=0, logprob=math.log(0.6)),
                RankedTokenLogprob(rank=1, token_id=1, logprob=math.log(0.3)),
            ),
            tail_logprob=math.log(0.1),
            logical_vocab_size=request.logical_vocab_size,
            temperature=request.target.temperature,
        )
        for position in request.selected_positions
    )
    return TeacherScoringResult.create(request=request, positions=rows)


class RecordingScorer:
    def __init__(self) -> None:
        self.requests: list[TeacherScoringRequest] = []

    async def score(
        self,
        request: TeacherScoringRequest,
    ) -> TeacherScoringResult:
        self.requests.append(request)
        return _result(request)


def _all_preparer(
    scorer: TeacherDistributionScorer,
    *,
    failure_policy: FailurePolicy | None = None,
    max_concurrency: int = 8,
) -> DistillationPreparer:
    return DistillationPreparer(
        scorer=scorer,
        teacher_name="teacher",
        target=TopK(k=2),
        select=AllCapturedGenerations(parts=(GenerationPart.ASSISTANT_TEXT,)),
        view=SameContext(),
        failure_policy=failure_policy,
        max_concurrency=max_concurrency,
    )


async def test_all_mode_builds_exact_local_requests_and_valid_artifact() -> None:
    scorer = RecordingScorer()
    candidate = _candidate()

    prepared = await _all_preparer(scorer).prepare(candidate, _context())

    assert [request.generation_id for request in scorer.requests] == [
        "generation-1",
        "generation-2",
    ]
    assert [request.selected_positions for request in scorer.requests] == [(1,), (1,)]
    assert [request.forced_token_ids for request in scorer.requests] == [
        (3, 4),
        (4, 5),
    ]
    assert [request.teacher_view for request in scorer.requests] == [
        candidate.generation("generation-1").context,
        candidate.generation("generation-2").context,
    ]

    loaded = PreparedTrainingBatch.from_bytes(prepared.to_bytes())
    payload = loaded.parsed_payload()
    assert loaded.batch_id == candidate.batch_id
    assert [(row.generation_id, row.position) for row in payload.targets] == [
        ("generation-1", 1),
        ("generation-2", 1),
    ]
    assert payload.report.selected_generations == 2
    assert payload.report.selected_tokens == payload.report.prepared_tokens == 2


async def test_materialized_hint_is_teacher_only_and_modes_are_exclusive() -> None:
    base = _candidate()
    generation = base.generation("generation-2")
    hinted_view = _view("context-2", hint="private feedback")
    example = Example.create(
        generation=generation,
        teacher_view=hinted_view,
        parts={GenerationPart.ASSISTANT_TEXT},
        provenance={"weave_call": "call-1"},
    )
    candidate = CandidateTrainingBatch.create(
        groups=base.groups,
        examples=(example,),
    )
    scorer = RecordingScorer()
    preparer = DistillationPreparer(
        scorer=scorer,
        teacher_name="teacher",
        target=TopK(k=2),
    )

    prepared = await preparer.prepare(candidate, _context())

    assert len(scorer.requests) == 1
    assert scorer.requests[0].generation_id == generation.generation_id
    assert scorer.requests[0].teacher_view == hinted_view
    serialized_groups = prepared.parsed_payload().groups[0].model_dump_json()
    assert "private feedback" not in serialized_groups
    target = prepared.parsed_payload().targets[0]
    assert target.provenance.to_dict() == {"weave_call": "call-1"}

    with pytest.raises(ValueError, match="either materialized examples"):
        await _all_preparer(RecordingScorer()).prepare(candidate, _context())
    with pytest.raises(ValueError, match="either materialized examples"):
        await preparer.prepare(base, _context())


async def test_concurrent_completion_keeps_candidate_order() -> None:
    release_first = asyncio.Event()
    second_finished = asyncio.Event()

    class OutOfOrderScorer:
        async def score(
            self,
            request: TeacherScoringRequest,
        ) -> TeacherScoringResult:
            if request.generation_id == "generation-1":
                await release_first.wait()
            else:
                second_finished.set()
                release_first.set()
            return _result(request)

    prepared = await _all_preparer(
        OutOfOrderScorer(),
        max_concurrency=2,
    ).prepare(_candidate(), _context())

    assert second_finished.is_set()
    assert [row.generation_id for row in prepared.parsed_payload().targets] == [
        "generation-1",
        "generation-2",
    ]


async def test_retry_produces_byte_identical_prepared_artifact() -> None:
    candidate = _candidate()

    first = await _all_preparer(RecordingScorer()).prepare(candidate, _context())
    second = await _all_preparer(RecordingScorer()).prepare(candidate, _context())

    assert first.preparation_id == second.preparation_id
    assert first.to_bytes() == second.to_bytes()


async def test_strict_and_partial_failure_coverage_are_explicit_and_sanitized() -> None:
    class FailingScorer:
        async def score(
            self,
            request: TeacherScoringRequest,
        ) -> TeacherScoringResult:
            if request.generation_id == "generation-1":
                raise RuntimeError("authorization: secret-value")
            return _result(request)

    with pytest.raises(DistillationPreparationError) as strict:
        await _all_preparer(FailingScorer()).prepare(_candidate(), _context())
    assert len(strict.value.issues) == 1
    assert "secret-value" not in str(strict.value)
    assert "secret-value" not in strict.value.issues[0].detail

    partial = await _all_preparer(
        FailingScorer(),
        failure_policy=FailurePolicy.mask_failed_generation(
            min_prepared_token_fraction=0.5
        ),
    ).prepare(_candidate(), _context())
    assert partial.report.selected_tokens == 2
    assert partial.report.prepared_tokens == 1
    assert partial.report.issue_count == 1
    assert len(partial.report.issues) == 1
    assert partial.report.issues[0].generation_id == "generation-1"
    assert partial.report.issues[0].teacher_name == "teacher"
    assert partial.report.issues[0].selected_positions == (1,)
    assert b"secret-value" not in partial.payload
    assert [target.generation_id for target in partial.parsed_payload().targets] == [
        "generation-2"
    ]

    with pytest.raises(DistillationPreparationError, match="coverage"):
        await _all_preparer(
            FailingScorer(),
            failure_policy=FailurePolicy.mask_failed_generation(
                min_prepared_token_fraction=0.75
            ),
        ).prepare(_candidate(), _context())


async def test_cancellation_propagates_and_cancels_sibling_scoring() -> None:
    started = 0
    both_started = asyncio.Event()
    cancelled = 0
    never = asyncio.Event()

    class BlockingScorer:
        async def score(
            self,
            request: TeacherScoringRequest,
        ) -> TeacherScoringResult:
            nonlocal started, cancelled
            started += 1
            if started == 2:
                both_started.set()
            try:
                await never.wait()
            finally:
                cancelled += 1
            raise AssertionError(request)

    task = asyncio.create_task(
        _all_preparer(BlockingScorer(), max_concurrency=2).prepare(
            _candidate(),
            _context(),
        )
    )
    await both_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == 2


async def test_malformed_scorer_echo_is_a_generation_failure() -> None:
    class BadEchoScorer:
        async def score(
            self,
            request: TeacherScoringRequest,
        ) -> TeacherScoringResult:
            result = _result(request)
            return result.model_copy(update={"teacher_revision": "wrong"})

    with pytest.raises(DistillationPreparationError) as raised:
        await _all_preparer(BadEchoScorer()).prepare(_candidate(), _context())

    assert len(raised.value.issues) == 2
    assert {issue.generation_id for issue in raised.value.issues} == {
        "generation-1",
        "generation-2",
    }


async def test_current_step_and_student_on_policy_are_enforced() -> None:
    prepared = await _all_preparer(RecordingScorer()).prepare(
        _candidate(),
        _context(consistency=CurrentStep(revision=7, session_id="session-1")),
    )
    assert prepared.constraints.consistency == CurrentStep(
        revision=7,
        session_id="session-1",
    )

    stale = _candidate()
    generation = _generation(1, trajectory_token_start=1, rollout_revision=6)
    old_trajectory = stale.groups[0].trajectories[0]
    changed_trajectory = old_trajectory.model_copy(
        update={"generations": (generation, old_trajectory.generations[1])}
    )
    stale = CandidateTrainingBatch.create(
        groups=(
            stale.groups[0].model_copy(update={"trajectories": (changed_trajectory,)}),
        )
    )
    with pytest.raises(ValueError, match="rollout revision"):
        await _all_preparer(RecordingScorer()).prepare(stale, _context())
