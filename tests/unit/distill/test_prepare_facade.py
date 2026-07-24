from contextlib import asynccontextmanager
import math
from typing import Any, cast

import pytest

from art import distill
from art.distill.candidate import CandidateTrainingBatch
from art.distill.scorer import (
    RankedTokenLogprob,
    RetryableTeacherScoringError,
    ScoredPosition,
    TeacherScoringRequest,
    TeacherScoringResult,
)
from art.model import Model, TrainableModel
from art.serving_capabilities import ServingCapabilities


def _capabilities(*, token_space: str = "tokens") -> ServingCapabilities:
    return ServingCapabilities(
        runtime="art_vllm",
        protocol_version=1,
        prompt_token_distributions=True,
        prompt_token_distribution_version=1,
        max_prompt_logprobs=32,
        prompt_distribution_temperature="unit_only",
        token_space_fingerprint=token_space,
        logical_vocab_size=8,
    )


def _candidate() -> CandidateTrainingBatch:
    view = distill.TeacherView.from_request(
        "chat_completions",
        {"messages": [{"content": "choose", "role": "user"}]},
    )
    generation = distill.CapturedGeneration.create(
        generation_id="generation-1",
        trajectory_fingerprint="trajectory-1",
        event_index=0,
        trajectory_token_start=1,
        protocol="chat_completions",
        continuation_token_ids=(2,),
        context=view,
        part_spans=(
            distill.PartSpan(
                part=distill.GenerationPart.ASSISTANT_TEXT,
                start=0,
                end=1,
            ),
        ),
        rollout_spans=(distill.RolloutRevisionSpan(start=0, end=1, revision=7),),
    )
    trajectory = distill.TrainingTrajectorySnapshot(
        trajectory_fingerprint="trajectory-1",
        token_ids=(1, 2),
        logprobs=(None, -0.1),
        token_flags=(0, 1),
        reward=1.0,
        advantage=0.0,
        generations=(generation,),
    )
    return CandidateTrainingBatch.create(
        groups=(
            distill.TrainingGroupSnapshot(
                group_id="group-1",
                trajectories=(trajectory,),
            ),
        ),
    )


class _Backend:
    def __init__(self) -> None:
        self.lease_active = False
        self.lease_calls: list[tuple[object, int]] = []
        self.current: distill.CurrentStep | None = None

    async def _get_step(self, _model: object) -> int:
        return 7

    async def _validate_current_step(
        self,
        _model: object,
        consistency: distill.CurrentStep,
    ) -> None:
        if consistency != self.current:
            raise ValueError("CurrentStep is not active")

    @asynccontextmanager
    async def exact_adapter_lease(self, model: object, step: int):
        self.lease_calls.append((model, step))
        self.lease_active = True
        try:
            yield
        finally:
            self.lease_active = False


class _Model:
    def __init__(
        self,
        *,
        name: str,
        backend: _Backend | None = None,
        capabilities: ServingCapabilities | None = None,
    ) -> None:
        self.name = name
        self.base_model = "base"
        self._internal_config = {}
        self._backend = backend
        self._serving_capabilities = capabilities
        self.inference_base_url = "http://teacher.test/v1"
        self.inference_api_key = "key"

    def backend(self) -> _Backend:
        assert self._backend is not None
        return self._backend

    def get_inference_name(self, step: int | str | None = None) -> str:
        if self._backend is not None and step is not None:
            assert self._backend.lease_active
        return self.name if step is None else f"{self.name}@{step}"


class _RecordingScorer:
    instances: list["_RecordingScorer"] = []
    backend: _Backend | None = None
    failures_remaining = 0

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.requests: list[TeacherScoringRequest] = []
        self.lease_active: bool | None = None
        self.__class__.instances.append(self)

    async def score(self, request: TeacherScoringRequest) -> TeacherScoringResult:
        self.requests.append(request)
        backend = getattr(self, "backend", None)
        self.lease_active = backend.lease_active if backend is not None else None
        if self.__class__.failures_remaining:
            self.__class__.failures_remaining -= 1
            raise RetryableTeacherScoringError("transient teacher failure")
        return TeacherScoringResult.create(
            request=request,
            positions=(
                ScoredPosition(
                    position=0,
                    forced_token_id=2,
                    entries=(
                        RankedTokenLogprob(
                            rank=0,
                            token_id=2,
                            logprob=math.log(0.75),
                        ),
                        RankedTokenLogprob(
                            rank=1,
                            token_id=1,
                            logprob=math.log(0.2),
                        ),
                    ),
                    tail_logprob=math.log(0.05),
                    logical_vocab_size=8,
                    temperature=1.0,
                ),
            ),
        )


@pytest.fixture
def facade(monkeypatch: pytest.MonkeyPatch) -> CandidateTrainingBatch:
    candidate = _candidate()
    monkeypatch.setattr(
        "art.distill.candidate.snapshot_groups",
        lambda *_args, **_kwargs: candidate,
    )
    _RecordingScorer.instances.clear()
    _RecordingScorer.failures_remaining = 0
    monkeypatch.setattr("art.distill.vllm.VLLMTeacherScorer", _RecordingScorer)
    return candidate


async def test_self_teacher_is_scored_inside_exact_revision_lease(
    facade: CandidateTrainingBatch,
) -> None:
    backend = _Backend()
    student = _Model(
        name="student",
        backend=backend,
        capabilities=_capabilities(),
    )
    _RecordingScorer.backend = backend

    prepared = await distill.prepare(
        cast(TrainableModel[Any, Any], student),
        (),
        teacher=cast(Model[Any, Any], student),
        select=distill.all_generations(),
        teacher_view=distill.same_context(),
        target=distill.TopK(k=2),
        consistency=distill.Frozen(revision=7),
    )

    assert prepared.batch_id == facade.batch_id
    assert backend.lease_calls == [(student, 7)]
    scorer = _RecordingScorer.instances[0]
    assert scorer.lease_active is True
    assert scorer.kwargs["model_name"] == "student@7"
    assert scorer.kwargs["render_model_name"] == "base"
    assert scorer.requests[0].teacher_revision == 7


async def test_advanced_learner_can_use_an_older_frozen_self_teacher(
    facade: CandidateTrainingBatch,
) -> None:
    backend = _Backend()
    student = _Model(
        name="student",
        backend=backend,
        capabilities=_capabilities(),
    )
    _RecordingScorer.backend = backend

    prepared = await distill.prepare(
        cast(TrainableModel[Any, Any], student),
        (),
        teacher=cast(Model[Any, Any], student),
        select=distill.all_generations(),
        teacher_view=distill.same_context(),
        target=distill.TopK(k=2),
        consistency=distill.Frozen(revision=5),
    )

    assert prepared.constraints.learner_revision == 7
    assert prepared.constraints.consistency == distill.Frozen(revision=5)
    assert backend.lease_calls == [(student, 5)]
    scorer = _RecordingScorer.instances[0]
    assert scorer.lease_active is True
    assert scorer.kwargs["model_name"] == "student@5"
    assert scorer.kwargs["render_model_name"] == "base"
    assert scorer.requests[0].teacher_revision == 5


async def test_public_retry_count_retries_the_same_scoring_request(
    facade: CandidateTrainingBatch,
) -> None:
    backend = _Backend()
    student = _Model(
        name="student",
        backend=backend,
        capabilities=_capabilities(),
    )
    _RecordingScorer.backend = backend
    _RecordingScorer.failures_remaining = 1

    prepared = await distill.prepare(
        cast(TrainableModel[Any, Any], student),
        (),
        teacher=cast(Model[Any, Any], student),
        select=distill.all_generations(),
        teacher_view=distill.same_context(),
        target=distill.TopK(k=2),
        consistency=distill.Frozen(revision=5),
        max_scoring_retries=1,
    )

    assert prepared.report.prepared_tokens == 1
    requests = _RecordingScorer.instances[0].requests
    assert len(requests) == 2
    assert requests[0] is requests[1]
    assert requests[0].request_id == requests[1].request_id


async def test_external_teacher_uses_asserted_revision_without_student_lease(
    facade: CandidateTrainingBatch,
) -> None:
    backend = _Backend()
    student = _Model(
        name="student",
        backend=backend,
        capabilities=_capabilities(),
    )
    teacher = _Model(name="teacher", capabilities=_capabilities())
    _RecordingScorer.backend = None

    prepared = await distill.prepare(
        cast(TrainableModel[Any, Any], student),
        (),
        teacher=cast(Model[Any, Any], teacher),
        select=distill.all_generations(),
        teacher_view=distill.same_context(),
        target=distill.TopK(k=2),
        consistency=distill.Frozen(revision="checkpoint-42"),
    )

    assert prepared.batch_id == facade.batch_id
    assert backend.lease_calls == []
    scorer = _RecordingScorer.instances[0]
    assert scorer.kwargs["model_name"] == "teacher"
    assert scorer.kwargs["render_model_name"] is None
    assert scorer.requests[0].teacher_revision == "checkpoint-42"


async def test_capability_mismatch_rejects_before_scoring(
    facade: CandidateTrainingBatch,
) -> None:
    backend = _Backend()
    student = _Model(
        name="student",
        backend=backend,
        capabilities=_capabilities(),
    )
    teacher = _Model(
        name="teacher",
        capabilities=_capabilities(token_space="different"),
    )

    with pytest.raises(RuntimeError, match="token spaces"):
        await distill.prepare(
            cast(TrainableModel[Any, Any], student),
            (),
            teacher=cast(Model[Any, Any], teacher),
            select=distill.all_generations(),
            teacher_view=distill.same_context(),
            target=distill.TopK(k=2),
            consistency=distill.Frozen(revision="checkpoint-42"),
        )

    assert not _RecordingScorer.instances
    assert backend.lease_calls == []


async def test_current_step_requires_active_session_and_non_unit_temperature_fails(
    facade: CandidateTrainingBatch,
) -> None:
    backend = _Backend()
    student = _Model(
        name="student",
        backend=backend,
        capabilities=_capabilities(),
    )

    with pytest.raises(ValueError, match="not active"):
        await distill.prepare(
            cast(TrainableModel[Any, Any], student),
            (),
            teacher=cast(Model[Any, Any], student),
            select=distill.all_generations(),
            teacher_view=distill.same_context(),
            target=distill.TopK(k=2),
            consistency=distill.CurrentStep(revision=7, session_id="session"),
        )
    backend.current = distill.CurrentStep(revision=7, session_id="session")
    prepared = await distill.prepare(
        cast(TrainableModel[Any, Any], student),
        (),
        teacher=cast(Model[Any, Any], student),
        select=distill.all_generations(),
        teacher_view=distill.same_context(),
        target=distill.TopK(k=2),
        consistency=backend.current,
    )
    assert prepared.constraints.consistency == backend.current
    assert prepared.constraints.learner_revision == 7
    assert _RecordingScorer.instances[0].requests[0].teacher_revision == 7
    _RecordingScorer.instances.clear()
    with pytest.raises(ValueError, match="temperature 1.0"):
        await distill.prepare(
            cast(TrainableModel[Any, Any], student),
            (),
            teacher=cast(Model[Any, Any], student),
            target=distill.TopK(k=2, temperature=2.0),
            consistency=distill.Frozen(revision=7),
        )
    with pytest.raises(ValueError, match="max_scoring_retries"):
        await distill.prepare(
            cast(TrainableModel[Any, Any], student),
            (),
            teacher=cast(Model[Any, Any], student),
            target=distill.TopK(k=2),
            consistency=distill.Frozen(revision=7),
            max_scoring_retries=-1,
        )

    assert not _RecordingScorer.instances
    assert backend.lease_calls == [(student, 7)]
