"""Concrete asynchronous preparation of immutable distillation batches."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from .artifact import PreparedTrainingBatch
from .candidate import CandidateTrainingBatch
from .prepare import PreparationContext
from .scorer import (
    TeacherDistributionScorer,
    TeacherScoringRequest,
    TeacherScoringResult,
)
from .types import (
    CanonicalJsonObject,
    CurrentStep,
    Frozen,
    GenerationPart,
    ImmutableModel,
    PartSpan,
    PreparationIssue,
    PreparationReport,
    PreparedConstraints,
    StudentOnPolicy,
    TeacherView,
    TopK,
    TopKTargetRow,
)


class AllCapturedGenerations(ImmutableModel):
    """Select specified model-generated parts from every captured generation."""

    kind: Literal["all_captured_generations"] = "all_captured_generations"
    parts: tuple[GenerationPart, ...] = tuple(GenerationPart)

    @field_validator("parts", mode="before")
    @classmethod
    def _canonical_parts(cls, value: Any) -> tuple[GenerationPart, ...]:
        parts = tuple(sorted(set(value), key=str))
        if not parts:
            raise ValueError("all-captured selection must include a generation part")
        return parts


class SameContext(ImmutableModel):
    """Use the exact context captured immediately before each generation."""

    kind: Literal["same_context"] = "same_context"


class FailurePolicy(ImmutableModel):
    """Control whether a failed generation rejects or reduces a preparation."""

    mode: Literal["strict", "mask_failed_generation"] = "strict"
    min_prepared_token_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @model_validator(mode="after")
    def _strict_requires_complete_coverage(self) -> Self:
        if self.mode == "strict" and self.min_prepared_token_fraction != 1.0:
            raise ValueError("strict failure policy always requires complete coverage")
        return self

    @classmethod
    def strict(cls) -> Self:
        return cls()

    @classmethod
    def mask_failed_generation(
        cls,
        *,
        min_prepared_token_fraction: float,
    ) -> Self:
        return cls(
            mode="mask_failed_generation",
            min_prepared_token_fraction=min_prepared_token_fraction,
        )


class DistillationPreparationError(RuntimeError):
    """Preparation failed without producing a partially trusted artifact."""

    def __init__(
        self,
        message: str,
        *,
        issues: tuple[PreparationIssue, ...] = (),
    ) -> None:
        super().__init__(message)
        self.issues = issues


@dataclass(frozen=True, slots=True)
class _Plan:
    generation_id: str
    teacher_view: TeacherView
    positions: tuple[int, ...]
    forced_token_ids: tuple[int, ...]
    provenance: CanonicalJsonObject


@dataclass(frozen=True, slots=True)
class _Outcome:
    request: TeacherScoringRequest
    result: TeacherScoringResult | None = None
    issue: PreparationIssue | None = None


class DistillationPreparer:
    """Score selected generations and build a validated prepared artifact."""

    def __init__(
        self,
        *,
        scorer: TeacherDistributionScorer,
        teacher_name: str,
        target: TopK,
        select: AllCapturedGenerations | None = None,
        view: SameContext | None = None,
        failure_policy: FailurePolicy | None = None,
        max_concurrency: int = 8,
    ) -> None:
        if not teacher_name:
            raise ValueError("teacher_name must not be empty")
        if (select is None) != (view is None):
            raise ValueError(
                "selection and teacher-view recipe must be configured together"
            )
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._scorer = scorer
        self._teacher_name = teacher_name
        self._target = target
        self._select = select
        self._view = view
        self._failure_policy = failure_policy or FailurePolicy.strict()
        self._max_concurrency = max_concurrency

    async def prepare(
        self,
        candidate_batch: CandidateTrainingBatch,
        context: PreparationContext,
    ) -> PreparedTrainingBatch:
        """Prepare all selected tokens without mutating learner or candidate data."""

        plans = self._plans(candidate_batch)
        teacher_revision = _teacher_revision(context, self._teacher_name)
        requests = tuple(
            TeacherScoringRequest.create(
                generation_id=plan.generation_id,
                teacher_view=plan.teacher_view,
                forced_token_ids=plan.forced_token_ids,
                selected_positions=plan.positions,
                teacher_name=self._teacher_name,
                teacher_revision=teacher_revision,
                token_space_fingerprint=context.token_space_fingerprint,
                logical_vocab_size=context.logical_vocab_size,
                target=self._target,
            )
            for plan in plans
        )
        _validate_rollout_revisions(candidate_batch, requests, context)

        outcomes = await self._score_all(requests)
        issues = tuple(
            outcome.issue for outcome in outcomes if outcome.issue is not None
        )
        if issues and self._failure_policy.mode == "strict":
            raise DistillationPreparationError(
                "Teacher scoring failed under the strict failure policy.",
                issues=issues,
            )

        selected_tokens = sum(len(request.selected_positions) for request in requests)
        successful = tuple(
            outcome
            for outcome in outcomes
            if outcome.result is not None and outcome.issue is None
        )
        prepared_tokens = sum(
            len(outcome.request.selected_positions) for outcome in successful
        )
        coverage = prepared_tokens / selected_tokens
        if prepared_tokens == 0 or (
            coverage < self._failure_policy.min_prepared_token_fraction
        ):
            raise DistillationPreparationError(
                "Prepared-token coverage is below the configured threshold.",
                issues=issues,
            )

        provenance_by_generation = {
            plan.generation_id: plan.provenance for plan in plans
        }
        targets = tuple(
            row
            for outcome in successful
            for row in _target_rows(
                outcome.request,
                _result(outcome),
                provenance=provenance_by_generation[outcome.request.generation_id],
            )
        )
        return PreparedTrainingBatch.create(
            groups=candidate_batch.groups,
            targets=targets,
            report=PreparationReport(
                selected_generations=len(requests),
                selected_tokens=selected_tokens,
                prepared_tokens=prepared_tokens,
                issue_count=len(issues),
                issues=issues,
            ),
            constraints=PreparedConstraints(
                learner_revision=context.learner_revision,
                token_space_fingerprint=context.token_space_fingerprint,
                logical_vocab_size=context.logical_vocab_size,
                rollout_requirement=context.rollout_requirement,
                consistency=context.consistency,
            ),
        )

    def _plans(self, candidate_batch: CandidateTrainingBatch) -> tuple[_Plan, ...]:
        has_materialized_examples = bool(candidate_batch.examples)
        has_recipe = self._select is not None
        if has_materialized_examples == has_recipe:
            raise ValueError(
                "provide either materialized examples or the explicit "
                "same-context/all-captured-generations recipe"
            )

        examples = {
            example.generation.generation_id: example
            for example in candidate_batch.examples
        }
        plans: list[_Plan] = []
        for group in candidate_batch.groups:
            for trajectory in group.trajectories:
                for generation in trajectory.generations:
                    if has_materialized_examples:
                        example = examples.get(generation.generation_id)
                        if example is None:
                            continue
                        teacher_view = example.teacher_view
                        parts = example.parts
                        provenance = example.provenance
                    else:
                        assert self._select is not None
                        assert self._view is not None
                        teacher_view = generation.context
                        parts = self._select.parts
                        provenance = CanonicalJsonObject.from_value({})
                    positions = _selected_positions(generation.part_spans, parts)
                    if not positions:
                        raise ValueError(
                            f"selected generation {generation.generation_id!r} "
                            "contains no tokens for the requested parts"
                        )
                    plans.append(
                        _Plan(
                            generation_id=generation.generation_id,
                            teacher_view=teacher_view,
                            positions=positions,
                            forced_token_ids=generation.continuation_token_ids,
                            provenance=provenance,
                        )
                    )
        if not plans:
            raise ValueError("distillation preparation selected no generations")
        return tuple(plans)

    async def _score_all(
        self,
        requests: tuple[TeacherScoringRequest, ...],
    ) -> tuple[_Outcome, ...]:
        semaphore = asyncio.Semaphore(self._max_concurrency)
        tasks: list[asyncio.Task[_Outcome]] = []
        async with asyncio.TaskGroup() as task_group:
            for request in requests:
                tasks.append(
                    task_group.create_task(self._score_one(request, semaphore))
                )
        return tuple(task.result() for task in tasks)

    async def _score_one(
        self,
        request: TeacherScoringRequest,
        semaphore: asyncio.Semaphore,
    ) -> _Outcome:
        try:
            async with semaphore:
                result = await self._scorer.score(request)
            return _Outcome(request=request, result=result.validate_for(request))
        except asyncio.CancelledError:
            raise
        except Exception:
            return _Outcome(
                request=request,
                issue=PreparationIssue(
                    generation_id=request.generation_id,
                    teacher_name=request.teacher_name,
                    selected_positions=request.selected_positions,
                ),
            )


def _selected_positions(
    spans: tuple[PartSpan, ...],
    parts: tuple[GenerationPart, ...],
) -> tuple[int, ...]:
    selected_parts = set(parts)
    positions: set[int] = set()
    for span in spans:
        if span.part in selected_parts:
            positions.update(range(span.start, span.end))
    return tuple(sorted(positions))


def _teacher_revision(
    context: PreparationContext,
    teacher_name: str,
) -> int | str:
    consistency = context.consistency
    if isinstance(consistency, Frozen):
        return consistency.revision_for(teacher_name)
    if isinstance(consistency, CurrentStep):
        return consistency.revision
    raise TypeError("unsupported distillation consistency mode")


def _validate_rollout_revisions(
    candidate_batch: CandidateTrainingBatch,
    requests: tuple[TeacherScoringRequest, ...],
    context: PreparationContext,
) -> None:
    if not (
        isinstance(context.rollout_requirement, StudentOnPolicy)
        or isinstance(context.consistency, CurrentStep)
    ):
        return
    for request in requests:
        generation = candidate_batch.generation(request.generation_id)
        for position in request.selected_positions:
            revisions = tuple(
                span.revision
                for span in generation.rollout_spans
                if span.start <= position < span.end
            )
            if revisions != (context.learner_revision,):
                raise ValueError(
                    "selected token does not have the required learner rollout revision"
                )


def _result(outcome: _Outcome) -> TeacherScoringResult:
    assert outcome.result is not None
    return outcome.result


def _target_rows(
    request: TeacherScoringRequest,
    result: TeacherScoringResult,
    *,
    provenance: CanonicalJsonObject,
) -> tuple[TopKTargetRow, ...]:
    return tuple(
        TopKTargetRow(
            generation_id=request.generation_id,
            position=row.position,
            sampled_token_id=row.forced_token_id,
            token_ids=tuple(entry.token_id for entry in row.entries),
            teacher_logprobs=tuple(entry.logprob for entry in row.entries),
            tail_logprob=row.tail_logprob,
            logical_vocab_size=row.logical_vocab_size,
            temperature=row.temperature,
            teacher_name=request.teacher_name,
            teacher_revision=request.teacher_revision,
            token_space_fingerprint=request.token_space_fingerprint,
            request_id=request.request_id,
            forced_token_sha256=request.forced_token_sha256,
            provenance=provenance,
        )
        for row in result.positions
    )
