"""Asynchronous preparation boundary for distillation training batches.

This module intentionally contains no teacher-scoring implementation.  It defines the
boundary that synchronous orchestration uses today and that a future asynchronous
pipeline can schedule without coupling preparation to backend training.
"""

from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from .artifact import PreparedTrainingBatch
from .types import (
    AnyRevision,
    CurrentStep,
    Example,
    Frozen,
    GenerationPart,
    StudentOnPolicy,
    TopK,
)

if TYPE_CHECKING:
    from art.model import Model, TrainableModel
    from art.trajectories import TrajectoryGroup

    from .preparer import AllCapturedGenerations, FailurePolicy, SameContext

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


def all_generations(
    *,
    parts: Iterable[GenerationPart] = tuple(GenerationPart),
) -> "AllCapturedGenerations":
    """Select model-generated parts from every exact captured generation."""

    from .preparer import AllCapturedGenerations

    return AllCapturedGenerations(parts=tuple(parts))


def same_context() -> "SameContext":
    """Use the exact context captured immediately before each generation."""

    from .preparer import SameContext

    return SameContext()


async def prepare(
    student: "TrainableModel[Any, Any]",
    groups: Iterable["TrajectoryGroup"],
    *,
    teacher: "Model[Any, Any]",
    target: TopK,
    consistency: Frozen | CurrentStep,
    rollouts: StudentOnPolicy | AnyRevision | None = None,
    select: "AllCapturedGenerations | None" = None,
    teacher_view: "SameContext | None" = None,
    examples: Iterable[Example] = (),
    failures: "FailurePolicy | None" = None,
    max_concurrency: int = 8,
    max_scoring_retries: int = 2,
) -> PreparedTrainingBatch:
    """Prepare immutable teacher targets for a later ``backend.train`` call.

    V1 supports a frozen teacher revision. A self-teacher is pinned under the
    student's backend lease for the complete render-and-score operation.
    """

    from art.serving_capabilities import discover_serving_capabilities

    from .candidate import snapshot_groups
    from .preparer import DistillationPreparer, FailurePolicy
    from .vllm import VLLMTeacherScorer

    if not isinstance(consistency, Frozen):
        raise NotImplementedError(
            "distill.prepare V1 supports Frozen consistency; "
            "CurrentStep orchestration is planned for the asynchronous pipeline"
        )
    if target.temperature != 1.0:
        raise ValueError(
            "distill.prepare V1 supports TopK temperature 1.0 because vLLM "
            "prompt distributions expose raw model log-probabilities"
        )
    if (
        not isinstance(max_concurrency, int)
        or isinstance(max_concurrency, bool)
        or max_concurrency <= 0
    ):
        raise ValueError("max_concurrency must be a positive integer")
    if (
        not isinstance(max_scoring_retries, int)
        or isinstance(max_scoring_retries, bool)
        or max_scoring_retries < 0
    ):
        raise ValueError("max_scoring_retries must be a non-negative integer")
    materialized_examples = tuple(examples)
    if (select is None) != (teacher_view is None):
        raise ValueError("select and teacher_view must be provided together")
    has_recipe = select is not None
    if bool(materialized_examples) == has_recipe:
        raise ValueError(
            "provide either materialized examples or select with teacher_view"
        )

    backend = student.backend()
    learner_revision = await backend._get_step(student)
    self_teacher = teacher is student
    teacher_name = teacher.name
    teacher_revision = consistency.revision_for(teacher_name)
    if self_teacher and not isinstance(teacher_revision, int):
        raise ValueError(
            "student self-distillation requires a frozen integer teacher revision"
        )

    rollout_requirement = rollouts or StudentOnPolicy()
    student_capabilities = await _capabilities_for(
        student,
        allow_openai_compatible=False,
        discover=discover_serving_capabilities,
    )
    teacher_capabilities = (
        student_capabilities
        if self_teacher
        else await _capabilities_for(
            teacher,
            allow_openai_compatible=True,
            discover=discover_serving_capabilities,
        )
    )
    _validate_distribution_capabilities(
        student_capabilities,
        teacher_capabilities,
        target=target,
    )

    candidate = snapshot_groups(
        groups,
        examples=materialized_examples,
        base_model=student.base_model,
        model=student.get_inference_name(),
        chat_template_kwargs=(student._internal_config or {}).get(
            "chat_template_kwargs"
        ),
    )
    context = PreparationContext(
        learner_revision=learner_revision,
        token_space_fingerprint=cast(str, student_capabilities.token_space_fingerprint),
        logical_vocab_size=cast(int, student_capabilities.logical_vocab_size),
        rollout_requirement=rollout_requirement,
        consistency=consistency,
        correlation_id=candidate.batch_id,
    )

    lease: AbstractAsyncContextManager[None]
    if self_teacher:
        lease = backend.exact_adapter_lease(student, cast(int, teacher_revision))
    else:
        lease = nullcontext()
    async with lease:
        preparer = DistillationPreparer(
            scorer=VLLMTeacherScorer(
                base_url=_runtime_base_url(teacher),
                capabilities=teacher_capabilities,
                model_name=teacher.get_inference_name(
                    cast(int, teacher_revision) if self_teacher else None
                ),
                headers=_authorization_headers(teacher),
            ),
            teacher_name=teacher_name,
            target=target,
            select=select,
            view=teacher_view,
            failure_policy=failures,
            max_concurrency=max_concurrency,
            max_retries=max_scoring_retries,
        )
        return await _prepare_with(preparer, candidate, context)


async def _capabilities_for(
    model: Any,
    *,
    allow_openai_compatible: bool,
    discover: Any,
) -> Any:
    capabilities = model._serving_capabilities
    if capabilities is not None:
        return capabilities
    base_url = _runtime_base_url(model)
    capabilities = await discover(
        base_url=base_url,
        headers=_authorization_headers(model),
        allow_openai_compatible=allow_openai_compatible,
    )
    object.__setattr__(model, "_serving_capabilities", capabilities)
    return capabilities


def _validate_distribution_capabilities(
    student: Any,
    teacher: Any,
    *,
    target: TopK,
) -> None:
    teacher.require(
        "prompt_token_distributions",
        operation="distillation preparation",
    )
    if not student.token_space_fingerprint or student.logical_vocab_size is None:
        raise RuntimeError(
            "student serving capabilities must expose token-space identity"
        )
    if teacher.token_space_fingerprint != student.token_space_fingerprint:
        raise RuntimeError("teacher and student token spaces do not match")
    if teacher.logical_vocab_size != student.logical_vocab_size:
        raise RuntimeError("teacher and student logical vocabulary sizes do not match")
    capacity = teacher.max_prompt_logprobs
    if capacity is None or target.k > capacity:
        raise RuntimeError(
            f"requested top-k width {target.k} exceeds teacher capacity {capacity}"
        )


def _runtime_base_url(model: Any) -> str:
    base_url = model.inference_base_url
    if not base_url:
        raise ValueError("distillation teacher must have an inference base URL")
    return base_url.removesuffix("/").removesuffix("/v1")


def _authorization_headers(model: Any) -> dict[str, str]:
    if not model.inference_api_key:
        return {}
    return {"Authorization": f"Bearer {model.inference_api_key}"}
