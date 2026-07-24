import dataclasses
from typing import Any, cast

from pydantic import ValidationError
import pytest

from art import TrainingObjectives, distill


def _view() -> distill.TeacherView:
    return distill.TeacherView.from_request(
        "chat_completions",
        {
            "messages": [{"content": "hello", "role": "user"}],
            "temperature": 0,
        },
    )


def _generation() -> distill.CapturedGeneration:
    view = _view()
    return distill.CapturedGeneration.create(
        generation_id="generation-1",
        trajectory_fingerprint="trajectory-sha256",
        event_index=0,
        trajectory_token_start=0,
        protocol="chat_completions",
        continuation_token_ids=(10, 11),
        context=view,
        part_spans=(
            distill.PartSpan(
                part=distill.GenerationPart.ASSISTANT_TEXT,
                start=0,
                end=2,
            ),
        ),
        rollout_spans=(distill.RolloutRevisionSpan(start=0, end=2, revision=7),),
    )


def test_teacher_view_and_provenance_are_recursively_immutable() -> None:
    request = {"messages": [{"role": "user", "content": "hello"}]}
    view = distill.TeacherView.from_request("chat_completions", request)
    request["messages"][0]["content"] = "mutated"

    first = cast(dict[str, Any], view.request())
    assert first["messages"][0]["content"] == "hello"
    decoded = cast(dict[str, Any], view.request())
    decoded["messages"][0]["content"] = "also mutated"
    fresh = cast(dict[str, Any], view.request())
    assert fresh["messages"][0]["content"] == "hello"

    example = distill.Example(
        generation=_generation(),
        teacher_view=view,
        parts=frozenset({distill.GenerationPart.ASSISTANT_TEXT}),
        provenance={"weave": {"call_id": "call-1"}},
    )
    provenance = cast(dict[str, Any], example.provenance.to_dict())
    provenance["weave"]["call_id"] = "changed"
    fresh_provenance = cast(dict[str, Any], example.provenance.to_dict())
    assert fresh_provenance["weave"]["call_id"] == "call-1"
    with pytest.raises(ValidationError):
        example.provenance = {}  # ty: ignore[invalid-assignment]


def test_capture_rejects_changed_tokens_and_overlapping_spans() -> None:
    generation = _generation()
    with pytest.raises(ValidationError, match="continuation hash"):
        generation.model_copy(
            update={"continuation_token_ids": (10, 12)}
        ).model_validate(
            generation.model_copy(
                update={"continuation_token_ids": (10, 12)}
            ).model_dump()
        )

    with pytest.raises(ValidationError, match="must not overlap"):
        distill.CapturedGeneration.create(
            generation_id="generation-1",
            trajectory_fingerprint="trajectory-sha256",
            event_index=0,
            trajectory_token_start=0,
            protocol="chat_completions",
            continuation_token_ids=(10, 11),
            context=_view(),
            part_spans=(
                distill.PartSpan(
                    part=distill.GenerationPart.REASONING,
                    start=0,
                    end=2,
                ),
                distill.PartSpan(
                    part=distill.GenerationPart.ASSISTANT_TEXT,
                    start=1,
                    end=2,
                ),
            ),
            rollout_spans=(distill.RolloutRevisionSpan(start=0, end=2, revision=7),),
        )


def test_target_validation_and_training_objectives() -> None:
    row = distill.TopKTargetRow(
        generation_id="generation-1",
        position=0,
        sampled_token_id=2,
        token_ids=(2, 1),
        teacher_logprobs=(-0.5108256237659907, -1.6094379124341003),
        tail_logprob=-1.6094379124341003,
        logical_vocab_size=4,
        temperature=1.0,
        teacher_name="teacher",
        teacher_revision="revision-1",
        token_space_fingerprint="token-space",
        request_id="request-1",
        forced_token_sha256="forced-sha256",
    )
    assert row.kind == "top_k"

    with pytest.raises(ValidationError, match="unique"):
        row.model_copy(update={"token_ids": (2, 2)}).model_validate(
            row.model_copy(update={"token_ids": (2, 2)}).model_dump()
        )
    with pytest.raises(ValidationError, match="sum to one"):
        row.model_copy(update={"tail_logprob": -4.0}).model_validate(
            row.model_copy(update={"tail_logprob": -4.0}).model_dump()
        )
    with pytest.raises(ValidationError, match="at least one"):
        TrainingObjectives()


def test_preparation_report_requires_complete_issue_ledger() -> None:
    issue = distill.PreparationIssue(
        generation_id="generation-1",
        teacher_name="teacher",
        selected_positions=(0,),
    )
    report = distill.PreparationReport(
        selected_generations=2,
        selected_tokens=2,
        prepared_tokens=1,
        issue_count=1,
        issues=(issue,),
    )
    assert report.issues == (issue,)

    with pytest.raises(ValidationError, match="issue count"):
        report.model_copy(update={"issue_count": 0}).model_validate(
            report.model_copy(update={"issue_count": 0}).model_dump()
        )


def test_public_values_are_frozen() -> None:
    target = distill.TopK()
    with pytest.raises(ValidationError):
        target.k = 8  # ty: ignore[invalid-assignment]

    context = distill.PreparationContext(
        learner_revision=0,
        token_space_fingerprint="tokens",
        logical_vocab_size=32,
        rollout_requirement=distill.StudentOnPolicy(),
        consistency=distill.Frozen(revision=0),
        correlation_id="correlation",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.learner_revision = 1  # ty: ignore[invalid-assignment]
