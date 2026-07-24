import json

import pytest

from art import distill


def _generation() -> distill.CapturedGeneration:
    view = distill.TeacherView.from_request(
        "chat_completions",
        {"messages": [{"content": "question", "role": "user"}]},
    )
    return distill.CapturedGeneration.create(
        generation_id="generation-1",
        trajectory_fingerprint="trajectory-1",
        event_index=0,
        trajectory_token_start=1,
        protocol="chat_completions",
        continuation_token_ids=(2, 3),
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


def _group() -> distill.TrainingGroupSnapshot:
    return distill.TrainingGroupSnapshot(
        group_id="group-1",
        trajectories=(
            distill.TrainingTrajectorySnapshot(
                trajectory_fingerprint="trajectory-1",
                token_ids=(1, 2, 3),
                logprobs=(None, -0.2, -0.3),
                token_flags=(0, 1, 1),
                reward=1.0,
                advantage=0.5,
                generations=(_generation(),),
            ),
        ),
    )


def _target(position: int = 0) -> distill.TopKTargetRow:
    generation = _generation()
    return distill.TopKTargetRow(
        generation_id="generation-1",
        position=position,
        sampled_token_id=generation.continuation_token_ids[position],
        token_ids=(2,),
        teacher_logprobs=(-0.35667494393873245,),
        tail_logprob=-1.2039728043259361,
        logical_vocab_size=4,
        temperature=1.0,
        teacher_name="teacher",
        teacher_revision="revision-1",
        token_space_fingerprint="token-space",
        request_id=f"request-{position}",
        forced_token_sha256=generation.continuation_sha256,
    )


def _artifact(
    groups: tuple[distill.TrainingGroupSnapshot, ...] | None = None,
    targets: tuple[distill.TopKTargetRow, ...] | None = None,
    constraints: distill.PreparedConstraints | None = None,
) -> distill.PreparedTrainingBatch:
    target_values = targets or (_target(),)
    return distill.PreparedTrainingBatch.create(
        groups=groups or (_group(),),
        targets=target_values,
        report=distill.PreparationReport(
            selected_generations=1,
            selected_tokens=len(target_values),
            prepared_tokens=len(target_values),
        ),
        constraints=constraints
        or distill.PreparedConstraints(
            learner_revision=7,
            token_space_fingerprint="token-space",
            logical_vocab_size=4,
            rollout_requirement=distill.StudentOnPolicy(),
            consistency=distill.Frozen(revision="revision-1"),
        ),
    )


def test_artifact_round_trip_is_byte_identical_and_validated() -> None:
    artifact = _artifact()
    encoded = artifact.to_bytes()
    loaded = distill.PreparedTrainingBatch.from_bytes(encoded)

    assert loaded == artifact
    assert loaded.to_bytes() == encoded
    assert loaded.report.prepared_tokens == 1
    assert loaded.parsed_payload().targets == (_target(),)


def test_artifact_rejects_payload_checksum_and_semantic_id_tampering() -> None:
    raw = json.loads(_artifact().to_bytes())
    raw["payload"] = raw["payload"].replace(
        '"prepared_tokens":1', '"prepared_tokens":0'
    )
    with pytest.raises(ValueError, match="checksum"):
        distill.PreparedTrainingBatch.from_bytes(
            json.dumps(raw, separators=(",", ":"), sort_keys=True).encode()
        )

    raw = json.loads(_artifact().to_bytes())
    raw["preparation_id"] = "0" * 64
    with pytest.raises(ValueError, match="preparation ID"):
        distill.PreparedTrainingBatch.from_bytes(
            json.dumps(raw, separators=(",", ":"), sort_keys=True).encode()
        )


def test_preparation_id_is_stable_and_order_sensitive() -> None:
    first = _target(0)
    second = _target(1)
    artifact = _artifact(targets=(first, second))
    identical = _artifact(targets=(first, second))
    reordered = _artifact(targets=(second, first))

    assert artifact.preparation_id == identical.preparation_id
    assert artifact.to_bytes() == identical.to_bytes()
    assert artifact.preparation_id != reordered.preparation_id


def test_training_snapshot_rejects_misalignment_and_application_fields() -> None:
    trajectory = _group().trajectories[0]
    with pytest.raises(ValueError, match="do not match trajectory tokens"):
        distill.TrainingTrajectorySnapshot.model_validate(
            trajectory.model_copy(update={"token_ids": (1, 9, 3)}).model_dump()
        )
    with pytest.raises(ValueError, match="Extra inputs"):
        distill.TrainingGroupSnapshot.model_validate(
            {
                **_group().model_dump(),
                "application_secret": "must not cross the snapshot boundary",
            }
        )


def test_artifact_rejects_report_mismatch_and_duplicate_positions() -> None:
    with pytest.raises(ValueError, match="target row count"):
        distill.PreparedTrainingBatch.create(
            groups=(_group(),),
            targets=(_target(),),
            report=distill.PreparationReport(
                selected_generations=1,
                selected_tokens=2,
                prepared_tokens=2,
            ),
            constraints=_artifact().constraints,
        )

    with pytest.raises(ValueError, match="unique generation positions"):
        _artifact(targets=(_target(), _target()))

    with pytest.raises(ValueError, match="require selected and prepared tokens"):
        distill.PreparedTrainingBatch.create(
            groups=(_group(),),
            targets=(),
            report=distill.PreparationReport(
                selected_generations=0,
                selected_tokens=0,
                prepared_tokens=0,
            ),
            constraints=_artifact().constraints,
        )


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"generation_id": "missing"}, "unknown generation"),
        ({"position": 2}, "position exceeds"),
        ({"sampled_token_id": 3}, "sampled token does not match"),
        ({"forced_token_sha256": "wrong"}, "forced-token hash"),
        ({"token_space_fingerprint": "wrong"}, "token space"),
        ({"logical_vocab_size": 5}, "vocabulary"),
        ({"teacher_revision": "wrong"}, "teacher revision"),
    ],
)
def test_artifact_rejects_cross_record_mismatches(
    update: dict[str, object],
    match: str,
) -> None:
    changed = _target().model_copy(update=update)
    with pytest.raises(ValueError, match=match):
        _artifact(targets=(changed,))


def test_frozen_constraints_support_multiple_teacher_revisions() -> None:
    first = _target(0).model_copy(
        update={"teacher_name": "specialist-a", "teacher_revision": "revision-a"}
    )
    second = _target(1).model_copy(
        update={"teacher_name": "specialist-b", "teacher_revision": "revision-b"}
    )
    constraints = distill.PreparedConstraints(
        learner_revision=7,
        token_space_fingerprint="token-space",
        logical_vocab_size=4,
        rollout_requirement=distill.StudentOnPolicy(),
        consistency=distill.Frozen(
            revisions=(
                distill.TeacherRevision(
                    teacher_name="specialist-a",
                    revision="revision-a",
                ),
                distill.TeacherRevision(
                    teacher_name="specialist-b",
                    revision="revision-b",
                ),
            )
        ),
    )

    artifact = _artifact(targets=(first, second), constraints=constraints)

    assert artifact.parsed_payload().targets == (first, second)


def test_artifact_rejects_duplicate_batch_source_identities() -> None:
    with pytest.raises(ValueError, match="group IDs"):
        _artifact(groups=(_group(), _group()))

    second_group = _group().model_copy(update={"group_id": "group-2"})
    with pytest.raises(ValueError, match="trajectory fingerprints"):
        _artifact(groups=(_group(), second_group))


def test_artifact_rejects_missing_or_stale_rollout_provenance() -> None:
    generation = _generation()
    for rollout_spans in (
        (),
        (distill.RolloutRevisionSpan(start=0, end=2, revision=6),),
    ):
        changed_generation = generation.model_copy(
            update={"rollout_spans": rollout_spans}
        )
        trajectory = (
            _group()
            .trajectories[0]
            .model_copy(update={"generations": (changed_generation,)})
        )
        group = _group().model_copy(update={"trajectories": (trajectory,)})
        with pytest.raises(ValueError, match="rollout revision"):
            _artifact(groups=(group,))


def test_current_step_revision_must_equal_learner_revision() -> None:
    constraints = distill.PreparedConstraints(
        learner_revision=7,
        token_space_fingerprint="token-space",
        logical_vocab_size=4,
        rollout_requirement=distill.StudentOnPolicy(),
        consistency=distill.CurrentStep(revision=6, session_id="session-1"),
    )

    with pytest.raises(ValueError, match="current-step teacher revision"):
        _artifact(constraints=constraints)


def test_current_step_rejects_stale_rollout_even_when_any_revision_requested() -> None:
    generation = _generation().model_copy(
        update={
            "rollout_spans": (distill.RolloutRevisionSpan(start=0, end=2, revision=6),)
        }
    )
    trajectory = (
        _group().trajectories[0].model_copy(update={"generations": (generation,)})
    )
    group = _group().model_copy(update={"trajectories": (trajectory,)})
    target = _target().model_copy(update={"teacher_revision": 7})
    constraints = distill.PreparedConstraints(
        learner_revision=7,
        token_space_fingerprint="token-space",
        logical_vocab_size=4,
        rollout_requirement=distill.AnyRevision(),
        consistency=distill.CurrentStep(revision=7, session_id="session-1"),
    )

    with pytest.raises(ValueError, match="rollout revision"):
        _artifact(
            groups=(group,),
            targets=(target,),
            constraints=constraints,
        )
