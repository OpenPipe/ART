import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art import distill
from art.local.backend import LocalBackend
from art.megatron.backend import MegatronBackend
from art.megatron.distillation import (
    CispoObjectiveConfig,
    DistillationObjectiveConfig,
    PolicyPackingConfig,
    pack_prepared_batch,
    packed_distillation_tensors_from_dir,
    validate_prepared_forward_kl,
    validate_standalone_forward_kl,
)
from art.megatron.runtime.jobs import (
    LORA_READY_EVENT,
    OPTIMIZER_READY_EVENT,
    MegatronDistillationJob,
    dump_megatron_job,
    load_megatron_job,
)
from art.megatron.service import (
    MegatronService,
    _validate_distillation_tensors_for_objective,
)
from art.megatron.writer_sessions import WriterLease
from art.types import LocalTrainResult, TrainConfig


def _artifact(
    *,
    revision: int = 7,
    consistency: distill.Frozen | distill.CurrentStep | None = None,
) -> distill.PreparedTrainingBatch:
    resolved_consistency = consistency or distill.Frozen(revision="frozen")
    view = distill.TeacherView.from_request(
        "chat_completions",
        {"messages": [{"content": "question", "role": "user"}]},
    )
    generation = distill.CapturedGeneration.create(
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
        rollout_spans=(distill.RolloutRevisionSpan(start=0, end=2, revision=revision),),
    )
    group = distill.TrainingGroupSnapshot(
        group_id="group-1",
        trajectories=(
            distill.TrainingTrajectorySnapshot(
                trajectory_fingerprint="trajectory-1",
                token_ids=(1, 2, 3),
                logprobs=(None, -0.2, -0.3),
                token_flags=(0, 1, 1),
                reward=1.0,
                advantage=0.5,
                generations=(generation,),
            ),
        ),
    )
    targets = tuple(
        distill.TopKTargetRow(
            generation_id="generation-1",
            position=position,
            sampled_token_id=generation.continuation_token_ids[position],
            token_ids=(0, 2),
            teacher_logprobs=(
                -1.6094379124341003,
                -0.6931471805599453,
            ),
            tail_logprob=-1.2039728043259361,
            logical_vocab_size=4,
            temperature=1.0,
            teacher_name="teacher",
            teacher_revision=(
                revision
                if isinstance(resolved_consistency, distill.CurrentStep)
                else "frozen"
            ),
            token_space_fingerprint="token-space",
            request_id=f"request-{position}",
            forced_token_sha256=generation.continuation_sha256,
        )
        for position in range(2)
    )
    return distill.PreparedTrainingBatch.create(
        groups=(group,),
        targets=targets,
        report=distill.PreparationReport(
            selected_generations=1,
            prepared_generations=1,
            selected_tokens=2,
            prepared_tokens=2,
        ),
        constraints=distill.PreparedConstraints(
            learner_revision=revision,
            token_space_fingerprint="token-space",
            logical_vocab_size=4,
            rollout_requirement=distill.StudentOnPolicy(),
            consistency=resolved_consistency,
        ),
    )


def _additive_artifact(
    *,
    revision: int = 7,
    successful_generation_ids: tuple[str, ...] = ("generation-0", "generation-1"),
    failed_generation_ids: tuple[str, ...] = (),
    rollout_revisions: tuple[int, int, int] | None = None,
    missing_logprob: tuple[int, int] | None = None,
    rewards: tuple[float, float, float] = (-1.0, 0.0, 1.0),
) -> distill.PreparedTrainingBatch:
    view = distill.TeacherView.from_request(
        "chat_completions",
        {"messages": [{"content": "question", "role": "user"}]},
    )
    resolved_revisions = rollout_revisions or (revision, revision, revision)
    trajectories = []
    generations: dict[str, distill.CapturedGeneration] = {}
    for index, (reward, rollout_revision) in enumerate(
        zip(rewards, resolved_revisions, strict=True)
    ):
        generation_id = f"generation-{index}"
        fingerprint = f"trajectory-{index}"
        continuation = (10 + index * 2, 11 + index * 2)
        generation = distill.CapturedGeneration.create(
            generation_id=generation_id,
            trajectory_fingerprint=fingerprint,
            event_index=0,
            trajectory_token_start=1,
            protocol="chat_completions",
            continuation_token_ids=continuation,
            context=view,
            part_spans=(
                distill.PartSpan(
                    part=distill.GenerationPart.ASSISTANT_TEXT,
                    start=0,
                    end=2,
                ),
            ),
            rollout_spans=(
                distill.RolloutRevisionSpan(
                    start=0,
                    end=2,
                    revision=rollout_revision,
                ),
            ),
        )
        generations[generation_id] = generation
        logprobs: list[float | None] = [None, -0.2, -0.3]
        if missing_logprob is not None and missing_logprob[0] == index:
            logprobs[1 + missing_logprob[1]] = None
        trajectories.append(
            distill.TrainingTrajectorySnapshot(
                trajectory_fingerprint=fingerprint,
                token_ids=(1, *continuation),
                logprobs=tuple(logprobs),
                token_flags=(int(1), int(3), int(3)),
                reward=reward,
                advantage=reward,
                generations=(generation,),
            )
        )

    group = distill.TrainingGroupSnapshot(
        group_id="group-0",
        trajectories=tuple(trajectories),
    )
    targets = tuple(
        distill.TopKTargetRow(
            generation_id=generation_id,
            position=0,
            sampled_token_id=generations[generation_id].continuation_token_ids[0],
            token_ids=(0, 2),
            teacher_logprobs=(
                -1.6094379124341003,
                -0.6931471805599453,
            ),
            tail_logprob=-1.2039728043259361,
            logical_vocab_size=32,
            temperature=1.0,
            teacher_name="teacher",
            teacher_revision="frozen",
            token_space_fingerprint="token-space",
            request_id=f"request-{generation_id}",
            forced_token_sha256=generations[generation_id].continuation_sha256,
        )
        for generation_id in successful_generation_ids
    )
    issues = tuple(
        distill.PreparationIssue(
            generation_id=generation_id,
            teacher_name="teacher",
            selected_positions=(0,),
        )
        for generation_id in failed_generation_ids
    )
    return distill.PreparedTrainingBatch.create(
        groups=(group,),
        targets=targets,
        report=distill.PreparationReport(
            selected_generations=len(targets) + len(issues),
            prepared_generations=len(targets),
            selected_tokens=len(targets) + len(issues),
            prepared_tokens=len(targets),
            issue_count=len(issues),
            issues=issues,
        ),
        constraints=distill.PreparedConstraints(
            learner_revision=revision,
            token_space_fingerprint="token-space",
            logical_vocab_size=32,
            rollout_requirement=distill.StudentOnPolicy(),
            consistency=distill.Frozen(revision="frozen"),
        ),
    )


def _validated_payload(
    artifact: distill.PreparedTrainingBatch,
    *,
    revision: int = 7,
    sequence_length: int = 8,
):
    return validate_standalone_forward_kl(
        batch=artifact,
        objectives=distill.TrainingObjectives(
            policy=None,
            distillation=distill.Loss(),
        ),
        expected_source_revision=revision,
        packed_sequence_length=sequence_length,
        tensor_parallel_size=1,
        context_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        expert_tensor_parallel_size=1,
    )


def _validated_additive_payload(
    artifact: distill.PreparedTrainingBatch,
    *,
    sequence_length: int = 8,
):
    return validate_prepared_forward_kl(
        batch=artifact,
        objectives=distill.TrainingObjectives(
            policy="cispo",
            distillation=distill.Loss(),
        ),
        expected_source_revision=7,
        packed_sequence_length=sequence_length,
    )


def test_service_releases_validation_tensors_before_worker_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _TrackingPacked(dict[str, torch.Tensor]):
        def clear(self) -> None:
            events.append("released")
            super().clear()

    packed = _TrackingPacked(
        policy_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    monkeypatch.setattr(
        "art.megatron.distillation.packed_distillation_tensors_from_dir",
        lambda _disk: packed,
    )

    _validate_distillation_tensors_for_objective(
        cast(Any, {"dir": "/unused"}),
        DistillationObjectiveConfig(coefficient=1.0),
    )

    assert events == ["released"]
    assert packed == {}


def _rebind_tensor_checksum(
    disk: dict[str, Any],
    directory: Path,
) -> None:
    hashes = [
        (path.stem, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(directory.iterdir())
    ]
    manifest = json.dumps(
        hashes, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    disk["tensors_sha256"] = hashlib.sha256(
        b"art-distill-tensors-v1\0" + manifest
    ).hexdigest()


def test_pack_is_exact_token_aligned_fixed_k_and_round_trips(tmp_path: Path) -> None:
    artifact = _artifact()
    disk = pack_prepared_batch(
        batch=artifact,
        payload=_validated_payload(artifact),
        sequence_length=8,
        output_dir=str(tmp_path / "packed"),
    )

    packed = packed_distillation_tensors_from_dir(disk)

    assert disk["target_count"] == 2
    assert disk["top_k_width"] == 2
    assert packed["tokens"].shape == (1, 8)
    assert packed["target_mask"].nonzero().tolist() == [[0, 1], [0, 2]]
    assert packed["distillation_weights"][0, :3].tolist() == [0.0, 1.0, 1.0]
    assert packed["topk_token_ids"][0, 0].tolist() == [-1, -1]
    assert packed["topk_token_ids"][0, 1].tolist() == [0, 2]
    assert int(packed["target_mask"].sum()) == len(artifact.parsed_payload().targets)
    assert disk["policy_count"] == 0
    assert "policy_kind" not in disk
    assert not bool(packed["policy_mask"].any())
    assert bool(torch.all(torch.isnan(packed["old_logprobs"])))


def test_additive_pack_has_independent_policy_and_kd_masks(tmp_path: Path) -> None:
    artifact = _additive_artifact()
    objectives = distill.TrainingObjectives(
        policy="cispo",
        distillation=distill.Loss(),
    )
    disk = pack_prepared_batch(
        batch=artifact,
        payload=_validated_additive_payload(artifact),
        sequence_length=8,
        output_dir=str(tmp_path / "packed"),
        objectives=objectives,
    )

    packed = packed_distillation_tensors_from_dir(disk)

    assert disk["policy_kind"] == "cispo"
    assert disk["policy_count"] == 4
    assert disk["target_count"] == 2
    assert packed["tokens"].shape == (3, 8)
    assert packed["policy_mask"].nonzero().tolist() == [
        [0, 1],
        [0, 2],
        [2, 1],
        [2, 2],
    ]
    assert packed["target_mask"].nonzero().tolist() == [[0, 1], [1, 1]]
    assert packed["source_group_ids"].tolist() == [0, 0, 0]
    assert packed["policy_group_ids"][packed["policy_mask"]].tolist() == [
        0,
        0,
        2,
        2,
    ]
    torch.testing.assert_close(
        packed["policy_advantages"][0, 1:3],
        torch.tensor([-1.0, -1.0]),
    )
    torch.testing.assert_close(
        packed["policy_advantages"][2, 1:3],
        torch.tensor([1.0, 1.0]),
    )
    torch.testing.assert_close(
        packed["policy_weights"][packed["policy_mask"]],
        torch.ones(4),
    )
    assert not bool(packed["policy_mask"][1].any())
    assert bool(packed["target_mask"][1, 1])
    assert bool(torch.all(torch.isnan(packed["old_logprobs"][1])))


def test_teacher_failure_does_not_change_policy_projection(tmp_path: Path) -> None:
    failed = _additive_artifact(failed_generation_ids=("generation-2",))
    unselected = _additive_artifact()
    objectives = distill.TrainingObjectives(
        policy="cispo",
        distillation=distill.Loss(),
    )
    failed_disk = pack_prepared_batch(
        batch=failed,
        payload=_validated_additive_payload(failed),
        sequence_length=8,
        output_dir=str(tmp_path / "failed"),
        objectives=objectives,
    )
    unselected_disk = pack_prepared_batch(
        batch=unselected,
        payload=_validated_additive_payload(unselected),
        sequence_length=8,
        output_dir=str(tmp_path / "unselected"),
        objectives=objectives,
    )

    failed_tensors = packed_distillation_tensors_from_dir(failed_disk)
    unselected_tensors = packed_distillation_tensors_from_dir(unselected_disk)
    for name in (
        "tokens",
        "token_mask",
        "source_group_ids",
        "policy_mask",
        "old_logprobs",
        "policy_advantages",
        "policy_weights",
        "policy_group_ids",
    ):
        torch.testing.assert_close(
            failed_tensors[name],
            unselected_tensors[name],
            equal_nan=True,
        )
    assert failed_disk["policy_count"] == unselected_disk["policy_count"] == 4
    assert not bool(failed_tensors["target_mask"][2].any())
    assert bool(failed_tensors["policy_mask"][2].all().item()) is False
    assert failed_tensors["policy_mask"][2, 1:3].tolist() == [True, True]


def test_zero_variance_rows_remain_eligible_for_kd_only(tmp_path: Path) -> None:
    artifact = _additive_artifact(rewards=(0.0, 0.0, 0.0))
    payload = validate_prepared_forward_kl(
        batch=artifact,
        objectives=distill.TrainingObjectives(distillation=distill.Loss()),
        expected_source_revision=7,
        packed_sequence_length=8,
    )
    disk = pack_prepared_batch(
        batch=artifact,
        payload=payload,
        sequence_length=8,
        output_dir=str(tmp_path / "packed"),
        objectives=distill.TrainingObjectives(distillation=distill.Loss()),
    )
    packed = packed_distillation_tensors_from_dir(disk)

    assert disk["target_count"] == 2
    assert disk["policy_count"] == 0
    assert packed["tokens"].shape[0] == 2
    assert packed["target_mask"].nonzero().tolist() == [[0, 1], [1, 1]]
    assert not bool(packed["policy_mask"].any())

    with pytest.raises(ValueError, match="zero token denominator"):
        validate_prepared_forward_kl(
            batch=artifact,
            objectives=distill.TrainingObjectives(
                policy="cispo",
                distillation=distill.Loss(),
            ),
            expected_source_revision=7,
            packed_sequence_length=8,
        )


def test_policy_validation_covers_unselected_trajectories() -> None:
    stale = _additive_artifact(rollout_revisions=(7, 7, 6))
    with pytest.raises(ValueError, match="rollout revision"):
        _validated_additive_payload(stale)

    missing = _additive_artifact(missing_logprob=(2, 1))
    with pytest.raises(ValueError, match="missing its rollout logprob"):
        _validated_additive_payload(missing)

    # The same missing value is legal on a zero-advantage, policy-inactive row.
    inactive_missing = _additive_artifact(missing_logprob=(1, 1))
    _validated_additive_payload(inactive_missing)


def test_loader_rejects_rebound_policy_sidecar_corruption(tmp_path: Path) -> None:
    artifact = _additive_artifact()
    directory = tmp_path / "packed"
    disk = pack_prepared_batch(
        batch=artifact,
        payload=_validated_additive_payload(artifact),
        sequence_length=8,
        output_dir=str(directory),
        objectives=distill.TrainingObjectives(
            policy="cispo",
            distillation=distill.Loss(),
        ),
    )
    values = torch.from_file(
        str(directory / "policy_mask.pt"),
        shared=True,
        size=3 * 8,
        dtype=torch.bool,
    ).view(3, 8)
    values[0, 1] = False
    _rebind_tensor_checksum(cast(dict[str, Any], disk), directory)

    with pytest.raises(ValueError, match="policy denominator mismatch"):
        packed_distillation_tensors_from_dir(disk)


def test_pack_is_byte_deterministic(tmp_path: Path) -> None:
    artifact = _artifact()
    payload = _validated_payload(artifact)
    first = pack_prepared_batch(
        batch=artifact,
        payload=payload,
        sequence_length=8,
        output_dir=str(tmp_path / "first"),
    )
    second = pack_prepared_batch(
        batch=artifact,
        payload=payload,
        sequence_length=8,
        output_dir=str(tmp_path / "second"),
    )

    assert first["tensors_sha256"] == second["tensors_sha256"]
    for path in (tmp_path / "first").iterdir():
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()


def test_loader_rejects_corruption_before_mapping(tmp_path: Path) -> None:
    artifact = _artifact()
    disk = pack_prepared_batch(
        batch=artifact,
        payload=_validated_payload(artifact),
        sequence_length=8,
        output_dir=str(tmp_path / "packed"),
    )
    path = tmp_path / "packed" / "tokens.pt"
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="invalid byte length"):
        packed_distillation_tensors_from_dir(disk)


def test_preflight_rejects_stale_revision_oversize_and_additive_policy() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="current model revision"):
        _validated_payload(artifact, revision=8)
    with pytest.raises(ValueError, match="never truncates"):
        _validated_payload(artifact, sequence_length=2)
    with pytest.raises(ValueError, match="standalone"):
        validate_standalone_forward_kl(
            batch=artifact,
            objectives=distill.TrainingObjectives(
                policy="cispo",
                distillation=distill.Loss(),
            ),
            expected_source_revision=7,
            packed_sequence_length=8,
            tensor_parallel_size=1,
            context_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            expert_tensor_parallel_size=1,
        )


def test_standalone_preflight_accepts_tp_cp_and_rejects_pipeline_parallel() -> None:
    artifact = _artifact()
    objective = distill.TrainingObjectives(distillation=distill.Loss())

    payload = validate_standalone_forward_kl(
        batch=artifact,
        objectives=objective,
        expected_source_revision=7,
        packed_sequence_length=8,
        tensor_parallel_size=2,
        context_parallel_size=2,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    assert payload.constraints.learner_revision == 7

    with pytest.raises(ValueError, match="PP=EP=ETP=1"):
        validate_standalone_forward_kl(
            batch=artifact,
            objectives=objective,
            expected_source_revision=7,
            packed_sequence_length=8,
            tensor_parallel_size=2,
            context_parallel_size=2,
            pipeline_parallel_size=2,
            expert_parallel_size=1,
            expert_tensor_parallel_size=1,
        )


def test_loader_rejects_tampered_active_distribution(tmp_path: Path) -> None:
    artifact = _artifact()
    disk = pack_prepared_batch(
        batch=artifact,
        payload=_validated_payload(artifact),
        sequence_length=8,
        output_dir=str(tmp_path / "packed"),
    )
    path = tmp_path / "packed" / "teacher_logprobs.pt"
    values = torch.from_file(
        str(path),
        shared=True,
        size=1 * 8 * 2,
        dtype=torch.float32,
    ).view(1, 8, 2)
    values[0, 1, 0] = 0.0
    # Rebinding the manifest checksum must not bypass semantic validation.
    hashes = []
    for tensor_path in sorted((tmp_path / "packed").iterdir()):
        hashes.append(
            (tensor_path.stem, hashlib.sha256(tensor_path.read_bytes()).hexdigest())
        )
    manifest = json.dumps(
        hashes, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    disk["tensors_sha256"] = hashlib.sha256(
        b"art-distill-tensors-v1\0" + manifest
    ).hexdigest()

    with pytest.raises(ValueError, match="not normalized"):
        packed_distillation_tensors_from_dir(disk)


def test_distillation_job_round_trips_revision_and_artifact_binding(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    disk = pack_prepared_batch(
        batch=artifact,
        payload=_validated_payload(artifact),
        sequence_length=8,
        output_dir=str(tmp_path / "packed"),
    )
    job = MegatronDistillationJob(
        step=8,
        source_policy_step=7,
        expected_source_revision=7,
        training_session_id="session",
        lora_path="/tmp/lora",
        optimizer_state_path="/tmp/optimizer",
        distillation_tensors=disk,
        config=TrainConfig(),
        objective=DistillationObjectiveConfig(coefficient=0.5),
        idempotency_key="stable-key",
        preparation_id=artifact.preparation_id,
        payload_sha256=artifact.payload_sha256,
    )

    loaded = load_megatron_job(dump_megatron_job(job))

    assert isinstance(loaded, MegatronDistillationJob)
    assert loaded.expected_source_revision == 7
    assert loaded.idempotency_key == "stable-key"
    assert loaded.preparation_id == artifact.preparation_id
    assert loaded.payload_sha256 == artifact.payload_sha256


def test_distillation_objective_rejects_nonfinite_coefficient() -> None:
    with pytest.raises(ValueError, match="finite number"):
        DistillationObjectiveConfig(coefficient=math.inf)


@pytest.mark.asyncio
async def test_backend_returns_committed_receipt_before_stale_revision_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _artifact(revision=7)
    backend = MegatronBackend(path=str(tmp_path))
    observed: dict[str, Any] = {}

    class _Service:
        async def committed_distillation_step(self, **kwargs: Any) -> int:
            observed.update(kwargs)
            return 8

    async def _get_service(_: Any) -> _Service:
        return _Service()

    async def _get_step(_: Any) -> int:
        raise AssertionError("committed replay must not check the current revision")

    monkeypatch.setattr(backend, "_get_service", _get_service)
    monkeypatch.setattr(backend, "_get_step", _get_step)
    result = await backend.train(
        cast(Any, SimpleNamespace()),
        artifact,
        objectives=distill.TrainingObjectives(distillation=distill.Loss()),
        idempotency_key="stable-key",
        save_checkpoint=False,
    )

    assert result.step == 8
    assert result.metrics["distill/idempotent_replay"] == 1.0
    assert observed["config"] == TrainConfig(optimizer_save_interval=1)


@pytest.mark.asyncio
async def test_backend_current_step_requires_explicit_single_use_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = MegatronBackend(path=str(tmp_path))
    model = SimpleNamespace(
        trainable=True,
        project="project",
        _storage_name=lambda: "student",
    )

    class _Service:
        def __init__(self) -> None:
            self.capability = b"c" * 32
            self.released = False

        async def acquire_current_step(
            self,
            *,
            revision: int,
            ttl_s: float,
        ) -> WriterLease:
            return WriterLease(
                model_identity="project/student",
                revision=revision,
                session_id="current-session",
                fence=1,
                expires_at=ttl_s,
                kind="current_step",
                capability=self.capability,
            )

        async def heartbeat_current_step(self, **_kwargs: Any) -> float:
            return 1_000.0

        async def committed_distillation_step(self, **_kwargs: Any) -> int:
            return 8

        async def release_current_step(self, **_kwargs: Any) -> None:
            self.released = True

    service = _Service()

    async def _get_service(_: Any) -> _Service:
        return service

    async def _get_step(_: Any) -> int:
        return 7

    monkeypatch.setattr(backend, "_get_service", _get_service)
    monkeypatch.setattr(backend, "_get_step", _get_step)

    async with backend.current_step(cast(Any, model)) as current:
        batch = _artifact(
            revision=7,
            consistency=distill.CurrentStep(current),
        )
        with pytest.raises(ValueError, match="same active"):
            await backend.train(
                cast(Any, model),
                batch,
                objectives=distill.TrainingObjectives(distillation=distill.Loss()),
                idempotency_key="current",
                save_checkpoint=False,
            )

        result = await backend.train(
            cast(Any, model),
            batch,
            objectives=distill.TrainingObjectives(distillation=distill.Loss()),
            idempotency_key="current",
            session=current,
            save_checkpoint=False,
        )
        assert result.step == 8
        with pytest.raises(RuntimeError, match="exactly one"):
            await backend.train(
                cast(Any, model),
                batch,
                objectives=distill.TrainingObjectives(distillation=distill.Loss()),
                idempotency_key="current",
                session=current,
                save_checkpoint=False,
            )

    assert service.released


@pytest.mark.asyncio
async def test_backend_resolves_additive_public_objective_before_receipt_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _additive_artifact()
    backend = MegatronBackend(path=str(tmp_path))
    observed: dict[str, Any] = {}

    class _Service:
        async def committed_distillation_step(self, **kwargs: Any) -> int:
            observed.update(kwargs)
            return 8

    async def _get_service(_: Any) -> _Service:
        return _Service()

    monkeypatch.setattr(backend, "_get_service", _get_service)
    result = await backend.train(
        cast(Any, SimpleNamespace()),
        artifact,
        objectives=distill.TrainingObjectives(
            policy="cispo",
            distillation=distill.Loss(coefficient=0.4),
        ),
        epsilon=0.2,
        epsilon_high=0.3,
        importance_sampling_level="token",
        idempotency_key="additive-key",
        save_checkpoint=False,
    )

    assert result.step == 8
    assert observed["objective"].coefficient == 0.4
    assert observed["objective"].policy is not None
    assert observed["objective"].policy.epsilon == 0.2
    assert observed["objective"].policy.epsilon_high == 0.3


@pytest.mark.asyncio
async def test_backend_rejects_non_durable_optimizer_interval_before_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = MegatronBackend(path=str(tmp_path))

    async def _get_service(_: Any) -> Any:
        raise AssertionError("invalid config must fail before service access")

    monkeypatch.setattr(backend, "_get_service", _get_service)
    with pytest.raises(ValueError, match="optimizer_save_interval=1"):
        await backend.train(
            cast(Any, SimpleNamespace()),
            _artifact(),
            objectives=distill.TrainingObjectives(distillation=distill.Loss()),
            idempotency_key="stable-key",
            optimizer_save_interval=2,
        )


@pytest.mark.parametrize(
    ("changed_config", "field"),
    [
        (TrainConfig(learning_rate=1e-5, optimizer_save_interval=1), "learning_rate"),
        (
            TrainConfig(
                grad_accumulation_sequences=2,
                optimizer_save_interval=1,
            ),
            "grad_accumulation_sequences",
        ),
    ],
)
def test_receipt_identity_rejects_changed_resolved_training_config(
    tmp_path: Path,
    changed_config: TrainConfig,
    field: str,
) -> None:
    path = tmp_path / "receipt.json"
    base_config = TrainConfig(optimizer_save_interval=1)
    common = {
        "idempotency_key": "stable",
        "expected_source_revision": 7,
        "preparation_id": "preparation",
        "payload_sha256": "payload",
        "objective": DistillationObjectiveConfig(coefficient=1.0),
    }
    original = MegatronService._distillation_receipt_binding(
        **common,
        config=base_config,
    )
    changed = MegatronService._distillation_receipt_binding(
        **common,
        config=changed_config,
    )
    MegatronService._write_json_atomic(
        path,
        {"binding": original, "state": "committed", "committed_step": 8},
    )

    with pytest.raises(ValueError, match="different distillation job"):
        MegatronService._read_distillation_receipt(
            cast(MegatronService, object()),
            path=path,
            binding=changed,
        )
    assert original["train_config"][field] != changed["train_config"][field]


def test_receipt_identity_rejects_changed_resolved_policy_config(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    common = {
        "idempotency_key": "stable",
        "expected_source_revision": 7,
        "preparation_id": "preparation",
        "payload_sha256": "payload",
        "config": TrainConfig(optimizer_save_interval=1),
    }
    original = MegatronService._distillation_receipt_binding(
        **common,
        objective=DistillationObjectiveConfig(
            coefficient=1.0,
            policy=CispoObjectiveConfig(epsilon=0.2),
        ),
    )
    changed = MegatronService._distillation_receipt_binding(
        **common,
        objective=DistillationObjectiveConfig(
            coefficient=1.0,
            policy=CispoObjectiveConfig(epsilon=0.4),
        ),
    )
    MegatronService._write_json_atomic(
        path,
        {"binding": original, "state": "committed", "committed_step": 8},
    )

    with pytest.raises(ValueError, match="different distillation job"):
        MegatronService._read_distillation_receipt(
            cast(MegatronService, object()),
            path=path,
            binding=changed,
        )


def test_receipt_reservation_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    binding = {"idempotency_key": "stable"}
    service = cast(MegatronService, object())

    MegatronService._reserve_distillation_receipt(
        service,
        path=path,
        binding=binding,
    )
    first_bytes = path.read_bytes()
    with pytest.raises(RuntimeError, match="already exists"):
        MegatronService._reserve_distillation_receipt(
            service,
            path=path,
            binding=binding,
        )

    assert path.read_bytes() == first_bytes
    assert json.loads(first_bytes) == {"binding": binding, "state": "pending"}


@pytest.mark.asyncio
async def test_two_distillation_updates_each_commit_optimizer_ready_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    disk = pack_prepared_batch(
        batch=artifact,
        payload=_validated_payload(artifact),
        sequence_length=8,
        output_dir=str(tmp_path / "packed"),
    )
    monkeypatch.setattr(
        MegatronService,
        "_validate_megatron_dependencies",
        lambda _self: None,
    )
    service = MegatronService(
        model_name="student",
        base_model="Qwen/Qwen3-0.6B",
        config={},
        output_dir=str(tmp_path),
        runtime_config=cast(
            Any,
            SimpleNamespace(
                topology=SimpleNamespace(tp=1, cp=1, pp=1, ep=1, etp=1),
            ),
        ),
    )
    service._latest_step = 7
    committed: list[tuple[int, int]] = []
    submitted: list[MegatronDistillationJob] = []

    monkeypatch.setattr(service, "_raise_if_child_failed", lambda: None)
    monkeypatch.setattr(service, "_data_parallel_world_size", lambda: 1)

    async def _prepare_for_training() -> str:
        return "/unused/source"

    monkeypatch.setattr(service, "_prepare_for_training", _prepare_for_training)
    monkeypatch.setattr(
        service,
        "_prepare_training_lora_dir",
        lambda _source, step: f"/unused/staging/{step}",
    )
    monkeypatch.setattr(
        service,
        "_create_megatron_job_paths",
        lambda: ("/unused/job.json", "/unused/job.log"),
    )
    monkeypatch.setattr(
        "art.megatron.service.write_megatron_job",
        lambda job, **_kwargs: submitted.append(job),
    )

    async def _completed_job(job: MegatronDistillationJob, **_kwargs: Any):
        yield {"event": LORA_READY_EVENT, "step": job.step}
        yield {"loss/distillation": 0.25}
        yield {
            "event": OPTIMIZER_READY_EVENT,
            "step": job.step,
            "world_size": 1,
        }

    monkeypatch.setattr("art.megatron.service.stream_megatron_job", _completed_job)

    async def _lora_ready(**kwargs: Any) -> str:
        return f"/unused/checkpoint/{kwargs['step']}"

    monkeypatch.setattr(service, "_handle_training_lora_ready", _lora_ready)

    async def _finish(**kwargs: Any) -> str:
        service._latest_step = int(kwargs["step"])
        return f"/unused/checkpoint/{kwargs['step']}"

    monkeypatch.setattr(service, "_finish_training_checkpoint", _finish)
    monkeypatch.setattr(
        service,
        "_commit_optimizer_checkpoint",
        lambda *, step, world_size: committed.append((step, world_size)),
    )
    config = TrainConfig(optimizer_save_interval=1)
    objective = DistillationObjectiveConfig(coefficient=1.0)

    async def _run(
        *,
        source_revision: int,
        key: str,
        current_lease: Any = None,
    ) -> list[dict[str, float]]:
        return [
            metrics
            async for metrics in service.train_distillation(
                disk,
                config,
                objective=objective,
                expected_source_revision=source_revision,
                idempotency_key=key,
                preparation_id=artifact.preparation_id,
                payload_sha256=artifact.payload_sha256,
                current_step_session_id=(
                    current_lease.session_id if current_lease is not None else None
                ),
                current_step_capability=(
                    current_lease.capability if current_lease is not None else None
                ),
            )
        ]

    current = await service.acquire_current_step(revision=7)
    first_metrics = await _run(
        source_revision=7,
        key="update-7",
        current_lease=current,
    )
    current_journal = service._writer_sessions.inspect()
    assert current_journal is not None
    assert current_journal["state"] == "committed"
    assert current_journal["result_revision"] == 8
    await service.release_current_step(
        session_id=current.session_id,
        capability=current.capability,
    )
    second_metrics = await _run(source_revision=8, key="update-8")

    assert first_metrics == [{"loss/distillation": 0.25}]
    assert second_metrics == [{"loss/distillation": 0.25}]
    assert [job.config.optimizer_save_interval for job in submitted] == [1, 1]
    assert committed == [(8, 1), (9, 1)]
    assert service._latest_step == 9
    for key, expected_step in (("update-7", 8), ("update-8", 9)):
        receipt = json.loads(service._distillation_receipt_path(key).read_text())
        assert receipt["state"] == "committed"
        assert receipt["committed_step"] == expected_step
        assert receipt["binding"]["train_config"] == config.model_dump(mode="json")


@pytest.mark.asyncio
async def test_megatron_legacy_rl_dispatch_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}

    async def _legacy_train(
        _: LocalBackend,
        model: Any,
        groups: Any,
        **kwargs: Any,
    ) -> LocalTrainResult:
        observed.update(model=model, groups=groups, kwargs=kwargs)
        return LocalTrainResult(step=3)

    monkeypatch.setattr(LocalBackend, "train", _legacy_train)
    monkeypatch.setattr(
        "art.megatron.backend.get_megatron_runtime_config",
        lambda: SimpleNamespace(packed_sequence_length=128),
    )
    backend = MegatronBackend(path=str(tmp_path))
    model = object()
    groups = [object()]

    result = await backend.train(cast(Any, model), cast(Any, groups), loss_fn="cispo")

    assert result.step == 3
    assert observed["model"] is model
    assert observed["groups"] is groups
    assert observed["kwargs"] == {
        "loss_fn": "cispo",
        "packed_sequence_length": 128,
    }
