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
    DistillationObjectiveConfig,
    pack_prepared_batch,
    packed_distillation_tensors_from_dir,
    validate_standalone_forward_kl,
)
from art.megatron.runtime.jobs import (
    MegatronDistillationJob,
    dump_megatron_job,
    load_megatron_job,
)
from art.megatron.service import MegatronService
from art.types import LocalTrainResult, TrainConfig


def _artifact(*, revision: int = 7) -> distill.PreparedTrainingBatch:
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
            teacher_revision="frozen",
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

    class _Service:
        async def committed_distillation_step(self, **_: Any) -> int:
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
