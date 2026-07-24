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
    LORA_READY_EVENT,
    OPTIMIZER_READY_EVENT,
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

    async def _run(*, source_revision: int, key: str) -> list[dict[str, float]]:
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
            )
        ]

    first_metrics = await _run(source_revision=7, key="update-7")
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
