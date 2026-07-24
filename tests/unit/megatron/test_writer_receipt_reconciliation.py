from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from art.megatron.backend import MegatronBackend
from art.megatron.distillation import DistillationObjectiveConfig
from art.megatron.optimizer_state import (
    PreparedCheckpointCommit,
    commit_optimizer_generation,
    optimizer_generation_files,
    read_optimizer_commit,
    write_prepared_checkpoint_commit,
)
from art.megatron.runtime.jobs import LORA_READY_EVENT, OPTIMIZER_READY_EVENT
from art.megatron.service import MegatronService
from art.megatron.writer_sessions import (
    AmbiguousWriterSessionError,
    WriterSessionValidationError,
)
from art.types import TrainConfig
from art.utils.output_dirs import get_step_checkpoint_dir


class _InjectedCrash(RuntimeError):
    pass


def _service(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> MegatronService:
    monkeypatch.setattr(
        MegatronService,
        "_validate_megatron_dependencies",
        lambda _self: None,
    )
    return MegatronService(
        model_name="student",
        base_model="Qwen/Qwen3-0.6B",
        config={},
        output_dir=str(root),
        runtime_config=cast(
            Any,
            SimpleNamespace(
                topology=SimpleNamespace(tp=1, cp=1, pp=1, ep=1, etp=1),
            ),
        ),
    )


def _bound_prepared_operation(
    service: MegatronService,
    *,
    revision: int = 4,
) -> tuple[dict[str, Any], dict[str, Any], Path, Any]:
    config = TrainConfig(optimizer_save_interval=1)
    objective = DistillationObjectiveConfig(coefficient=1.0)
    binding = service._distillation_receipt_binding(
        idempotency_key="update-4",
        expected_source_revision=revision,
        preparation_id="prepared-4",
        payload_sha256="a" * 64,
        objective=objective,
        config=config,
    )
    receipt_path = service._distillation_receipt_path("update-4")
    service._reserve_distillation_receipt(path=receipt_path, binding=binding)
    lease = service._writer_sessions.acquire_current_step(
        revision=revision,
        ttl_s=30.0,
    )
    operation = service._writer_operation_for_job(
        route="prepared",
        source_revision=revision,
        target_revision=revision + 1,
        identity={"receipt_binding": binding},
    )
    service._writer_sessions.bind(lease, operation_identity=operation)
    return operation, binding, receipt_path, lease


def _write_source_optimizer(service: MegatronService, revision: int) -> None:
    checkpoint = Path(get_step_checkpoint_dir(service.output_dir, revision))
    checkpoint.mkdir(parents=True, exist_ok=True)
    optimizer_dir = Path(service._get_optimizer_state_path())
    files = optimizer_generation_files(revision, 1)
    (optimizer_dir / files[0]).write_bytes(b"source optimizer")
    commit_optimizer_generation(
        str(optimizer_dir),
        step=revision,
        world_size=1,
        files=files,
    )


async def _run_with_injected_crash(
    service: MegatronService,
    monkeypatch: pytest.MonkeyPatch,
    *,
    boundary: str,
    sidecar_dir: Path,
) -> None:
    service._latest_step = 4
    sidecar_dir.mkdir()
    _write_source_optimizer(service, 4)
    optimizer_dir = Path(service._get_optimizer_state_path())
    monkeypatch.setattr(service, "_raise_if_child_failed", lambda: None)
    monkeypatch.setattr(service, "_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(service, "_status", lambda _message: None)
    monkeypatch.setattr(
        "art.megatron.service._validate_distillation_tensors_for_objective",
        lambda _tensors, _objective: None,
    )

    async def _prepare_for_training() -> str:
        return "/unused/source"

    staging_dir = Path(service._staging_lora_dir(5))

    def _prepare_training_lora_dir(_source: str, _step: int) -> str:
        staging_dir.mkdir(parents=True)
        (staging_dir / "adapter_model.safetensors").write_bytes(b"lora")
        return str(staging_dir)

    async def _finish_checkpoint(
        *,
        checkpoint_dir: str | None,
        staging_lora_path: str,
        step: int,
    ) -> str:
        assert checkpoint_dir is None
        target = Path(get_step_checkpoint_dir(service.output_dir, step))
        target.parent.mkdir(parents=True, exist_ok=True)
        Path(staging_lora_path).rename(target)
        service._latest_step = 5
        if boundary == "checkpoint_published":
            raise _InjectedCrash("crash after checkpoint publication")
        return str(target)

    async def _stream_job(*_args: Any, **_kwargs: Any):
        target_files = optimizer_generation_files(5, 1)
        (optimizer_dir / target_files[0]).write_bytes(b"target optimizer")
        yield {
            "event": LORA_READY_EVENT,
            "step": 5,
        }
        yield {
            "event": OPTIMIZER_READY_EVENT,
            "step": 5,
            "world_size": 1,
        }

    async def _aclose() -> None:
        return None

    monkeypatch.setattr(service, "_prepare_for_training", _prepare_for_training)
    monkeypatch.setattr(
        service,
        "_prepare_training_lora_dir",
        _prepare_training_lora_dir,
    )
    monkeypatch.setattr(
        service,
        "_create_megatron_job_paths",
        lambda: ("/unused/job.json", "/unused/job.log"),
    )
    monkeypatch.setattr(service, "_finish_training_checkpoint", _finish_checkpoint)
    monkeypatch.setattr(service, "_ensure_lora_adapter_config", lambda _path: None)
    from art.megatron import service as service_module

    original_write_marker = service_module.write_prepared_checkpoint_commit

    def _write_marker(path: str, marker: Any) -> None:
        original_write_marker(path, marker)
        if boundary == "outputs_ready" and marker.state == "outputs_ready":
            raise _InjectedCrash("crash after outputs-ready marker")

    monkeypatch.setattr(
        "art.megatron.service.write_prepared_checkpoint_commit",
        _write_marker,
    )
    monkeypatch.setattr(service, "aclose", _aclose)
    monkeypatch.setattr(
        "art.megatron.service.MegatronDistillationJob",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def _write_job(*_args: Any, **_kwargs: Any) -> None:
        if boundary == "submitted":
            raise _InjectedCrash("crash after submitted marker")

    monkeypatch.setattr("art.megatron.service.write_megatron_job", _write_job)
    monkeypatch.setattr("art.megatron.service.stream_megatron_job", _stream_job)

    original_commit_optimizer = service._commit_optimizer_checkpoint

    def _commit_optimizer(
        *,
        step: int,
        world_size: int,
        operation_identity: dict[str, Any] | None = None,
    ) -> None:
        assert (step, world_size) == (5, 1)
        original_commit_optimizer(
            step=step,
            world_size=world_size,
            operation_identity=operation_identity,
        )
        if boundary == "optimizer_published":
            raise _InjectedCrash("crash after optimizer publication")

    monkeypatch.setattr(service, "_commit_optimizer_checkpoint", _commit_optimizer)
    original_write_receipt = service._write_json_atomic

    def _write_receipt(path: Path, value: dict[str, Any]) -> None:
        original_write_receipt(path, value)
        if boundary == "receipt_committed" and value.get("state") == "committed":
            raise _InjectedCrash("crash after receipt commit")

    monkeypatch.setattr(service, "_write_json_atomic", _write_receipt)
    original_commit = service._writer_sessions.commit
    original_release = service._writer_sessions.release

    def _commit_writer(*args: Any, **kwargs: Any) -> None:
        original_commit(*args, **kwargs)
        if boundary == "writer_committed":
            raise _InjectedCrash("crash after writer commit")

    released_result: Any = None

    def _release_writer(*args: Any, **kwargs: Any):
        nonlocal released_result
        if (
            boundary == "writer_committed"
            and service._writer_sessions.recovery_status().action == "closed_committed"
        ):
            # Model process death drops the OS gate without writing "closed".
            service._writer_sessions._release_gate(args[0].session_id)
            return service._writer_sessions.recovery_status()
        if released_result is not None:
            return released_result
        result = original_release(*args, **kwargs)
        released_result = result
        if boundary == "writer_released":
            raise _InjectedCrash("crash after writer release")
        return result

    monkeypatch.setattr(service._writer_sessions, "commit", _commit_writer)
    monkeypatch.setattr(service._writer_sessions, "release", _release_writer)

    with pytest.raises(_InjectedCrash, match="crash after"):
        async for _metrics in service.train_distillation(
            cast(Any, {"dir": str(sidecar_dir)}),
            TrainConfig(optimizer_save_interval=1),
            objective=DistillationObjectiveConfig(coefficient=1.0),
            expected_source_revision=4,
            idempotency_key="update-4",
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
        ):
            pass


@pytest.mark.parametrize(
    "crash_boundary",
    [
        "checkpoint_published",
        "optimizer_published",
        "receipt_commit_started",
    ],
)
def test_pre_receipt_crash_boundaries_remain_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    _operation, _binding, _receipt_path, lease = _bound_prepared_operation(owner)
    # Checkpoint and optimizer markers are deliberately not reconciliation
    # authority. Until the exact receipt is atomic and committed, every crash
    # boundary must remain ambiguous.
    (tmp_path / crash_boundary).touch()
    owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    with pytest.raises(AmbiguousWriterSessionError, match="receipt reconciliation"):
        recovered._writer_sessions.acquire_ordinary_open(
            revision=5,
            ttl_s=30.0,
        )
    assert recovered._writer_sessions.recovery_status().action == (
        "ambiguous_after_bind"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("crash_boundary", "resolution"),
    [
        ("outputs_ready", "committed"),
        ("checkpoint_published", "committed"),
        ("optimizer_published", "committed"),
        ("receipt_committed", "committed"),
        ("writer_committed", "committed"),
        ("writer_released", "committed"),
    ],
)
async def test_train_crash_recovery_never_resubmits_worker_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
    resolution: str,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    await _run_with_injected_crash(
        owner,
        monkeypatch,
        boundary=crash_boundary,
        sidecar_dir=tmp_path / "sidecars",
    )
    crashed_journal = owner._writer_sessions.inspect()
    assert crashed_journal is not None
    assert crashed_journal["state"] in {"bound", "committed", "closed"}

    recovered = _service(tmp_path, monkeypatch)
    recovered._latest_step = 4
    activated: list[int] = []

    async def _activate(step: int) -> None:
        activated.append(step)
        recovered._latest_step = step

    async def _forbid_submission(*_args: Any, **_kwargs: Any):
        raise AssertionError("recovery must not submit another Megatron job")
        yield {}

    monkeypatch.setattr(
        recovered,
        "_activate_committed_distillation_step",
        _activate,
    )
    monkeypatch.setattr(
        "art.megatron.service.stream_megatron_job",
        _forbid_submission,
    )
    retry_sidecars = tmp_path / "retry-sidecars"
    retry_sidecars.mkdir()
    retry = recovered.train_distillation(
        cast(Any, {"dir": str(retry_sidecars)}),
        TrainConfig(optimizer_save_interval=1),
        objective=DistillationObjectiveConfig(coefficient=1.0),
        expected_source_revision=4,
        idempotency_key="update-4",
        preparation_id="prepared-4",
        payload_sha256="a" * 64,
    )
    metrics = [sample async for sample in retry]
    assert metrics == [
        {
            "distillation/idempotent_replay": 1.0,
            "distillation/committed_step": 5.0,
            "data/step_num_gradient_steps": 0.0,
        }
    ]
    assert activated == [5]
    assert not retry_sidecars.exists()
    commit = read_optimizer_commit(recovered._get_optimizer_state_path())
    assert commit is not None
    assert commit.schema_version == 2
    assert commit.step == 5
    assert commit.operation_identity == crashed_journal["operation_identity"]
    next_revision = 5

    later = recovered._writer_sessions.acquire_ordinary_open(
        revision=next_revision,
        ttl_s=30.0,
    )
    assert later.revision == next_revision
    assert later.fence == cast(int, crashed_journal["fence"]) + 1
    recovered._writer_sessions.release(later)


def test_receipt_commit_before_writer_commit_reconciles_without_resubmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    _operation, binding, receipt_path, lease = _bound_prepared_operation(owner)
    owner._write_json_atomic(
        receipt_path,
        {
            "binding": binding,
            "state": "committed",
            "committed_step": 5,
        },
    )
    owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    later = recovered._writer_sessions.acquire_ordinary_open(
        revision=5,
        ttl_s=30.0,
    )

    journal = recovered._writer_sessions.inspect()
    assert journal is not None
    assert later.fence == lease.fence + 1
    assert journal["state"] == "open"
    recovered._writer_sessions.release(later)
    with pytest.raises(WriterSessionValidationError, match="not active"):
        owner._writer_sessions.commit(
            lease,
            operation_identity=_operation,
            result_revision=5,
        )


@pytest.mark.parametrize("crash_boundary", ["writer_commit", "writer_release"])
def test_post_receipt_writer_boundaries_allow_exactly_one_later_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    operation, binding, receipt_path, lease = _bound_prepared_operation(owner)
    owner._write_json_atomic(
        receipt_path,
        {
            "binding": binding,
            "state": "committed",
            "committed_step": 5,
        },
    )
    owner._writer_sessions.commit(
        lease,
        operation_identity=operation,
        result_revision=5,
    )
    if crash_boundary == "writer_release":
        assert owner._writer_sessions.release(lease).action == "closed_committed"
    else:
        owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    later = recovered._writer_sessions.acquire_ordinary_open(
        revision=5,
        ttl_s=30.0,
    )
    assert later.fence == lease.fence + 1
    recovered._writer_sessions.release(later)


@pytest.mark.asyncio
async def test_idempotent_replay_reconciles_receipt_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    _operation, binding, receipt_path, lease = _bound_prepared_operation(owner)
    owner._write_json_atomic(
        receipt_path,
        {
            "binding": binding,
            "state": "committed",
            "committed_step": 5,
        },
    )
    owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    recovered._latest_step = 4
    activated: list[int] = []

    async def _activate(step: int) -> None:
        activated.append(step)
        recovered._latest_step = step

    monkeypatch.setattr(
        recovered,
        "_activate_committed_distillation_step",
        _activate,
    )
    sidecars = tmp_path / "replay-sidecars"
    sidecars.mkdir()
    metrics = [
        result
        async for result in recovered.train_distillation(
            cast(Any, {"dir": str(sidecars)}),
            TrainConfig(optimizer_save_interval=1),
            objective=DistillationObjectiveConfig(coefficient=1.0),
            expected_source_revision=4,
            idempotency_key="update-4",
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
        )
    ]
    assert activated == [5]

    assert metrics == [
        {
            "distillation/idempotent_replay": 1.0,
            "distillation/committed_step": 5.0,
            "data/step_num_gradient_steps": 0.0,
        }
    ]
    assert not sidecars.exists()
    assert recovered._writer_sessions.recovery_status().action == "none"
    later = recovered._writer_sessions.acquire_ordinary_open(
        revision=5,
        ttl_s=30.0,
    )
    recovered._writer_sessions.release(later)


@pytest.mark.asyncio
async def test_exact_committed_replay_uses_live_current_step_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    service._latest_step = 4
    objective = DistillationObjectiveConfig(coefficient=1.0)
    config = TrainConfig(optimizer_save_interval=1)
    binding = service._distillation_receipt_binding(
        idempotency_key="live-current",
        expected_source_revision=4,
        preparation_id="prepared-4",
        payload_sha256="a" * 64,
        objective=objective,
        config=config,
    )
    lease = await service.acquire_current_step(revision=4)
    operation = service._writer_operation_for_job(
        route="prepared",
        source_revision=4,
        target_revision=5,
        identity={"receipt_binding": binding},
    )
    service._writer_sessions.bind(lease, operation_identity=operation)
    service._writer_sessions.commit(
        lease,
        operation_identity=operation,
        result_revision=5,
    )
    service._current_writer_lease = lease
    service._latest_step = 5
    service._write_json_atomic(
        service._distillation_receipt_path("live-current"),
        {
            "binding": binding,
            "state": "committed",
            "committed_step": 5,
        },
    )

    assert (
        await service.committed_distillation_step(
            idempotency_key="live-current",
            expected_source_revision=4,
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
            objective=objective,
            config=config,
        )
        == 5
    )
    await service.release_current_step(
        session_id=lease.session_id,
        capability=lease.capability,
    )


@pytest.mark.asyncio
async def test_current_step_submitted_recovery_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    operation, _binding, _receipt_path, lease = _bound_prepared_operation(owner)
    _write_source_optimizer(owner, 4)
    staging = Path(owner._staging_lora_dir(5))
    staging.mkdir(parents=True)
    write_prepared_checkpoint_commit(
        staging,
        PreparedCheckpointCommit(
            state="submitted",
            step=5,
            operation_identity=operation,
        ),
    )
    owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="pending with an ambiguous outcome"):
        await recovered.committed_distillation_step(
            idempotency_key="update-4",
            expected_source_revision=4,
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
            objective=DistillationObjectiveConfig(coefficient=1.0),
            config=TrainConfig(optimizer_save_interval=1),
        )

    journal = recovered._writer_sessions.inspect()
    assert journal is not None
    assert journal["state"] == "bound"
    assert journal["close_reason"] is None
    assert journal["revision"] == 4
    assert journal["result_revision"] is None
    with pytest.raises(AmbiguousWriterSessionError, match="receipt reconciliation"):
        recovered._writer_sessions.acquire_current_step(
            revision=4,
            ttl_s=30.0,
        )


@pytest.mark.asyncio
async def test_pending_receipt_before_bind_is_recovered_without_poisoning_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    owner._latest_step = 4
    config = TrainConfig(optimizer_save_interval=1)
    objective = DistillationObjectiveConfig(coefficient=1.0)
    binding = owner._distillation_receipt_binding(
        idempotency_key="pre-bind-crash",
        expected_source_revision=4,
        preparation_id="prepared-4",
        payload_sha256="a" * 64,
        objective=objective,
        config=config,
    )
    lease = owner._writer_sessions.acquire_ordinary_open(
        revision=4,
        ttl_s=600.0,
    )
    receipt_path = owner._distillation_receipt_path("pre-bind-crash")
    owner._reserve_distillation_receipt(path=receipt_path, binding=binding)
    # Simulate process death: the OS gate disappears, but neither a bound
    # operation nor a submission marker was ever published.
    owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    recovered._latest_step = 4
    assert (
        await recovered.committed_distillation_step(
            idempotency_key="pre-bind-crash",
            expected_source_revision=4,
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
            objective=objective,
            config=config,
        )
        is None
    )

    monkeypatch.setattr(recovered, "_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        "art.megatron.service._validate_distillation_tensors_for_objective",
        lambda _tensors, _objective: None,
    )

    async def _observe_same_key_reservation() -> str:
        receipt = recovered._read_distillation_receipt(
            path=receipt_path,
            binding=binding,
        )
        assert receipt is not None
        assert receipt["state"] == "pending"
        raise _InjectedCrash("stop after exact-key re-reservation")

    monkeypatch.setattr(
        recovered,
        "_prepare_for_training",
        _observe_same_key_reservation,
    )
    sidecars = tmp_path / "retry-sidecars"
    sidecars.mkdir()
    with pytest.raises(_InjectedCrash, match="exact-key re-reservation"):
        async for _metrics in recovered.train_distillation(
            cast(Any, {"dir": str(sidecars)}),
            config,
            objective=objective,
            expected_source_revision=4,
            idempotency_key="pre-bind-crash",
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
        ):
            pass

    receipt = recovered._read_distillation_receipt(
        path=receipt_path,
        binding=binding,
    )
    assert receipt is not None
    assert receipt["state"] == "failed"
    assert receipt["failure"] == "_InjectedCrash"
    retry = recovered._writer_sessions.acquire_ordinary_open(
        revision=4,
        ttl_s=30.0,
    )
    recovered._writer_sessions.release(retry)


@pytest.mark.asyncio
async def test_exact_current_replacement_recovers_prebind_receipt_without_reentering_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    owner._latest_step = 4
    config = TrainConfig(optimizer_save_interval=1)
    objective = DistillationObjectiveConfig(coefficient=1.0)
    binding = owner._distillation_receipt_binding(
        idempotency_key="replacement-pre-bind",
        expected_source_revision=4,
        preparation_id="prepared-4",
        payload_sha256="a" * 64,
        objective=objective,
        config=config,
    )
    old_lease = await owner.acquire_current_step(revision=4)
    receipt_path = owner._distillation_receipt_path("replacement-pre-bind")
    owner._reserve_distillation_receipt(
        path=receipt_path,
        binding=binding,
        lease=old_lease,
    )
    # The old process dies after receipt reservation but before publishing a
    # marker or binding the writer operation.
    owner._writer_sessions._release_gate(old_lease.session_id)

    resumed = _service(tmp_path, monkeypatch)
    resumed._latest_step = 4
    recovery = resumed._writer_sessions.reconcile()
    assert recovery.action == "abandoned_before_submit"
    assert recovery.session_id == old_lease.session_id

    replacement = await resumed.acquire_current_step(revision=4)
    assert replacement.fence > old_lease.fence
    # This query must use the replacement lease as recovery evidence instead
    # of trying to recursively acquire the gate it already owns.
    assert (
        await resumed.committed_distillation_step(
            idempotency_key="replacement-pre-bind",
            expected_source_revision=4,
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
            objective=objective,
            config=config,
        )
        is None
    )

    monkeypatch.setattr(resumed, "_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(resumed, "_status", lambda _message: None)
    monkeypatch.setattr(
        "art.megatron.service._validate_distillation_tensors_for_objective",
        lambda _tensors, _objective: None,
    )
    source = tmp_path / "source-lora"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"lora")

    async def _prepare_for_training() -> str:
        return str(source)

    async def _aclose() -> None:
        return None

    observed_binding: dict[str, Any] | None = None

    def _observe_bound_submission(_job: Any, *, job_path: str) -> None:
        del job_path
        nonlocal observed_binding
        journal = resumed._writer_sessions.inspect_active(replacement)
        assert journal["state"] == "bound"
        operation = cast(dict[str, Any], journal["operation_identity"])
        observed_binding = cast(dict[str, Any], operation["identity"])[
            "receipt_binding"
        ]
        receipt = resumed._read_distillation_receipt(
            path=receipt_path,
            binding=binding,
        )
        assert receipt is not None
        assert receipt["state"] == "pending"
        assert receipt["writer_session"] == {
            "session_id": replacement.session_id,
            "revision": replacement.revision,
            "fence": replacement.fence,
        }
        raise _InjectedCrash("stop after replacement submission was bound")

    monkeypatch.setattr(resumed, "_prepare_for_training", _prepare_for_training)
    monkeypatch.setattr(resumed, "aclose", _aclose)
    monkeypatch.setattr(
        "art.megatron.service.write_megatron_job",
        _observe_bound_submission,
    )
    sidecars = tmp_path / "replacement-sidecars"
    sidecars.mkdir()
    disk_tensors = {
        "schema_version": 1,
        "dir": str(sidecars),
        "num_sequences": 1,
        "sequence_length": 1,
        "top_k_width": 1,
        "target_count": 1,
        "logical_vocab_size": 1,
        "tensors_sha256": "b" * 64,
    }
    with pytest.raises(_InjectedCrash, match="replacement submission was bound"):
        async for _metrics in resumed.train_distillation(
            cast(Any, disk_tensors),
            config,
            objective=objective,
            expected_source_revision=4,
            idempotency_key="replacement-pre-bind",
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
            current_step_session_id=replacement.session_id,
            current_step_capability=replacement.capability,
        ):
            pass
    assert observed_binding == binding


@pytest.mark.asyncio
async def test_live_current_bound_pending_receipt_is_never_recovered_as_unsubmitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    service._latest_step = 4
    config = TrainConfig(optimizer_save_interval=1)
    objective = DistillationObjectiveConfig(coefficient=1.0)
    binding = service._distillation_receipt_binding(
        idempotency_key="live-bound",
        expected_source_revision=4,
        preparation_id="prepared-4",
        payload_sha256="a" * 64,
        objective=objective,
        config=config,
    )
    lease = await service.acquire_current_step(revision=4)
    receipt_path = service._distillation_receipt_path("live-bound")
    service._reserve_distillation_receipt(
        path=receipt_path,
        binding=binding,
        lease=lease,
    )
    operation = service._writer_operation_for_job(
        route="prepared",
        source_revision=4,
        target_revision=5,
        identity={"receipt_binding": binding},
    )
    staging = Path(service._staging_lora_dir(5))
    staging.mkdir(parents=True)
    write_prepared_checkpoint_commit(
        staging,
        PreparedCheckpointCommit(
            state="submitted",
            step=5,
            operation_identity=operation,
        ),
    )
    service._writer_sessions.bind(lease, operation_identity=operation)

    with pytest.raises(RuntimeError, match="pending with an ambiguous outcome"):
        await service.committed_distillation_step(
            idempotency_key="live-bound",
            expected_source_revision=4,
            preparation_id="prepared-4",
            payload_sha256="a" * 64,
            objective=objective,
            config=config,
        )
    receipt = service._read_distillation_receipt(
        path=receipt_path,
        binding=binding,
    )
    assert receipt is not None
    assert receipt["state"] == "pending"
    assert service._writer_sessions.inspect_active(lease)["state"] == "bound"


@pytest.mark.asyncio
async def test_production_resume_reconciles_outputs_before_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    operation, _binding, _receipt_path, lease = _bound_prepared_operation(owner)
    _write_source_optimizer(owner, 4)
    staging = Path(owner._staging_lora_dir(5))
    staging.mkdir(parents=True)
    (staging / "adapter_model.safetensors").write_bytes(b"target lora")
    optimizer_dir = Path(owner._get_optimizer_state_path())
    files = optimizer_generation_files(5, 1)
    (optimizer_dir / files[0]).write_bytes(b"target optimizer")
    write_prepared_checkpoint_commit(
        staging,
        PreparedCheckpointCommit(
            state="outputs_ready",
            step=5,
            operation_identity=operation,
            world_size=1,
            files=files,
        ),
    )
    owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(recovered, "_ensure_lora_adapter_config", lambda _path: None)
    info = await recovered.prepare_resume_state()

    assert info.step == 5
    assert recovered._latest_step == 5
    assert info.latest_lora_step == 5
    assert info.optimizer_step == 5
    assert info.quarantined_lora_steps == ()
    assert Path(get_step_checkpoint_dir(str(tmp_path), 5)).is_dir()
    assert not (tmp_path / "unpaired_checkpoints").exists()
    receipt = recovered._read_distillation_receipt(
        path=recovered._distillation_receipt_path("update-4"),
        binding=cast(
            dict[str, Any],
            cast(dict[str, Any], operation["identity"])["receipt_binding"],
        ),
    )
    assert receipt is not None
    assert receipt["state"] == "committed"


@pytest.mark.asyncio
async def test_backend_get_step_delegates_resume_recovery_to_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = MegatronBackend(path=str(tmp_path))
    model = SimpleNamespace(
        trainable=True,
        project="project",
        _storage_name=lambda: "student",
    )
    output_dir = tmp_path / "project" / "models" / "student"
    calls: list[str] = []

    class _Service:
        async def prepare_resume_state(self) -> SimpleNamespace:
            calls.append("prepare")
            Path(get_step_checkpoint_dir(str(output_dir), 5)).mkdir(
                parents=True,
            )
            return SimpleNamespace(
                step=5,
                latest_lora_step=5,
                optimizer_step=5,
                used_unpaired_override=False,
                quarantined_lora_steps=(),
            )

    async def _get_service(_model: Any) -> _Service:
        return _Service()

    monkeypatch.setattr(backend, "_get_service", _get_service)
    assert await backend._get_step(cast(Any, model)) == 5
    assert calls == ["prepare"]


@pytest.mark.parametrize("failure", ["incomplete", "mismatch"])
def test_incomplete_or_mismatched_output_evidence_remains_fenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    operation, _binding, _receipt_path, lease = _bound_prepared_operation(owner)
    _write_source_optimizer(owner, 4)
    staging = Path(owner._staging_lora_dir(5))
    staging.mkdir(parents=True)
    marker_operation = operation
    if failure == "mismatch":
        marker_operation = owner._writer_operation_for_job(
            route="prepared",
            source_revision=4,
            target_revision=5,
            identity={"receipt_binding": {"idempotency_key": "different"}},
        )
    target_files = optimizer_generation_files(5, 1)
    write_prepared_checkpoint_commit(
        staging,
        PreparedCheckpointCommit(
            state="outputs_ready",
            step=5,
            operation_identity=marker_operation,
            world_size=1,
            files=target_files,
        ),
    )
    owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    if failure == "mismatch":
        with pytest.raises(
            WriterSessionValidationError,
            match="different operation",
        ):
            recovered._writer_sessions.reconcile()
    else:
        assert recovered._writer_sessions.reconcile().action == "ambiguous_after_bind"
    journal = recovered._writer_sessions.inspect()
    assert journal is not None
    assert journal["state"] == "bound"
    assert journal["result_revision"] is None
    optimizer_commit = read_optimizer_commit(recovered._get_optimizer_state_path())
    assert optimizer_commit is not None
    assert optimizer_commit.step == 4
    assert not Path(get_step_checkpoint_dir(recovered.output_dir, 5)).exists()


@pytest.mark.parametrize("mismatch", ["binding", "revision"])
def test_receipt_mismatch_never_unfences_bound_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    owner = _service(tmp_path, monkeypatch)
    _operation, binding, receipt_path, lease = _bound_prepared_operation(owner)
    committed_binding = dict(binding)
    committed_step = 5
    if mismatch == "binding":
        committed_binding["payload_sha256"] = "b" * 64
    else:
        committed_step = 6
    owner._write_json_atomic(
        receipt_path,
        {
            "binding": committed_binding,
            "state": "committed",
            "committed_step": committed_step,
        },
    )
    owner._writer_sessions._release_gate(lease.session_id)

    recovered = _service(tmp_path, monkeypatch)
    expected_error = (
        ValueError if mismatch == "binding" else WriterSessionValidationError
    )
    with pytest.raises(expected_error):
        recovered._writer_sessions.acquire_ordinary_open(
            revision=5,
            ttl_s=30.0,
        )
    assert recovered._writer_sessions.recovery_status().action == (
        "ambiguous_after_bind"
    )
