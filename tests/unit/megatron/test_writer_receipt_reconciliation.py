from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from art.megatron.distillation import DistillationObjectiveConfig
from art.megatron.runtime.jobs import OPTIMIZER_READY_EVENT
from art.megatron.service import MegatronService
from art.megatron.writer_sessions import (
    AmbiguousWriterSessionError,
    WriterSessionValidationError,
)
from art.types import TrainConfig


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


async def _run_with_injected_crash(
    service: MegatronService,
    monkeypatch: pytest.MonkeyPatch,
    *,
    boundary: str,
    sidecar_dir: Path,
) -> None:
    service._latest_step = 4
    sidecar_dir.mkdir()
    monkeypatch.setattr(service, "_raise_if_child_failed", lambda: None)
    monkeypatch.setattr(service, "_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(service, "_status", lambda _message: None)
    monkeypatch.setattr(
        "art.megatron.service._validate_distillation_tensors_for_objective",
        lambda _tensors, _objective: None,
    )

    async def _prepare_for_training() -> str:
        return "/unused/source"

    async def _finish_checkpoint(**_kwargs: Any) -> str:
        service._latest_step = 5
        if boundary == "checkpoint_published":
            raise _InjectedCrash("crash after checkpoint publication")
        return "/unused/checkpoint/5"

    async def _stream_job(*_args: Any, **_kwargs: Any):
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
        lambda _source, _step: "/unused/staging/5",
    )
    monkeypatch.setattr(
        service,
        "_create_megatron_job_paths",
        lambda: ("/unused/job.json", "/unused/job.log"),
    )
    monkeypatch.setattr(
        service,
        "_get_optimizer_state_path",
        lambda: "/unused/optimizer",
    )
    monkeypatch.setattr(service, "_finish_training_checkpoint", _finish_checkpoint)
    monkeypatch.setattr(service, "aclose", _aclose)
    monkeypatch.setattr(
        "art.megatron.service.MegatronDistillationJob",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "art.megatron.service.write_megatron_job", lambda *_a, **_k: None
    )
    monkeypatch.setattr("art.megatron.service.stream_megatron_job", _stream_job)

    def _commit_optimizer(*, step: int, world_size: int) -> None:
        assert (step, world_size) == (5, 1)
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
    ("crash_boundary", "recoverable"),
    [
        ("checkpoint_published", False),
        ("optimizer_published", False),
        ("receipt_committed", True),
        ("writer_committed", True),
        ("writer_released", True),
    ],
)
async def test_train_crash_ordering_uses_receipt_as_reconciliation_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
    recoverable: bool,
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
    if not recoverable:
        assert crashed_journal["state"] == "bound", crashed_journal

    recovered = _service(tmp_path, monkeypatch)
    if not recoverable:
        with pytest.raises(
            AmbiguousWriterSessionError,
            match="receipt reconciliation",
        ):
            recovered._writer_sessions.acquire_ordinary_open(
                revision=5,
                ttl_s=30.0,
            )
        return

    later = recovered._writer_sessions.acquire_ordinary_open(
        revision=5,
        ttl_s=30.0,
    )
    assert later.fence == 2
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

    assert metrics == [
        {
            "distill/idempotent_replay": 1.0,
            "distill/committed_step": 5.0,
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
