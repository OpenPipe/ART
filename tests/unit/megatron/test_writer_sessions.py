from __future__ import annotations

from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from art.megatron.writer_sessions import (
    AmbiguousWriterSessionError,
    WriterBindingMismatchError,
    WriterBusyError,
    WriterJournalCorruptionError,
    WriterSessionStore,
    WriterSessionValidationError,
)


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _store(
    root: Path,
    *,
    model: str = "project/model",
    clock: FakeClock | None = None,
    secret: bytes = b"s" * 32,
) -> WriterSessionStore:
    return WriterSessionStore(
        root=root,
        model_identity=model,
        clock=clock or FakeClock(),
        secret_factory=lambda _size: secret,
    )


def _operation(name: str = "update-1") -> dict[str, object]:
    return {
        "idempotency_key": name,
        "preparation_id": f"prepared-{name}",
        "objectives": {"policy": "cispo", "distillation": "forward_kl"},
    }


def _crash_with_open_session(root: str) -> None:
    store = WriterSessionStore(
        root=Path(root),
        model_identity="project/model",
        clock=lambda: 0.0,
        secret_factory=lambda size: b"c" * size,
    )
    store.acquire_current_step(revision=3, ttl_s=1.0)


def test_current_and_ordinary_writers_share_one_os_gate(tmp_path: Path) -> None:
    first = _store(tmp_path)
    second = _store(tmp_path)
    current = first.acquire_current_step(revision=4, ttl_s=30.0)

    with pytest.raises(WriterBusyError, match="another process"):
        second.acquire_ordinary(
            revision=4,
            ttl_s=30.0,
            operation_identity=_operation(),
        )

    abandoned = first.release(current)
    assert abandoned.action == "abandoned_before_submit"
    ordinary = second.acquire_ordinary(
        revision=4,
        ttl_s=30.0,
        operation_identity=_operation(),
    )
    assert ordinary.fence == current.fence + 1
    assert second.release(ordinary).action == "ambiguous_after_bind"


def test_open_ordinary_failure_before_submission_is_safely_retryable(
    tmp_path: Path,
) -> None:
    first = _store(tmp_path)
    lease = first.acquire_ordinary_open(revision=4, ttl_s=30.0)

    assert first.release(lease).action == "abandoned_before_submit"
    retry = _store(tmp_path)
    replacement = retry.acquire_ordinary_open(revision=4, ttl_s=30.0)
    assert replacement.fence == lease.fence + 1
    retry.release(replacement)


def test_model_scoped_locks_do_not_block_another_model(tmp_path: Path) -> None:
    model_a = _store(tmp_path, model="project/a", secret=b"a" * 32)
    model_b = _store(tmp_path, model="project/b", secret=b"b" * 32)

    lease_a = model_a.acquire_current_step(revision=1, ttl_s=30.0)
    lease_b = model_b.acquire_current_step(revision=8, ttl_s=30.0)
    assert lease_a.fence == 1
    assert lease_b.fence == 1
    model_a.release(lease_a)
    model_b.release(lease_b)


def test_forged_stale_expired_and_reused_capabilities_are_rejected(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock=clock)
    lease = store.acquire_current_step(revision=2, ttl_s=10.0)

    forged_secret = replace(lease, capability=b"f" * 32)
    with pytest.raises(WriterSessionValidationError, match="secret"):
        store.bind(forged_secret, operation_identity=_operation())

    forged_model = replace(lease, model_identity="project/other")
    with pytest.raises(WriterSessionValidationError, match="model_identity"):
        store.bind(forged_model, operation_identity=_operation())

    renewed = store.heartbeat(lease, ttl_s=20.0)
    with pytest.raises(WriterSessionValidationError, match="expires_at"):
        store.bind(lease, operation_identity=_operation())

    clock.advance(21.0)
    with pytest.raises(WriterSessionValidationError, match="expired"):
        store.bind(renewed, operation_identity=_operation())

    assert store.release(renewed).action == "abandoned_before_submit"
    with pytest.raises(WriterSessionValidationError, match="not active"):
        store.heartbeat(renewed, ttl_s=10.0)


def test_binding_is_idempotent_but_mismatch_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = store.acquire_current_step(revision=5, ttl_s=30.0)
    operation = _operation()

    store.bind(lease, operation_identity=operation)
    store.bind(lease, operation_identity=dict(reversed(tuple(operation.items()))))
    with pytest.raises(WriterBindingMismatchError, match="different operation"):
        store.bind(lease, operation_identity=_operation("update-2"))

    assert store.release(lease).action == "ambiguous_after_bind"
    reopened = _store(tmp_path)
    assert reopened.recovery_status().action == "ambiguous_after_bind"
    with pytest.raises(AmbiguousWriterSessionError, match="receipt reconciliation"):
        reopened.acquire_current_step(revision=5, ttl_s=30.0)


def test_commit_allows_only_one_revision_transition_and_identical_replay(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lease = store.acquire_current_step(revision=7, ttl_s=30.0)
    operation = _operation()
    store.bind(lease, operation_identity=operation)

    with pytest.raises(WriterSessionValidationError, match="exactly one"):
        store.commit(
            lease,
            operation_identity=operation,
            result_revision=9,
        )
    with pytest.raises(WriterBindingMismatchError, match="different operation"):
        store.commit(
            lease,
            operation_identity=_operation("different"),
            result_revision=8,
        )

    store.commit(lease, operation_identity=operation, result_revision=8)
    store.commit(lease, operation_identity=operation, result_revision=8)
    assert store.release(lease).action == "closed_committed"

    with pytest.raises(WriterSessionValidationError, match="not active"):
        store.commit(lease, operation_identity=operation, result_revision=8)


def test_bound_owner_can_commit_known_result_after_heartbeat_expiry(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock=clock)
    lease = store.acquire_current_step(revision=7, ttl_s=5.0)
    operation = _operation()
    store.bind(lease, operation_identity=operation)
    clock.advance(6.0)

    with pytest.raises(WriterSessionValidationError, match="expired"):
        store.heartbeat(lease, ttl_s=5.0)
    store.commit(lease, operation_identity=operation, result_revision=8)
    assert store.release(lease).action == "closed_committed"


def test_fence_persists_across_reopen_and_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    first = _store(tmp_path)
    lease_1 = first.acquire_current_step(revision=0, ttl_s=30.0)
    first.release(lease_1)

    second = _store(tmp_path)
    lease_2 = second.acquire_current_step(revision=0, ttl_s=30.0)
    assert lease_2.fence == lease_1.fence + 1
    second.release(lease_2)

    second.journal_path.write_text("{broken", encoding="utf-8")
    fence_before = second.fence_path.read_text(encoding="ascii")
    corrupted = _store(tmp_path)
    with pytest.raises(WriterJournalCorruptionError, match="unreadable"):
        corrupted.acquire_current_step(revision=0, ttl_s=30.0)
    assert corrupted.fence_path.read_text(encoding="ascii") == fence_before


def test_fence_corruption_is_never_reset(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = store.acquire_current_step(revision=0, ttl_s=30.0)
    store.release(lease)
    store.fence_path.write_text("not-an-integer", encoding="ascii")

    reopened = _store(tmp_path)
    with pytest.raises(WriterJournalCorruptionError, match="fencing counter"):
        reopened.acquire_current_step(revision=0, ttl_s=30.0)
    assert reopened.fence_path.read_text(encoding="ascii") == "not-an-integer"


def test_process_crash_releases_file_lock_and_expired_open_is_abandoned(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_with_open_session,
        args=(str(tmp_path),),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0

    clock = FakeClock(now=2.0)
    recovered = _store(tmp_path, clock=clock)
    assert recovered.recovery_status().action == "abandoned_before_submit"
    new_lease = recovered.acquire_current_step(revision=3, ttl_s=10.0)
    assert new_lease.fence == 2
    recovered.release(new_lease)


def test_crash_after_bind_is_ambiguous_but_committed_crash_is_recoverable(
    tmp_path: Path,
) -> None:
    bound_owner = _store(tmp_path)
    bound = bound_owner.acquire_current_step(revision=10, ttl_s=30.0)
    bound_owner.bind(bound, operation_identity=_operation())
    bound_owner._release_gate(bound.session_id)

    bound_reopen = _store(tmp_path)
    with pytest.raises(AmbiguousWriterSessionError):
        bound_reopen.acquire_current_step(revision=10, ttl_s=30.0)

    other_root = tmp_path / "committed"
    committed_owner = _store(other_root)
    committed = committed_owner.acquire_current_step(revision=10, ttl_s=30.0)
    committed_owner.bind(committed, operation_identity=_operation())
    committed_owner.commit(
        committed,
        operation_identity=_operation(),
        result_revision=11,
    )
    committed_owner._release_gate(committed.session_id)

    committed_reopen = _store(other_root)
    next_lease = committed_reopen.acquire_current_step(revision=11, ttl_s=30.0)
    assert next_lease.fence == committed.fence + 1
    committed_reopen.release(next_lease)


@pytest.mark.parametrize("kind", ["current_step", "ordinary"])
def test_exact_committed_receipt_reconciles_bound_crash_before_later_writer(
    tmp_path: Path,
    kind: str,
) -> None:
    operation = _operation()
    owner = _store(tmp_path)
    if kind == "current_step":
        crashed = owner.acquire_current_step(revision=10, ttl_s=30.0)
        owner.bind(crashed, operation_identity=operation)
    else:
        crashed = owner.acquire_ordinary(
            revision=10,
            ttl_s=30.0,
            operation_identity=operation,
        )
    owner._release_gate(crashed.session_id)
    observed: list[tuple[dict[str, object], int]] = []

    def _resolve(
        candidate: dict[str, Any],
        result_revision: int,
    ) -> bool:
        observed.append((candidate, result_revision))
        return candidate == operation and result_revision == 11

    recovered = WriterSessionStore(
        root=tmp_path,
        model_identity="project/model",
        clock=FakeClock(),
        secret_factory=lambda _size: b"r" * 32,
        committed_operation_resolver=_resolve,
    )
    later = recovered.acquire_ordinary_open(revision=11, ttl_s=30.0)

    assert observed == [(operation, 11)]
    assert later.fence == crashed.fence + 1
    recovered.release(later)


def test_explicit_reconciliation_closes_only_exact_committed_operation(
    tmp_path: Path,
) -> None:
    operation = _operation()
    owner = _store(tmp_path)
    crashed = owner.acquire_current_step(revision=4, ttl_s=30.0)
    owner.bind(crashed, operation_identity=operation)
    owner._release_gate(crashed.session_id)

    recovered = WriterSessionStore(
        root=tmp_path,
        model_identity="project/model",
        clock=FakeClock(),
        committed_operation_resolver=(
            lambda candidate, result: candidate == operation and result == 5
        ),
    )
    result = recovered.reconcile()

    assert result.action == "closed_committed"
    journal = recovered.inspect()
    assert journal is not None
    assert journal["state"] == "closed"
    assert journal["operation_identity"] == operation
    assert journal["result_revision"] == 5
    assert journal["close_reason"] == "recovered_committed"


def test_missing_committed_receipt_keeps_bound_crash_fail_closed(
    tmp_path: Path,
) -> None:
    operation = _operation()
    owner = _store(tmp_path)
    crashed = owner.acquire_current_step(revision=4, ttl_s=30.0)
    owner.bind(crashed, operation_identity=operation)
    owner._release_gate(crashed.session_id)

    recovered = WriterSessionStore(
        root=tmp_path,
        model_identity="project/model",
        committed_operation_resolver=lambda _candidate, _result: False,
    )

    assert recovered.reconcile().action == "ambiguous_after_bind"
    assert recovered.recovery_status().action == "ambiguous_after_bind"
    with pytest.raises(AmbiguousWriterSessionError, match="receipt reconciliation"):
        recovered.acquire_ordinary_open(revision=5, ttl_s=30.0)
    journal = recovered.inspect()
    assert journal is not None
    assert journal["state"] == "bound"
    assert journal["result_revision"] is None


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (WriterBindingMismatchError("receipt binding mismatch"), "binding mismatch"),
        (
            WriterSessionValidationError("receipt revision mismatch"),
            "revision mismatch",
        ),
    ],
)
def test_receipt_mismatch_during_reconciliation_fails_closed(
    tmp_path: Path,
    failure: Exception,
    message: str,
) -> None:
    owner = _store(tmp_path)
    crashed = owner.acquire_ordinary(
        revision=4,
        ttl_s=30.0,
        operation_identity=_operation(),
    )
    owner._release_gate(crashed.session_id)

    def _reject(
        _candidate: dict[str, Any],
        _result_revision: int,
    ) -> bool:
        raise failure

    recovered = WriterSessionStore(
        root=tmp_path,
        model_identity="project/model",
        committed_operation_resolver=_reject,
    )

    with pytest.raises(type(failure), match=message):
        recovered.acquire_current_step(revision=5, ttl_s=30.0)
    assert recovered.recovery_status().action == "ambiguous_after_bind"
    assert recovered.fence_path.read_text(encoding="ascii") == str(crashed.fence)


def test_heartbeat_is_durable_and_old_lease_is_fenced(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock=clock)
    original = store.acquire_current_step(revision=6, ttl_s=5.0)
    clock.advance(3.0)
    renewed = store.heartbeat(original, ttl_s=20.0)

    journal = store.inspect()
    assert journal is not None
    assert journal["heartbeat_at"] == 103.0
    assert journal["expires_at"] == 123.0
    with pytest.raises(WriterSessionValidationError, match="expires_at"):
        store.bind(original, operation_identity=_operation())
    store.bind(renewed, operation_identity=_operation())
    assert store.release(renewed).action == "ambiguous_after_bind"


def test_capability_secret_is_never_persisted(tmp_path: Path) -> None:
    secret = b"plain-text-capability-secret!!###"
    assert len(secret) >= 32
    store = _store(tmp_path, secret=secret)
    lease = store.acquire_current_step(revision=1, ttl_s=30.0)

    journal_bytes = store.journal_path.read_bytes()
    journal = json.loads(journal_bytes)
    assert secret not in journal_bytes
    assert secret.hex().encode() not in journal_bytes
    assert lease.capability == secret
    assert journal["capability_sha256"] != secret.decode()
    assert "capability" not in journal
    store.release(lease)


def test_context_managers_surface_ambiguous_exit_and_close_committed_work(
    tmp_path: Path,
) -> None:
    current_store = _store(tmp_path / "current")
    with pytest.raises(AmbiguousWriterSessionError, match="uncommitted"):
        with current_store.current_step(revision=2, ttl_s=30.0) as lease:
            current_store.bind(lease, operation_identity=_operation())
    assert current_store.recovery_status().action == "ambiguous_after_bind"

    ordinary_store = _store(tmp_path / "ordinary")
    operation = _operation()
    with ordinary_store.ordinary(
        revision=2,
        ttl_s=30.0,
        operation_identity=operation,
    ) as lease:
        ordinary_store.commit(
            lease,
            operation_identity=operation,
            result_revision=3,
        )
    assert ordinary_store.recovery_status().action == "none"
