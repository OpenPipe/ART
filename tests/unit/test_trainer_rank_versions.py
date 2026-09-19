from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import gc
from pathlib import Path
import time
from types import SimpleNamespace
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from art.trainer_rank import TrainerRank, TrainerRankSlotStateError
from art.trainer_rank._impl import _CheckpointSlot


def _trainer() -> tuple[TrainerRank, torch.nn.Parameter]:
    trainer = TrainerRank.__new__(TrainerRank)
    parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    trainer._checkpoint_slots = {"student": _CheckpointSlot(params=(parameter,))}
    trainer.runtime = SimpleNamespace(model=[], optimizer=None)
    return trainer, parameter


@pytest.mark.parametrize("reentrant", (False, True))
def test_snapshot_recompute_routes_original_gradient_after_update(
    reentrant: bool,
) -> None:
    trainer, current = _trainer()
    version = trainer._capture_checkpoint_version("student")
    old = trainer._snapshot_parameter(current, version)
    x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    loss = checkpoint(lambda x: x * old.square(), x, use_reentrant=reentrant)
    with torch.no_grad():
        current.fill_(3)
    trainer._checkpoint_slots["student"].revision += 1
    with trainer._gradient_transaction():
        loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor(12.0, dtype=torch.float64))
    torch.testing.assert_close(x.grad, torch.tensor(4.0, dtype=torch.float64))
    assert current.item() == 3


def test_coupled_versions_accumulate_and_repeated_backward_routes_once() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    with torch.no_grad():
        current.fill_(3)
    trainer._checkpoint_slots["student"].revision += 1
    new = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    loss = (old.square() - new).square()
    with trainer._gradient_transaction():
        loss.backward(retain_graph=True)
    assert current.grad is not None
    assert current.grad.item() == 6
    with trainer._gradient_transaction():
        loss.backward()
    assert current.grad is not None
    assert current.grad.item() == 12
    assert old.grad is None and new.grad is None


def test_stale_backward_preserves_existing_current_gradient() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student"), 0
    )
    loss = old.square()
    current.grad = torch.tensor(7.0, dtype=torch.float64)
    trainer._checkpoint_slots["student"].revision += 1
    with pytest.raises(TrainerRankSlotStateError, match="staleness 1"):
        with trainer._gradient_transaction():
            loss.backward()
    assert current.grad.item() == 7


def test_failed_backward_discards_staged_gradients_and_releases_batch() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )

    def fail(_gradient: torch.Tensor) -> None:
        raise RuntimeError("local autograd failed")

    old.register_hook(fail)
    with pytest.raises(RuntimeError, match="local autograd failed"):
        with trainer._gradient_transaction():
            old.square().backward()
    assert current.grad is None
    assert trainer._version_state()._transaction is None


@pytest.mark.parametrize("reentrant", (False, True))
@pytest.mark.parametrize("transaction", (False, True))
def test_snapshot_backward_requires_atomic_scope_for_outer_failure(
    reentrant: bool, transaction: bool
) -> None:
    trainer, p = _trainer()
    q = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    trainer._checkpoint_slots["student"].params += (q,)
    version = trainer._capture_checkpoint_version("student")
    old_p = trainer._snapshot_parameter(p, version)
    old_q = trainer._snapshot_parameter(q, version, 0)
    loss = checkpoint(lambda x: x * old_p.square(), old_q, use_reentrant=reentrant)
    p.grad, q.grad = torch.full_like(p, 7), torch.full_like(q, 11)
    previous_p, previous_q = p.grad, q.grad
    trainer._checkpoint_slots["student"].revision += 1
    message = "staleness 1" if transaction else "requires TrainerRank.backward"
    with pytest.raises(RuntimeError, match=message):
        with trainer._gradient_transaction() if transaction else nullcontext():
            loss.backward()
    assert p.grad is previous_p and p.grad.item() == 7
    assert q.grad is previous_q and q.grad.item() == 11
    assert trainer._version_state()._origins == {}
    assert trainer._version_state()._transaction is None


def test_reentrant_backward_transaction_rolls_back_completed_nested_task() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    loss = checkpoint(lambda x: x * old.square(), x, use_reentrant=True)
    with pytest.raises(RuntimeError, match="later backward failed"):
        with trainer._gradient_transaction():
            loss.backward()
            assert current.grad is None
            raise RuntimeError("later backward failed")
    assert current.grad is None


def test_cotangent_batch_preflights_all_versions_before_any_mutation() -> None:
    trainer, current = _trainer()
    version = trainer._capture_checkpoint_version("student")
    trainer._checkpoint_slots["student"].revision += 1
    now = trainer._capture_checkpoint_version("student")
    with pytest.raises(TrainerRankSlotStateError, match="staleness"):
        trainer._commit_versioned_gradients(
            [
                (now, 0, current, torch.ones_like(current)),
                (version, 0, current, torch.ones_like(current)),
            ]
        )
    assert current.grad is None


def test_replacement_invalidates_origin_and_old_target() -> None:
    trainer, current = _trainer()
    version = trainer._capture_checkpoint_version("student")
    snapshot = trainer._snapshot_parameter(current, version)
    new = torch.nn.Parameter(torch.tensor(5.0, dtype=torch.float64))
    trainer._checkpoint_slots["student"] = _CheckpointSlot(params=(new,), generation=1)
    with pytest.raises(TrainerRankSlotStateError, match="replaced"):
        with trainer._gradient_transaction():
            snapshot.square().backward()
    assert current.grad is None and new.grad is None


def test_replaced_target_identity_rejects_entire_gradient_batch() -> None:
    trainer, current = _trainer()
    version = trainer._capture_checkpoint_version("student")
    replacement = torch.nn.Parameter(current.detach().clone())
    trainer._checkpoint_slots["student"].params = (replacement,)
    with pytest.raises(TrainerRankSlotStateError, match="target was replaced"):
        trainer._commit_versioned_gradients(
            [
                (version, 2, replacement, torch.ones_like(replacement)),
                (version, 2, current, torch.ones_like(current)),
            ]
        )
    assert current.grad is None and replacement.grad is None


def test_accumulated_origin_is_checked_before_optimizer_mutation() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student"), 0
    )
    with trainer._gradient_transaction():
        old.square().backward()
    trainer._checkpoint_slots["student"].revision += 1
    with pytest.raises(TrainerRankSlotStateError, match="staleness"):
        trainer._dynamic_optim_step(["student"], params={}, scale_grads={})
    assert current.grad is not None
    assert current.item() == 2 and current.grad.item() == 4
    trainer.zero_grad()
    trainer._version_state().validate_accumulated(["student"])


def test_snapshot_lifetime_follows_graph_references() -> None:
    trainer, current = _trainer()
    snapshot = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    reference = weakref.ref(snapshot)
    loss = snapshot.square()
    del snapshot
    gc.collect()
    assert reference() is not None
    with trainer._gradient_transaction():
        loss.backward()
    del loss
    gc.collect()
    assert reference() is None


def test_replay_capture_does_not_reset_origin_age() -> None:
    trainer, current = _trainer()
    origin = trainer._capture_checkpoint_version("student")
    trainer._checkpoint_slots["student"].revision = 2
    replay = trainer._snapshot_parameter(current, origin, 2)
    trainer._checkpoint_slots["student"].revision = 3
    with pytest.raises(TrainerRankSlotStateError, match="staleness 3"):
        with trainer._gradient_transaction():
            replay.square().backward()
    assert current.grad is None


def test_collective_preflight_failure_does_not_commit_local_gradients() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )

    def other_rank_failed(validate) -> None:
        validate()
        assert current.grad is None
        raise RuntimeError("another rank rejected stale gradients")

    with pytest.raises(RuntimeError, match="another rank"):
        with trainer._gradient_transaction(before_commit=other_rank_failed):
            old.square().backward()
    assert current.grad is None
    assert not trainer._version_state()._origins


def test_gradient_dtype_is_validated_before_any_accumulator_is_published() -> None:
    trainer, current = _trainer()
    other = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    trainer._checkpoint_slots["student"].params += (other,)
    version = trainer._capture_checkpoint_version("student")
    with pytest.raises(ValueError, match="dtype"):
        trainer._commit_versioned_gradients(
            [
                (version, 2, current, torch.ones_like(current)),
                (version, 2, other, torch.tensor(2.0, dtype=torch.float32)),
            ]
        )
    assert current.grad is None and other.grad is None


def test_nested_transaction_still_participates_in_collective_preflight() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    calls = []

    def collective(validate) -> None:
        validate()
        calls.append(True)
        assert current.grad is None

    with trainer._gradient_transaction():
        with trainer._gradient_transaction(before_commit=collective):
            old.square().backward()
        assert calls == [True]
        assert current.grad is None
    assert current.grad is not None
    assert current.grad.item() == 4


def test_caught_nested_backward_failure_invalidates_whole_transaction() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    with pytest.raises(RuntimeError, match="nested gradient transaction failed"):
        with trainer._gradient_transaction():
            with pytest.raises(RuntimeError, match="failed"):
                with trainer._gradient_transaction():
                    old.square().backward()
                    raise RuntimeError("failed")
    assert current.grad is None


def test_explicit_head_cotangents_join_model_gradient_transaction() -> None:
    trainer, current = _trainer()
    version = trainer._capture_checkpoint_version("student")
    old = trainer._snapshot_parameter(current, version)
    with pytest.raises(RuntimeError, match="model backward failed"):
        with trainer._gradient_transaction():
            trainer._commit_versioned_gradients(
                [(version, 2, current, torch.ones_like(current))]
            )
            old.square().backward()
            assert current.grad is None
            raise RuntimeError("model backward failed")
    assert current.grad is None
    with trainer._gradient_transaction():
        trainer._commit_versioned_gradients(
            [(version, 2, current, torch.ones_like(current))]
        )
        old.square().backward()
    assert current.grad.item() == 5


def _divergent_version_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer, current = _trainer()
        version = trainer._capture_checkpoint_version("student")
        if rank == 0:
            trainer._commit_versioned_gradients(
                [(version, 0, current, torch.full_like(current, 3))]
            )
        else:
            current.grad = torch.full_like(current, 3)
        trainer._checkpoint_slots["student"].revision = 1
        with pytest.raises(RuntimeError, match="staleness|Another rank failed"):
            trainer._dynamic_optim_step(["student"], params={}, scale_grads={})
        assert current.grad is not None
        assert current.item() == 2 and current.grad.item() == 3
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


def test_divergent_optimizer_provenance_rejects_collectively_before_mutation(
    tmp_path: Path,
) -> None:
    processes = mp.spawn(
        _divergent_version_worker,
        args=(f"file://{tmp_path / 'versions'}",),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if processes.join(timeout=1):
            return
    for process in processes.processes:
        process.terminate()
    pytest.fail("Divergent version preflight did not complete collectively")


def test_transaction_coalesces_many_children_but_keeps_each_origin() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    trainer._checkpoint_slots["student"].revision = 1
    new = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    current.grad = torch.full_like(current, 7)
    pointer = None
    with trainer._gradient_transaction():
        for _ in range(50):
            old.square().backward()
            new.sum().backward()
            batch = trainer._version_state()._transaction
            assert batch is not None
            assert len(batch.gradients) == 1 and len(batch.origins) == 2
            staged = batch.gradients[id(current)][1]
            if pointer is None:
                pointer = staged.data_ptr()
            assert staged.data_ptr() == pointer
            assert current.grad.item() == 7
    assert current.grad.item() == 257
    assert len(trainer._version_state()._origins["student"]) == 2
    assert batch.gradients == {} and batch.origins == set()


def test_retained_failed_transaction_traceback_releases_staging() -> None:
    trainer, current = _trainer()
    old = trainer._snapshot_parameter(
        current, trainer._capture_checkpoint_version("student")
    )
    saved_error = None
    reference = None
    try:
        with trainer._gradient_transaction():
            old.square().backward()
            batch = trainer._version_state()._transaction
            assert batch is not None
            reference = weakref.ref(batch.gradients[id(current)][1])
            raise RuntimeError("failed after staged child")
    except RuntimeError as exc:
        saved_error = exc
    assert saved_error is not None and saved_error.__traceback__ is not None
    gc.collect()
    assert reference is not None and reference() is None
    assert current.grad is None and trainer._version_state()._origins == {}


def test_csr_cotangent_rejected_before_any_gradient_publication() -> None:
    trainer, _ = _trainer()
    first, second = [torch.nn.Parameter(torch.ones(2, 2)) for _ in range(2)]
    first.grad = torch.full_like(first, 7)
    previous = first.grad
    trainer._checkpoint_slots["student"].params = (first, second)
    version = trainer._capture_checkpoint_version("student")
    with pytest.raises(ValueError, match="strided"):
        trainer._commit_versioned_gradients(
            [
                (version, 2, first, torch.ones_like(first)),
                (version, 2, second, torch.ones_like(second).to_sparse_csr()),
            ]
        )
    assert first.grad is previous and torch.all(first.grad == 7)
    assert second.grad is None and trainer._version_state()._origins == {}


def test_gradient_assignment_failure_rolls_back_earlier_publication() -> None:
    class RejectGradient(torch.nn.Parameter):
        def __setattr__(self, name: str, value: object) -> None:
            if name == "grad":
                raise RuntimeError("injected gradient publication failure")
            super().__setattr__(name, value)

    trainer, first = _trainer()
    second = RejectGradient(torch.ones_like(first))
    trainer._checkpoint_slots["student"].params += (second,)
    first.grad = torch.full_like(first, 7)
    previous = first.grad
    version = trainer._capture_checkpoint_version("student")
    with pytest.raises(RuntimeError, match="publication failure"):
        trainer._commit_versioned_gradients(
            [
                (version, 2, first, torch.ones_like(first)),
                (version, 2, second, torch.ones_like(second)),
            ]
        )
    assert first.grad is previous and first.grad.item() == 7
    assert second.grad is None and trainer._version_state()._origins == {}


def _transaction_exit_failure_worker(rank: int, rendezvous: str, nested: bool) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        trainer, current = _trainer()
        version = trainer._capture_checkpoint_version("student")
        trainer._commit_versioned_gradients(
            [(version, 2, current, torch.full_like(current, 7))]
        )
        previous = current.grad
        origins = trainer._version_state()._origins.copy()
        snapshot = trainer._snapshot_parameter(current, version)
        calls = []

        def coordinate(validate) -> None:
            calls.append(True)
            error = None
            try:
                validate()
            except BaseException as exc:
                error = exc
            errors = [None, None]
            dist.all_gather_object(errors, None if error is None else str(error))
            if error is not None:
                raise error
            if any(errors):
                raise RuntimeError(f"another rank failed: {errors}")

        saved_error = None
        reference = None
        try:
            with trainer._gradient_transaction() if nested else nullcontext():
                with trainer._gradient_transaction(before_commit=coordinate):
                    snapshot.square().backward()
                    batch = trainer._version_state()._transaction
                    assert batch is not None
                    reference = weakref.ref(batch.gradients[id(current)][1])
                    if rank == 0:
                        raise ValueError("injected second-child replay failure")
                    snapshot.sum().backward()
        except (ValueError, RuntimeError) as exc:
            saved_error = exc
        assert saved_error is not None and "second-child replay failure" in str(
            saved_error
        )
        if rank == 0:
            assert isinstance(saved_error, ValueError)
        assert calls == [True]
        assert current.grad is previous and current.grad is not None
        assert current.grad.item() == 7
        assert trainer._version_state()._origins == origins
        gc.collect()
        assert reference is not None and reference() is None
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("nested", [False, True])
def test_local_replay_failure_uses_same_collective_exit_phase_on_every_rank(
    tmp_path: Path,
    nested: bool,
) -> None:
    processes = mp.spawn(
        _transaction_exit_failure_worker,
        args=(f"file://{tmp_path / 'exit'}", nested),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if processes.join(timeout=1):
            return
    for process in processes.processes:
        process.terminate()
    pytest.fail("Transaction success/failure exit phases did not match")
