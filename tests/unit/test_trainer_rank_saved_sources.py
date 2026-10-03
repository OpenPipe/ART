"""Remembered immutable sources survive every disposable cache tier."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
import threading
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import (
    MaterializedCheckpoint,
    TrainerRank,
    TrainerRankSlotStateError,
    _checkpoint,
)
from art.trainer_rank._impl import _CheckpointSlot
from tests.unit.test_trainer_rank_validation import _runtime, _slot_ref


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trainer = TrainerRank(_runtime())
    monkeypatch.setattr(trainer, "_slot_ref", _slot_ref)
    entered, exited, loads = [], [], []

    def source(name: str, value: int):
        @contextmanager
        def prepare():
            root = tmp_path / name
            # A deleted disk tier is retrieved again from the immutable source.
            root.write_text(str(value))
            entered.append(name)
            try:
                yield cast(Any, root)
            finally:
                exited.append(name)

        trainer._register_checkpoint_source(name, "immutable:" + name, prepare)

    def load(rank, root, name, *, forward_only=False, **_ownership):
        assert entered.count(name) == exited.count(name) + 1
        loads.append(name)
        rank._checkpoint_slots[name] = _CheckpointSlot(
            params=(
                torch.nn.Parameter(
                    torch.tensor(float(root.read_text())),
                    requires_grad=not forward_only,
                ),
            ),
            config=cast(Any, {}),
            snapshot=forward_only,
        )

    monkeypatch.setattr(_checkpoint, "load_checkpoint", load)
    return trainer, source, entered, exited, loads


def test_registration_is_lazy_idempotent_and_independent_of_mutation_queue(
    tmp_path, monkeypatch
):
    trainer, source, entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        _checkpoint,
        "_gather",
        lambda *_: pytest.fail("registration entered a collective"),
    )
    with trainer._checkpoint_mutation_lock:
        worker = threading.Thread(target=lambda: source("saved", 3))
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
    source("saved", 3)
    assert not entered and not trainer._checkpoint_prefetches
    assert list(trainer._checkpoint_sources) == ["saved"]
    with pytest.raises(TrainerRankSlotStateError, match="source changed"):
        trainer._register_checkpoint_source("saved", "changed", lambda: None)


def test_eviction_reload_preserves_immutable_bytes_and_live_training(
    tmp_path, monkeypatch
):
    trainer, source, entered, exited, loads = fixture(tmp_path, monkeypatch)
    mutable = _CheckpointSlot(params=(torch.nn.Parameter(torch.tensor(9.0)),))
    trainer._checkpoint_slots["training"] = mutable
    trainer._set_default_slot(_slot_ref("training"))
    source("saved", 3)
    trainer._ensure_checkpoint_slots(("saved",))
    assert trainer._checkpoint_slots["saved"].params[0].item() == 3
    assert not trainer._checkpoint_slots["saved"].params[0].requires_grad
    trainer._discard_snapshot_checkpoint("saved")
    (tmp_path / "saved").unlink()
    mutable.params[0].data.add_(1)
    trainer._ensure_checkpoint_slots(("saved",))
    assert trainer._checkpoint_slots["saved"].params[0].item() == 3
    assert trainer._checkpoint_slots["training"] is mutable
    assert mutable.params[0].item() == 10
    assert trainer._default_slot_ref.name == "training"
    assert entered == exited == loads == ["saved", "saved"]
    assert not trainer._checkpoint_prefetches


def test_saved_public_load_and_push_reuse_frozen_slot(tmp_path, monkeypatch):
    trainer, source, _entered, _exited, loads = fixture(tmp_path, monkeypatch)
    source("saved", 3)
    trainer.load_checkpoint("saved")
    slot = trainer._checkpoint_slots["saved"]
    trainer.load_checkpoint("saved")
    trainer._push_checkpoint("saved")
    trainer.pop_checkpoint()
    assert trainer._checkpoint_slots["saved"] is slot and loads == ["saved"]
    with pytest.raises(TrainerRankSlotStateError, match="forward-only"):
        trainer._guard_slot_can_load(_slot_ref("saved"))


async def test_saved_prefetch_is_already_admitted_without_retaining_cpu_result(
    tmp_path, monkeypatch
):
    trainer, source, entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    source("saved", 3)
    await trainer.prefetch_checkpoints("saved")
    assert not entered and not trainer._checkpoint_prefetches
    trainer._ensure_checkpoint_slots(("saved",))
    assert entered == ["saved"]


def test_forward_only_preparation_skips_custom_optimizer_tensor_loading(
    tmp_path, monkeypatch
):
    import json

    from safetensors.torch import save_file

    from tests.unit.test_trainer_rank_validation import _canonical_checkpoint

    root = tmp_path / "saved"
    manifest = _canonical_checkpoint(root)
    record = dict(
        kind="parameter",
        tensor_keys=["temperature"],
        trainable_keys=["temperature"],
        parameter_aliases=[["temperature"]],
        buffer_aliases=[],
        persistent_buffer_keys=[],
    )
    manifest["format_version"] = 3
    manifest["custom_tensors"] = cast(Any, {"temperature": record})
    save_file({"temperature": torch.tensor(3.0)}, root / "custom_tensors.safetensors")
    save_file(
        {
            f"{component}/temperature": torch.tensor(1.0)
            for component in ("master", "exp_avg", "exp_avg_sq", "step")
        },
        root / "optimizer/custom.safetensors",
    )
    for relative in ("custom_tensors.safetensors", "optimizer/custom.safetensors"):
        manifest["files"][relative] = _checkpoint._file_digest(root / relative)
    manifest["digest"] = _checkpoint._manifest_digest(manifest)
    (root / "checkpoint.json").write_text(json.dumps(manifest))
    original = __import__("safetensors.torch", fromlist=["load_file"]).load_file
    reads = []

    def load(path, **kwargs):
        reads.append(Path(path).relative_to(root).as_posix())
        return original(path, **kwargs)

    monkeypatch.setattr("safetensors.torch.load_file", load)
    prepared = _checkpoint.prepare_checkpoint(str(root), forward_only=True)
    assert prepared.custom is not None and not prepared.custom.optimizer
    assert reads == ["custom_tensors.safetensors"]
    restored = _checkpoint.prepare_checkpoint(str(root))
    assert restored.custom is not None and restored.custom.optimizer
    # Even readonly loading authenticates the persisted optimizer files.
    (root / "optimizer/custom.safetensors").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="file digest mismatch"):
        _checkpoint.prepare_checkpoint(str(root), forward_only=True)


def test_mutable_resident_collision_does_not_freeze_or_replace_weights(
    tmp_path, monkeypatch
):
    trainer, source, entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    slot = _CheckpointSlot(params=(torch.nn.Parameter(torch.tensor(19.0)),))
    trainer._checkpoint_slots["saved"] = slot
    with pytest.raises(TrainerRankSlotStateError, match="different ownership"):
        source("saved", 3)
    assert trainer._checkpoint_slots["saved"] is slot
    assert slot.params[0].item() == 19 and slot.params[0].requires_grad
    assert not entered and not trainer._checkpoint_sources


@pytest.mark.parametrize("fails", [False, True])
def test_source_admission_cannot_race_legacy_slot_publication(
    tmp_path, monkeypatch, fails
):
    trainer, remember, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    trainer._register_checkpoint_prefetch(
        "saved", str(tmp_path), lambda: cast(Any, tmp_path)
    ).result()
    entered, release = threading.Event(), threading.Event()
    errors = []

    def publish(rank, _prepared, name, *, forward_only=False):
        entered.set()
        assert release.wait(5)
        if fails:
            raise OSError("native load failed")
        rank._checkpoint_slots[name] = _CheckpointSlot(snapshot=forward_only)

    def load():
        try:
            trainer.load_checkpoint("saved")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(_checkpoint, "load_checkpoint", publish)
    worker = threading.Thread(target=load)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(TrainerRankSlotStateError, match="in-flight"):
            remember("saved", 3)
        assert not trainer._checkpoint_sources
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not trainer._checkpoint_slot_writes
    if fails:
        assert len(errors) == 1 and isinstance(errors[0], OSError)
        remember("saved", 3)
    else:
        assert not errors and not trainer._checkpoint_slots["saved"].snapshot
        with pytest.raises(TrainerRankSlotStateError, match="different ownership"):
            remember("saved", 3)


def test_native_snapshot_publication_and_remembered_names_exclude_each_other(
    tmp_path, monkeypatch
):
    trainer, remember, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    trainer._checkpoint_slots["training"] = _CheckpointSlot()
    entered, release = threading.Event(), threading.Event()
    errors = []

    def publish(rank, _source, destination):
        entered.set()
        assert release.wait(5)
        rank._checkpoint_slots[destination] = _CheckpointSlot(snapshot=True)
        return True

    def snapshot():
        try:
            trainer.snapshot_checkpoint("training", "saved")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(_checkpoint, "_snapshot_checkpoint", publish)
    worker = threading.Thread(target=snapshot)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(TrainerRankSlotStateError, match="in-flight"):
            remember("saved", 3)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not errors and not trainer._checkpoint_slot_writes
    with pytest.raises(TrainerRankSlotStateError, match="different ownership"):
        remember("saved", 3)
    remember("remembered", 3)
    with pytest.raises(TrainerRankSlotStateError, match="immutable source"):
        trainer.snapshot_checkpoint("training", "remembered")


def test_direct_native_load_cannot_replace_remembered_source(tmp_path, monkeypatch):
    native_load = _checkpoint.load_checkpoint
    trainer, remember, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    remember("saved", 3)
    with pytest.raises(TrainerRankSlotStateError, match="immutable source"):
        native_load(trainer, cast(Any, object()), "saved")
    assert not trainer._checkpoint_slot_writes


def test_source_admission_cannot_race_snapshot_discard_rollback(tmp_path, monkeypatch):
    trainer, remember, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    slot = _CheckpointSlot(snapshot=True)
    trainer._checkpoint_slots["saved"] = slot
    entered, release = threading.Event(), threading.Event()
    errors = []
    raise_distributed = _checkpoint.raise_distributed

    def pause(error, phase, group):
        if phase == "discard checkpoint snapshot":
            assert "saved" not in trainer._checkpoint_slots
            entered.set()
            assert release.wait(5)
            raise RuntimeError("discard failed")
        return raise_distributed(error, phase, group)

    def discard():
        try:
            trainer._discard_snapshot_checkpoint("saved")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(_checkpoint, "raise_distributed", pause)
    worker = threading.Thread(target=discard)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(TrainerRankSlotStateError, match="in-flight"):
            remember("saved", 3)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and len(errors) == 1
    assert trainer._checkpoint_slots["saved"] is slot
    assert not trainer._checkpoint_sources and not trainer._checkpoint_slot_writes


def test_public_load_reserves_ownership_before_collective_branch(tmp_path, monkeypatch):
    trainer, remember, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    trainer._register_checkpoint_prefetch(
        "saved", str(tmp_path), lambda: cast(Any, tmp_path)
    ).result()
    entered, release = threading.Event(), threading.Event()
    errors = []
    gather = _checkpoint._gather

    def pause(value, group):
        if isinstance(value, tuple) and value == ("saved", None, False):
            entered.set()
            assert release.wait(5)
        return gather(value, group)

    def load():
        try:
            trainer.load_checkpoint("saved")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(_checkpoint, "_gather", pause)
    monkeypatch.setattr(
        _checkpoint,
        "load_checkpoint",
        lambda rank, _source, name: rank._checkpoint_slots.update(
            {name: _CheckpointSlot()}
        ),
    )
    worker = threading.Thread(target=load)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(TrainerRankSlotStateError, match="in-flight"):
            remember("saved", 3)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not errors and not trainer._checkpoint_slot_writes


async def test_remembrance_detaches_eager_future_and_late_registration_is_admission(
    tmp_path, monkeypatch
):
    import gc
    import weakref

    trainer, remember, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    tensor = torch.ones(3)
    reference = weakref.ref(tensor)
    future = trainer._register_checkpoint_prefetch(
        "saved", str(tmp_path), lambda: cast(Any, tensor)
    )
    assert future.result() is tensor
    remember("saved", 3)
    assert (
        not trainer._checkpoint_prefetch_sources and not trainer._checkpoint_prefetches
    )
    assert reference() is not None  # The already admitted waiter still owns its result.
    del tensor, future
    gc.collect()
    assert reference() is None
    future = trainer._register_checkpoint_prefetch(
        "saved",
        str(tmp_path),
        lambda: pytest.fail("remembered source prepared eagerly"),
    )
    assert future.done() and future.result() is None
    await trainer._checkpoint_prefetch_waiter("saved")
    assert (
        not trainer._checkpoint_prefetch_sources and not trainer._checkpoint_prefetches
    )


def test_remembrance_preserves_other_waiters_and_inflight_future_custody(
    tmp_path, monkeypatch
):
    trainer, remember, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def prepare():
        entered.set()
        assert release.wait(5)
        return cast(Any, tmp_path)

    future = trainer._register_checkpoint_prefetch("saved", str(tmp_path), prepare)
    assert entered.wait(5)
    assert trainer._register_checkpoint_prefetch("other", str(tmp_path)) is future
    try:
        remember("saved", 3)
        assert list(trainer._checkpoint_prefetch_sources) == ["other"]
        assert list(trainer._checkpoint_prefetches.values()) == [future]
        remember("other", 3)
        assert not trainer._checkpoint_prefetches
        assert not future.cancelled() and not future.done()
    finally:
        release.set()
    assert future.result(timeout=5) == tmp_path


def test_failed_preparation_can_retry_and_always_releases_load_lease(
    tmp_path, monkeypatch
):
    trainer, _source, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    attempts, releases = [], []

    @contextmanager
    def prepare():
        attempts.append(1)
        try:
            if len(attempts) == 1:
                raise OSError("temporary retrieval failure")
            root = tmp_path / "saved"
            root.write_text("3")
            yield root
        finally:
            releases.append(1)

    trainer._register_checkpoint_source("saved", "immutable:saved", prepare)
    with pytest.raises(OSError, match="temporary retrieval failure"):
        trainer._ensure_checkpoint_slots(("saved",))
    monkeypatch.setattr(
        _checkpoint,
        "load_checkpoint",
        lambda rank, _source, name, **_: rank._checkpoint_slots.update(
            {name: _CheckpointSlot(snapshot=True)}
        ),
    )
    trainer._ensure_checkpoint_slots(("saved",))
    assert len(attempts) == len(releases) == 2
    assert "saved" in trainer._checkpoint_sources
    assert not trainer._checkpoint_prefetches


@pytest.mark.parametrize("protected", ["selected", "pushed", "graph", "custom"])
def test_residency_cache_evicts_only_safe_unrequested_snapshots(
    tmp_path, monkeypatch, protected
):
    trainer, source, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    trainer._checkpoint_snapshot_cache_size = 2
    for name in ("held", "evict", "new"):
        source(name, 3)
    trainer._ensure_checkpoint_slots(("held", "evict"))
    if protected == "selected":
        trainer._set_default_slot(_slot_ref("held"))
    elif protected == "pushed":
        trainer._push_checkpoint("held")
    elif protected == "custom":
        trainer._checkpoint_slots["held"].custom["handle"] = cast(Any, object())
    else:
        trainer._has_live_slot_graph = lambda ref: ref.name == "held"
    trainer._ensure_checkpoint_slots(("new",))
    assert set(trainer._checkpoint_slots) == {"held", "new"}
    assert set(trainer._checkpoint_sources) == {"held", "evict", "new"}
    trainer._ensure_checkpoint_slots(("evict",))
    assert set(trainer._checkpoint_slots) == {"held", "evict"}


def test_multi_snapshot_request_can_exceed_soft_residency_limit(tmp_path, monkeypatch):
    trainer, source, _entered, _exited, _loads = fixture(tmp_path, monkeypatch)
    trainer._checkpoint_snapshot_cache_size = 1
    for name in ("one", "two", "three"):
        source(name, 3)
    trainer._ensure_checkpoint_slots(("one", "two", "three"))
    assert len(trainer._checkpoint_slots) == 3
    trainer._ensure_checkpoint_slots(("three",))
    assert set(trainer._checkpoint_slots) == {"three"}


def _mixed_residency_worker(index: int, directory: str) -> None:
    root = Path(directory) / str(index)
    root.mkdir()
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=index,
        world_size=2,
        init_method=f"file://{directory}/rendezvous",
        timeout=timedelta(seconds=15),
    )
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            trainer, source, entered, exited, loads = fixture(root, monkeypatch)
            source("saved", 3)
            trainer._ensure_checkpoint_slots(("saved",))
            # Simulate a lost cache tier on just one process.
            if index == 1:
                trainer._checkpoint_slots.pop("saved")
                trainer._checkpoint_snapshot_lru.pop("saved")
            trainer._ensure_checkpoint_slots(("saved",))
            assert entered == exited == loads == ["saved", "saved"]
            assert trainer._checkpoint_slots["saved"].params[0].item() == 3
            if index == 1:
                trainer._checkpoint_slots.pop("saved")
            else:
                monkeypatch.setattr(trainer, "_has_live_slot_graph", lambda _ref: True)
            # Surviving graph-held state cannot be replaced to repair a miss.
            with pytest.raises(TrainerRankSlotStateError, match="live outputs"):
                trainer._ensure_checkpoint_slots(("saved",))
            assert "saved" in trainer._checkpoint_sources
            trainer._register_checkpoint_prefetch(
                "partial", str(root), lambda: cast(Any, root)
            ).result()
            if index == 0:
                source("partial", 3)
            with pytest.raises(
                TrainerRankSlotStateError, match="source differs across ranks"
            ):
                trainer.load_checkpoint(MaterializedCheckpoint("partial", str(root)))
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_mixed_rank_saved_residency_reloads_collectively_and_protects_graphs(tmp_path):
    mp.spawn(_mixed_residency_worker, args=(str(tmp_path),), nprocs=2, join=True)
