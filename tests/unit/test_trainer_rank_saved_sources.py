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

from art.trainer_rank import TrainerRank, TrainerRankSlotStateError, _checkpoint
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

    def load(rank, root, name, *, forward_only=False):
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
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_mixed_rank_saved_residency_reloads_collectively_and_protects_graphs(tmp_path):
    mp.spawn(_mixed_residency_worker, args=(str(tmp_path),), nprocs=2, join=True)
