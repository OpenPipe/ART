"""Local rollback and publication readiness for checkpoint transactions."""

from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, cast
import weakref

import pytest
from test_trainer_rank_calibration_seed import _rank
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
import torch.multiprocessing as mp

from art.megatron.lora import LoRA
from art.trainer_rank import _calibration_protocol as protocol
from art.trainer_rank import _checkpoint
from art.trainer_rank._impl import _CheckpointSlot


def _worker(index, directory, mode, fault, existing=True, world=2, failed_rank=1):
    calibration = mode != "outside"
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        rank=index,
        world_size=world,
        init_method="file://" + str(Path(directory, "gloo")),
        timeout=timedelta(seconds=8),
    )
    group = dist.new_group(backend="gloo")
    assert group is not dist.group.WORLD
    rank = _rank()
    owner: Any = LoRA.__new__(LoRA)
    torch.nn.Module.__init__(owner)
    owner._slot_keys = {}
    owner._slot_modules = torch.nn.ModuleDict()
    rank.runtime.model[0].add_module("transaction_lora", owner)
    ref = rank._slot_ref("adapter")
    old: Any = torch.nn.Module()
    old.ref = ref
    old.weight = torch.nn.Parameter(torch.tensor([1.0]))
    old_state = None
    if existing:
        owner._slot_keys[ref] = "old"
        owner._slot_modules["old"] = old
        old_state = _CheckpointSlot(params=(old.weight,), config=None, revision=7)
        rank._checkpoint_slots["adapter"] = old_state
    references = []
    caught = []
    applying = False
    row: dict[str, Any] = {
        "injections": 0,
        "publications": 0,
        "ready_sequence": None,
        "votes": [],
        "wire": [],
    }
    real_commit = _checkpoint._commit_slot
    real_vote = protocol.Protocol._vote
    real_reduce, real_gather = dist.all_reduce, c10d.all_gather
    real_decode = c10d._tensor_to_object

    def checksum():
        if ref not in owner._slot_keys:
            return None
        tensor = owner._slot_modules[owner._slot_keys[ref]].weight.detach()
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

    before = checksum()

    def inject(label):
        if index == failed_rank and not row["injections"] and fault == label:
            row["injections"] += 1
            raise MemoryError("injected checkpoint transaction failure: " + label)

    def stage(name, *args, **kwargs):
        staged: Any = torch.nn.Module()
        staged.ref = rank._slot_ref(name)
        staged.weight = torch.nn.Parameter(torch.tensor([2.0]))
        references.extend([weakref.ref(staged), weakref.ref(staged.weight)])
        owner._slot_keys[staged.ref] = "staged"
        owner._slot_modules["staged"] = staged
        inject("stage")
        return 1

    def validate_config(*args):
        inject("config")

    def commit(*args):
        current = protocol.active()
        if current is not None:
            row["ready_sequence"] = current.sequence + 2
        inject("commit")
        result = real_commit(*args)
        if not calibration:
            row["publications"] += 1
        return result

    def vote(current, stage, *args, **kwargs):
        result = real_vote(current, stage, *args, **kwargs)
        row["votes"].append(
            dict(stage=stage, sequence=current.sequence, failed=bool(result[0][0]))
        )
        if (
            calibration
            and row["ready_sequence"] == current.sequence
            and stage == "entry"
        ):
            assert checksum() == before, "adapter published before its readiness vote"
        return result

    def record(op, tensor, group_arg):
        row["wire"].append(
            dict(
                op=op,
                dtype=str(tensor.dtype),
                shape=list(tensor.shape),
                group="checkpoint" if group_arg is group else "replica",
            )
        )

    def reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        record(str(op), tensor, kwargs.get("group"))
        return real_reduce(tensor, op=op, **kwargs)

    def gather(outputs, tensor, **kwargs):
        record("ALLGATHER", tensor, kwargs.get("group"))
        return real_gather(outputs, tensor, **kwargs)

    def decode(*args, **kwargs):
        if row["ready_sequence"] is not None:
            inject("ready")
        return real_decode(*args, **kwargs)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_checkpoint, "_ensure_group", lambda trainer: group)
            patch.setattr(rank, "_checkpoint_group", lambda: group)
            patch.setattr(rank, "_local_lora_adapter_templates", lambda: {})
            patch.setattr(
                rank,
                "_validate_checkpoint_adapter_config",
                lambda *a, **k: {"base_model_name_or_path": "model", "lora_alpha": 1.0},
            )
            patch.setattr(rank.runtime, "model_identifier", "model", raising=False)
            patch.setattr(rank, "_prepare_adapter_model", lambda *a, **k: {})
            patch.setattr(rank, "_guard_slot_can_load", lambda *a: None)
            patch.setattr(_checkpoint, "_load_adapter", lambda *a: {})
            patch.setattr(rank, "_load_checkpoint_slot", stage)
            patch.setattr(
                rank,
                "_validate_checkpoint_consistency",
                lambda name, *a: (
                    owner._slot_modules[owner._slot_keys[rank._slot_ref(name)]].weight,
                ),
            )
            patch.setattr(rank, "_validate_loaded_checkpoint_config", validate_config)
            patch.setattr(_checkpoint, "_commit_slot", commit)
            patch.setattr(protocol.Protocol, "_vote", vote)
            patch.setattr(dist, "all_reduce", reduce)
            patch.setattr(dist, "all_gather", gather)
            patch.setattr(c10d, "all_gather", gather)
            patch.setattr(c10d, "_tensor_to_object", decode)
            changes = getattr(_checkpoint, "_CheckpointChanges", None)
            if changes is not None:
                real_apply = changes.apply

                def apply(plan):
                    nonlocal applying
                    publication = any(
                        namespace is rank.__dict__
                        and key == "_checkpoint_slots"
                        and value.get("adapter") is not old_state
                        for namespace, key, value in plan.updates
                    )
                    if publication:
                        assert row["votes"][-1] == dict(
                            stage="entry", sequence=row["ready_sequence"], failed=False
                        ), "publication preceded cohort readiness"
                        row["publications"] += 1
                    applying = True
                    try:
                        return real_apply(plan)
                    finally:
                        applying = False

                patch.setattr(changes, "apply", apply)
                for factory in ("empty", "empty_like", "zeros", "tensor"):
                    original = getattr(torch, factory)

                    def allocate(*a, _original=original, **k):
                        assert not applying, (
                            "tensor allocation during reference publication/rollback"
                        )
                        return _original(*a, **k)

                    patch.setattr(torch, factory, allocate)
            source = _checkpoint.PreparedCheckpoint(
                Path(directory),
                {"base_model_name_or_path": "model", "lora_alpha": 1.0},
                (),
                None,
                "synthetic-digest",
            )
            began = time.monotonic()
            try:
                with protocol.Driver(rank) if calibration else nullcontext():
                    _checkpoint.load_checkpoint(rank, source, "adapter")
            except BaseException as exc:
                row.update(error=type(exc).__name__, message=str(exc))
                if calibration:
                    caught.append(
                        exc
                    )  # Keep causes/tracebacks alive during reclamation checks.
            gc.collect()
            row.update(
                slots=list(rank._checkpoint_slots),
                model_slots=[r.name for r in owner._slot_keys],
                original_slot=rank._checkpoint_slots.get("adapter") is old_state,
                weight=cast(
                    torch.Tensor, owner._slot_modules[owner._slot_keys[ref]].weight
                ).item()
                if ref in owner._slot_keys
                else None,
                checksum=checksum(),
                before_checksum=before,
                active=protocol.active() is not None,
                temporary_names=[
                    name
                    for name in rank._checkpoint_slots
                    if name.startswith("__art_loading_")
                ]
                + [
                    r.name
                    for r in owner._slot_keys
                    if r.name.startswith("__art_loading_")
                ],
                staged_alive=sum(item() is not None for item in references),
                elapsed=time.monotonic() - began,
            )
            try:
                row["outer"] = rank._recovery_reduce(
                    [float("error" in row)], op="MAX", sync_across_dp=True
                )
            except BaseException as exc:
                row["outer_error"] = str(exc)
    finally:
        Path(directory, f"rank{index}.json").write_text(json.dumps(row))
        dist.destroy_process_group()


def _run(tmp_path, mode, fault, existing=True, world=2, failed_rank=1):
    _rank()
    mp.set_forkserver_preload(
        ["art.trainer_rank._impl", "test_trainer_rank_calibration_seed"]
    )
    context = mp.start_processes(
        _worker,
        args=(str(tmp_path), mode, fault, existing, world, failed_rank),
        nprocs=world,
        join=False,
        start_method="forkserver",
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=0.1):
            assert time.monotonic() < deadline, (
                "checkpoint transaction exceeded 120 seconds"
            )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(world)]
    failed = fault != "none"
    assert all(
        not r["active"] and r.get("outer") == [float(failed)] and "outer_error" not in r
        for r in rows
    ), rows
    assert all(r["elapsed"] < 45 and not r["temporary_names"] for r in rows), rows
    assert all(r["wire"] == rows[0]["wire"] for r in rows), rows
    if failed:
        assert [r["injections"] for r in rows] == [
            int(i == failed_rank) for i in range(world)
        ], rows
        assert all(
            r["original_slot"]
            and r["checksum"] == r["before_checksum"]
            and r["staged_alive"] == 0
            for r in rows
        ), rows
        if mode != "outside":
            assert all(
                r["error"] == "RuntimeError"
                and r["message"] == "Calibration failed on one or more ranks"
                and r["publications"] == 0
                for r in rows
            ), rows
    else:
        assert all(
            "error" not in r
            and r["weight"] == 2.0
            and r["staged_alive"] == 2
            and r["publications"] == 1
            for r in rows
        ), rows
    assert all(r["checksum"] == rows[0]["checksum"] for r in rows), rows
    return rows


@pytest.mark.parametrize("fault", ["config", "commit"])
@pytest.mark.parametrize("mode", ["outside", "current"])
def test_failed_load_preserves_checkpoint_transaction(tmp_path, fault, mode):
    _run(tmp_path, mode, fault)


@pytest.mark.parametrize("mode", ["outside", "current"])
def test_fresh_adapter_commit_failure_leaves_no_publication(tmp_path, mode):
    _run(tmp_path, mode, "commit", False)


CASES = [
    (world, mode, existing, fault, rank)
    for world in (2, 8)
    for mode in ("current", "outside")
    for existing in (False, True)
    for fault in ("stage", "commit")
    for rank in range(world)
]


@pytest.mark.parametrize("world,mode,existing,fault,failed_rank", CASES)
def test_transaction_failure_on_each_rank(
    tmp_path, world, mode, existing, fault, failed_rank
):
    _run(tmp_path, mode, fault, existing, world, failed_rank)


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("mode", ["current", "outside"])
@pytest.mark.parametrize("existing", [False, True])
def test_healthy_transaction_publishes_after_readiness(tmp_path, world, mode, existing):
    _run(tmp_path, mode, "none", existing, world, world - 1)


@pytest.mark.parametrize("world", [2, 8])
def test_ready_result_failure_rolls_back_before_publication(tmp_path, world):
    _run(tmp_path, "current", "ready", True, world, world - 1)


@pytest.mark.parametrize("site", ["writer", "coordination"])
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_deferred_save_finalization_releases_locally_cleaned_state(
    tmp_path, monkeypatch, site, cleanup_error
):
    """Deferring the error vote must still acknowledge completed local cleanup."""
    from concurrent.futures import Future

    rank = _rank()
    destination = str(tmp_path / "checkpoint")
    snapshot, reservation = tmp_path / "snapshot", tmp_path / "reservation"
    snapshot.mkdir()
    reservation.mkdir()
    writer = Future()
    if site == "writer":
        writer.set_exception(MemoryError("injected snapshot writer failure"))
    else:
        writer.set_result(())
    prepared = _checkpoint._PreparedSave(
        0, snapshot, reservation, Path(destination), {}, (), None, writer=writer
    )
    rank._prepared_checkpoint_saves[destination] = prepared
    monkeypatch.setattr(_checkpoint, "_ensure_finalize_group", lambda trainer: None)
    monkeypatch.setattr(_checkpoint, "_finish", lambda *a: None)
    real_gather, real_cleanup = _checkpoint._gather, _checkpoint._cleanup_paths

    def gather(value, group=None):
        if site == "coordination" and value == (None, None):
            raise MemoryError("injected cleanup coordination failure")
        return real_gather(value, group)

    def cleanup(paths):
        result = real_cleanup(paths)
        assert result is None
        return MemoryError("injected local cleanup failure") if cleanup_error else None

    monkeypatch.setattr(_checkpoint, "_gather", gather)
    monkeypatch.setattr(_checkpoint, "_cleanup_paths", cleanup)
    with pytest.raises((MemoryError, BaseExceptionGroup)):
        with protocol.Driver(rank):
            _checkpoint.finish_checkpoint_save(rank, destination)
    assert not snapshot.exists() and not reservation.exists()
    assert not rank._checkpoint_finalizing_saves
    assert (destination in rank._prepared_checkpoint_saves) == cleanup_error
    if not cleanup_error:
        assert destination not in rank._checkpoint_save_outcomes
        assert rank._finalized_checkpoint_saves[destination].outcome == (
            "abort" if site == "writer" else "finish"
        )
