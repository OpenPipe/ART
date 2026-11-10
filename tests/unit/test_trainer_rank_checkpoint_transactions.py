"""Local rollback and publication readiness for checkpoint transactions."""

from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, cast
from unittest.mock import patch as mock_patch
import weakref

import pytest
from test_trainer_rank_calibration_seed import _rank
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
import torch.multiprocessing as mp

from art.megatron import lora as lora_module
from art.megatron.lora import LoRA
from art.trainer_rank import _calibration_protocol as protocol
from art.trainer_rank import _checkpoint
from art.trainer_rank._impl import _AdapterConfig, _CheckpointSlot


def checkpoint_rank(existing=True, gap=False):
    """CPU weights with production slot loading, compaction and validation."""
    rank = _rank()
    with (
        mock_patch.object(
            lora_module.ps, "get_expert_model_parallel_rank", return_value=0
        ),
        mock_patch.object(
            lora_module.ps, "get_tensor_model_parallel_world_size", return_value=1
        ),
    ):
        owner = LoRA(
            "layer.projection",
            2,
            2,
            1,
            1.0,
            torch.float32,
            torch.device("cpu"),
            allreduce=False,
        )
    rank.runtime.model[0].add_module("transaction_lora", owner)
    rank.runtime.model_identifier = "model"
    from types import SimpleNamespace

    rank.runtime.model_support_handler = SimpleNamespace(
        canonicalize_loaded_lora_state=lambda weights, model: dict(weights),
        from_vllm_lora_tensors=lambda weights, **kwargs: dict(weights),
    )
    config = dict(
        base_model_name_or_path="model",
        r=1,
        lora_alpha=1.0,
        target_modules=["projection"],
    )

    def weights(value):
        return {
            "layer.projection.lora_A.weight": torch.full((1, 2), value),
            "layer.projection.lora_B.weight": torch.full((2, 1), value),
        }

    names = [
        ("retained", 3.0),
        *([("adapter", 1.0)] if existing else []),
        ("other", 4.0),
    ]
    for name, value in names:
        ref = rank._slot_ref(name)
        assert owner.load_lora_slot(ref, weights(value), alpha=1.0, requires_grad=True)
        rank._checkpoint_slots[name] = _CheckpointSlot(
            params=tuple(owner.lora_slot_params(ref)),
            config=cast(_AdapterConfig, config),
            revision=7,
        )
    if gap:
        entries = list(owner._slot_keys.items())
        owner._slot_modules = torch.nn.ModuleDict(
            {
                f"slot_{2 + 3 * i}": owner._slot_modules[key]
                for i, (_ref, key) in reversed(list(enumerate(entries)))
            }
        )
        owner._slot_keys = {
            ref: f"slot_{2 + 3 * i}" for i, (ref, _key) in enumerate(entries)
        }
    return rank, owner, config, weights(2.0)


def _worker(
    index, directory, mode, fault, existing=True, world=2, failed_rank=1, gap=False
):
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
    rank, owner, config, adapter = checkpoint_rank(existing, gap)
    ref = rank._slot_ref("adapter")
    old_state = rank._checkpoint_slots.get("adapter")
    original_container, original_keys, original_registry = (
        owner._slot_modules,
        owner._slot_keys,
        rank._checkpoint_slots,
    )
    original_entries = tuple(original_container.items())
    original_key_entries = tuple(original_keys.items())
    original_registry_entries = tuple(original_registry.items())
    prior_slots = {
        name: owner._slot(rank._slot_ref(name)) for name in original_registry
    }
    references = []
    caught = []
    applying = False
    row: dict[str, Any] = {
        "injections": 0,
        "publications": 0,
        "ready_sequence": None,
        "votes": [],
        "wire": [],
        "gap": gap,
    }
    real_validate_config = rank._validate_loaded_checkpoint_config
    real_commit = _checkpoint._commit_slot
    real_vote = protocol.Protocol._vote
    real_reduce, real_gather = dist.all_reduce, c10d.all_gather
    real_decode = c10d._tensor_to_object

    def slot_checksums():
        result = {}
        for name in prior_slots:
            try:
                params = owner.lora_slot_params(rank._slot_ref(name))
                result[name] = hashlib.sha256(
                    b"".join(p.detach().numpy().tobytes() for p in params)
                ).hexdigest()
            except KeyError:
                result[name] = "missing-slot"
        return result

    def prior_lookups():
        try:
            return all(
                owner._slot(rank._slot_ref(name)) is slot
                for name, slot in prior_slots.items()
            )
        except KeyError:
            return False

    def checksum():
        try:
            if owner._slot(ref) is None:
                return None
            return hashlib.sha256(
                b"".join(
                    p.detach().numpy().tobytes() for p in owner.lora_slot_params(ref)
                )
            ).hexdigest()
        except KeyError:
            return "missing-slot"

    real_stage = rank._load_checkpoint_slot
    real_compact = rank._compact_lora_slot_keys
    before_all = slot_checksums()
    before = checksum()

    def inject(label):
        if index == failed_rank and not row["injections"] and fault == label:
            row["injections"] += 1
            raise MemoryError("injected checkpoint transaction failure: " + label)

    def stage_then_fail(name, *args, **kwargs):
        result = real_stage(name, *args, **kwargs)
        staged = owner._slot(rank._slot_ref(name))
        assert staged is not None
        references.extend(
            [weakref.ref(staged), *(weakref.ref(p) for p in staged.parameters())]
        )
        inject("stage")
        return result

    def compact():
        row["compactions"] = row.get("compactions", 0) + 1
        return real_compact()

    def validate_config(*args):
        real_validate_config(*args)
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

        if tensor.dtype == torch.float64 and tensor.numel() == 5:
            row["wire"][-1]["phase_tags"] = tensor[-2:].tolist()

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
            patch.setattr(rank, "_load_checkpoint_slot", stage_then_fail)
            patch.setattr(rank, "_compact_lora_slot_keys", compact)
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
            from safetensors.torch import save_file

            source_path = Path(directory) / f"source-{index}"
            source_path.mkdir()
            save_file(adapter, source_path / "adapter_model.safetensors")
            (source_path / "adapter_config.json").write_text(json.dumps(config))
            source = _checkpoint.PreparedCheckpoint(
                source_path,
                config,
                tuple(adapter),
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
                original_container=owner._slot_modules is original_container,
                original_keys=owner._slot_keys is original_keys,
                original_registry=rank._checkpoint_slots is original_registry,
                exact_module_order=tuple(owner._slot_modules.items())
                == original_entries,
                exact_key_order=tuple(owner._slot_keys.items()) == original_key_entries,
                exact_registry_order=tuple(rank._checkpoint_slots)
                == tuple(name for name, _slot in original_registry_entries)
                and all(
                    rank._checkpoint_slots.get(name) is slot
                    for name, slot in original_registry_entries
                ),
                all_prior_lookups=prior_lookups(),
                prior_checksums=slot_checksums(),
                before_prior_checksums=before_all,
                weight=cast(
                    torch.Tensor, owner._slot_modules[owner._slot_keys[ref]].A_T
                )[0, 0].item()
                if ref in owner._slot_keys
                and owner._slot_keys[ref] in owner._slot_modules
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


def _run(tmp_path, mode, fault, existing=True, world=2, failed_rank=1, gap=False):
    _rank()
    mp.set_forkserver_preload(
        ["art.trainer_rank._impl", "test_trainer_rank_calibration_seed"]
    )
    context = mp.start_processes(
        _worker,
        args=(str(tmp_path), mode, fault, existing, world, failed_rank, gap),
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
    assert all(r.get("compactions") == 1 for r in rows), rows
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
                r["original_container"]
                and r["original_keys"]
                and r["original_registry"]
                and r["exact_module_order"]
                and r["exact_key_order"]
                and r["exact_registry_order"]
                and r["all_prior_lookups"]
                and r["prior_checksums"] == r["before_prior_checksums"]
                for r in rows
            ), rows
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
            and r["staged_alive"] == 3
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


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fault", ["stage", "commit"])
def test_gapped_slot_rollback(tmp_path, world, existing, fault):
    _run(tmp_path, "current", fault, existing, world, world - 1, gap=True)


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("existing", [False, True])
def test_healthy_gapped_slot_compaction(tmp_path, world, existing):
    _run(tmp_path, "current", "none", existing, world, world - 1, gap=True)
