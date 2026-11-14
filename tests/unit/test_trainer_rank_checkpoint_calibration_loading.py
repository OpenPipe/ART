"""Checkpoint loading defers rank-local failures to its calibration owner."""

from datetime import timedelta
import json
from pathlib import Path
import sys
import time
from typing import Any

import pytest
from test_trainer_rank_calibration_seed import _rank
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
import torch.multiprocessing as mp

from art.trainer_rank import _calibration_protocol as protocol
from art.trainer_rank import _checkpoint


def _worker(index, world, directory, mode, fault):
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=world,
        timeout=timedelta(seconds=6),
    )
    group = dist.new_group(backend="gloo")
    assert group is not dist.group.WORLD
    rank = _rank()
    row: dict[str, Any] = {"injections": 0, "target_gathers": [], "rollback_calls": 0}
    armed = False
    real_resize, real_empty, real_gather = (
        torch.Tensor.resize_,
        torch.empty,
        c10d.all_gather,
    )
    real_prepare = rank._prepare_adapter_model

    def inject():
        row["injections"] += 1
        current = protocol.active()
        assert current is not None
        row["open_at_failure"] = current.open
        raise MemoryError("injected one-time checkpoint payload allocation failure")

    def resize(tensor, *args, **kwargs):
        if (
            armed
            and index == world - 1
            and not row["injections"]
            and fault == "resize"
            and tensor.dtype == torch.uint8
            and sys._getframe(1).f_code.co_filename == protocol.__file__
        ):
            inject()
        return real_resize(tensor, *args, **kwargs)

    def empty(*args, **kwargs):
        if (
            armed
            and index == world - 1
            and not row["injections"]
            and fault == "storage"
            and kwargs.get("dtype") == torch.uint8
            and sys._getframe(1).f_code.co_filename == protocol.__file__
        ):
            inject()
        return real_empty(*args, **kwargs)

    def gather(outputs, tensor, **kwargs):
        if armed:
            assert kwargs.get("group") is group
            row["target_gathers"].append(str(tensor.dtype))
        return real_gather(outputs, tensor, **kwargs)

    def prepare(*args, **kwargs):
        nonlocal armed
        armed = mode == "load"
        try:
            return real_prepare(*args, **kwargs)
        finally:
            armed = False

    def stage(*args, **kwargs):
        nonlocal armed
        if mode == "transaction":
            armed = True
            try:
                real_prepare("adapter", {}, canonicalized=True)
            finally:
                armed = False
        return ()

    def rollback(*args, **kwargs):
        row["rollback_calls"] += 1
        pytest.fail(
            "a failed calibration must reach its owner before distributed rollback"
        )

    def local_action():
        if fault != "none" and index == world - 1:
            row["injections"] += 1
            raise MemoryError("injected non-calibration action failure")
        return "ok"

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.Tensor, "resize_", resize)
            patch.setattr(torch, "empty", empty)
            patch.setattr(dist, "all_gather", gather)
            patch.setattr(c10d, "all_gather", gather)
            began = time.monotonic()
            try:
                if mode == "outside":
                    row["result"] = _checkpoint._phase(
                        local_action, "local action", group
                    )
                else:
                    patch.setattr(_checkpoint, "_ensure_group", lambda trainer: group)
                    patch.setattr(rank, "_checkpoint_group", lambda: group)
                    patch.setattr(rank, "_local_lora_adapter_templates", lambda: {})
                    patch.setattr(
                        rank,
                        "_validate_checkpoint_adapter_config",
                        lambda *a, **k: {
                            "base_model_name_or_path": "model",
                            "lora_alpha": 1.0,
                        },
                    )
                    patch.setattr(
                        rank.runtime, "model_identifier", "model", raising=False
                    )
                    patch.setattr(
                        rank.runtime.model_support_handler,
                        "canonicalize_loaded_lora_state",
                        lambda state, model: state,
                        raising=False,
                    )
                    patch.setattr(rank, "_guard_slot_can_load", lambda *a: None)
                    patch.setattr(_checkpoint, "_load_adapter", lambda *a: {})
                    patch.setattr(rank, "_prepare_adapter_model", prepare)
                    patch.setattr(rank, "_load_checkpoint_slot", stage)
                    patch.setattr(
                        rank, "_validate_checkpoint_consistency", lambda *a: ()
                    )
                    patch.setattr(
                        rank, "_validate_loaded_checkpoint_config", lambda *a: None
                    )
                    patch.setattr(_checkpoint, "_commit_slot", lambda *a: None)
                    patch.setattr(_checkpoint, "_rollback_load", rollback)
                    source = _checkpoint.PreparedCheckpoint(
                        Path(directory),
                        {"base_model_name_or_path": "model", "lora_alpha": 1.0},
                        (),
                        None,
                        "synthetic-digest",
                    )
                    with protocol.Driver(rank):
                        _checkpoint.load_checkpoint(rank, source, "adapter")
                    row["committed"] = "adapter" in rank._checkpoint_slots
            except BaseException as exc:
                row.update(
                    error=type(exc).__name__,
                    message=str(exc),
                    cause=repr(exc.__cause__),
                )
            row.update(
                elapsed=time.monotonic() - began, active=protocol.active() is not None
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


def _run(tmp_path, world, mode, fault):
    _rank()
    mp.set_forkserver_preload(
        ["art.trainer_rank._impl", "test_trainer_rank_calibration_seed"]
    )
    context = mp.start_processes(
        _worker,
        args=(world, str(tmp_path), mode, fault),
        nprocs=world,
        join=False,
        start_method="forkserver",
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=0.1):
            assert time.monotonic() < deadline, (
                "checkpoint calibration exceeded the 120-second deadline"
            )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(world)]
    failed = fault != "none"
    assert all(
        not row["active"]
        and "outer_error" not in row
        and row["outer"] == [float(failed)]
        for row in rows
    ), rows
    assert all(row["elapsed"] < 45 for row in rows), rows
    assert [row["injections"] for row in rows] == [0] * (world - 1) + [int(failed)], (
        rows
    )
    if failed:
        if mode == "outside":
            assert rows[-1]["error"] == "MemoryError", rows
            assert all(
                row["error"] == "RuntimeError"
                and "Another rank failed to local action" in row["message"]
                for row in rows[:-1]
            ), rows
        else:
            assert rows[-1]["open_at_failure"], rows
            assert all(
                row.get("error") == "RuntimeError"
                and row.get("message") == "Calibration failed on one or more ranks"
                for row in rows
            ), rows
            assert all(
                row["target_gathers"] == ["torch.int64"] and row["rollback_calls"] == 0
                for row in rows
            ), rows
    else:
        assert all("error" not in row for row in rows), rows
        if mode == "outside":
            assert all(row["result"] == "ok" for row in rows), rows
        else:
            assert all(
                row["committed"]
                and row["target_gathers"] == ["torch.int64", "torch.uint8"]
                for row in rows
            ), rows


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("mode", ["load", "transaction"])
@pytest.mark.parametrize("fault", ["resize", "storage", "none"])
def test_checkpoint_loader_obeys_the_active_calibration_owner(
    tmp_path, world, mode, fault
):
    _run(tmp_path, world, mode, fault)


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("fault", ["storage", "none"])
def test_checkpoint_error_exchange_outside_calibration_is_unchanged(
    tmp_path, world, fault
):
    _run(tmp_path, world, "outside", fault)
