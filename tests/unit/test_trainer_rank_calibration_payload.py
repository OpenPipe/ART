"""Payload allocation failures coordinate before the payload collective."""

from datetime import timedelta
import json
from pathlib import Path
import sys
import time
from typing import Any

import pytest
from test_trainer_rank_calibration_seed import _rank, fake_device
from test_trainer_rank_calibration_trust_region import request
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
import torch.multiprocessing as mp

from art.trainer_rank import _calibration_protocol as protocol
from art.trainer_rank import _checkpoint, _memory


def _worker(index, world, directory, site, fail):
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=world,
        timeout=timedelta(seconds=6),
    )
    rank = _rank()
    rank._available_memory_bytes = lambda sample=None: 71_000_000_000
    row: dict[str, Any] = {"injected": False, "gathers": [], "allocations": []}
    real_empty, real_gather = torch.empty, c10d.all_gather

    def empty(*args, **kwargs):
        caller = sys._getframe(1)
        if (
            caller.f_code.co_filename == protocol.__file__
            and kwargs.get("dtype") == torch.uint8
        ):
            current = protocol.active()
            row["allocations"].append(current is not None and current.open)
            if fail and index == world - 1:
                row["injected"] = True
                raise MemoryError("injected payload storage failure")
        return real_empty(*args, **kwargs)

    def gather(outputs, tensor, **kwargs):
        row["gathers"].append([str(tensor.dtype), list(tensor.shape)])
        return real_gather(outputs, tensor, **kwargs)

    try:
        with (
            pytest.MonkeyPatch.context() as patch,
            _memory.fixed_slope_memory_profiles(),
        ):
            fake_device(patch, rank, 1_000_000)
            patch.setattr(torch, "empty", empty)
            patch.setattr(dist, "all_gather", gather)
            patch.setattr(c10d, "all_gather", gather)
            began = time.monotonic()
            try:
                if site == "count":
                    with rank.calibration_rung([request(16384)], no_grad=True) as rung:
                        list(rung.forward_micro_batches())
                else:
                    with protocol.Driver(rank):
                        if site == "checkpoint":
                            value = b"" if index == 0 else {"data": "x" * index * 200}
                            result = _checkpoint._gather(value, group=dist.group.WORLD)
                            assert result == tuple(
                                b"" if i == 0 else {"data": "x" * i * 200}
                                for i in range(world)
                            )
                        else:
                            patch.setattr(
                                rank, "_local_lora_adapter_templates", lambda: {}
                            )
                            patch.setattr(
                                rank, "_checkpoint_group", lambda: dist.group.WORLD
                            )
                            assert (
                                rank._prepare_adapter_model(
                                    "adapter", {}, canonicalized=True
                                )
                                == {}
                            )
            except BaseException as exc:
                row.update(error=type(exc).__name__, message=str(exc))
            row["elapsed"] = time.monotonic() - began
            try:
                row["outer"] = rank._recovery_reduce(
                    [float("error" in row)], op="MAX", sync_across_dp=True
                )
            except BaseException as exc:
                row["outer_error"] = str(exc)
    finally:
        Path(directory, f"rank{index}.json").write_text(json.dumps(row))
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("site", ["count", "checkpoint", "adapter"])
@pytest.mark.parametrize("fail", [True, False], ids=["oom", "no-fault"])
def test_payload_allocation_is_in_phase(tmp_path, world, site, fail):
    _rank()
    mp.set_forkserver_preload(
        ["art.trainer_rank._impl", "test_trainer_rank_calibration_seed"]
    )
    context = mp.start_processes(
        _worker,
        args=(world, str(tmp_path), site, fail),
        nprocs=world,
        join=False,
        start_method="forkserver",
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=0.1):
            if time.monotonic() > deadline:
                pytest.fail("payload allocation exceeded the 120-second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(world)]
    if fail:
        assert rows[-1]["injected"], rows
        assert all(row.get("error") == "RuntimeError" for row in rows), rows
        assert all(
            row.get("message") == "Calibration failed on one or more ranks"
            for row in rows
        ), rows
        assert all(row["gathers"] == [["torch.int64", [1]]] for row in rows), rows
    else:
        assert all("error" not in row and not row["injected"] for row in rows), rows
        assert all(row["gathers"] == rows[0]["gathers"] for row in rows), rows
        assert rows[0]["gathers"][0] == ["torch.int64", [1]], rows
        assert rows[0]["gathers"][1][0] == "torch.uint8", rows
    assert all(row["allocations"] and all(row["allocations"]) for row in rows), rows
    assert all(
        row.get("outer") == [float(fail)] and "outer_error" not in row for row in rows
    ), rows
    assert all(row["elapsed"] < 45 for row in rows), rows
