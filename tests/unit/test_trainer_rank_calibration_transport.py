"""Interrupt and object-serialization failures stay in calibration phases."""

from datetime import timedelta
import inspect
import json
from pathlib import Path
import sys
import time

import pytest
from test_trainer_rank_calibration_seed import _rank, fake_device
from test_trainer_rank_calibration_trust_region import request
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
import torch.multiprocessing as mp

from art.trainer_rank import _calibration_protocol as protocol
from art.trainer_rank import _checkpoint, _memory


def _worker(index, world, directory, site, public):
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=world,
        timeout=timedelta(seconds=6),
    )
    rank, row = _rank(), {}
    rank._available_memory_bytes = lambda sample=None: 71_000_000_000
    injected = False
    lines, first = inspect.getsourcelines(protocol.Protocol.start)
    needle = "self.open = True" if site == "after_increment" else "self.sequence"
    target = next(
        first + i
        for i, line in enumerate(lines)
        if (
            line.strip() == needle
            if site == "after_increment"
            else line.strip() in ("self.sequence += 1", "self.sequence = sequence")
        )
    )

    def trace(frame, event, arg):
        nonlocal injected
        if (
            not injected
            and index == world - 1
            and event == "line"
            and frame.f_code is protocol.Protocol.start.__code__
            and frame.f_lineno == target
        ):
            injected = True
            raise KeyboardInterrupt("injected at phase increment boundary")
        return trace

    def fail(*args, **kwargs):
        nonlocal injected
        injected = True
        raise MemoryError("injected local object serialization failure")

    try:
        with (
            pytest.MonkeyPatch.context() as patch,
            _memory.fixed_slope_memory_profiles(),
        ):
            fake_device(patch, rank, 1_000_000)
            began = time.monotonic()
            try:
                if site in ("increment", "after_increment"):
                    if public:
                        with rank.calibration_rung(
                            [request(16384)], no_grad=True
                        ) as rung:
                            sys.settrace(trace)
                            list(rung.forward_micro_batches())
                    else:
                        with protocol.Driver(rank) as driver:
                            sys.settrace(trace)
                            driver.transition()
                elif site == "count":
                    with rank.calibration_rung([request(16384)], no_grad=True) as rung:
                        if index == world - 1:
                            patch.setattr(c10d, "_object_to_tensor", fail)
                        list(rung.forward_micro_batches())
                else:
                    with protocol.Driver(rank):
                        if index == world - 1:
                            patch.setattr(c10d, "_object_to_tensor", fail)
                        if site == "checkpoint":
                            _checkpoint._gather({"rank": index}, group=dist.group.WORLD)
                        else:
                            patch.setattr(
                                rank, "_local_lora_adapter_templates", lambda: {}
                            )
                            patch.setattr(
                                rank, "_checkpoint_group", lambda: dist.group.WORLD
                            )
                            rank._prepare_adapter_model(
                                "adapter", {}, canonicalized=True
                            )
            except BaseException as exc:
                row.update(error=type(exc).__name__, message=str(exc))
            finally:
                sys.settrace(None)
            row.update(injected=injected, elapsed=time.monotonic() - began)
            try:
                row["outer"] = rank._recovery_reduce(
                    [float("error" in row)], op="MAX", sync_across_dp=True
                )
            except BaseException as exc:
                row["outer_error"] = str(exc)
    finally:
        Path(directory, f"rank{index}.json").write_text(json.dumps(row))
        dist.destroy_process_group()


def _run(tmp_path, world, site, public=False):
    _rank()
    mp.set_forkserver_preload(
        ["art.trainer_rank._impl", "test_trainer_rank_calibration_seed"]
    )
    context = mp.start_processes(
        _worker,
        args=(world, str(tmp_path), site, public),
        nprocs=world,
        join=False,
        start_method="forkserver",
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=0.1):
            if time.monotonic() > deadline:
                pytest.fail("calibration transport exceeded the 120-second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(world)]
    assert rows[-1]["injected"], rows
    assert all(row.get("error") == "RuntimeError" for row in rows), rows
    assert all(
        row.get("message") == "Calibration failed on one or more ranks" for row in rows
    ), rows
    assert all(
        row.get("outer") == [1.0] and "outer_error" not in row for row in rows
    ), rows
    assert all(row["elapsed"] < 45 for row in rows), rows


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("public", [False, True], ids=["direct", "callback"])
@pytest.mark.parametrize("boundary", ["increment", "after_increment"])
def test_phase_increment_interrupt_is_coordinated(tmp_path, world, public, boundary):
    _run(tmp_path, world, boundary, public)


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("site", ["count", "checkpoint", "adapter"])
def test_object_serialization_error_precedes_transport(tmp_path, world, site):
    _run(tmp_path, world, site)
