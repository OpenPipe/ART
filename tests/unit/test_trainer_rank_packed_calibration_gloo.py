"""Packed calibration completion and callback failure agree across CP peers."""

from datetime import timedelta
import json
from pathlib import Path
import time

import pytest
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_request_floor import prepared_rank
from test_trainer_rank_packed_calibration import marker_device, packed_requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import _memory as memory


def worker(index, world, directory, phase):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    row = {}
    try:
        with (
            pytest.MonkeyPatch.context() as patch,
            memory.fixed_slope_memory_profiles(),
        ):
            rank = prepared_rank("moe")
            marker_device(patch, rank)

            def fail(at):
                if phase == at and index == world - 1:
                    raise OSError("packed rate journal unavailable")

            def pending():
                fail("pending")

            def execute(rung):
                fail("intent")
                stream = rung.forward_micro_batches()
                if phase == "partial":
                    next(stream)
                    fail("partial")
                for batch in stream:
                    sum(o.target_logprobs.sum() for o in batch.outputs).backward()
                fail("completion")

            try:
                decisions = rank.calibrate_packed_target_rates(
                    request(1),
                    execute=execute,
                    check_pending=pending,
                )
                row["decisions"] = [d.candidate_packed_tokens for d in decisions]
                plan = rank._plan_flat_forward(packed_requests(9))
                check = rank._memory_check(plan)
                row["required"] = check.estimated_required_bytes
                assert (
                    check.fits
                    and check.estimated_required_bytes < memory._MEMORY_DEFERRED_BYTES
                )
                row["error"] = None
            except Exception as exc:
                row["error"] = type(exc).__name__
                row["message"] = str(exc)
            row["outer"] = rank._recovery_reduce(
                [float(row["error"] is not None)],
                op="MAX",
                sync_across_dp=True,
            )
    except BaseException as exc:
        row["unexpected"] = repr(exc)
    finally:
        Path(directory, f"rank{index}.json").write_text(json.dumps(row))
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize(
    "phase", ["healthy", "pending", "intent", "partial", "completion"]
)
def test_packed_rate_cohort_completion(tmp_path, world, phase):
    started = time.monotonic()
    processes = mp.start_processes(
        worker,
        args=(world, str(tmp_path), phase),
        nprocs=world,
        join=False,
        start_method="fork",
    )
    try:
        while not processes.join(timeout=1):
            assert time.monotonic() - started < 120, "packed calibration deadline"
    finally:
        for child in processes.processes:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(world)]
    assert all("unexpected" not in row for row in rows), rows
    assert all(row == rows[0] for row in rows), rows
    if phase == "healthy":
        assert rows[0]["error"] is None and rows[0]["decisions"] == [16384, 8192]
        assert rows[0]["outer"] == [0.0]
    else:
        assert rows[0]["error"] == "RuntimeError"
        assert rows[0]["outer"] == [1.0]
