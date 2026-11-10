"""Real Gloo faults across calibration phases and explicit driver close."""

from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path
import time
from typing import Any, cast

import pytest
from test_trainer_rank_calibration_seed import _rank, fake_device
from test_trainer_rank_calibration_trust_region import request
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import _impl, _micro_batch_planner
from art.trainer_rank import _memory as memory

LOCAL_FAULTS = {
    "backward": "_backward_work",
    "harvest": "_backward_work",
    "telemetry": "_snapshot_planning_telemetry",
    "telemetry_reset": "_reset_planning_telemetry",
    "groups": "_group_active_request_indices",
    "tensor_inputs": "_forward_item",
    "model": "_run_flat_plan_with_memory_tracking",
    "telemetry_signature": "_telemetry_signature",
    "guard": "_guard_forward_collective",
    "checkpoint": "_ensure_checkpoint_slots_for",
    "handoff_preparation": "_cache_recovery_episode",
    "vote_tensor": "_forward_collective_device",
    "attach": "_backward_work",
    "swallowed": "_backward_work",
}


def _worker(index, world, directory, site):
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=world,
        timeout=timedelta(seconds=8),
    )
    row, events, phases = {}, [], []
    rank = _rank()
    rank._available_memory_bytes = lambda sample=None: 71_000_000_000
    real_reduce = dist.all_reduce

    def reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        events.append([str(op), list(tensor.shape), str(tensor.dtype)])
        if op == dist.ReduceOp.MAX and tensor.numel() == 5:
            values = tensor.tolist()
            if values[3] >= 1024 and values[3] == -values[4]:
                phases.append(int(values[3]))
        return real_reduce(tensor, op=op, **kwargs)

    def fail(*args, **kwargs):
        raise MemoryError("injected local calibration preparation failure")

    try:
        with (
            pytest.MonkeyPatch.context() as patch,
            memory.fixed_slope_memory_profiles(),
        ):
            executed, _ = fake_device(patch, rank, 1_000_000)
            patch.setattr(dist, "all_reduce", reduce)
            armed = False
            if site in ("before_entry", "exit_vote"):
                from art.trainer_rank import _calibration_protocol as protocol

                original = protocol._phase_payload
                injected = False

                def payload(stage, *args):
                    nonlocal injected
                    if (
                        index == world - 1
                        and not injected
                        and (
                            (site == "before_entry" and stage == "entry")
                            or (site == "exit_vote" and stage == "exit" and armed)
                        )
                    ):
                        injected = True
                        fail()
                    return original(stage, *args)

                patch.setattr(protocol, "_phase_payload", payload)
            error = None
            rung = None
            began = time.monotonic()
            try:
                if index == world - 1:
                    if site == "plan":
                        patch.setattr(rank, "_plan_flat_forward", fail)
                    elif site == "admission":
                        patch.setattr(rank, "_available_memory_bytes", fail)
                    elif site == "rung":
                        patch.setattr(_micro_batch_planner, "CalibrationRung", fail)
                scope = rank.calibration_rung([request(16384)], no_grad=True)
                with scope as rung:
                    if site == "mismatched_phase":
                        # The outer caller's phase cannot masquerade as this
                        # scope's exit, even though both use MAX float64.
                        if index == world - 1:
                            rank._recovery_reduce([0.0], op="MAX", sync_across_dp=True)
                        else:
                            scope.close()
                    if site == "collective_close":
                        iterator = rung.forward_micro_batches()
                        next(iterator)
                        scope.close()
                        iterator = None
                        continue_execution = False
                    else:
                        continue_execution = True
                    if index == world - 1:
                        if site in LOCAL_FAULTS:
                            if site in ("harvest", "attach"):

                                class BrokenWork:
                                    def harvest(self):
                                        if site == "harvest":
                                            fail()

                                    def attach(self, outputs):
                                        fail()

                                    def region(self, **kwargs):
                                        return nullcontext()

                                patch.setattr(
                                    rank, "_backward_work", lambda: BrokenWork()
                                )
                            elif site == "vote_tensor":
                                original_device = rank._forward_collective_device

                                def failed_device():
                                    patch.setattr(
                                        rank,
                                        "_forward_collective_device",
                                        original_device,
                                    )
                                    fail()

                                patch.setattr(
                                    rank, "_forward_collective_device", failed_device
                                )
                            else:
                                patch.setattr(rank, LOCAL_FAULTS[site], fail)
                        elif site == "materialize":
                            patch.setattr(_impl, "_materialize", fail)
                        elif site == "unflatten":
                            patch.setattr(_impl, "_unflatten", fail)
                        elif site == "serialization":
                            from torch.distributed import distributed_c10d as c10d

                            patch.setattr(c10d, "_object_to_tensor", fail)
                        elif site == "logging":
                            patch.setattr(_impl, "_telemetry_phase", fail)
                        elif site == "stats":
                            patch.setattr(_impl, "MicroBatchStats", fail)
                        elif site == "result":
                            patch.setattr(_impl, "MicroBatch", fail)
                        elif site == "after_entry":
                            fail()
                    if not continue_execution:
                        pass
                    elif site in ("local_close", "iterator_close", "iterator_throw"):
                        iterator = rung.forward_micro_batches()
                        next(iterator)
                        if index == world - 1:
                            before = len(events)
                            try:
                                if site == "iterator_close":
                                    cast(Any, iterator).close()
                                elif site == "iterator_throw":
                                    cast(Any, iterator).throw(GeneratorExit)
                                else:
                                    cast(Any, scope).gen.close()
                            finally:
                                row["close_collectives"] = len(events) - before
                        else:
                            list(iterator)
                    else:
                        iterator = rung.forward_micro_batches()
                        if site == "swallowed":
                            try:
                                list(iterator)
                            except Exception:
                                pass  # The failed phase must still fail scope exit.
                        else:
                            list(iterator)
                            if site == "completed_close" and index == world - 1:
                                before = len(events)
                                cast(Any, iterator).close()
                                row["close_collectives"] = len(events) - before
                    armed = True
            except BaseException as exc:
                error = exc
                row["error"] = type(exc).__name__
                row["message"] = str(exc)
            row["elapsed"] = time.monotonic() - began
            row["expired"] = rung is None or not rung._active
            row["executed"] = len(executed)
            try:
                row["outer"] = rank._recovery_reduce(
                    [float(error is not None)], op="MAX", sync_across_dp=True
                )
            except BaseException as exc:
                row["outer_error"] = str(exc)
    finally:
        row.update(events=events, phases=phases)
        Path(directory, f"rank{index}.json").write_text(json.dumps(row))
        dist.destroy_process_group()


def _run(tmp_path, world, site):
    _rank()
    mp.set_forkserver_preload(
        [
            "art.trainer_rank._impl",
            "test_trainer_rank_calibration_seed",
        ]
    )
    context = mp.start_processes(
        _worker,
        args=(world, str(tmp_path), site),
        nprocs=world,
        join=False,
        start_method="forkserver",
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=0.1):
            if time.monotonic() >= deadline:
                pytest.fail("calibration phase fault exceeded the 120-second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(world)]
    if site == "mismatched_phase":
        assert all(row.get("error") == "RuntimeError" for row in rows), rows
        assert all(
            row.get("message") == "Calibration phase mismatch" for row in rows
        ), rows
        assert all(row.get("outer") == [1.0] for row in rows), rows
        assert all(row["elapsed"] < 45 for row in rows), rows
        return
    if site in ("collective_close", "completed_close"):
        assert all("error" not in row and row["outer"] == [0.0] for row in rows), rows
        assert all(row["expired"] and row["elapsed"] < 45 for row in rows), rows
        if site == "completed_close":
            assert rows[-1]["close_collectives"] == 0, rows
        return
    assert all(row.get("error") == "RuntimeError" for row in rows), rows
    assert all(
        row.get("outer") == [1.0] and "outer_error" not in row for row in rows
    ), rows
    assert all("Calibration failed" in row["message"] for row in rows), rows
    assert all(row["expired"] and row["elapsed"] < 45 for row in rows), rows
    assert all(row["events"] == rows[0]["events"] for row in rows), rows
    if site in ("local_close", "iterator_close", "iterator_throw"):
        assert rows[-1]["close_collectives"] == 0, rows
    for row in rows:
        assert row["phases"], rows
        entries = [value for value in row["phases"] if value % 2 == 0]
        exits = [value for value in row["phases"] if value % 2]
        assert len(entries) == len(set(entries))
        assert exits == [value + 1 for value in entries], row


@pytest.mark.parametrize("site", ["backward", "telemetry", "local_close"])
def test_prior_collective_seams_are_coordinated(tmp_path, site):
    _run(tmp_path, 2, site)


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize(
    "site",
    [
        "before_entry",
        "after_entry",
        "backward",
        "harvest",
        "telemetry",
        "telemetry_reset",
        "groups",
        "tensor_inputs",
        "materialize",
        "unflatten",
        "model",
        "telemetry_signature",
        "guard",
        "checkpoint",
        "handoff_preparation",
        "vote_tensor",
        "logging",
        "exit_vote",
        "local_close",
        "iterator_close",
        "iterator_throw",
        "plan",
        "admission",
        "rung",
        "stats",
        "result",
        "attach",
        "swallowed",
        "serialization",
    ],
)
def test_calibration_fault_matrix(tmp_path, world, site):
    _run(tmp_path, world, site)


@pytest.mark.parametrize("world", [2, 8])
def test_explicit_collective_close_is_the_only_close_path(tmp_path, world):
    _run(tmp_path, world, "collective_close")


@pytest.mark.parametrize("world", [2, 8])
def test_completed_iterator_close_is_local_and_keeps_the_scope_owned(tmp_path, world):
    _run(tmp_path, world, "completed_close")


@pytest.mark.parametrize("world", [2, 8])
def test_phase_tag_rejects_an_outer_vote_in_the_callback_phase(tmp_path, world):
    _run(tmp_path, world, "mismatched_phase")
