"""Calibration callback failures agree before the next model or caller vote."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import time
from typing import Any, cast

import pytest
from test_trainer_rank_calibration_seed import fake_device
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_request_floor import prepared_rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import _memory as memory


def _worker(index, directory, phase, failing, direct):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=6),
    )
    result, operations, rungs = {}, [], []
    finished = False
    original_reduce = dist.all_reduce

    def all_reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        operations.append((str(op), list(tensor.shape), str(tensor.dtype)))
        return original_reduce(tensor, op=op, **kwargs)

    def fail(at):
        if phase == at and index in failing:
            raise OSError("local calibration journal failure: " + at)

    try:
        with (
            memory.fixed_slope_memory_profiles(),
            pytest.MonkeyPatch.context() as patch,
        ):
            rank = prepared_rank("dense", include_floor=False)
            fake_device(patch, rank, 100_000_000)
            patch.setattr(dist, "all_reduce", all_reduce)

            def check_pending():
                fail("pending")

            def execute(rung):
                nonlocal finished
                rungs.append(rung)
                fail("intent")
                if phase == "skip" and index in failing:
                    return
                iterator = rung.forward_micro_batches()
                if phase == "partial":
                    next(iterator)
                    # Deliberately retain the iterator in the exception frame.
                    # The calibration scope must close it before its error vote.
                    fail("partial")
                list(iterator)
                finished = True
                fail("completion")
                fail("checkpoint")

            template = request(1)
            if phase == "inventory" and index in failing:
                template = replace(
                    template, input_tokens=template.input_tokens.reshape(1, 1)
                )
            if phase == "materialize":

                def inputs():
                    fail("materialize")
                    yield template

                selection = inputs()
            else:
                selection = [template]
            if phase in ("lookup", "postlookup"):
                original_floor = memory._cp_measured_request_floor

                def floor(*args, **kwargs):
                    if phase == "lookup" or finished:
                        fail(phase)
                    return original_floor(*args, **kwargs)

                patch.setattr(memory, "_cp_measured_request_floor", floor)
            error = None
            try:
                if direct:
                    with rank.calibration_rung(
                        selection, no_grad=True, seed_uncovered_cp_family=True
                    ) as rung:
                        execute(rung)
                else:
                    rank.calibrate_request_floors(
                        template, execute=execute, check_pending=check_pending
                    )
            except Exception as exc:
                error = type(exc).__name__
                result["message"] = str(exc)
                result["cause"] = (
                    type(exc.__cause__).__name__ if exc.__cause__ else None
                )
            result["error_type"] = error
            result["outer"] = rank._recovery_reduce(
                [float(error is not None)], op="MAX", sync_across_dp=True
            )
            result["rungs"] = len(rungs)
            result["expired"] = all(not rung._active for rung in rungs)
            result["permission_leaked"] = any(
                key in rank.__dict__
                for key in ("_calibration_seed_plan", "_calibration_coverage_guard")
            )
    except Exception as exc:
        result["setup_error"] = str(exc)
    finally:
        result["operations"] = operations
        Path(directory, f"rank{index}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.parametrize("failing", [(0,), (1,), (0, 1)], ids=["rank0", "rank1", "all"])
@pytest.mark.parametrize("phase", ["completion", "intent", "partial"])
@pytest.mark.parametrize("direct", [False, True], ids=["floor-pass", "rung"])
def test_callback_error_preserves_outer_error_exchange(
    tmp_path, monkeypatch, phase, failing, direct
):
    _run(tmp_path, monkeypatch, phase, failing, direct)


@pytest.mark.parametrize("failing", [(0,), (1,), (0, 1)], ids=["rank0", "rank1", "all"])
def test_pending_error_preserves_outer_error_exchange(tmp_path, monkeypatch, failing):
    _run(tmp_path, monkeypatch, "pending", failing, False)


@pytest.mark.parametrize(
    ("phase", "direct"),
    [
        ("inventory", False),
        ("lookup", False),
        ("postlookup", False),
        ("materialize", True),
        ("checkpoint", False),
        ("skip", False),
    ],
)
def test_local_preparation_error_preserves_outer_error_exchange(
    tmp_path, monkeypatch, phase, direct
):
    _run(tmp_path, monkeypatch, phase, (1,), direct)


def test_successful_floor_pass_preserves_outer_error_exchange(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "success", (), False)


def test_abandoned_calibration_scope_closes_without_a_collective(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank("dense", include_floor=False)
        fake_device(monkeypatch, rank, 100_000_000)
        scope = rank.calibration_rung(
            [request(1)], no_grad=True, seed_uncovered_cp_family=True
        )
        rung = scope.__enter__()
        iterator = rung.forward_micro_batches()
        next(iterator)
        monkeypatch.setattr(
            rank,
            "_recovery_reduce",
            lambda *a, **kw: pytest.fail("destruction introduced a collective"),
        )
        cast(Any, scope).gen.close()
        assert not rung._active
        assert "_calibration_seed_plan" not in rank.__dict__


def _run(tmp_path, monkeypatch, phase, failing, direct):
    prepared_rank("dense", include_floor=False)
    monkeypatch.setenv("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    mp.set_forkserver_preload(
        ["art.trainer_rank._impl", "art.megatron.context_parallel.types"]
    )
    context = mp.start_processes(
        _worker,
        args=(str(tmp_path), phase, failing, direct),
        nprocs=2,
        join=False,
        start_method="forkserver",
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=0.1):
            if time.monotonic() >= deadline:
                pytest.fail("calibration error exchange did not finish before timeout")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    assert not any(row.get("setup_error") for row in rows), rows
    assert rows[0]["operations"] == rows[1]["operations"], rows
    assert all(row["outer"] == [float(bool(failing))] for row in rows), rows
    assert all(row["expired"] and not row["permission_leaked"] for row in rows), rows
    assert all(
        row["error_type"] == ("RuntimeError" if failing else None) for row in rows
    ), rows
    if failing:
        assert all("Calibration" in row["message"] for row in rows), rows
    else:
        assert all(row["rungs"] == 62 for row in rows), rows
