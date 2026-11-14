"""Unavailable caller samples do not affect replicated wave execution."""

from dataclasses import replace
from datetime import timedelta
import json
import logging
from pathlib import Path
import time
from unittest.mock import patch

import pytest
from test_trainer_rank_active_memory import _rank
from test_trainer_rank_split_peak import _counter_split
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.context_parallel.types import ParallelTopology
from art.trainer_rank import ForwardInput, ForwardOutput, _impl
from art.trainer_rank import _memory as memory


def _sampling_worker(index, directory):
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    rank = _rank()
    rank._dp_rank_and_size = lambda: (0, 1)
    rank._topology_key = lambda: (1, 1, 2, 1)
    rank._topology = lambda: ParallelTopology(cp=2)
    rank._plan_group_rows = lambda plan: tuple(
        ((rank._physical_tokens(g.packed.tokens.numel()) + 1) // 2, g.grad_enabled)
        for g in plan.groups
    )
    rank._plan_retained_tokens = lambda plan: (plan.packed_tokens + 1) // 2
    rank._plan_head_workspace_bytes = lambda plan: 0
    rank._plan_hybridep_growth_bytes = lambda plan: 0
    rank._available_memory_bytes = lambda sample=None: 9_000_000_000_000_000_000
    request = ForwardInput(
        input_tokens=torch.arange(128), hidden_states=True, no_grad=True
    )
    plan = rank._plan_flat_forward([request])
    select = rank._select_next_micro_batch
    rank._select_next_micro_batch = lambda items, start, **kwargs: select(
        items[: start + 1], start, **kwargs
    )
    rank._submit_speculative_wave_planning = lambda *a, **k: None
    armed = False
    executed = 0
    peak = 0
    operations, warnings = [], []
    original_reduce, original_gather = dist.all_reduce, dist.all_gather_object

    def reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        operations.append((str(op), list(tensor.shape), str(tensor.dtype)))
        return original_reduce(tensor, op=op, **kwargs)

    def gather(objects, value, **kwargs):
        operations.append(("ALL_GATHER_OBJECT", [len(objects)], "object"))
        return original_gather(objects, value, **kwargs)

    def sample(*args):
        if index == 0 and armed:
            raise ValueError("caller sample unavailable")
        return peak

    def execute(candidate, **kwargs):
        nonlocal executed, peak
        executed += 1
        rank._peak_resets = executed
        rank._peak_reading = (executed, 100_000_000)
        peak = 100_000_000
        return [ForwardOutput(None, None, None, torch.ones((1, 8)))], 0

    class Handler(logging.Handler):
        def emit(self, record):
            if "caller memory sample" in record.getMessage():
                warnings.append(record.getMessage())

    handler = Handler()
    _impl.logger.addHandler(handler)
    rank._run_flat_plan_with_memory_tracking = execute
    result = {}
    try:
        with (
            patch.object(dist, "all_reduce", reduce),
            patch.object(dist, "all_gather_object", gather),
            patch.object(torch.cuda, "max_memory_allocated", sample),
            memory.fixed_slope_memory_profiles(),
        ):
            rank._update_memory_profile(
                replace(plan, packed_tokens=8192, logical_tokens=8192, output_bytes=0),
                100_000_000,
                retained_bytes=None,
            )
            previous = rank._memory_profiles[plan.signature]
            iterator = rank.forward_micro_batches([request] * 3, no_grad=True)
            yielded = 0
            try:
                for _ in iterator:
                    yielded += 1
                    armed = True
                    peak = 200_000_000 + 100_000_000 * yielded
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                iterator.close()
            profile = rank._memory_profiles[plan.signature]
            result.update(
                yielded=yielded,
                executed=executed,
                unchanged=profile == previous,
                callers=profile.caller_plans,
                rates=profile.rate_observations,
                suppressed=getattr(rank, "_suppressed_caller_memory_samples", 0),
                warnings=warnings,
            )
            try:
                result["outer"] = rank._recovery_reduce(
                    [float("error" in result)], op="MAX", sync_across_dp=True
                )
            except Exception as exc:
                result["outer_error"] = str(exc)
    finally:
        result["operations"] = operations
        Path(directory, f"rank{index}.json").write_text(json.dumps(result))
        _impl.logger.removeHandler(handler)
        dist.destroy_process_group()


def test_sampling_failure_keeps_all_three_cp_waves_matched(tmp_path):
    _rank()
    context = mp.start_processes(
        _sampling_worker,
        args=(str(tmp_path),),
        nprocs=2,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 30
    try:
        while not context.join(timeout=0.1):
            if time.monotonic() >= deadline:
                pytest.fail("sampling workers exceeded the deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    assert not any(row.get("error") or row.get("outer_error") for row in rows), rows
    assert all(
        row["yielded"] == row["executed"] == 3 and row["outer"] == [0.0] for row in rows
    )
    assert rows[0]["operations"] == rows[1]["operations"]
    assert rows[0]["unchanged"] and rows[0]["callers"] == 0
    assert rows[0]["suppressed"] == 3 and len(rows[0]["warnings"]) == 1
    assert "ValueError" in rows[0]["warnings"][0]
    assert not rows[1]["unchanged"] and rows[1]["callers"] == 3
    assert rows[1]["suppressed"] == 0 and rows[1]["warnings"] == []


def _iterator(monkeypatch):
    rank = _rank()
    rank._dp_rank_and_size = lambda: (0, 1)
    rank._available_memory_bytes = lambda sample=None: 9_000_000_000_000_000_000
    request = ForwardInput(
        input_tokens=torch.arange(8192), hidden_states=True, no_grad=True
    )

    def execute(plan, **kwargs):
        rank._peak_reading = (0, 100)
        return [ForwardOutput(None, None, None, torch.ones((1, 8)))], 0

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", execute)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 300_000_000)
    plan = rank._plan_flat_forward([request])
    rank._update_memory_profile(plan, 100_000_000, retained_bytes=None)
    iterator = rank.forward_micro_batches([request], no_grad=True)
    next(iterator)
    return rank, plan, iterator


@pytest.mark.parametrize("failure", ["sample", "metadata", "update"])
def test_failed_caller_update_restores_the_prior_profile(monkeypatch, caplog, failure):
    with memory.fixed_slope_memory_profiles():
        rank, plan, iterator = _iterator(monkeypatch)
        previous = rank._memory_profiles[plan.signature]
        reading = rank._peak_reading
        observation = {"peak": 7, "window_open": True}
        rank._planner_observation = observation

        def fail(*args, **kwargs):
            raise ValueError("sample/update unavailable")

        if failure == "sample":
            monkeypatch.setattr(torch.cuda, "max_memory_allocated", fail)
        elif failure == "metadata":
            monkeypatch.setattr(memory, "_rate_learning_eligible", fail)
        else:
            original = rank._update_memory_profile

            def update(*args, **kwargs):
                original(*args, **kwargs)
                raise ValueError("failed after profile publication")

            monkeypatch.setattr(rank, "_update_memory_profile", update)
        assert list(iterator) == []
        assert rank._memory_profiles[plan.signature] is previous
        assert rank._peak_reading == reading and observation["peak"] == 7
        assert rank._suppressed_caller_memory_samples == 1
        assert (
            sum(
                "caller memory sample (ValueError)" in r.message for r in caplog.records
            )
            == 1
        )


@pytest.mark.parametrize("field", ["bounded_caller_peak", "bounded_forward_peak"])
@pytest.mark.parametrize("enabled", [False, True])
def test_specialized_profile_construction_is_atomic(monkeypatch, field, enabled):
    with memory.fixed_slope_memory_profiles(enabled):
        rank, plan, iterator = _iterator(monkeypatch)
        iterator.close()
        key = memory._ScopedMemorySignature(**vars(plan.signature), **{field: True})
        plan = replace(plan, signature=key)
        rank._update_memory_profile(
            plan, 100_000_000, retained_bytes=None, caller_phase=True
        )
        previous = rank._memory_profiles[key]
        name = (
            "_CallerMemoryProfile"
            if field == "bounded_caller_peak"
            else "_BoundedMemoryProfile"
        ) + ("V2" if enabled else "")
        original = getattr(memory, name)

        class Broken(original):
            def __new__(cls, *args, **kwargs):
                raise ValueError("specialization unavailable")

        monkeypatch.setattr(memory, name, Broken)
        with pytest.raises(ValueError, match="specialization unavailable"):
            rank._update_memory_profile(
                plan, 200_000_000, retained_bytes=None, caller_phase=True
            )
        assert rank._memory_profiles[key] is previous


@pytest.mark.parametrize("failure", ["sample", "record", "key"])
def test_split_caller_failure_restores_floor_state(monkeypatch, caplog, failure):
    rank, requests, _ = _counter_split(monkeypatch)
    iterator = rank.forward_micro_batches([requests], yield_empty=True)
    next(iterator)
    profiles, floors, status = (
        dict(rank._memory_profiles),
        dict(rank._split_memory_floors),
        rank._split_memory_floor_status,
    )

    def fail(*args, **kwargs):
        raise ValueError("split sample unavailable")

    if failure == "sample":
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", fail)
    elif failure == "key":
        monkeypatch.setattr(rank, "_split_memory_key", fail)
    else:
        original = rank._record_split_memory_floor

        def record(*args, **kwargs):
            original(*args, **kwargs)
            raise ValueError("failed after floor publication")

        monkeypatch.setattr(rank, "_record_split_memory_floor", record)
    assert list(iterator) == []
    assert rank._memory_profiles == profiles and rank._split_memory_floors == floors
    assert rank._split_memory_floor_status == status
    assert rank._suppressed_caller_memory_samples == 1
    assert (
        sum("caller memory sample (ValueError)" in r.message for r in caplog.records)
        == 1
    )


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
@pytest.mark.parametrize("split", [False, True])
def test_process_control_exceptions_still_propagate(monkeypatch, exception, split):
    if split:
        rank, requests, _ = _counter_split(monkeypatch)
        iterator = rank.forward_micro_batches([requests], yield_empty=True)
        next(iterator)
    else:
        rank, _, iterator = _iterator(monkeypatch)
    original = exception("stop sampling")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fail)
    with pytest.raises(exception) as caught:
        next(iterator)
    assert caught.value is original
    assert rank._suppressed_caller_memory_samples == 0
    assert not rank._caller_memory_sample_warning_logged
