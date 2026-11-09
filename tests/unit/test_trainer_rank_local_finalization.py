"""Local cleanup never enters a process group or replaces caller failures."""

from dataclasses import replace
from datetime import timedelta
import gc
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch
import weakref

import pytest
from test_trainer_rank_active_memory import _rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.context_parallel.types import ParallelTopology
from art.trainer_rank import ForwardInput, ForwardOutput
from art.trainer_rank import _memory as memory


def _gc_worker(index, cp, tp, dp, directory, case):
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=cp * tp * dp,
        timeout=timedelta(seconds=5),
    )
    rank = _rank()
    rank._dp_rank_and_size = lambda: ((index // (cp * tp)) % dp, dp)
    rank._topology_key = lambda: (dp, tp, cp, 1)
    rank._topology = lambda: ParallelTopology(dp=dp, tp=tp, cp=cp)
    rank._plan_group_rows = lambda plan: tuple(
        (
            (rank._physical_tokens(g.packed.tokens.numel()) + cp - 1) // cp,
            g.grad_enabled,
        )
        for g in plan.groups
    )
    rank._plan_retained_tokens = lambda plan: (plan.packed_tokens + cp - 1) // cp
    rank._plan_head_workspace_bytes = lambda plan: 0
    rank._plan_hybridep_growth_bytes = lambda plan: 0
    rank._available_memory_bytes = lambda sample=None: 9_000_000_000_000_000_000

    def execute(plan, **kwargs):
        rank._peak_resets = 1
        rank._peak_reading = (1, 1_000_000)
        return [
            ForwardOutput(None, None, None, torch.ones((1, 8)))
            for _ in range(plan.request_count)
        ], 0

    rank._run_flat_plan_with_memory_tracking = execute
    operations, toggles, errors, holder = [], [], [], []
    original_reduce, original_scope = (
        dist.all_reduce,
        memory.fixed_slope_memory_profiles,
    )
    original_collect = gc.collect
    gc.disable()
    result = {}

    def all_reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        operations.append(
            (
                str(op),
                list(tensor.shape),
                str(tensor.dtype),
                threading.current_thread().name,
            )
        )
        if case == "inside_collective" and index == 0 and list(tensor.shape) == [1]:
            original_collect()
        return original_reduce(tensor, op=op, **kwargs)

    def scope(*args, **kwargs):
        toggles.append(threading.current_thread().name)
        return original_scope(*args, **kwargs)

    try:
        with (
            patch.object(torch.cuda, "max_memory_allocated", return_value=1_000_000),
            memory.fixed_slope_memory_profiles(),
            patch.object(dist, "all_reduce", all_reduce),
            patch.object(
                sys, "unraisablehook", lambda e: errors.append(str(e.exc_value))
            ),
        ):
            iterator = rank.forward_micro_batches(
                [
                    ForwardInput(
                        input_tokens=torch.arange(3), hidden_states=True, no_grad=True
                    )
                ],
                yield_empty=True,
            )
            next(iterator)
            reference = weakref.ref(iterator)
            operations.clear()
            with patch.object(memory, "fixed_slope_memory_profiles", scope):
                if case == "thread":
                    holder.append(iterator)
                    del iterator
                    if index == 0:
                        thread = threading.Thread(
                            target=holder.clear, name="gc-finalizer"
                        )
                        thread.start()
                        thread.join(timeout=10)
                        result["thread_alive"] = thread.is_alive()
                else:
                    cycle = [iterator]
                    cycle.append(cycle)
                    del iterator
                    if index == 0:
                        del cycle
                try:
                    result["outer"] = rank._recovery_reduce(
                        [0.0], op="MAX", sync_across_dp=True
                    )
                except Exception as exc:
                    result["outer_error"] = str(exc)
                result.update(
                    operations=list(operations),
                    toggles=list(toggles),
                    collected=reference() is None,
                )
                dist.destroy_process_group()
                holder.clear()
                if "cycle" in locals():
                    del cycle
                original_collect()
    except Exception as exc:
        result["setup_error"] = str(exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        result["unraisable"] = errors
        Path(directory, f"rank{index}.json").write_text(json.dumps(result))


@pytest.mark.parametrize("geometry", [(2, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("case", ["thread", "inside_collective"])
def test_gc_finalization_is_local(tmp_path, geometry, case):
    _rank()
    cp, tp, dp = geometry
    context = mp.start_processes(
        _gc_worker,
        args=(cp, tp, dp, str(tmp_path), case),
        nprocs=cp * tp * dp,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 35
    try:
        while not context.join(timeout=0.1):
            if time.monotonic() >= deadline:
                pytest.fail("GC workers exceeded the deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [
        json.loads((tmp_path / f"rank{i}.json").read_text())
        for i in range(cp * tp * dp)
    ]
    assert not any(
        row.get("setup_error") or row.get("outer_error") or row["unraisable"]
        for row in rows
    ), rows
    # The sole logged operation is the caller's outer MAX, on the main thread.
    expected = [["RedOpType.MAX", [1], "torch.float64", "MainThread"]]
    assert all(
        row["operations"] == expected and row["toggles"] == [] for row in rows
    ), rows
    assert rows[0]["collected"] and not rows[0].get("thread_alive", False)
    assert all(row["outer"] == [0.0] for row in rows)


@pytest.mark.parametrize("seed", ["warm", "borrowed"])
@pytest.mark.parametrize("action", ["throw", "unwind", "close", "resume"])
def test_caller_completion_never_exchanges_or_masks_failure(monkeypatch, seed, action):
    rank = _rank()
    rank._dp_rank_and_size = lambda: (0, 1)
    rank._available_memory_bytes = lambda sample=None: 9_000_000_000_000_000_000
    requests = [
        ForwardInput(input_tokens=torch.arange(512), hidden_states=True, no_grad=False)
    ]
    target = rank._plan_flat_forward(requests)
    source = replace(target, packed_tokens=8192, logical_tokens=8192, output_bytes=0)
    if seed == "borrowed":
        source = replace(
            source, signature=replace(source.signature, request_mix=("target:single",))
        )

    def execute(plan, **kwargs):
        rank._peak_resets = 1
        rank._update_peak_memory_profile(plan, 0, retained_after=plan.output_bytes)
        return [ForwardOutput(None, None, None, torch.ones((1, 8)))], 0

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", execute)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 500_000_000)
    with memory.fixed_slope_memory_profiles():
        rank._update_memory_profile(
            source, 1_000_000_000, retained_bytes=None, caller_phase=True
        )
        iterator = rank.forward_micro_batches(requests, no_grad=False)
        next(iterator)
        before = rank._memory_profiles[target.signature]
        calls = []

        def exchange(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("completion exchange failure")

        monkeypatch.setattr(rank, "_recovery_reduce", exchange)
        original = torch.cuda.OutOfMemoryError("caller allocation failed")
        if action == "throw":
            with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
                iterator.throw(original)
            assert caught.value is original
        elif action == "unwind":
            with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
                try:
                    raise original
                finally:
                    iterator.close()
            assert caught.value is original
        elif action == "close":
            iterator.close()
        else:
            assert list(iterator) == []
        iterator.close()
        assert calls == []
        after = rank._memory_profiles[target.signature]
        if action != "resume":
            assert after == before
        else:
            assert after.caller_plans == before.caller_plans + (seed == "warm")
