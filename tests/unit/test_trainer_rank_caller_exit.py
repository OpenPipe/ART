"""Real Gloo ordering around caller exceptions and deterministic iterator exit."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import time
from unittest.mock import patch

import pytest
from test_trainer_rank_active_memory import _rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.context_parallel.types import ParallelTopology
from art.trainer_rank import ForwardInput, ForwardOutput
from art.trainer_rank import _memory as memory


def _caller_exit_worker(index, cp, tp, dp, directory, case):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=cp * tp * dp,
        timeout=timedelta(seconds=6),
    )
    operations = []
    original_all_reduce = dist.all_reduce

    def all_reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        operations.append((str(op), list(tensor.shape), str(tensor.dtype)))
        return original_all_reduce(tensor, op=op, **kwargs)

    result = {}
    try:
        rank = _rank()
        rank._dp_rank_and_size = lambda: ((index // (cp * tp)) % dp, dp)
        rank._topology_key = lambda: (dp, tp, cp, 1)
        rank._topology = lambda: ParallelTopology(dp=dp, tp=tp, cp=cp)
        rank._plan_group_rows = lambda plan: tuple(
            (
                (rank._physical_tokens(group.packed.tokens.numel()) + cp - 1) // cp,
                group.grad_enabled,
            )
            for group in plan.groups
        )
        rank._plan_retained_tokens = lambda plan: (plan.packed_tokens + cp - 1) // cp
        rank._plan_head_workspace_bytes = lambda plan: 0
        rank._plan_hybridep_growth_bytes = lambda plan: 0
        # Explicit CPU collection budget; execution alone is synthetic.
        rank._available_memory_bytes = lambda sample=None: 9_000_000_000_000_000_000
        requests = [
            ForwardInput(
                input_tokens=torch.arange(512), hidden_states=True, no_grad=False
            )
        ]
        target = rank._plan_flat_forward(requests)
        source = replace(
            target,
            signature=replace(target.signature, request_mix=("target:single",)),
            packed_tokens=8192,
            logical_tokens=8192,
            output_bytes=0,
        )

        def execute(plan, **kwargs):
            rank._peak_resets = rank.__dict__.get("_peak_resets", 0) + 1
            rank._update_peak_memory_profile(plan, 0, retained_after=plan.output_bytes)
            return [
                ForwardOutput(None, None, None, torch.ones((1, 8), requires_grad=True))
                for _ in range(plan.request_count)
            ], 0

        rank._run_flat_plan_with_memory_tracking = execute
        rank._submit_speculative_wave_planning = lambda *args, **kwargs: None
        if case == "multiwave":
            requests *= 3
            select = rank._select_next_micro_batch
            rank._select_next_micro_batch = lambda items, start, **kwargs: select(
                items[: start + 1], start, **kwargs
            )
        with (
            patch.object(dist, "all_reduce", all_reduce),
            patch.object(torch.cuda, "max_memory_allocated", return_value=500_000_000),
            memory.fixed_slope_memory_profiles(),
        ):
            rank._update_memory_profile(
                source, 1_000_000_000, retained_bytes=None, caller_phase=True
            )
            # Public wrapper owns and closes its inner iterator exactly once.
            inner = rank._forward_micro_batches
            closes = []

            class ClosingIterator:
                def __init__(self, *args, **kwargs):
                    self.iterator = inner(*args, **kwargs)

                def __next__(self):
                    return next(self.iterator)

                def close(self):
                    closes.append(True)
                    self.iterator.close()

            rank._forward_micro_batches = ClosingIterator
            iterator = rank.forward_micro_batches(
                requests, no_grad=False, yield_empty=case == "multiwave"
            )
            caller_error = None
            yielded = 0
            try:
                for batch in iterator:
                    yielded += 1
                    if batch.outputs:
                        batch.outputs[0].hidden_states.sum().backward()
                    if case == "multiwave":
                        # The caller's matched status exchange owns the abort;
                        # iterator cleanup must add no collective before it.
                        if index == 0:
                            caller_error = "ValueError"
                            iterator.close()
                        result["outer"] = rank._recovery_reduce(
                            [float(caller_error is not None)],
                            op="MAX",
                            sync_across_dp=True,
                        )
                        break
                    if index == 0:
                        if case == "oom":
                            raise torch.cuda.OutOfMemoryError(
                                "caller allocation failed"
                            )
                        if case in ("value", "multiwave"):
                            raise ValueError("caller loss failed")
                        if case == "throw":
                            iterator.throw(ValueError("caller loss failed"))
                        if case == "outside_context":
                            # Close outside collection remains purely local.
                            with memory.fixed_slope_memory_profiles(False):
                                iterator.close()
                        break
            except Exception as exc:
                caller_error = type(exc).__name__
            finally:
                iterator.close()
                iterator.close()
            result.update(error=caller_error, yielded=yielded, closes=len(closes))
            result["callers"] = sum(
                profile.caller_plans
                for key, profile in rank._memory_profiles.items()
                if key != source.signature
            )
            result["rates"] = [
                profile.rate_observations
                for key, profile in rank._memory_profiles.items()
                if key != source.signature
            ]
            try:
                if case != "multiwave":
                    result["outer"] = rank._recovery_reduce(
                        [float(caller_error is not None)], op="MAX", sync_across_dp=True
                    )
            except Exception as exc:
                result["outer_error"] = str(exc)
    except Exception as exc:
        result["setup_error"] = str(exc)
    finally:
        result["operations"] = operations
        Path(directory, f"rank{index}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.parametrize("geometry", [(2, 1, 1), (2, 2, 2)], ids=["cp2", "cp2-tp2-dp2"])
@pytest.mark.parametrize(
    "case", ["oom", "value", "close", "throw", "outside_context", "multiwave"]
)
def test_caller_exit_precedes_outer_error_exchange(
    tmp_path, monkeypatch, geometry, case
):
    _rank()  # Prime optional imports before starting bounded distributed work.
    monkeypatch.setenv("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    cp, tp, dp = geometry
    context = mp.start_processes(
        _caller_exit_worker,
        args=(cp, tp, dp, str(tmp_path), case),
        nprocs=cp * tp * dp,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 35
    try:
        while not context.join(timeout=0.1):
            if time.monotonic() >= deadline:
                pytest.fail("caller exit did not complete within the timeout")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    rows = [
        json.loads((tmp_path / f"rank{i}.json").read_text())
        for i in range(cp * tp * dp)
    ]
    assert not any(row.get("setup_error") or row.get("outer_error") for row in rows), (
        rows
    )
    assert all(row["operations"] == rows[0]["operations"] for row in rows), rows
    assert len(rows[0]["operations"]) == 16
    assert rows[0]["operations"][-1] == ["RedOpType.MAX", [1], "torch.float64"]
    failed = case in ("oom", "value", "throw", "multiwave")
    assert all(row["outer"] == [float(failed)] and row["closes"] == 1 for row in rows)
    assert all(row["callers"] == 0 and not any(row["rates"]) for row in rows)
    assert rows[0]["error"] == (
        "OutOfMemoryError"
        if case == "oom"
        else "ValueError"
        if case in ("value", "throw", "multiwave")
        else None
    )
    assert all(row["error"] is None for row in rows[1:])
    assert all(row["yielded"] <= 1 for row in rows)
