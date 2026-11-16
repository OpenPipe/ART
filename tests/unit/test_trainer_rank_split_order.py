"""Replicated split execution with rank-local memory profiles, on CPU/Gloo."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from test_trainer_rank_active_memory import _rank
import torch
import torch.distributed as dist

pytest.importorskip("megatron")

from art.megatron.context_parallel.types import ParallelTopology
from art.trainer_rank import ForwardInput, ForwardOutput, _impl


def _cp_rank(patch):
    rank = _rank()
    patch.setattr(rank, "_topology_key", lambda: (1, 1, 2, 1))
    patch.setattr(rank, "_topology", lambda: ParallelTopology(cp=2))
    patch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    # Synthetic memory observations isolate ordering from model memory floors.
    patch.setattr(
        rank,
        "_plan_group_rows",
        lambda p: tuple((p.packed_tokens, g.grad_enabled) for g in p.groups),
    )
    patch.setattr(rank, "_plan_retained_tokens", lambda p: p.packed_tokens // 2)
    patch.setattr(rank, "_plan_head_workspace_bytes", lambda p: 0)
    patch.setattr(rank, "_plan_hybridep_growth_bytes", lambda p: 0)
    patch.setattr(rank, "_available_memory_bytes", lambda sample=None: 3_000_000_000)
    return rank


def _requests(lengths=(8192, 8192)):
    return [
        ForwardInput(
            input_tokens=torch.arange(lengths[0]), hidden_states=True, no_grad=True
        ),
        ForwardInput(
            input_tokens=torch.arange(lengths[1]),
            target_tokens=torch.arange(lengths[1]),
            no_grad=True,
        ),
    ]


def _learn(rank, requests, peaks):
    plans = [rank._plan_flat_forward([request]) for request in requests]
    assert plans[0].signature != plans[1].signature
    for plan, peak in zip(plans, peaks, strict=True):
        rank._update_memory_profile(plan, peak, retained_bytes=plan.output_bytes)
    return plans


@pytest.mark.parametrize("lengths", [(8192, 8192), (8192, 12288), (12288, 8192)])
def test_split_order_depends_only_on_replicated_plan_data(monkeypatch, lengths):
    requests = _requests(lengths)
    orders = []
    local_cost_orders = []
    for peaks in [(1_000_000_000, 2_000_000_000), (2_000_000_000, 1_000_000_000)]:
        rank = _cp_rank(monkeypatch)
        plans = _learn(rank, requests, peaks)
        # packed_tokens counts global rows, before the CP=2 shard.
        assert tuple(plan.packed_tokens for plan in plans) == lengths
        costs = [rank._plan_cost(plan) for plan in plans]
        local_cost_orders.append(sorted(range(2), key=lambda i: -costs[i].ephemeral))
        split, check = rank._admit_split_rung(
            ((0,), (1,)),
            requests,
            [request.input_tokens for request in requests],
            checkpoint=None,
        )
        assert check.fits
        assert isinstance(split, _impl._SplitForwardPlan)
        orders.append(split.request_indices)
    assert local_cost_orders[0] != local_cost_orders[1]
    expected = tuple((i,) for i in sorted(range(2), key=lambda i: (-lengths[i], i)))
    assert orders == [expected, expected]


def test_split_order_and_collective_sequence_agree_across_cp_ranks(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    children = []
    logs = []
    try:
        for index in range(2):
            log = (tmp_path / f"rank-{index}.log").open("w")
            logs.append(log)
            env = os.environ | {
                "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                # Report mismatched tensor shapes instead of aborting in Gloo.
                "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
            }
            children.append(
                subprocess.Popen(
                    [sys.executable, __file__, str(index), str(tmp_path)],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for index, child in enumerate(children):
            assert child.wait(timeout=45) == 0, (
                tmp_path / f"rank-{index}.log"
            ).read_text()
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for log in logs:
            log.close()
    rows = [json.loads((tmp_path / f"rank-{i}.json").read_text()) for i in range(2)]
    assert rows[0].get("order") == rows[1].get("order") == [[0], [1]], rows
    assert rows[0]["operations"] == rows[1]["operations"], rows
    assert all(row["error"] is None for row in rows), rows
    assert rows[0]["costs"] != rows[1]["costs"]
    assert all(row["packed_tokens"] == [8192, 8192] for row in rows)
    assert rows[0]["operations"][-2:] == [
        ["RedOpType.SUM", [8192, 8], "torch.float32"],
        ["RedOpType.SUM", [8192], "torch.float32"],
    ]


def _worker(index, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    result = {"error": None}
    operations = []
    original = dist.all_reduce

    def reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        operations.append((str(op), list(tensor.shape), str(tensor.dtype)))
        return original(tensor, op=op, **kwargs)

    def execute(plan, **kwargs):
        request = plan.groups[0].items[0].request
        length = int(request.input_tokens.numel())
        positions = torch.arange(index, length, 2)
        shape = (len(positions), 8) if request.hidden_states else (len(positions),)
        # Production CP gather: hidden rows and target rows use different shapes.
        gathered = _impl._GatherContextParallelRows.apply(
            torch.ones(shape), positions, length, dist.group.WORLD
        )
        assert torch.equal(gathered, torch.ones_like(gathered))
        return [ForwardOutput(None, None, None, None)], None

    try:
        with pytest.MonkeyPatch.context() as patch:
            rank = _cp_rank(patch)
            patch.setattr(rank, "_run_flat_plan_with_memory_tracking", execute)
            patch.setattr(dist, "all_reduce", reduce)
            requests = _requests()
            plans = _learn(
                rank,
                requests,
                [1_000_000_000 if i == index else 2_000_000_000 for i in range(2)],
            )
            result["costs"] = [rank._plan_cost(plan).required for plan in plans]
            result["packed_tokens"] = [plan.packed_tokens for plan in plans]
            # Force the full search onto its split ladder using a real profile.
            unsplit = rank._plan_flat_forward(requests)
            rank._update_memory_profile(
                unsplit, 4_000_000_000, retained_bytes=unsplit.output_bytes
            )
            found = rank._find_admissible_forward(
                requests, checkpoint=None, refusal_prefix="split order regression"
            )
            assert not isinstance(found, _impl._ForwardRefusal), found
            split, check = found
            assert isinstance(split, _impl._SplitForwardPlan)
            assert check.fits
            result["order"] = split.request_indices
            rank._execute_split_plan_with_memory_tracking(
                split, check=check, context="split order regression"
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["operations"] = operations
        (directory / f"rank-{index}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


if __name__ == "__main__":
    _worker(int(sys.argv[1]), Path(sys.argv[2]))
