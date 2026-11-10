"""Incremental CPU review witnesses; no GPU execution or measured GPU claims."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from test_trainer_rank_calibration_seed import metadata_rank
from test_trainer_rank_subfloor_cp import requests, short_kernel
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import _impl
from art.trainer_rank import _memory as memory


def observed_rank():
    rank = metadata_rank("moe")
    anchor = rank._plan_flat_forward(requests([16384], True))
    rank._update_memory_profile(
        anchor, 2_500_000_000, retained_bytes=None, caller_phase=True
    )
    assert isinstance(
        rank._memory_profiles[anchor.signature], memory._CallerMemoryProfileV2
    )
    return rank, anchor.signature


@pytest.mark.parametrize("collect", [False, True])
def test_nonbinding_preview_keeps_budget_and_empirical_split_floor(collect):
    with memory.fixed_slope_memory_profiles():
        rank, signature = observed_rank()
        rank._memory_profiles[signature] = memory._block_cp_pair(
            rank._memory_profiles[signature],
            "unrelated-request",
            "unrelated-observation",
        )
        with short_kernel(), memory.fixed_slope_memory_profiles(collect):
            inputs = requests([256, 256], True)
            rows = [r.input_tokens for r in inputs]
            lower = [
                rank._split_chunk_lower_cost([r], [r.input_tokens], checkpoint=None)
                for r in inputs
            ]
            assert all(c.required == c.retained == 0 for c in lower)
            costs = [rank._plan_cost(rank._plan_flat_forward([r])) for r in inputs]
            exact = rank._split_required_memory(costs)
            assert 0 < exact < 300_000_000
            rank._available_memory_bytes = lambda sample=None: exact - 1
            denied, check = rank._admit_split_rung(
                ((0,), (1,)), inputs, rows, checkpoint=None
            )
            assert denied is None and not check.fits
            rank._available_memory_bytes = lambda sample=None: exact + 1
            accepted, check = rank._admit_split_rung(
                ((0,), (1,)), inputs, rows, checkpoint=None
            )
            assert isinstance(accepted, _impl._SplitForwardPlan) and check.fits
            key = rank._split_memory_key(accepted)
            assert key is not None
            rank._split_memory_floors[key] = exact + 1_000_000
            denied, check = rank._admit_split_rung(
                ((0,), (1,)), inputs, rows, checkpoint=None
            )
            assert denied is None and not check.fits
            assert check.estimated_required_bytes >= exact + 1_000_000


@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("length", [72, 16384])
def test_disabled_dominance_preserves_exact_equality_and_unknown_refusal(
    collect, length
):
    with memory.fixed_slope_memory_profiles():
        rank, signature = observed_rank()
        rank._memory_profiles[signature] = replace(
            rank._memory_profiles[signature], dominance_disabled=True
        )
        with memory.fixed_slope_memory_profiles(collect):
            inputs = requests([length, length], True)
            rank._available_memory_bytes = lambda sample=None: 71_000_000_000
            lower = [
                rank._split_chunk_lower_cost([r], [r.input_tokens], checkpoint=None)
                for r in inputs
            ]
            assert all(c.required == 0 for c in lower)
            result, check = rank._admit_split_rung(
                ((0,), (1,)), inputs, [r.input_tokens for r in inputs], checkpoint=None
            )
            if length == 16384:
                assert isinstance(result, _impl._SplitForwardPlan) and check.fits
            else:
                assert result is None and not check.fits
                assert check.estimated_required_bytes >= memory._MEMORY_DEFERRED_BYTES


def mixed_worker(index, directory, scenario):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with memory.fixed_slope_memory_profiles():
            rank, signature = observed_rank()
            if index == 0:
                profile = rank._memory_profiles[signature]
                rank._memory_profiles[signature] = (
                    replace(profile, dominance_disabled=True)
                    if scenario == "disabled"
                    else memory._block_cp_pair(
                        profile, "unrelated-request", "unrelated-observation"
                    )
                )
            rank._available_memory_bytes = lambda sample=None: (
                1 if scenario == "budget" and index == 0 else 300_000_000
            )
            operations = []
            original = dist.all_reduce

            def reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
                operations.append((str(op), list(tensor.shape)))
                return original(tensor, op=op, **kwargs)

            with short_kernel(), patch.object(dist, "all_reduce", reduce):
                inputs = requests([256, 256], True)
                result, check = rank._admit_split_rung(
                    ((0,), (1,)),
                    inputs,
                    [r.input_tokens for r in inputs],
                    checkpoint=None,
                )
            Path(directory, f"rank{index}.json").write_text(
                json.dumps(
                    {
                        "split": result is not None,
                        "fits": check.fits,
                        "required": check.estimated_required_bytes,
                        "ops": operations,
                    }
                )
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["admit", "budget", "disabled"])
def test_mixed_rank_revocation_flags_agree_on_exact_admission(tmp_path, scenario):
    mp.start_processes(
        mixed_worker,
        args=(str(tmp_path), scenario),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    a, b = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    assert a == b
    assert a["fits"] is (scenario == "admit")
    assert a["split"] is (scenario == "admit")
    assert len(a["ops"]) == (4 if scenario == "admit" else 6)
