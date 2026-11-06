"""Real CP ownership and collective admission with synthetic workspace pressure.

The pressure is model/backward workspace, not caller-loss storage. These CPU
policy counterexamples do not assert that any particular CUDA model OOMs.
"""

from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_moe_long_root_caller import classify, memory_rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.context_parallel.runtime import (
    context_parallel_rank_model_token_counts,
)
from art.megatron.context_parallel.types import ContextParallelConfig, ParallelTopology
from art.trainer_rank import _memory as m

GIB = 2**30


def counts(cp, length):
    metadata = torch.ones((1, length), dtype=torch.int64)
    return context_parallel_rank_model_token_counts(
        group_ids=metadata,
        parent_ids=metadata,
        topology=ParallelTopology(cp=cp),
        config=ContextParallelConfig(planner_owned_token_ms=0.0003),
        original_seq_len=length,
        build_gdn_execution_spec=False,
    )


def observe(rank, signature, length, peak):
    plan = SimpleNamespace(
        signature=signature,
        packed_tokens=length,
        active_logical_tokens=length,
        output_bytes=4 * length,
    )
    m._update_memory_profile(
        cast(Any, rank),
        cast(Any, plan),
        peak + plan.output_bytes,
        retained_bytes=None,
        caller_phase=True,
    )


def admission_worker(rank, cp, init_path, output_dir, demands, budgets):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=cp,
        init_method="file://" + init_path,
        timeout=timedelta(seconds=60),
    )
    try:
        owner = SimpleNamespace(
            device=torch.device("cpu"),
            _forward_memory_group=lambda: dist.group.WORLD,
            _available_memory_bytes=lambda: budgets[rank],
            _forward_collective_device=lambda: "cpu",
        )
        check = m._memory_check_required(cast(Any, owner), demands[rank])
        (Path(output_dir) / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "fits": check.fits,
                    "local_required": check.local_required_bytes,
                    "required": check.estimated_required_bytes,
                    "available": check.available_bytes,
                    "local_available": check.local_available_bytes,
                }
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp", (2, 4, 8))
def test_new_ownership_load_needs_covering_measurement(cp, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_checkpoint_layers", lambda *args: 40)
    short, long = cp * 512, cp * 512 + 128
    requested, measured = counts(cp, short), counts(cp, long)
    assert requested[-1] == 512 == 4 * measured[-1]
    demands = []
    for local_rows in measured:
        rank = memory_rank()
        rank._topology_key = lambda: (1, 1, cp, 1)
        rank._coefficient_version, rank._coefficient_table = 2, "independent"
        rank._slot_memory_shapes = lambda _: ()
        rank._parallel_shape = SimpleNamespace(cp=cp, ep=1, etp=1)
        rank._moe_layers, rank._moe_memory_supported = 40, True
        rank._dense_fc1_adapted = False
        signature = classify(rank)
        observe(rank, signature, 128, 16 * GIB)
        observe(rank, signature, 128, 16 * GIB)
        observe(rank, signature, long, max(16 * GIB, local_rows * 64 * 2**20))
        demands.append(
            m._estimate_required_memory_bytes_from_values(
                cast(Any, rank),
                packed_tokens=short,
                logical_tokens=short,
                output_bytes=4 * short,
                signature=signature,
                group_rows=((max(requested), True),),
            )
        )
    budgets = [100 * GIB] * (cp - 1) + [30 * GIB]
    assert max(16 * GIB, requested[-1] * 64 * 2**20) + 4 * short > budgets[-1]
    mp.start_processes(
        admission_worker,
        args=(cp, str(tmp_path / "gloo"), str(tmp_path), demands, budgets),
        nprocs=cp,
        join=True,
        start_method="fork",
    )
    results = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(cp)]
    assert not any(row["fits"] for row in results), (
        "Admitted a rank whose larger local workload has no covering measurement"
    )
