"""Actual CP2 graph residency and constrained complete-root placement."""

from dataclasses import asdict, replace
from datetime import timedelta
import gc
import json
import os
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("megatron.core")

from art.megatron.context_parallel import executor  # noqa: E402
from art.megatron.context_parallel.runtime import (
    prepare_megatron_context_parallel_state,
)  # noqa: E402
from art.megatron.context_parallel.types import (  # noqa: E402
    ContextParallelConfig,
    ParallelTopology,
)
from art.megatron.flex_attn import compiled  # noqa: E402
from art.megatron.runtime.compile_cache import configure_reusable_backward  # noqa: E402
from art.preprocessing.pack import PackedTensors  # noqa: E402
from art.trainer_rank._graphs import GraphCache  # noqa: E402
from art.trainer_rank._memory_policy import (  # noqa: E402
    ForwardMemoryCost,
    placement_cost,
)


@pytest.mark.skipif(
    os.environ.get("ART_GRAPH_GPU_TEST") != "1" or torch.cuda.device_count() < 2,
    reason="requires two reserved GPUs",
)
def test_cp_cpu_residency_constrains_complete_root(tmp_path):
    mp.spawn(
        _worker, args=(f"file://{tmp_path / 'cp-residency'}",), nprocs=2, join=True
    )


def _worker(rank, rendezvous):
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    configure_reusable_backward()
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
        device_id=device,
    )
    try:
        with (
            patch.object(compiled, "_FORCED_FLEX_BACKEND", "TRITON"),
            patch.object(
                compiled,
                "sparse_compiled_flex_attention",
                compiled.triton_sparse_compiled_flex_attention,
            ),
        ):
            _check(rank, device)
    finally:
        dist.destroy_process_group()


def _check(rank, device):
    length, heads, dim = 512, 2, 64
    micro = cast(
        PackedTensors,
        {
            "tokens": torch.arange(length)[None],
            "group_ids": torch.ones((1, length), dtype=torch.long),
            "parent_ids": torch.ones((1, length), dtype=torch.long),
            "input_pos": torch.arange(length)[None],
        },
    )
    state, plan, _, _ = prepare_megatron_context_parallel_state(
        micro=micro,
        topology=ParallelTopology(cp=2),
        config=ContextParallelConfig(
            planner_chunk_size=128, planner_owned_token_ms=1.0
        ),
        cp_group=dist.group.WORLD,
        cp_rank=rank,
        target_device=device,
    )
    indices = torch.tensor(
        [
            index
            for start, end, _local_start in plan.token_layout_index.ownership_ranges_by_rank[
                rank
            ]
            for index in range(start, end)
        ],
        device=device,
    )
    assert indices.numel() > 0
    assert indices.numel() == sum(plan.local_valid_lengths)
    executor.prepare_context_parallel_execution_state(state=state, device=device)
    torch.manual_seed(841)
    full = tuple(torch.randn(length, 1, heads, dim).to(device) for _ in range(3))
    local = tuple(value.index_select(0, indices) for value in full)
    weight = torch.nn.Parameter(torch.ones((), device=device))
    reference_weight = torch.ones((), device=device, requires_grad=True)
    q, k, v = (value[:, 0].transpose(0, 1) * reference_weight for value in full)
    scores = (q @ k.transpose(-1, -2)) * dim**-0.5
    mask = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    reference = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v).sum()
    (expected,) = torch.autograd.grad(reference, reference_weight)
    expected = expected.detach()
    del q, k, v, scores, mask, reference, reference_weight, full
    cache = GraphCache()

    def execute(inputs):
        q, k, v = (value * weight for value in inputs)
        return (
            executor.run_context_parallel(
                query=q,
                key=k,
                value=v,
                state=state,
                scale=dim**-0.5,
                enable_gqa=False,
                compile_enabled=True,
            ).sum(),
        )

    def run(retention, count=1):
        weight.grad = None
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        records = [
            cache.run(
                execute,
                local,
                retention=retention,
                output_device="cpu",
                cuda_devices=[rank],
                keep_on_device=lambda tensor: (
                    tensor.untyped_storage().data_ptr()
                    == weight.untyped_storage().data_ptr()
                ),
            )
            for _ in range(count)
        ]
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - baseline
        states = [cache.state(handle) for handle, _ in records]
        for handle, outputs in records:
            cache.backward(handle, tuple(torch.ones_like(value) for value in outputs))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        assert weight.grad is not None
        observed = weight.grad.detach().clone()
        dist.all_reduce(observed)
        torch.testing.assert_close(observed, expected * count, atol=3e-4, rtol=3e-4)
        assert not cache.handles()
        return dict(retained=retained, peak=peak, states=states)

    run("gpu")  # Compile and initialize communication before measurements.
    run("cpu")
    gpu, cpu = run("gpu"), run("cpu")
    print(
        "CP_RESIDENCY_PROBE="
        + json.dumps(
            dict(
                rank=rank,
                gpu_retained=gpu["retained"],
                cpu_retained=cpu["retained"],
                reported_cpu_state=asdict(cpu["states"][0]),
            )
        ),
        flush=True,
    )
    resident = getattr(cpu["states"][0], "non_offloadable_bytes", None)
    assert resident is not None and resident > 0
    assert cpu["retained"] <= cpu["states"][0].gpu_bytes + 64 * 1024
    assert cpu["retained"] < gpu["retained"]
    peak = int(max(gpu["peak"], cpu["peak"]) * 1.2) + 64 * 1024
    cost = ForwardMemoryCost(
        peak_bytes=peak,
        retained_bytes=max(int(gpu["states"][0].gpu_bytes * 1.2), int(resident * 1.2)),
        output_bytes=4,
        cpu_resident_bytes=int(resident * 1.2),
    )
    count = 8
    corrected = placement_cost(
        [cost] * count, backward_state="cpu", output_device="cpu"
    )
    old = placement_cost(
        [replace(cost, cpu_resident_bytes=0)] * count,
        backward_state="cpu",
        output_device="cpu",
    )
    cap = old.gpu_required_bytes + resident
    assert old.gpu_required_bytes <= cap < corrected.gpu_required_bytes
    replay_plan = placement_cost(
        [cost] * count, backward_state="replay", output_device="cpu"
    )
    assert replay_plan.gpu_required_bytes <= cap
    assert replay_plan.cpu_required_bytes <= 1 << 40
    replay = run("replay", count)
    assert replay["peak"] <= cap
    retained = placement_cost([cost] * count, backward_state="gpu", output_device="cpu")
    assert corrected.gpu_required_bytes < retained.gpu_required_bytes
    assert corrected.cpu_required_bytes <= 1 << 40
    partial = run("cpu", count)
    assert cap < partial["peak"] <= corrected.gpu_required_bytes
    print(
        "CP_RESIDENCY="
        + json.dumps(
            dict(
                rank=rank,
                gpu_retained=gpu["retained"],
                cpu_retained=cpu["retained"],
                reported_cpu_state=asdict(cpu["states"][0]),
                old_required=old.gpu_required_bytes,
                corrected_required=corrected.gpu_required_bytes,
                cap=cap,
                replay_peak=replay["peak"],
                partial_cpu_peak=partial["peak"],
                children=count,
            )
        ),
        flush=True,
    )
