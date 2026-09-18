"""Actual CP collectives and compiled attention against a dense manual oracle."""

from datetime import timedelta
import os
from pathlib import Path
from typing import cast
from unittest.mock import patch
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("megatron.core")

from art.megatron.context_parallel import executor  # noqa: E402
from art.megatron.context_parallel.runtime import (  # noqa: E402
    prepare_megatron_context_parallel_state,
)
from art.megatron.context_parallel.types import (  # noqa: E402
    ContextParallelConfig,
    ParallelTopology,
)
from art.megatron.flex_attn import compiled  # noqa: E402
from art.megatron.runtime.compile_cache import configure_reusable_backward  # noqa: E402
from art.preprocessing.pack import PackedTensors  # noqa: E402
from art.trainer_rank._graphs import GraphCache  # noqa: E402


def test_cp_retained_failure_releases_original_records(monkeypatch):
    contexts, saved = [], []

    def recorded(*, query, key, value, **kwargs):
        output = query * key + value
        saved.append(weakref.ref(output))
        return output, output.detach(), output.detach(), [{"stage_out": output}]

    def fail(*, replay_records, **kwargs):
        # Stage cleanup already consumed a per-backward dictionary when a
        # later operation fails. The untouched originals must also be released.
        replay_records[0].clear()
        raise RuntimeError("injected CP backward failure")

    monkeypatch.setattr(executor, "_run_context_parallel_forward_recorded", recorded)
    monkeypatch.setattr(executor, "_run_context_parallel_backward", fail)
    weight = torch.nn.Parameter(torch.tensor(2.0))

    def execute(_):
        output = executor.ArtContextParallelFn.apply(
            weight, weight, weight, None, None, 1.0, False, True, None, ()
        )
        contexts.append(output.grad_fn)
        return (output,)

    cache = GraphCache()
    handle, (output,) = cache.run(execute, ())
    with pytest.raises(RuntimeError, match="injected CP backward failure"):
        cache.backward(handle, (torch.ones_like(output),), retain_graph=True)
    assert cache.handles() == ()
    assert getattr(contexts[0], "replay_records") is None
    assert all(reference() is None for reference in saved)
    assert weight.grad is None


@pytest.mark.skipif(
    os.environ.get("ART_GRAPH_GPU_TEST") != "1" or torch.cuda.device_count() < 2,
    reason="requires two reserved GPUs",
)
@pytest.mark.parametrize("backend,dim", [("TRITON", 64), ("FLASH", 64), ("FLASH", 128)])
def test_cp_retained_backward_matches_dense_attention(
    tmp_path: Path, backend: str, dim: int
):
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", backend, dim),
        nprocs=2,
        join=True,
    )


def _worker(rank: int, init_method: str, backend: str, dim: int) -> None:
    # Exercise group ranks independently from CUDA device numbering.
    device = torch.device("cuda", 1 - rank)
    torch.cuda.set_device(device)
    configure_reusable_backward()
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
        device_id=device,
    )
    try:
        if backend == "TRITON":
            with (
                patch.object(compiled, "_FORCED_FLEX_BACKEND", "TRITON"),
                patch.object(
                    compiled,
                    "sparse_compiled_flex_attention",
                    compiled.triton_sparse_compiled_flex_attention,
                ),
            ):
                _check_repeated_backward(rank, device, backend, dim)
        else:
            _check_repeated_backward(rank, device, backend, dim)
    finally:
        dist.destroy_process_group()


def _check_repeated_backward(
    rank: int, device: torch.device, backend: str, dim: int
) -> None:
    length, heads = 512, 2
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
            for start, end, _ in plan.token_layout_index.ownership_ranges_by_rank[rank]
            for index in range(start, end)
        ],
        device=device,
    )
    assert indices.numel() > 0
    assert indices.numel() == sum(plan.local_valid_lengths)
    executor.prepare_context_parallel_execution_state(state=state, device=device)
    torch.manual_seed(841)
    dtype = torch.bfloat16 if backend == "FLASH" else torch.float32
    full = tuple(
        torch.randn(length, 1, heads, dim, dtype=dtype).to(device) for _ in range(3)
    )
    local = tuple(value.index_select(0, indices).requires_grad_() for value in full)
    refs = tuple(value.float().requires_grad_() for value in full)
    q, k, v = (value[:, 0].transpose(0, 1) for value in refs)
    scores = (q @ k.transpose(-1, -2)) * dim**-0.5
    mask = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    reference = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v).transpose(0, 1)[
        :, None
    ]
    with patch.object(
        executor, "_forward_stage_records", wraps=executor._forward_stage_records
    ) as forwards:
        output = executor.run_context_parallel(
            query=local[0],
            key=local[1],
            value=local[2],
            state=state,
            scale=dim**-0.5,
            enable_gqa=False,
            compile_enabled=True,
        )
        context = output.grad_fn
        assert context is not None
        records = getattr(context, "replay_records")
        saved_refs = [weakref.ref(record["stage_out"]) for record in records]
        del records
        atol, rtol = (0.012, 0.025) if backend == "FLASH" else (3e-5, 3e-4)
        torch.testing.assert_close(
            output.float(), reference.index_select(0, indices), atol=atol, rtol=rtol
        )
        for step, retain in enumerate((True, True, False)):
            torch.manual_seed(920 + step)
            cotangent = torch.randn(length, 1, heads, dim, dtype=dtype).to(device)
            expected = torch.autograd.grad(
                reference, refs, cotangent.float(), retain_graph=True
            )
            actual = torch.autograd.grad(
                output, local, cotangent.index_select(0, indices), retain_graph=retain
            )
            torch.cuda.synchronize(device)
            for observed, wanted in zip(actual, expected, strict=True):
                torch.testing.assert_close(
                    observed.float(),
                    wanted.index_select(0, indices),
                    atol=atol,
                    rtol=rtol,
                )
            assert forwards.call_count == 1
            if retain:
                assert getattr(context, "replay_records")
            else:
                assert getattr(context, "replay_records") is None
                assert all(ref() is None for ref in saved_refs)
            print(
                f"rank={rank} policy={backend} dim={dim} backward={step + 1} retain={retain} passed",
                flush=True,
            )
