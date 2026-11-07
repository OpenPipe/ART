"""CPU regression for short CP rows leaving the last GDN island layer."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
from art.megatron.context_parallel.executor import run_context_parallel
from art.megatron.context_parallel.runtime import _build_runtime_plan
from art.megatron.context_parallel.types import (
    ArtContextParallelState,
    ContextParallelConfig,
    ParallelTopology,
)
from art.megatron.gdn import operator


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("length", [1, 3, 7, 8, 9, 16, 72, 513])
@pytest.mark.parametrize("datums", [1, 3])
def test_short_row_ownership(cp_size, length, datums):
    # Independent roots, including a long root followed by one-token roots.
    groups = torch.tensor([[0] * length + list(range(1, datums))])
    spec = build_prefix_tree_attention_spec(group_ids=groups, parent_ids=groups)
    plans = _build_runtime_plan(
        spec,
        topology=ParallelTopology(cp=cp_size),
        config=ContextParallelConfig(),
        original_seq_len=groups.shape[1],
    )
    counts = [p.local_valid_lengths[0] for p in plans]
    assert sum(counts) == groups.numel()
    assert (min(counts) > 0) == (groups.numel() >= cp_size)
    if groups.numel() < cp_size:
        assert counts == [0] * (cp_size - 1) + [groups.numel()]


@pytest.mark.parametrize("valid,physical", [(0, 1), (0, 4), (1, 1), (1, 4), (3, 4)])
@pytest.mark.parametrize("recompute", [False, True])
def test_restore_cp_padding_preserves_zero_vjp(valid, physical, recompute, monkeypatch):
    from megatron.core.tensor_parallel import random

    # Only CUDA RNG bookkeeping is replaced; use the installed production
    # CheckpointFunction, including its filter/zip in backward.
    monkeypatch.setattr(random, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(random, "_set_all_rng_states", lambda *args: None)
    monkeypatch.setattr(random, "_fork_rng", nullcontext)
    hidden = torch.arange(physical * 4.0).reshape(physical, 1, 4).requires_grad_()

    def layer(x):
        flat = x[:valid, 0] * 2
        return operator._restore_hidden_from_cp_flat(flat, tuple(x.shape))

    output = random.checkpoint(layer, False, hidden) if recompute else layer(hidden)
    expected = torch.zeros_like(hidden)
    expected[:valid] = hidden.detach()[:valid] * 2
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.sum().backward()
    expected_grad = torch.zeros_like(hidden)
    expected_grad[:valid] = 2
    torch.testing.assert_close(hidden.grad, expected_grad, rtol=0, atol=0)


def test_restore_empty_no_grad():
    with torch.no_grad():
        output = operator._restore_hidden_from_cp_flat(torch.empty(0, 4), (1, 1, 4))
    assert output.shape == (1, 1, 4)
    assert not output.requires_grad
    assert output.count_nonzero() == 0


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("valid", [1, 8, 513])
def test_long_physical_row_and_large_chunks_use_valid_token_ownership(cp_size, valid):
    groups = torch.full((1, 1024), -1, dtype=torch.long)
    groups[:, :valid] = 0
    spec = build_prefix_tree_attention_spec(group_ids=groups, parent_ids=groups)
    plans = _build_runtime_plan(
        spec,
        topology=ParallelTopology(cp=cp_size),
        config=ContextParallelConfig(planner_chunk_size=1048576),
        original_seq_len=1024,
    )
    counts = [p.local_valid_lengths[0] for p in plans]
    assert sum(counts) == valid
    assert (min(counts) > 0) == (valid >= cp_size)


@pytest.mark.parametrize("mode", ["forward", "backward", "recompute"])
def test_empty_attention_rank_keeps_checkpoint_edge(mode, monkeypatch):
    from megatron.core.tensor_parallel import random

    monkeypatch.setattr(random, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(random, "_set_all_rng_states", lambda *args: None)
    monkeypatch.setattr(random, "_fork_rng", nullcontext)
    groups = torch.zeros(1, 3, dtype=torch.long)
    config = ContextParallelConfig()
    spec = build_prefix_tree_attention_spec(group_ids=groups, parent_ids=groups)
    rank = _build_runtime_plan(
        spec, topology=ParallelTopology(cp=8), config=config, original_seq_len=3
    )[1]
    assert rank.local_valid_lengths == (0,)
    # Isolate the empty attention operator on one CPU process. The global plan
    # also has empty communication waves, covered by the gloo layout test.
    assert all(
        not any(s.kv_fetch_plan.send_splits + s.kv_fetch_plan.recv_splits)
        for s in rank.stage_plans
        if s.kv_fetch_plan is not None
    )
    rank = replace(
        rank,
        stage_plans=tuple(s for s in rank.stage_plans if s.is_local_stage),
        backward_stage_indices=(0,),
    )
    state = ArtContextParallelState(
        rank_plan=rank,
        cp_group=None,
        config=config,
        group_ids=groups,
        parent_ids=groups,
        input_pos=torch.arange(3)[None],
    )
    inputs = [torch.randn(1, 1, 2, 4, requires_grad=True) for _ in range(3)]

    def layer(q, k, v):
        return run_context_parallel(
            query=q,
            key=k,
            value=v,
            state=state,
            scale=0.5,
            enable_gqa=False,
            compile_enabled=True,
        )

    with torch.set_grad_enabled(mode != "forward"):
        output = (
            random.checkpoint(layer, False, *inputs)
            if mode == "recompute"
            else layer(*inputs)
        )
    assert output.count_nonzero() == 0
    if mode == "forward":
        assert not output.requires_grad
    else:
        output.sum().backward()
        for tensor in inputs:
            torch.testing.assert_close(
                tensor.grad, torch.zeros_like(tensor), rtol=0, atol=0
            )
