"""CPU/gloo oracle for GDN island layout edges, including Megatron recompute.

The stateful CPU toy replaces CUDA GDN math. Planning, layout all-to-all,
physical padding restoration and CheckpointFunction are production code.
"""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
from torch.distributed import all_reduce, destroy_process_group, init_process_group
import torch.multiprocessing as mp

from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
from art.megatron.context_parallel.runtime import _build_runtime_plan
from art.megatron.context_parallel.types import ContextParallelConfig, ParallelTopology
from art.megatron.gdn.gdn_prefix_tree import (
    build_gdn_rank_execution_plan,
    parse_gdn_prefix_tree_segments,
)
from art.megatron.gdn.operator import (
    gdn_cp_attention_to_gdn_layout,
    gdn_cp_gdn_to_attention_layout,
)

from .distributed_init import file_init_method
from .oracles import ToyGdnConfig, ToyStatefulGdn


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_short_cp_gdn_island_matches_cp1(cp_size, tmp_path):
    mp.start_processes(
        _worker,
        args=(cp_size, file_init_method(tmp_path, f"short_cp{cp_size}")),
        nprocs=cp_size,
        join=True,
        start_method="spawn",
    )


def _root_blocks(model, hidden, lengths):
    outputs = []
    for part in hidden.split(lengths):
        output, _, _ = model.forward_segment(
            part,
            conv_initial=model.zero_conv_state(part),
            recurrent_initial=model.zero_recurrent_state(part),
        )
        outputs.append(output)
    # An empty rank must still participate in the layout exchange backward.
    return torch.cat(outputs) if outputs else hidden * 0


def _worker(rank, cp_size, init_method):
    from megatron.core.tensor_parallel import random

    random._get_all_rng_states = lambda: ()
    random._set_all_rng_states = lambda *args: None
    random._fork_rng = nullcontext
    torch.set_num_threads(1)
    init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=cp_size,
        timeout=timedelta(seconds=30),
    )
    try:
        for lengths in [
            (1,),
            (3,),
            (cp_size - 1,),
            (cp_size,),
            (cp_size + 1,),
            (16,),
            (72,),
            (1, 1),
            (2, 1),
            (64, 1, 1),
        ]:
            groups = torch.repeat_interleave(
                torch.arange(len(lengths)), torch.tensor(lengths)
            )[None]
            spec = build_prefix_tree_attention_spec(group_ids=groups, parent_ids=groups)
            attention = _build_runtime_plan(
                spec,
                topology=ParallelTopology(cp=cp_size),
                config=ContextParallelConfig(),
                original_seq_len=sum(lengths),
            )[rank]
            gdn_spec = parse_gdn_prefix_tree_segments(groups, groups)
            plan = build_gdn_rank_execution_plan(
                gdn_spec,
                cp_size=cp_size,
                cp_rank=rank,
                device="cpu",
                attention_token_layout_index=attention.token_layout_index,
            )
            # These short roots use local whole-segment GDN, without CP chains.
            indices = tuple(plan.gdn_token_indices)
            owned_lengths, start = [], 0
            for length in lengths:
                if start in indices:
                    assert all(i in indices for i in range(start, start + length))
                    owned_lengths.append(length)
                start += length
            local_indices = [
                i
                for r in attention.local_row_ranges
                if r is not None
                for i in range(r.start, r.end)
            ]
            for mode in ["forward", "backward", "recompute"]:
                torch.manual_seed(163)
                model = ToyStatefulGdn(ToyGdnConfig(hidden_size=4, conv_width=3))
                reference = deepcopy(model)
                full = torch.randn(sum(lengths), 4)
                hidden = full[local_indices].reshape(-1, 1, 4)
                if not local_indices:
                    hidden = full.new_zeros(1, 1, 4)
                hidden.requires_grad_()

                def island(x):
                    gdn_hidden, shape = gdn_cp_attention_to_gdn_layout(x, plan, None)
                    out = _root_blocks(model, gdn_hidden[:, 0], owned_lengths)
                    return gdn_cp_gdn_to_attention_layout(
                        out[:, None], plan, shape, None
                    )

                with torch.set_grad_enabled(mode != "forward"):
                    output = (
                        random.checkpoint(island, False, hidden)
                        if mode == "recompute"
                        else island(hidden)
                    )
                ref_hidden = full.detach().clone().requires_grad_()
                expected = _root_blocks(reference, ref_hidden, lengths)
                expected_local = expected[local_indices].reshape(-1, 1, 4)
                if not local_indices:
                    expected_local = expected.new_zeros(1, 1, 4)
                torch.testing.assert_close(output, expected_local, rtol=0, atol=0)
                if mode == "forward":
                    assert not output.requires_grad
                    continue
                output.square().sum().backward()
                expected.square().sum().backward()
                assert ref_hidden.grad is not None
                wanted = ref_hidden.grad[local_indices].reshape(-1, 1, 4)
                if not local_indices:
                    wanted = torch.zeros_like(hidden)
                torch.testing.assert_close(hidden.grad, wanted, rtol=1e-6, atol=1e-7)
                for parameter, ref_parameter in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    gradient = (
                        parameter.grad
                        if parameter.grad is not None
                        else torch.zeros_like(parameter)
                    )
                    all_reduce(gradient)
                    assert ref_parameter.grad is not None
                    torch.testing.assert_close(
                        gradient, ref_parameter.grad, rtol=1e-6, atol=1e-7
                    )
    finally:
        destroy_process_group()
