"""Exercise empty-peer scans with real CPU collectives and a serial oracle."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.gdn import fla_cp as cp


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("world_size", [2, 4])
def test_empty_peer_summary_scans(world_size, tmp_path):
    mp.spawn(
        _summary_worker,
        args=(world_size, f"file://{tmp_path / 'gloo'}"),
        nprocs=world_size,
    )


def _summary_worker(rank, world_size, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=60),
    )
    try:
        half = world_size // 2
        masks = [
            [[r < half] for r in range(world_size)],
            [[r >= half] for r in range(world_size)],
            [[r in (0, world_size - 1)] for r in range(world_size)],
            [[r < half, r >= half] for r in range(world_size)],
        ]
        for active in masks:
            torch.manual_seed(732)
            roots = len(active[0])
            initial = torch.randn(roots, 1, 2, 3, requires_grad=True)
            matrices, offsets = [], []
            for flags in active:
                matrix = torch.randn(roots, 1, 2, 2) * 0.1 + torch.eye(2)
                offset = torch.randn(roots, 1, 2, 3)
                for root, enabled in enumerate(flags):
                    if not enabled:
                        matrix[root] = torch.eye(2)
                        offset[root] = 0
                matrices.append(matrix)
                offsets.append(offset)
            states = [initial]
            for matrix, offset in zip(matrices, offsets):
                states.append(matrix @ states[-1] + offset)
            summary = torch.cat((offsets[rank], matrices[rank]), dim=-1)
            prefix = cp._prefix_summary_exclusive(summary, dist.group.WORLD)
            local_initial = cp._scan_fwd_initial_state(prefix, initial.detach())
            torch.testing.assert_close(local_initial, states[rank])
            final = cp._broadcast_chain_final_state(
                matrices[rank] @ local_initial + offsets[rank], dist.group.WORLD
            )
            torch.testing.assert_close(final, states[-1])

            gradient = torch.randn_like(final)
            (expected,) = torch.autograd.grad((states[-1] * gradient).sum(), initial)
            backward_summary = torch.cat(
                (torch.zeros_like(offsets[rank]), matrices[rank].transpose(-1, -2)),
                dim=-1,
            )
            _, full = cp._suffix_summary_exclusive_and_full(
                backward_summary, dist.group.WORLD
            )
            external = cp._external_final_state_grad(
                gradient / world_size, final, group=dist.group.WORLD, enabled=True
            )
            torch.testing.assert_close(
                cp._scan_bwd_initial_state_grad(full, external), expected
            )
    finally:
        dist.destroy_process_group()
