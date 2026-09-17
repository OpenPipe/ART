"""Native FLA CP collectives must address peers by global rank.

Context-parallel groups are only ``range(cp_size)`` when TP = DP = 1. With
tensor parallelism the groups are strided (TP 2 on four ranks: {0, 2} and
{1, 3}); with data parallelism they are offset ({2, 3}). ``torch.distributed``
``src`` arguments are global ranks, so a CP-local index raises
``ValueError: Global rank N is not part of group`` (OpenPipe/ART#919).
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.gdn.fla_cp import (
    _broadcast_chain_final_state,
    _suffix_summary_exclusive_and_full,
)


@pytest.mark.parametrize(
    "layout",
    ("tp2_cp2", "dp2_cp2"),
)
def test_fla_cp_collectives_use_global_ranks(layout: str, tmp_path) -> None:
    mp.spawn(
        _worker,
        args=(layout, f"file://{tmp_path / 'init'}"),
        nprocs=4,
        join=True,
    )


def _cp_groups(layout: str) -> list[list[int]]:
    if layout == "tp2_cp2":
        # Megatron orders TP fastest: CP peers are two ranks apart.
        return [[0, 2], [1, 3]]
    if layout == "dp2_cp2":
        # Two CP groups of contiguous ranks; the second never contains rank 0.
        return [[0, 1], [2, 3]]
    raise AssertionError(layout)


def _worker(rank: int, layout: str, init_method: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        groups = [dist.new_group(ranks) for ranks in _cp_groups(layout)]
        members = next(r for r in _cp_groups(layout) if rank in r)
        group = groups[_cp_groups(layout).index(members)]
        owner = members[-1]

        # Chain final state: every member must receive the last member's state.
        final_state = torch.full((2, 3, 3), float(rank))
        received = _broadcast_chain_final_state(final_state, group)
        assert torch.equal(received, torch.full((2, 3, 3), float(owner)))

        # Suffix summaries: the full-chain summary is broadcast from the first
        # member; all members must agree on it.
        summary = torch.zeros((2, 3, 5))
        summary[..., :3] = torch.eye(3) * (1.0 + rank)
        summary[..., 3:] = float(rank)
        _exclusive, full = _suffix_summary_exclusive_and_full(summary, group)
        gathered = [torch.empty_like(full) for _ in members]
        dist.all_gather(gathered, full, group=group)
        for other in gathered:
            assert torch.equal(other, full)
    finally:
        dist.destroy_process_group()
