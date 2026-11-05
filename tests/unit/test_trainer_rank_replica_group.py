"""Forward admission of one replica cannot communicate with idle replicas."""

from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp

from art.trainer_rank import TrainerRank

dist = cast(Any, torch_dist)


def _worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=5),
    )
    try:
        groups = [dist.new_group(ranks, backend="gloo") for ranks in ([0, 1], [2, 3])]
        trainer = TrainerRank.__new__(TrainerRank)
        trainer.device = torch.device("cpu")
        trainer._parallel_shape = SimpleNamespace(tp=1, cp=1, ep=1, etp=1)
        trainer.set_forward_replica_group(groups[rank // 2])
        if rank < 2:
            assert trainer._dp_rank_and_size() == (rank, 2)
            trainer._validate_replicated_top_level_count(1, yield_empty=False)
            assert not trainer._all_ranks_true(rank == 0)
            assert trainer._recovery_reduce(
                [rank + 1.0], op="SUM", sync_across_dp=True
            ) == [3.0]
            assert trainer._admission_outcome(rank + 1) == 1
            trainer._available_memory_bytes = lambda: 1024
            check = trainer._memory_check_required(128, sync_across_dp=True)
            assert check.estimated_required_bytes == 128
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_gloo_replica_admission_isolated_from_idle_nodes(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rdzv"),), nprocs=4, join=True)


def test_replica_group_rejects_inner_parallelism():
    trainer = TrainerRank.__new__(TrainerRank)
    trainer._parallel_shape = SimpleNamespace(tp=2, cp=1, ep=1, etp=1)
    with pytest.raises(ValueError, match="TP1/CP1/EP1"):
        trainer.set_forward_replica_group(object())
