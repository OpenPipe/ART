from __future__ import annotations

from datetime import timedelta
import importlib
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parents[2] / "dev"))
_compare_outputs = importlib.import_module("trainer_rank_check")._compare_outputs
all_ranks_checked = importlib.import_module("trainer_rank_diag").all_ranks_checked


def _topk_output(values: torch.Tensor, tokens: torch.Tensor) -> dict[str, object]:
    return {
        "target": None,
        "topk_logprobs": values,
        "topk_tokens": tokens,
        "logits": None,
        "hidden": None,
    }


def test_topk_only_comparison_rejects_token_corruption() -> None:
    expected = _topk_output(torch.tensor([[-0.1, -0.2]]), torch.tensor([[1, 2]]))
    actual = _topk_output(torch.tensor([[-0.1, -0.2]]), torch.tensor([[1, 3]]))

    with pytest.raises(AssertionError, match="top-k tokens differ"):
        _compare_outputs([actual], [expected], tolerance=1e-4)


def test_topk_only_comparison_rejects_logprob_corruption() -> None:
    expected = _topk_output(torch.tensor([[-0.1, -0.2]]), torch.tensor([[1, 2]]))
    actual = _topk_output(torch.tensor([[-0.1, -2.0]]), torch.tensor([[1, 2]]))

    with pytest.raises(AssertionError, match="topk_logprobs mean_abs_pct"):
        _compare_outputs([actual], [expected], tolerance=1e-4)


def test_all_ranks_checked_terminates_on_one_rank_failure(tmp_path: Path) -> None:
    mp.spawn(
        _all_ranks_checked_worker,
        args=(2, f"file://{tmp_path / 'all-ranks'}"),
        nprocs=2,
        join=True,
    )


def _all_ranks_checked_worker(
    rank: int,
    world_size: int,
    init_method: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:

        def check() -> None:
            if rank == 1:
                raise AssertionError("injected rank-local failure")

        with pytest.raises(AssertionError, match="injected rank-local failure"):
            all_ranks_checked("injected", check)
        dist.barrier()
    finally:
        dist.destroy_process_group()
