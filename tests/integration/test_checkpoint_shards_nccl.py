"""Optional GPU qualification of the same checkpoint shard protocol."""

import pytest
import torch
import torch.distributed as dist

from ..unit.test_checkpoint_shards import run_shards


@pytest.mark.parametrize("world", [2, 4])
def test_checkpoint_shards_nccl(tmp_path, world):
    if not dist.is_nccl_available() or torch.cuda.device_count() < world:
        pytest.skip(f"requires NCCL and {world} CUDA devices")
    run_shards(tmp_path, world, "nccl", torch.bfloat16)
