"""Real checkpoint pack/unpack collectives; no Megatron or TE required."""

import math

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from art.megatron.context_parallel.checkpoint import (
    current_checkpoint_partition,
    shard_saved_activations,
)

# Pytest imports this file directly or through the integration package.
if __package__:
    from .trainer_rank_test_support import process_group, spawn_and_join
else:
    from trainer_rank_test_support import process_group, spawn_and_join


def _worker(rank, rendezvous, world, backend, dtype):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    with process_group(rank, rendezvous, world_size=world, backend=backend):
        torch.manual_seed(321)
        layers = 5
        model = torch.nn.Sequential(
            *[
                torch.nn.Sequential(torch.nn.Linear(7, 7), torch.nn.Tanh())
                for _ in range(layers)
            ]
        ).to(device=device, dtype=dtype)
        x = torch.randn(13, 1, 7, device=device, dtype=dtype, requires_grad=True)

        def gradients():
            result = []
            for parameter in (x, *model.parameters()):
                assert parameter.grad is not None
                result.append(parameter.grad.detach().clone())
            return result

        reference = model(x) + x.sin()
        reference.square().mean().backward()
        expected = gradients()
        model.zero_grad(set_to_none=True)
        x.grad = None
        with shard_saved_activations(
            dist.group.WORLD, shape=(x.shape[0], x.shape[1], x.shape[2]), key="proof"
        ) as stats:
            assert current_checkpoint_partition() == (world, tuple(x.shape))
            # This saves the same shape, but is not a checkpoint boundary.
            ordinary = x.sin()
            value = x
            for layer in model:
                value = checkpoint(layer, value, use_reentrant=True)
            value = value + ordinary
        assert current_checkpoint_partition() is None
        # Unpack hooks must survive scope exit, while recompute saves stay local.
        value.square().mean().backward()
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for actual, wanted in zip(gradients(), expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert x.numel() % world  # Exercise padding on every group size.
        assert stats.tensors == layers
        assert stats.original_bytes == layers * x.numel() * x.element_size()
        assert stats.restored_bytes == stats.original_bytes
        assert (
            stats.stored_bytes
            == layers * math.ceil(x.numel() / world) * x.element_size()
        )

        # The same hook also handles contiguous copies of non-contiguous inputs.
        view = x.transpose(0, 2)
        with shard_saved_activations(
            dist.group.WORLD,
            shape=(view.shape[0], view.shape[1], view.shape[2]),
            key="view",
        ) as view_stats:
            restored = checkpoint(lambda v: v.sin(), view, use_reentrant=True)
        restored.sum().backward()
        assert view_stats.tensors == 1
        assert view_stats.restored_bytes == view.numel() * view.element_size()

        for mismatch in ("ordinal", "identity"):
            key = f"rank-{rank}" if mismatch == "identity" else "fence"
            with shard_saved_activations(
                dist.group.WORLD, shape=(x.shape[0], x.shape[1], x.shape[2]), key=key
            ):
                a = checkpoint(model[0], x, use_reentrant=True)
                b = checkpoint(model[0], x, use_reentrant=True)
            selected = b if mismatch == "ordinal" and rank % 2 else a
            with pytest.raises(
                RuntimeError, match="mismatched identity or backward order"
            ):
                selected.sum().backward()


def run_shards(tmp_path, world, backend, dtype=torch.float32):
    spawn_and_join(
        _worker,
        (f"file://{tmp_path / 'rendezvous'}", world, backend, dtype),
        nprocs=world,
        timeout=120,
        failure="checkpoint shard workers did not finish",
    )


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_shards_gloo(tmp_path, world, dtype):
    if not dist.is_gloo_available():
        pytest.skip("PyTorch was built without Gloo")
    run_shards(tmp_path, world, "gloo", dtype)


def test_overlapping_scopes_rejected_and_reset(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    with shard_saved_activations(object(), shape=(3, 1, 7), key="outer"):
        with pytest.raises(RuntimeError, match="cannot overlap"):
            with shard_saved_activations(object(), shape=(3, 1, 7), key="inner"):
                pass
    assert current_checkpoint_partition() is None


@pytest.mark.parametrize("shape", [(0, 1, 7), (3, 2, 7), (3, 7)])
def test_invalid_shapes_rejected(shape):
    with pytest.raises(ValueError, match="activation shape"):
        with shard_saved_activations(object(), shape=shape, key="bad"):
            pass
