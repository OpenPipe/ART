"""Real Gloo/NCCL proof of checkpoint storage sharding, including order fencing."""

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from art.megatron.context_parallel.checkpoint import shard_saved_activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if args.cuda else "gloo")
    torch.manual_seed(321)
    model = torch.nn.Sequential(
        *[
            torch.nn.Sequential(torch.nn.Linear(15, 15), torch.nn.Tanh())
            for _ in range(8)
        ]
    ).to(device)
    x = torch.randn(133, 1, 15, device=device, requires_grad=True)

    def gradients():
        result = []
        for parameter in (x, *model.parameters()):
            assert parameter.grad is not None
            result.append(parameter.grad.detach().clone())
        return result

    reference = model(x)
    reference.square().mean().backward()
    expected = gradients()
    model.zero_grad(set_to_none=True)
    x.grad = None
    with shard_saved_activations(
        dist.group.WORLD, shape=(133, 1, 15), key="proof"
    ) as stats:
        value = x
        for layer in model:
            value = checkpoint(layer, value, use_reentrant=True)
    # Unpack must work after the forward-only pack scope closes.
    value.square().mean().backward()
    observed = gradients()
    torch.testing.assert_close(value, reference, rtol=0, atol=0)
    for actual, wanted in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert stats.tensors == 8 and stats.restored_bytes == stats.original_bytes
    assert (
        stats.stored_bytes
        <= stats.original_bytes / dist.get_world_size()
        + stats.tensors * x.element_size()
    )
    with shard_saved_activations(dist.group.WORLD, shape=(133, 1, 15), key="fence"):
        a = checkpoint(model[0], x, use_reentrant=True)
        b = checkpoint(model[0], x, use_reentrant=True)
    try:
        (a if dist.get_rank() % 2 == 0 else b).sum().backward()
    except RuntimeError as error:
        assert "mismatched identity or backward order" in str(error)
    else:
        raise AssertionError("mismatched checkpoint order was not rejected")
    if dist.get_rank() == 0:
        print(
            dict(
                world=dist.get_world_size(),
                tensors=stats.tensors,
                original_bytes=stats.original_bytes,
                stored_bytes=stats.stored_bytes,
                exact_gradients=True,
                order_fence=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
