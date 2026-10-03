"""Raw autograd context storage is nonoffloadable and must never be restored twice."""

import gc
import os

import pytest
import torch

from art._tensor_residency import record_resident_tensors
from art.trainer_rank._graphs import GraphCache

pytestmark = pytest.mark.skipif(
    os.environ.get("ART_GRAPH_GPU_TEST") != "1", reason="reserved GPU required"
)


class _RawContext(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.raw = value * 2
        ctx.save_for_backward(ctx.raw)
        record_resident_tensors((ctx.raw, ctx.raw[1:]))
        return ctx.raw.sum()

    @staticmethod
    def backward(ctx, *grad_outputs):
        (gradient,) = grad_outputs
        (saved,) = ctx.saved_tensors
        assert (
            saved.untyped_storage().data_ptr() == ctx.raw.untyped_storage().data_ptr()
        )
        return gradient.expand_as(saved) * 2


@pytest.mark.parametrize("retention", ["gpu", "cpu"])
def test_raw_context_residency_and_saved_alias_restore(retention):
    cache = GraphCache()
    inputs = torch.ones(4 * 1024**2, device="cuda", requires_grad=True)
    handle, (output,) = cache.run(
        lambda value: (_RawContext.apply(value),),
        inputs,
        retention=retention,
        output_device="cpu",
    )
    state = cache.state(handle)
    assert state.non_offloadable_bytes == inputs.numel() * 4 + 4
    cache.offload(handle)
    offloaded = cache.state(handle)
    assert offloaded.gpu_bytes == state.non_offloadable_bytes
    assert offloaded.cpu_bytes == offloaded.replay_bytes
    cache.backward(handle, (torch.ones_like(output),), retain_graph=True)
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    cache.evict(handle)
    torch.cuda.synchronize()
    assert before - torch.cuda.memory_allocated() >= inputs.numel() * 4
    assert cache.state(handle).gpu_bytes == 0
    cache.backward(handle, (torch.ones_like(output),))
    assert not cache.handles()
