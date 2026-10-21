from types import SimpleNamespace

import pytest
import torch

from art.megatron import segmented_gdn as gdn


@pytest.mark.parametrize("packed", [False, True])
def test_gate_reductions_keep_native_sequence_shapes_and_backward_scope(packed):
    calls = []

    def cumsum(value, chunk_size, **kwargs):
        calls.append((tuple(value.shape), kwargs))
        return value.cumsum(dim=1)

    class Function:
        @staticmethod
        def forward(ctx, value, **kwargs):
            ctx.value, ctx.kwargs = value, kwargs
            return module.chunk_local_cumsum(value, 64, **kwargs)

        @staticmethod
        def backward(ctx):
            return module.chunk_local_cumsum(ctx.value, 64, **ctx.kwargs)

    module = SimpleNamespace(
        ChunkGatedDeltaRuleFunction=Function, chunk_local_cumsum=cumsum
    )
    gdn._install(module)
    installed = module.chunk_local_cumsum
    gdn._install(module)
    assert module.chunk_local_cumsum is installed
    values = torch.arange(24.0).reshape(2, 3, 4)
    expected = values.cumsum(dim=1)
    kwargs = {}
    if packed:
        values = values.reshape(1, 6, 4)
        kwargs = {
            "cu_seqlens": torch.tensor([0, 3, 6]),
            "chunk_indices": torch.ones(2, 2),
        }
        expected = expected.reshape_as(values)
    ctx = SimpleNamespace()
    token = gdn._ACTIVE.set(True)
    try:
        actual = Function.forward(ctx, values, **kwargs)
    finally:
        gdn._ACTIVE.reset(token)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert [shape for shape, _ in calls] == [(1, 3, 4), (1, 3, 4)]
    calls.clear()
    torch.testing.assert_close(Function.backward(ctx), expected, atol=0, rtol=0)
    assert [shape for shape, _ in calls] == [(1, 3, 4), (1, 3, 4)]
    assert not gdn._ACTIVE.get()
    calls.clear()
    ordinary = SimpleNamespace()
    Function.forward(ordinary, values, **kwargs)
    Function.backward(ordinary)
    assert [shape for shape, _ in calls] == [tuple(values.shape)] * 2


def test_backward_scope_restores_after_failure():
    ctx = SimpleNamespace(_art_native_gdn_reductions=True)

    def fail(ctx):
        assert gdn._ACTIVE.get()
        raise RuntimeError("backward failure")

    with pytest.raises(RuntimeError, match="backward failure"):
        gdn._backward(fail, ctx)
    assert not gdn._ACTIVE.get()


def test_art_checkpoint_capture_restores_reduction_scope_on_worker_thread():
    from concurrent.futures import ThreadPoolExecutor

    from art.megatron.lora import _with_captured_lora_slot

    ordinary = _with_captured_lora_slot(lambda: gdn._ACTIVE.get())
    token = gdn._ACTIVE.set(True)
    try:
        captured = _with_captured_lora_slot(lambda: gdn._ACTIVE.get())
    finally:
        gdn._ACTIVE.reset(token)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(captured).result() is True
        assert pool.submit(gdn._ACTIVE.get).result() is False
        assert pool.submit(ordinary).result() is False
    assert not gdn._ACTIVE.get()
