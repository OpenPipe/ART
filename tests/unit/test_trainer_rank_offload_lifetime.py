"""CPU storage-ownership probes; synthetic device labels do not qualify CUDA IO."""

import asyncio
import gc
from typing import Any, cast

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from art.trainer_rank import _graphs as graphs


@pytest.mark.parametrize("failure_type", [MemoryError, asyncio.CancelledError])
@pytest.mark.parametrize("stage", ["allocation", "copy"])
@pytest.mark.parametrize("late", [False, True], ids=["initial", "later"])
def test_offload_failure_storage_lifetime(monkeypatch, failure_type, stage, late):
    class SyntheticCuda(torch.Tensor):
        __torch_function__ = cast(Any, torch._C._disabled_torch_function_impl)

        @property
        def device(self):
            return torch.device("cuda")

    class Destination(torch.Tensor):
        __torch_function__ = cast(Any, torch._C._disabled_torch_function_impl)

        def copy_(
            self, other: torch.Tensor, non_blocking: bool = False
        ) -> torch.Tensor:
            if failing and attempts == (2 if late else 1) and stage == "copy":
                del self, other
                raise error from cause
            return super().copy_(other, non_blocking=non_blocking)

    class SavedTensor(graphs._SavedTensor):
        def __init__(self, tensor, *args):
            super().__init__(tensor, *args)
            if self.managed:
                self.tensor = tensor.as_subclass(SyntheticCuda)
                sources.append(StorageWeakRef(tensor.untyped_storage()))

    original_empty, original_like = torch.empty, torch.empty_like
    sources, destinations = [], []
    attempts = 0
    failing = True
    error, cause = failure_type("injected offload failure"), ValueError("copy cause")

    def empty(*args, **kwargs):
        if kwargs.get("device") == torch.device("cuda"):
            kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs).as_subclass(SyntheticCuda)
        return original_empty(*args, **kwargs)

    def empty_like(tensor, **kwargs):
        nonlocal attempts
        if kwargs.pop("pin_memory", False):
            attempts += 1
            if failing and attempts == (2 if late else 1) and stage == "allocation":
                del tensor
                raise error from cause
            result = original_like(tensor, **kwargs).as_subclass(Destination)
            destinations.append(StorageWeakRef(result.untyped_storage()))
            return result
        return original_like(tensor, **kwargs)

    weight = torch.nn.Parameter(torch.tensor(2.0))
    borrowed = StorageWeakRef(weight.untyped_storage())

    def execute(value):
        activation = None
        try:
            activation = weight * value + 1
            return (activation.square(),)
        finally:
            del activation, value

    cache = graphs.GraphCache()
    older_weight = torch.nn.Parameter(torch.tensor(3.0))
    older, (older_output,) = cache.run(lambda _: (older_weight.square(),), ())
    monkeypatch.setattr(graphs, "_SavedTensor", SavedTensor)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    inputs = torch.arange(1.0, 4.0, requires_grad=True)

    def keep_on_device(tensor):
        return tensor.data_ptr() == weight.data_ptr()

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        handle = None
        if late:
            handle, (output,) = cache.run(
                execute, inputs, keep_on_device=keep_on_device
            )
        with pytest.raises(failure_type) as failure:
            if handle is None:
                cache.run(
                    execute, inputs, retention="cpu", keep_on_device=keep_on_device
                )
            else:
                cache.offload(handle)
        assert failure.value is error and error.__cause__ is cause
        assert error.__traceback__ is not None
        assert cache.transfer_stats.offload_count == int(late)
        assert cache.transfer_stats.restore_count == 0
        assert not borrowed.expired() and weight.item() == 2
        if late:
            assert handle is not None
            assert cache.handles() == (older, handle)
            # A valid partial graph owns the earlier copy and the failed source.
            assert not destinations[0].expired()
            assert not sources[-1].expired()
            failing = False
            cache.backward(handle, (torch.ones_like(output),))
            torch.testing.assert_close(weight.grad, torch.tensor(68.0))
            weight.grad = None
        assert cache.handles() == (older,)
        assert sources and all(storage.expired() for storage in sources)
        assert all(storage.expired() for storage in destinations)
        failing = False
        handle, (output,) = cache.run(
            execute, inputs, retention="cpu", keep_on_device=keep_on_device
        )
        torch.testing.assert_close(output, torch.tensor([9.0, 25.0, 49.0]))
        cache.backward(handle, (torch.ones_like(output),), retain_graph=True)
        torch.testing.assert_close(weight.grad, torch.tensor(68.0))
        weight.grad = None
        assert cache.handles() == (older, handle)
        cache.backward(handle, (torch.ones_like(output),))
        torch.testing.assert_close(weight.grad, torch.tensor(68.0))
        assert cache.handles() == (older,)
        assert all(storage.expired() for storage in sources + destinations)
        assert not borrowed.expired() and weight.item() == 2
        torch.testing.assert_close(older_output, torch.tensor(9.0))
        cache.backward(older, (torch.ones_like(older_output),))
        torch.testing.assert_close(older_weight.grad, torch.tensor(6.0))
        assert cache.handles() == ()
    finally:
        if was_enabled:
            gc.enable()
